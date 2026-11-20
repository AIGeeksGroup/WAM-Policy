import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from wam_go2.control.policy import RamboPolicy  # noqa: E402


def _make_checkpoint(path, num_obs=405, num_actions=18, with_normalizer=True):
    torch.manual_seed(0)
    dims = [num_obs, 512, 256, 128]
    policy_dict = {}
    for k, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
        policy_dict[f"policy_latent_net.{2 * k}.weight"] = torch.randn(o, i) * 0.01
        policy_dict[f"policy_latent_net.{2 * k}.bias"] = torch.randn(o) * 0.01
    policy_dict["action_mean_net.weight"] = torch.randn(num_actions, 128) * 0.01
    policy_dict["action_mean_net.bias"] = torch.randn(num_actions) * 0.01
    policy_dict["log_std"] = torch.zeros(num_actions)
    ckpt = {"policy_dict": policy_dict, "optimizer_state_dict": {},
            "iteration": 1, "infos": None}
    if with_normalizer:
        ckpt["obs_normalizer"] = {
            "_mean": torch.randn(1, num_obs) * 0.1,
            "_var": torch.ones(1, num_obs),
            "_std": torch.ones(1, num_obs) * 2.0,
        }
    torch.save(ckpt, path)
    return ckpt


def _reference_forward(ckpt, obs):
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    if "obs_normalizer" in ckpt:
        n = ckpt["obs_normalizer"]
        x = (x - n["_mean"]) / (n["_std"] + 1e-2)
    pd = ckpt["policy_dict"]
    for k in (0, 2, 4):
        x = torch.nn.functional.elu(
            torch.nn.functional.linear(x, pd[f"policy_latent_net.{k}.weight"],
                                       pd[f"policy_latent_net.{k}.bias"]))
    x = torch.nn.functional.linear(x, pd["action_mean_net.weight"],
                                   pd["action_mean_net.bias"])
    return x.squeeze(0).numpy()


def test_crl2_checkpoint_load_and_forward():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_100.pt")
        ckpt = _make_checkpoint(path)
        policy = RamboPolicy(path)
        obs = np.random.default_rng(5).normal(size=405)
        out = policy(obs)
        ref = _reference_forward(ckpt, obs)
        assert out.shape == (18,)
        assert np.allclose(out, ref, atol=1e-5)


def test_checkpoint_without_normalizer():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_1.pt")
        ckpt = _make_checkpoint(path, with_normalizer=False)
        policy = RamboPolicy(path)
        obs = np.zeros(405)
        out = policy(obs)
        ref = _reference_forward(ckpt, obs)
        assert np.allclose(out, ref, atol=1e-6)


def test_dimension_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_2.pt")
        _make_checkpoint(path, num_obs=300)
        with pytest.raises(ValueError, match="dimensions"):
            RamboPolicy(path)


class _ExportPolicy(torch.nn.Module):
    def __init__(self, num_obs=405, num_actions=18, nonfinite=0.0, extra_axis=False):
        super().__init__()
        self.linear = torch.nn.Linear(num_obs, num_actions)
        self.nonfinite = nonfinite
        self.extra_axis = extra_axis

    def forward(self, obs):
        out = self.linear(obs) + self.nonfinite
        return out.unsqueeze(0) if self.extra_axis else out


@pytest.mark.parametrize("num_obs,num_actions,nonfinite,extra_axis", [
    (400, 18, 0.0, False), (405, 17, 0.0, False),
    (405, 18, float("nan"), False), (405, 18, float("inf"), False),
    (405, 18, 0.0, True),
])
def test_invalid_torchscript_rejected_during_loading(
        tmp_path, num_obs, num_actions, nonfinite, extra_axis):
    path = tmp_path / "export.pt"
    model = _ExportPolicy(num_obs, num_actions, nonfinite, extra_axis).eval()
    torch.jit.script(model).save(str(path))
    with pytest.raises(ValueError, match="startup"):
        RamboPolicy(str(path))


def test_valid_torchscript_load_and_forward(tmp_path):
    path = tmp_path / "export.pt"
    model = _ExportPolicy().eval()
    torch.jit.script(model).save(str(path))
    policy = RamboPolicy(str(path))
    obs = np.ones(405, dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    np.testing.assert_allclose(policy(obs), expected)


@pytest.mark.parametrize("field,value", [("_mean", float("nan")),
                                           ("_std", float("inf")),
                                           ("_std", -0.01)])
def test_invalid_checkpoint_normalizer_rejected_at_startup(tmp_path, field, value):
    path = tmp_path / "invalid_norm.pt"
    checkpoint = _make_checkpoint(str(path))
    checkpoint["obs_normalizer"][field][0, 0] = value
    torch.save(checkpoint, str(path))
    with pytest.raises(ValueError, match="normalizer"):
        RamboPolicy(str(path))


def _install_onnx_session(monkeypatch, *, input_name="obs", input_shape=(1, 405),
                          output_shape=(1, 18), input_type="tensor(float)",
                          output_type="tensor(float)", output=None):
    # onnxruntime is optional, so it is stubbed. Model metadata and inference are its
    # external boundary; validation here is the real RamboPolicy implementation.
    from types import SimpleNamespace

    class Session:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name=input_name, shape=list(input_shape),
                                    type=input_type)]

        def get_outputs(self):
            return [SimpleNamespace(name="actions", shape=list(output_shape),
                                    type=output_type)]

        def run(self, names, feed):
            assert list(feed) == [input_name]
            assert feed[input_name].shape == (1, 405)
            return [np.zeros((1, 18), dtype=np.float32) if output is None else output]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))


@pytest.mark.parametrize("metadata", [
    {"input_name": "observations"}, {"input_shape": (1, 400)},
    {"input_shape": (2, 405)}, {"input_shape": (405,)},
    {"output_shape": (1, 17)}, {"output_shape": (18,)},
    {"input_type": "tensor(double)"}, {"output_type": "tensor(int64)"},
    {"output": np.full((1, 18), np.nan)},
    {"output": np.full((1, 18), np.inf)},
    {"output": np.zeros((1, 17))}, {"output": np.zeros((1, 1, 18))},
])
def test_invalid_onnx_rejected_during_loading(monkeypatch, metadata):
    _install_onnx_session(monkeypatch, **metadata)
    with pytest.raises(ValueError, match="startup"):
        RamboPolicy("export.onnx")


def test_dynamic_batch_onnx_load_and_forward(monkeypatch):
    _install_onnx_session(monkeypatch, input_shape=("batch", 405),
                          output_shape=("batch", 18))
    policy = RamboPolicy("export.onnx")
    assert policy(np.zeros(405)).shape == (18,)


def test_onnx_runtime_does_not_flatten_invalid_output(monkeypatch):
    _install_onnx_session(monkeypatch)
    policy = RamboPolicy("export.onnx")
    policy._session.run = lambda *args: [np.zeros((2, 9))]
    with pytest.raises(ValueError, match="policy output"):
        policy(np.zeros(405))
