"""RAMBO policy loading and deterministic inference (deployment side).

Standalone loader for crl2 checkpoints - no crl2 / isaaclab import needed.
It reads the checkpoint layout of these crl2 modules:
- crl2/modules/policy.py       (Policy: policy_latent_net + action_mean_net)
- crl2/modules/normalizer.py   (EmpiricalNormalization: (x-mean)/(std+1e-2))
- crl2/algorithms/ppo.py       (checkpoint keys: policy_dict, obs_normalizer)

Supported checkpoint formats:
- crl2 training checkpoint  model_<it>.pt  (dict with policy_dict/obs_normalizer)
- TorchScript export        (normalizer embedded; from export_policy_as_jit)
- ONNX export               model.onnx     (input "obs", output "actions")
"""

import os

import numpy as np


class PolicyOutputError(ValueError):
    """A rejected raw action, retained before any clipping or reshaping."""

    def __init__(self, reason, raw_action):
        self.raw_action = np.array(raw_action, copy=True)
        super().__init__(f"Invalid policy output: {reason}; raw={self.raw_action!r}")


def validate_policy_output(raw_action, expected_shape):
    """Validate the original interface shape and values before processing."""
    action = np.asarray(raw_action)
    if action.shape != expected_shape:
        raise PolicyOutputError(
            f"shape {action.shape}, expected {expected_shape}", action)
    if action.dtype.kind not in "fiu" or not np.all(np.isfinite(action)):
        raise PolicyOutputError("non-finite or non-numeric values", action)
    return action


class RamboPolicy:
    """Deterministic policy: obs (405,) float32 -> action (18,)."""

    def __init__(self, checkpoint_path, num_obs=405, num_actions=18,
                 hidden_dims=(512, 256, 128)):
        self.num_obs = num_obs
        self.num_actions = num_actions
        self._backend = None
        ext = os.path.splitext(checkpoint_path)[1].lower()
        try:
            if ext == ".onnx":
                self._init_onnx(checkpoint_path)
            else:
                self._init_torch(checkpoint_path, hidden_dims)
            # Every backend must execute the configured interface before the
            # node is constructed and can accept an active-motion request.
            self(np.zeros(self.num_obs, dtype=np.float32))
        except Exception as exc:
            raise ValueError(
                f"Policy startup validation failed at {checkpoint_path}: {exc}") from exc

    # ------------------------------------------------------------------
    def _init_torch(self, path, hidden_dims):
        import torch
        import torch.nn as nn

        self._torch = torch
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(loaded, torch.jit.ScriptModule) or isinstance(
                loaded, torch.jit.RecursiveScriptModule):
            # jit export already embeds the normalizer and outputs the mean
            self._model = loaded
            self._model.eval()
            self._normalizer = None
            self._backend = "jit"
            self.normalization_source = "embedded in export"
            return

        if not isinstance(loaded, dict) or "policy_dict" not in loaded:
            raise ValueError(
                f"Unrecognized checkpoint format at {path}: expected a crl2 "
                "training checkpoint (dict with 'policy_dict'), a TorchScript "
                "module, or an .onnx file")

        policy_dict = loaded["policy_dict"]
        # infer dimensions from the weights (authoritative over the defaults)
        in_features = policy_dict["policy_latent_net.0.weight"].shape[1]
        out_features = policy_dict["action_mean_net.weight"].shape[0]
        if in_features != self.num_obs or out_features != self.num_actions:
            raise ValueError(
                f"Checkpoint dimensions ({in_features} obs, {out_features} "
                f"actions) do not match the configured RAMBO task "
                f"({self.num_obs} obs, {self.num_actions} actions). The "
                "checkpoint was trained with different qp_env.py settings.")
        hidden = [policy_dict[f"policy_latent_net.{i}.weight"].shape[0]
                  for i in (0, 2, 4)]

        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        self._latent = nn.Sequential(*layers)
        self._mean = nn.Linear(prev, out_features)

        with torch.no_grad():
            for i, li in enumerate((0, 2, 4)):
                self._latent[2 * i].weight.copy_(policy_dict[f"policy_latent_net.{li}.weight"])
                self._latent[2 * i].bias.copy_(policy_dict[f"policy_latent_net.{li}.bias"])
            self._mean.weight.copy_(policy_dict["action_mean_net.weight"])
            self._mean.bias.copy_(policy_dict["action_mean_net.bias"])
        self._latent.eval()
        self._mean.eval()

        # EmpiricalNormalization buffers (crl2/modules/normalizer.py)
        self._norm_mean = None
        self._norm_std = None
        if "obs_normalizer" in loaded and loaded["obs_normalizer"] is not None:
            norm = loaded["obs_normalizer"]
            self._norm_mean = norm["_mean"].numpy().reshape(-1).astype(np.float32)
            self._norm_std = norm["_std"].numpy().reshape(-1).astype(np.float32)
            if (self._norm_mean.shape != (self.num_obs,)
                    or self._norm_std.shape != (self.num_obs,)
                    or not np.all(np.isfinite(self._norm_mean))
                    or not np.all(np.isfinite(self._norm_std))
                    or np.any(self._norm_std < 0.0)):
                raise ValueError("invalid checkpoint normalizer dimensions or values")
        self.normalization_source = ("checkpoint" if self._norm_mean is not None
                                     else "none")
        self._backend = "torch"

    def _init_onnx(self, path):
        import onnxruntime  # noqa: deferred heavy import

        self._session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs":
            raise ValueError("ONNX requires exactly one input named 'obs'")
        if len(outputs) != 1 or outputs[0].name != "actions":
            raise ValueError("ONNX requires exactly one output named 'actions'")
        for item, width in ((inputs[0], self.num_obs), (outputs[0], self.num_actions)):
            shape = item.shape
            # Symbolic dimensions are checked by the startup inference below.
            if (item.type != "tensor(float)" or len(shape) != 2
                    or (isinstance(shape[0], int) and shape[0] != 1)
                    or (isinstance(shape[1], int) and shape[1] != width)):
                raise ValueError(
                    f"ONNX {item.name} requires float32 shape (1, {width}), "
                    f"got {item.type} {shape}")
        self.normalization_source = "embedded in export"
        self._backend = "onnx"

    # ------------------------------------------------------------------
    def __call__(self, obs):
        """obs: (num_obs,) numpy -> action (num_actions,) numpy (gaussian mean)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.num_obs,) or not np.all(np.isfinite(obs)):
            raise ValueError(f"Policy observation must be finite shape ({self.num_obs},)")
        if self._backend == "onnx":
            out = self._session.run(["actions"], {"obs": obs[None, :]})[0]
            return validate_policy_output(out, (1, self.num_actions))[0].astype(np.float64)

        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(obs).unsqueeze(0)
            if self._backend == "jit":
                out = self._model(x)
            else:
                if self._norm_mean is not None:
                    # (x - mean) / (std + eps), eps = 1e-2 (normalizer.py:36-39)
                    x = (x - torch.from_numpy(self._norm_mean)) / (
                        torch.from_numpy(self._norm_std) + 1e-2)
                out = self._mean(self._latent(x))
            if not isinstance(out, torch.Tensor):
                raise ValueError(f"Invalid policy output: expected a tensor, got {type(out).__name__}")
            raw = out.numpy()
            return validate_policy_output(raw, (1, self.num_actions))[0].astype(np.float64)
