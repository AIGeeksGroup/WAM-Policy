"""The bridge's episode driver against this repository's real policy-server session (no model, no GPU).

Imports ``wam_policy.inference.session.OnlineSession`` from the repository's ``src/`` (deploy/ sits in the
WAM-Policy repository) and drives it with the bridge's own episode driver (``wam_go2.wam.episode``). Only
what needs the model is replaced: the sampler (16 fixed native9 commands), the streaming VAE codec (checks
the two views and returns one latent group) and the prompt encoder (refuses any instruction without a cached
embedding, as ``wam_policy.inference.loader.load_prompts`` does). ``wan_va.modules.utils`` is stubbed when it
cannot be imported. Everything else is the server's code: the session state machine, ``PolicyHistory``, the
observation checks and the preprocessing profiles. The repository code is only imported, never changed.
"""
import importlib
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("qlm_bench.live", reason="QLM-Bench live package not found (set QLM_BENCH_SRC)")

from wam_go2.wam import episode as ep  # noqa: E402
from wam_go2.wam import mock_service as ms  # noqa: E402
from wam_go2.wam import settings  # noqa: E402
from test_wam_episode import FakeRuntime  # noqa: E402

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))


def _drop(prefixes):
    for name in [n for n in sys.modules if n.split(".")[0] in prefixes]:
        del sys.modules[name]


@pytest.fixture(scope="module")
def session_module():
    if not os.path.isfile(os.path.join(SRC, "wam_policy", "inference", "session.py")):
        pytest.skip("deploy/ is not inside the WAM-Policy repository (no ../src/wam_policy)")
    saved_path = list(sys.path)
    sys.path.insert(0, SRC)
    try:
        try:
            module = importlib.import_module("wam_policy.inference.session")
        except ImportError:
            _drop(("wam_policy", "wan_va"))
            for name in ("wan_va", "wan_va.modules"):
                package = types.ModuleType(name)
                package.__path__ = []
                sys.modules[name] = package
            utils = types.ModuleType("wan_va.modules.utils")
            utils.WanVAEStreamingWrapper = type("WanVAEStreamingWrapper", (), {})
            sys.modules["wan_va.modules.utils"] = utils
            module = importlib.import_module("wam_policy.inference.session")
        yield module
    finally:
        sys.path[:] = saved_path
        _drop(("wam_policy", "wan_va"))


class StubCodec:
    def __init__(self, rgb_profile=settings.RGB_PROFILE):
        self.rgb_profile = rgb_profile
        self.groups = []

    def reset(self):
        pass

    def push(self, views):
        assert set(views) == {"go2_ego", "d435i_rgb_task"}
        self.groups.append({k: tuple(v.shape) for k, v in views.items()})
        return np.zeros((1, 48, 1, 24, 20), np.float32)


class StubSampler:
    def __init__(self):
        self.committed = []

    def reset(self, latent, text_emb, *, seed, empty_text_emb=None):
        self.seed = seed

    def predict(self):
        actions = np.zeros((16, 9), np.float32)
        actions[:, 0] = 0.10
        actions[:, 3:6] = (0.1934, 0.142, 0.05)
        return actions

    def commit(self, latent, executed):
        self.committed.append(len(executed))

    def clear(self):
        pass

    def cache_summary(self):
        return {}


def prompt_encoder(instruction):
    if instruction != settings.INSTRUCTION:
        raise ValueError(f"No cached prompt embedding for instruction {instruction!r}")
    return {"text_emb": None}


def _episode(module, *, instruction=settings.INSTRUCTION, max_chunks=2):
    session = module.OnlineSession(StubSampler(), StubCodec(), prompt_encoder, policy_seed=42,
                                   info={"policy": "opsd_student"})
    runtime = FakeRuntime(ego_period=1 / 60, d435i_period=1 / 60)
    client = ms.LocalClient(session.handle, advance_clock=lambda dt: runtime.sleep_until(runtime.now() + dt))
    driver = ep.EpisodeDriver(runtime, client, session_id="go2-real", episode_id="go2-real:1", reset_epoch=1,
                              instruction=instruction, max_chunks=max_chunks)
    return session, driver


def test_the_server_session_runs_two_chunks_from_the_bridge(session_module):
    session, driver = _episode(session_module)
    reply = driver.reset_episode()
    assert reply["status"] == "OBSERVED" and reply["policy_info"] == {"policy": "opsd_student"}
    result = driver.run()
    assert (result.exit_code, result.chunks, result.commands_completed) == (0, 2, 32)
    assert session.sampler.committed == [16, 16]
    assert session.pipeline.codec.groups == [{"go2_ego": (1, 360, 640, 3), "d435i_rgb_task": (1, 360, 640, 3)}] + \
        [{"go2_ego": (4, 360, 640, 3), "d435i_rgb_task": (4, 360, 640, 3)}] * 2
    assert session.history.state == "ENDED"


def test_the_server_session_refuses_a_prompt_it_has_no_embedding_for(session_module):
    _, driver = _episode(session_module, instruction="Push the box into the target area.")
    with pytest.raises(ep.EpisodeAbort) as e:
        driver.reset_episode()
    assert e.value.reason == "rpc_error" and "prompt" in e.value.detail
