"""One native Euler generator for actor, learner, teacher targets and deployment."""
import hashlib
import torch
from wan_va.utils import FlowMatchScheduler, data_seq_to_patch
from wam_policy.opsd.history import STATE_SHAPE, ACTION_SHAPE


def scheduler(config, action=False, *, steps=None):
    """Native FM schedule; `steps` overrides the student step count (teacher targets)."""
    config.validate()
    if steps is None:
        steps = config.action_steps if action else config.video_steps
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive integer sampler step count required")
    result = FlowMatchScheduler(shift=1. if action else 5., sigma_min=0., extra_one_step=True)
    result.set_timesteps(steps)
    return result


def _manifest(config, video_steps, action_steps):
    result = {}
    for name, action, steps in (("state", False, video_steps), ("action", True, action_steps)):
        flow = scheduler(config, action, steps=steps)
        result[name] = {"sigmas": flow.sigmas.tolist(), "timesteps": flow.timesteps.tolist(), "endpoint": 0.,
                        "steps": steps, "shift": flow.shift,
                        "guidance": 1.0,
                        "window_units": config.window_units, "solver": "native_euler",
                        "parameterization": "noise_minus_clean_velocity"}
    return result


def sampler_manifest(config):
    """Student/deployment sampler (1/2 steps)."""
    return _manifest(config, config.video_steps, config.action_steps)


def teacher_sampler_manifest(config):
    """Teacher target sampler (25/50 steps); part of the target cache reuse key."""
    return _manifest(config, config.teacher_video_steps, config.teacher_action_steps)


def velocity(context, noisy, rho, *, action=False, count="score"):
    if action and not context.plan_set:
        raise ValueError("Action requires the completed detached student plan")
    output = context.forward(noisy, float(rho) * 1000., action=action, count=count)
    if action:
        return output.transpose(1, 2).reshape(ACTION_SHAPE)
    return data_seq_to_patch((1, 2, 2), output, 1, 24, 20, batch_size=1)


def draw_noise(generator, runtime, *, action=False):
    """Initial Gaussian noise in the runtime dtype (bf16 on GPU), the exact tensor
    the actor would draw inside generate(); stored with every regression target."""
    shape = ACTION_SHAPE if action else STATE_SHAPE
    return torch.randn(shape, generator=generator, device=runtime.device, dtype=runtime.dtype)


def generate(context, generator, *, action=False, initial_noise=None, steps=None, trace=None, velocity_trace=None):
    runtime = context.runtime
    shape = ACTION_SHAPE if action else STATE_SHAPE
    value = (draw_noise(generator, runtime, action=action)
             if initial_noise is None else initial_noise.to(runtime.device, runtime.dtype))
    if tuple(value.shape) != shape:
        raise ValueError("Native generator noise geometry mismatch")
    flow = scheduler(runtime.config, action, steps=steps)
    for t in flow.timesteps:
        if trace is not None:
            trace.append(value)
        pred = velocity(context, value, t / 1000., action=action, count="denoise")
        if velocity_trace is not None:
            pred.retain_grad()
            velocity_trace.append(pred)
        value = flow.step(pred, t, value).to(runtime.dtype)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite native FM endpoint")
    # Cache writes are explicit caller operations; never mutate before backward.
    return value


class RNGStreams:
    """Counter-keyed streams: all draws reproduce from seed/rank and saved counters."""
    def __init__(self, seed, rank, device):
        self.seed, self.rank, self.device = seed, rank, torch.device(device)

    def generator(self, *key):
        # "opd4-rng-v1" is the RNG label of the paper runs; it seeds the on-policy target noise and the
        # per-update teacher-cache draw order (not resets, episodes, actor noise or the held-out choice).
        raw = repr(("opd4-rng-v1", self.seed, self.rank, key)).encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)
        return torch.Generator(device=self.device).manual_seed(seed)

    def state_dict(self):
        return {"schema": "opd4-rng-v1", "seed": self.seed, "rank": self.rank}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("RNG stream state belongs to another seed or rank")
