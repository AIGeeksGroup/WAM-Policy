"""One functional native Euler kernel for actor, teacher target generation and differentiable learner."""
from dataclasses import dataclass
import torch
from wan_va.utils.scheduler import FlowMatchScheduler
from arm_opsd.config import ArmContract
from arm_opsd.model.codec import project, time_vector


def sigma_grid(steps, shift):
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(steps)
    return torch.cat([scheduler.sigmas, torch.zeros(1)])


@dataclass
class StateSample:
    endpoint: torch.Tensor
    history: object
    step_outputs: tuple
    noise: torch.Tensor = None


@dataclass
class ActionSample:
    endpoint: torch.Tensor
    history: object
    step_outputs: tuple
    noise: torch.Tensor = None


@dataclass
class JointSample:
    state: torch.Tensor
    action: torch.Tensor
    post_history: object
    state_steps: tuple
    action_steps: tuple
    denoising_calls: int
    clean_calls: int
    # The initial noises actually integrated: a teacher rollout's (noise, endpoint) pairs are
    # regression targets as they are.
    state_noise: torch.Tensor = None
    action_noise: torch.Tensor = None


class NativeSampler:
    """Steps are fixed per instance. The student uses 1/2, the target generator the teacher's
    20/50 (the paper uses no other steps)."""

    def __init__(self, contract=None, *, state_steps=1, action_steps=2):
        self.contract = contract or ArmContract()
        if type(state_steps) is not int or type(action_steps) is not int or min(state_steps, action_steps) < 1:
            raise ValueError("Positive integer video/action step counts required")
        self.state_steps, self.action_steps = state_steps, action_steps

    def _noise(self, shape, role, generator):
        return torch.randn(shape, generator=generator, device=role.device, dtype=role.dtype)

    def _euler(self, role, view, value, *, action, first, initial_latent=None,
               steps, guidance):
        shift = self.contract.action_shift if action else self.contract.video_shift
        grid = sigma_grid(steps, shift)
        outputs = []
        for index in range(steps):
            value = project(value, action=action, first=first,
                            initial_latent=initial_latent, contract=self.contract)
            times = time_vector(value, grid[index] * 1000, first=first)
            velocity = role.velocity(value, times, view, action=action, guidance=guidance)
            if velocity.shape != value.shape:
                raise ValueError("Velocity shape changed")
            outputs.append(velocity)
            value = value + velocity * (grid[index + 1] - grid[index]).to(value.device)
            value = project(value, action=action, first=first,
                            initial_latent=initial_latent, contract=self.contract)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite Euler endpoint")
        return value, tuple(outputs)

    def state(self, role, history, *, first, initial_latent, generator=None, noise=None):
        prepared = role.reserve(history, self.contract.video_tokens, update_cache=0)
        value = self._noise(self.contract.state_shape, role, generator) if noise is None else noise
        if tuple(value.shape) != self.contract.state_shape:
            raise ValueError("State noise shape differs from the native state")
        endpoint, steps = self._euler(role, prepared, value, action=False, first=first,
                                      initial_latent=initial_latent, steps=self.state_steps,
                                      guidance=self.contract.video_cfg)
        return StateSample(endpoint, prepared, steps, value)

    @torch.no_grad()
    def plan(self, role, history, state):
        video_history = role.reserve(history, self.contract.video_tokens, update_cache=0)
        return role.encode_clean(state.detach(), video_history, action=False, predicted=True)

    def action(self, role, plan_view, *, first, generator=None, noise=None):
        prepared = role.reserve(plan_view, self.contract.action_tokens, update_cache=0)
        value = self._noise(self.contract.action_shape, role, generator) if noise is None else noise
        if tuple(value.shape) != self.contract.action_shape:
            raise ValueError("Action noise shape differs from the native action chunk")
        endpoint, steps = self._euler(role, prepared, value, action=True, first=first,
                                      steps=self.action_steps, guidance=self.contract.action_cfg)
        return ActionSample(endpoint, prepared, steps, value)

    def joint(self, role, history, *, first, initial_latent, generator=None,
              state_noise=None, action_noise=None, actor_clean_action=True):
        from arm_opsd.model.kv_views import clear_predictions, with_metadata
        state = self.state(role, history, first=first, initial_latent=initial_latent,
                           generator=generator, noise=state_noise)
        plan = self.plan(role, state.history, state.endpoint)
        action = self.action(role, plan, first=first, generator=generator, noise=action_noise)
        prepared = action.history
        if actor_clean_action:
            with torch.no_grad():
                end_view = role.encode_clean(action.endpoint.detach(), prepared, action=True, predicted=True)
        else:
            # The action clean value is unused by the learner, but its native
            # reservation/write/clear metadata event must still be simulated.
            metadata = prepared.metadata.reserve(self.contract.action_tokens, 1).after.clear_pred()
            end_view = with_metadata(prepared, metadata)
        post = clear_predictions(end_view)
        return JointSample(state.endpoint, action.endpoint, post, state.step_outputs, action.step_outputs,
                           self.state_steps + self.action_steps, 2 if actor_clean_action else 1,
                           state.noise, action.noise)
