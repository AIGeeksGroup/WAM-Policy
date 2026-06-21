from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import MethodType
from types import ModuleType
from typing import Any, Mapping

import torch
from torch import nn

from wam_policy.utils.paths import resolve_repo_root
from wam_policy.utils.runtime import resolve_device, resolve_lingbot_attn_mode, resolve_torch_dtype


class LingBotVAAdapter:
    """Adapter boundary for LingBot-VA internals."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.repo_root = resolve_repo_root(cfg)
        self.repo_path = self._resolve_lingbot_repo()
        self.checkpoint_path = self._resolve_checkpoint_path()
        self.device = self._resolve_device()
        self.torch_dtype = self._resolve_precision()
        self.lingbot_attn_mode = resolve_lingbot_attn_mode(cfg)

    def load_pretrained_video_action_model(self) -> nn.Module:
        """Load the LingBot-VA transformer on server/runtime environments."""
        self._install_flash_attention_stub_if_needed()
        try:
            from wan_va.modules.utils import load_transformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Unable to import LingBot-VA runtime modules. Ensure "
                "bootstrap_external_repos(cfg) has added external/lingbot-va to sys.path "
                "and install LingBot-VA dependencies such as diffusers and transformers "
                "on the server environment."
            ) from exc

        try:
            model = load_transformer(
                str(self.checkpoint_path),
                torch_dtype=self.torch_dtype,
                torch_device=self.device,
                attn_mode=self.lingbot_attn_mode,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LingBot-VA transformer from {self.checkpoint_path} "
                f"with attn_mode={self.lingbot_attn_mode!r}."
            ) from exc

        if not isinstance(model, nn.Module):
            raise TypeError("LingBot-VA load_transformer did not return a torch.nn.Module.")
        return model

    def build_video_context(self, batch: Mapping[str, Any]) -> Any:
        return batch.get("context")

    def attach_video_time_pair_conditioner(
        self,
        model: nn.Module,
        conditioner: nn.Module,
    ) -> nn.Module:
        model.wam_time_pair_conditioner = conditioner
        self._configure_conditioner_from_model(model, conditioner)
        self._patch_time_embed(model, conditioner)
        self.install_video_flowmap_hook(model, conditioner)
        return model

    def install_video_flowmap_hook(
        self,
        model: nn.Module,
        conditioner: nn.Module,
    ) -> nn.Module:
        model.wam_time_pair_conditioner = conditioner

        def wam_forward_video_flowmap(
            *,
            z_src: torch.Tensor,
            s_src: torch.Tensor,
            s_tgt: torch.Tensor,
            context: Any,
        ) -> torch.Tensor:
            input_dict = self.build_lingbot_video_input(model, z_src, s_src, s_tgt, context)
            model._wam_current_s_src = s_src
            model._wam_current_s_tgt = s_tgt
            try:
                output = _call_lingbot_forward(
                    model,
                    input_dict,
                    update_cache=0,
                    cache_name=self._cache_name(),
                    action_mode=False,
                    train_mode=False,
                )
            finally:
                model._wam_current_s_src = None
                model._wam_current_s_tgt = None
            return self.lingbot_output_to_latent(output, z_src.shape, _patch_size(model))

        model.wam_forward_video_flowmap = wam_forward_video_flowmap
        return model

    def build_lingbot_video_input(
        self,
        model: nn.Module,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> dict[str, torch.Tensor]:
        _validate_video_flowmap_inputs(z_src, s_src, s_tgt)
        if z_src.ndim != 5:
            raise ValueError("z_src must have shape [B, C, F, H, W].")

        patch_size = _patch_size(model)
        batch_size, _, frames, height, width = z_src.shape
        patch_f, patch_h, patch_w = patch_size
        if frames % patch_f or height % patch_h or width % patch_w:
            raise ValueError(
                f"z_src shape {tuple(z_src.shape)} is not divisible by patch_size {patch_size}."
            )

        token_frames = frames // patch_f
        timesteps = (1.0 - s_src.to(dtype=torch.float32)) * self._num_train_timesteps()
        timesteps = timesteps[:, None].expand(batch_size, token_frames).to(device=z_src.device)
        grid_id = _get_mesh_id(
            token_frames,
            height // patch_h,
            width // patch_w,
            time_type=0,
            device=z_src.device,
        )
        grid_id = grid_id[None].expand(batch_size, -1, -1)
        text_emb = self.normalize_video_context(
            context,
            batch_size=batch_size,
            device=z_src.device,
            dtype=z_src.dtype,
        )
        return {
            "noisy_latents": z_src,
            "timesteps": timesteps,
            "grid_id": grid_id,
            "text_emb": text_emb,
        }

    def normalize_video_context(
        self,
        context: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        text_emb = _extract_text_emb(context)
        if isinstance(text_emb, list):
            text_emb = torch.stack([_normalize_text_emb_tensor(item) for item in text_emb], dim=0)
        else:
            text_emb = _normalize_text_emb_tensor(text_emb)

        if text_emb.ndim == 2:
            if batch_size != 1:
                raise ValueError("A single [L, D] text_emb is only valid for batch_size=1.")
            text_emb = text_emb[None]
        if text_emb.ndim != 3:
            raise ValueError("text_emb must have shape [L, D] or [B, L, D].")
        if text_emb.shape[0] != batch_size:
            raise ValueError(
                f"text_emb batch size must match z_src batch size: {text_emb.shape[0]} != {batch_size}"
            )
        return text_emb.to(device=device, dtype=dtype)

    def lingbot_output_to_latent(
        self,
        output: Any,
        z_shape: torch.Size | tuple[int, ...],
        patch_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("LingBot video forward must return a tensor or tuple/list of tensors.")
        if tuple(output.shape) == tuple(z_shape):
            return output
        if output.ndim == 3:
            return self.video_sequence_to_latent(output, z_shape, patch_size)
        raise ValueError(f"Unsupported LingBot video output shape: {tuple(output.shape)}")

    def video_sequence_to_latent(
        self,
        sequence: torch.Tensor,
        z_shape: torch.Size | tuple[int, ...],
        patch_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if len(z_shape) != 5:
            raise ValueError("z_shape must be [B, C, F, H, W].")
        batch_size, channels, frames, height, width = tuple(z_shape)
        patch_f, patch_h, patch_w = patch_size
        if frames % patch_f or height % patch_h or width % patch_w:
            raise ValueError(f"z_shape {tuple(z_shape)} is not divisible by patch_size {patch_size}.")

        post_frames = frames // patch_f
        post_height = height // patch_h
        post_width = width // patch_w
        expected_tokens = post_frames * post_height * post_width * patch_f * patch_h * patch_w
        if sequence.shape != (batch_size, expected_tokens, channels):
            raise ValueError(
                f"sequence shape must be {(batch_size, expected_tokens, channels)}, "
                f"got {tuple(sequence.shape)}."
            )

        latent = sequence.reshape(
            batch_size,
            post_frames,
            post_height,
            post_width,
            patch_f,
            patch_h,
            patch_w,
            channels,
        )
        latent = latent.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return latent.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward_video_flowmap(
        self,
        model: nn.Module,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        _validate_video_flowmap_inputs(z_src, s_src, s_tgt)

        hook = getattr(model, "wam_forward_video_flowmap", None)
        if hook is None:
            raise NotImplementedError(
                "LingBot video flow-map forward binding is not implemented yet. "
                "Attach model.wam_forward_video_flowmap during server runtime debug."
            )

        pred_u = hook(z_src=z_src, s_src=s_src, s_tgt=s_tgt, context=context)
        if not isinstance(pred_u, torch.Tensor):
            raise TypeError("LingBot video flow-map hook must return a torch.Tensor.")
        return pred_u

    def _resolve_lingbot_repo(self) -> Path:
        paths_cfg = self.cfg.get("paths")
        if not isinstance(paths_cfg, Mapping):
            raise ValueError("Config must contain a 'paths' mapping.")

        raw_path = paths_cfg.get("lingbot_repo")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Config paths.lingbot_repo must be a non-empty string.")
        return _resolve_under_repo(self.repo_root, raw_path)

    def _resolve_checkpoint_path(self) -> Path:
        teacher_cfg = self.cfg.get("teacher")
        if not isinstance(teacher_cfg, Mapping):
            raise ValueError("Config must contain a 'teacher' mapping.")

        raw_path = teacher_cfg.get("checkpoint_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Config teacher.checkpoint_path must be a non-empty string.")
        return _resolve_under_repo(self.repo_root, raw_path)

    def _resolve_device(self) -> torch.device:
        return resolve_device(self.cfg)

    def _resolve_precision(self) -> torch.dtype:
        return resolve_torch_dtype(self.cfg)

    def _cache_name(self) -> str:
        runtime_cfg = self.cfg.get("runtime")
        value = "pos"
        if isinstance(runtime_cfg, Mapping):
            raw_value = runtime_cfg.get("lingbot_cache_name", value)
            if isinstance(raw_value, str) and raw_value:
                value = raw_value
        return value

    def _install_flash_attention_stub_if_needed(self) -> None:
        if self.lingbot_attn_mode != "torch":
            return
        _install_flash_attention_stub_if_missing()

    def _num_train_timesteps(self) -> float:
        flowmap_cfg = self.cfg.get("flowmap")
        value = 1000
        if isinstance(flowmap_cfg, Mapping):
            value = flowmap_cfg.get("lingbot_num_train_timesteps", value)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("flowmap.lingbot_num_train_timesteps must be positive.")
        return float(value)

    def _configure_conditioner_from_model(self, model: nn.Module, conditioner: nn.Module) -> None:
        configure = getattr(conditioner, "configure_output_dims", None)
        if not callable(configure):
            return
        try:
            time_proj = model.condition_embedder.time_proj
            timestep_proj_dim = int(time_proj.out_features)
            hidden_dim = timestep_proj_dim // 6
        except (AttributeError, TypeError, ValueError):
            return
        if hidden_dim <= 0 or timestep_proj_dim <= 0:
            return
        configure(hidden_dim=hidden_dim, timestep_proj_dim=timestep_proj_dim)

    def _patch_time_embed(self, model: nn.Module, conditioner: nn.Module) -> None:
        self._patch_condition_embedder_forward(model, conditioner)
        if hasattr(model, "_wam_original_time_embed"):
            return
        original_time_embed = getattr(model, "_time_embed", None)
        if not callable(original_time_embed):
            return

        model._wam_original_time_embed = original_time_embed

        def patched_time_embed(
            model_self: nn.Module,
            timesteps: torch.Tensor,
            height: int,
            width: int,
            dtype: torch.dtype,
            action_mode: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            model_self._wam_conditioning_was_applied = False
            temb, timestep_proj = model_self._wam_original_time_embed(
                timesteps,
                height,
                width,
                dtype,
                action_mode=action_mode,
            )
            if action_mode:
                return temb, timestep_proj

            s_src = getattr(model_self, "_wam_current_s_src", None)
            s_tgt = getattr(model_self, "_wam_current_s_tgt", None)
            if s_src is None or s_tgt is None:
                return temb, timestep_proj
            if getattr(model_self, "_wam_conditioning_was_applied", False):
                return temb, timestep_proj

            s_src_tokens = _expand_time_to_tokens(s_src, temb.shape[1]).to(device=temb.device)
            s_tgt_tokens = _expand_time_to_tokens(s_tgt, temb.shape[1]).to(device=temb.device)
            temb_delta, timestep_proj_delta = conditioner.forward_conditioning(
                s_src_tokens,
                s_tgt_tokens,
                timestep_proj.shape,
            )
            return (
                temb + temb_delta.to(device=temb.device, dtype=temb.dtype),
                timestep_proj
                + timestep_proj_delta.to(device=timestep_proj.device, dtype=timestep_proj.dtype),
            )

        model._time_embed = MethodType(patched_time_embed, model)

    def _patch_condition_embedder_forward(
        self,
        model: nn.Module,
        conditioner: nn.Module,
    ) -> None:
        condition_embedder = getattr(model, "condition_embedder", None)
        original_forward = getattr(condition_embedder, "forward", None)
        if condition_embedder is None or not callable(original_forward):
            return
        if hasattr(condition_embedder, "_wam_original_forward"):
            return

        condition_embedder._wam_original_forward = original_forward

        def patched_condition_forward(
            condition_self: nn.Module,
            timestep: torch.Tensor,
            dtype: torch.dtype | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            temb, timestep_proj = condition_self._wam_original_forward(timestep, dtype=dtype)
            model._wam_conditioning_was_applied = False

            s_src = getattr(model, "_wam_current_s_src", None)
            s_tgt = getattr(model, "_wam_current_s_tgt", None)
            if s_src is None or s_tgt is None:
                return temb, timestep_proj

            s_src_tokens = _expand_time_to_tokens(s_src, temb.shape[1]).to(device=temb.device)
            s_tgt_tokens = _expand_time_to_tokens(s_tgt, temb.shape[1]).to(device=temb.device)
            temb_delta, timestep_proj_delta = conditioner.forward_conditioning(
                s_src_tokens,
                s_tgt_tokens,
                timestep_proj.shape,
            )
            model._wam_conditioning_was_applied = True
            return (
                temb + temb_delta.to(device=temb.device, dtype=temb.dtype),
                timestep_proj
                + timestep_proj_delta.to(device=timestep_proj.device, dtype=timestep_proj.dtype),
            )

        condition_embedder.forward = MethodType(patched_condition_forward, condition_embedder)


def _resolve_under_repo(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _call_lingbot_forward(
    model: nn.Module,
    input_dict: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> Any:
    forward = getattr(model, "forward", None)
    if not callable(forward):
        raise TypeError("LingBot model must expose a callable forward method.")

    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        accepted_kwargs = kwargs
    else:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted_kwargs = (
            kwargs
            if accepts_kwargs
            else {key: value for key, value in kwargs.items() if key in signature.parameters}
        )
    return model(input_dict, **accepted_kwargs)


def _install_flash_attention_stub_if_missing() -> None:
    if (
        importlib.util.find_spec("flash_attn_interface") is not None
        or importlib.util.find_spec("flash_attn") is not None
    ):
        return

    def _unavailable_flash_attn_func(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "flash-attn is unavailable in this environment. WAM-Policy installed a "
            "stub only so LingBot-VA can be imported with attn_mode='torch'. "
            "Use runtime.lingbot_attn_mode='torch' or install a ROCm-compatible "
            "attention backend before selecting flashattn."
        )

    for module_name in ("flash_attn_interface", "flash_attn"):
        module = ModuleType(module_name)
        module.flash_attn_func = _unavailable_flash_attn_func
        sys.modules[module_name] = module


def _validate_video_flowmap_inputs(
    z_src: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
) -> None:
    if not isinstance(z_src, torch.Tensor):
        raise TypeError("z_src must be a torch.Tensor.")
    if z_src.ndim == 0:
        raise ValueError("z_src must have a batch dimension.")
    _validate_time_tensor("s_src", s_src, z_src.shape[0], z_src.device)
    _validate_time_tensor("s_tgt", s_tgt, z_src.shape[0], z_src.device)
    if not torch.all(s_src < s_tgt).item():
        raise ValueError("s_src must be strictly smaller than s_tgt for every sample.")


def _validate_time_tensor(
    name: str,
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B].")
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match z_src batch size.")
    if value.device != device:
        raise ValueError(f"{name} must be on the same device as z_src.")


def _patch_size(model: nn.Module) -> tuple[int, int, int]:
    raw_patch_size = getattr(model, "patch_size", (1, 2, 2))
    if not isinstance(raw_patch_size, (tuple, list)) or len(raw_patch_size) != 3:
        raise ValueError("LingBot model patch_size must have three dimensions.")
    patch_size = tuple(int(value) for value in raw_patch_size)
    if any(value <= 0 for value in patch_size):
        raise ValueError("LingBot model patch_size values must be positive.")
    return patch_size


def _get_mesh_id(
    frames: int,
    height: int,
    width: int,
    time_type: int,
    device: torch.device,
) -> torch.Tensor:
    frame_idx = torch.arange(frames, device=device)
    height_idx = torch.arange(height, device=device)
    width_idx = torch.arange(width, device=device)
    frame_grid, height_grid, width_grid = torch.meshgrid(
        frame_idx,
        height_idx,
        width_idx,
        indexing="ij",
    )
    grid_id = torch.cat(
        [
            frame_grid.unsqueeze(0),
            height_grid.unsqueeze(0),
            width_grid.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], time_type)], dim=0)


def _extract_text_emb(context: Any) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(context, torch.Tensor):
        return context
    if isinstance(context, Mapping):
        if "text_emb" not in context:
            raise ValueError("video context dict must contain 'text_emb'.")
        return context["text_emb"]
    if isinstance(context, list):
        if not context:
            raise ValueError("video context list must not be empty.")
        extracted: list[torch.Tensor] = []
        for item in context:
            value = _extract_text_emb(item)
            if isinstance(value, list):
                raise ValueError("nested video context lists are not supported.")
            extracted.append(value)
        return extracted
    raise TypeError("video context must be a tensor, dict, or list of dict/tensor items.")


def _normalize_text_emb_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("text_emb must be a torch.Tensor.")
    if value.ndim == 3 and value.shape[0] == 1:
        return value.squeeze(0)
    if value.ndim in {2, 3}:
        return value
    raise ValueError("text_emb tensor must have shape [L, D] or [B, L, D].")


def _expand_time_to_tokens(s: torch.Tensor, token_count: int) -> torch.Tensor:
    if s.ndim == 1:
        return s[:, None].expand(s.shape[0], token_count)
    if s.ndim == 2 and s.shape[1] == token_count:
        return s
    raise ValueError("s_src/s_tgt cannot be expanded to the requested token count.")
