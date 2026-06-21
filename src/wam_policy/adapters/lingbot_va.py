from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from wam_policy.utils.paths import resolve_repo_root


class LingBotVAAdapter:
    """Adapter boundary for LingBot-VA internals."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.repo_root = resolve_repo_root(cfg)
        self.repo_path = self._resolve_lingbot_repo()
        self.checkpoint_path = self._resolve_checkpoint_path()
        self.device = self._resolve_device()
        self.torch_dtype = self._resolve_precision()

    def load_pretrained_video_action_model(self) -> nn.Module:
        """Load the LingBot-VA transformer on server/runtime environments."""
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
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LingBot-VA transformer from {self.checkpoint_path}."
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
        return model

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
        runtime_cfg = self.cfg.get("runtime")
        device_name = "auto"
        if isinstance(runtime_cfg, Mapping):
            raw_device = runtime_cfg.get("device", "auto")
            if isinstance(raw_device, str) and raw_device:
                device_name = raw_device

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_name)

    def _resolve_precision(self) -> torch.dtype:
        runtime_cfg = self.cfg.get("runtime")
        precision = "bf16"
        if isinstance(runtime_cfg, Mapping):
            raw_precision = runtime_cfg.get("precision", "bf16")
            if isinstance(raw_precision, str) and raw_precision:
                precision = raw_precision

        precision_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if precision not in precision_map:
            raise ValueError(f"Unsupported runtime.precision: {precision}")
        return precision_map[precision]


def _resolve_under_repo(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


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
