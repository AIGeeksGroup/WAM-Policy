from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.models.flow_map_heads import FlowTimePairEmbedding


class LingBotVideoFlowMapStudent(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: LingBotVAAdapter,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter
        self.base_model: nn.Module | None = None
        self.time_pair_conditioner = FlowTimePairEmbedding(
            hidden_dim=_model_int(cfg, "time_pair_hidden_dim", default=256),
            mlp_hidden_dim=_model_optional_int(cfg, "time_pair_mlp_hidden_dim"),
        )

    def build(self) -> None:
        self.base_model = self.adapter.load_pretrained_video_action_model()
        self.base_model = self.adapter.attach_video_time_pair_conditioner(
            self.base_model,
            self.time_pair_conditioner,
        )
        self.freeze_by_config()

    def freeze_by_config(self) -> None:
        if self.base_model is None:
            raise RuntimeError("Cannot freeze before build(): base_model is not set.")

        model_cfg = self.cfg.get("model")
        freeze_backbone = True
        trainable_modules: list[str] = []
        if isinstance(model_cfg, Mapping):
            freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
            raw_modules = model_cfg.get("trainable_modules", [])
            if isinstance(raw_modules, list):
                trainable_modules = [item for item in raw_modules if isinstance(item, str)]

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        for name, param in self.base_model.named_parameters():
            if any(module_name in name for module_name in trainable_modules):
                param.requires_grad = True

        for param in self.time_pair_conditioner.parameters():
            param.requires_grad = True

    def forward_video_flowmap(
        self,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        if self.base_model is None:
            raise RuntimeError("LingBotVideoFlowMapStudent.build() must be called before forward.")

        pred_u = self.adapter.forward_video_flowmap(
            self.base_model,
            z_src=z_src,
            s_src=s_src,
            s_tgt=s_tgt,
            context=context,
        )
        if pred_u.shape != z_src.shape:
            raise ValueError(f"pred_u shape must match z_src shape: {pred_u.shape} != {z_src.shape}")
        return pred_u


class LingBotActionFlowMapStudent(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: LingBotVAAdapter,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter
        self.base_model: nn.Module | None = None
        self.time_pair_conditioner = FlowTimePairEmbedding(
            hidden_dim=_model_int(cfg, "time_pair_hidden_dim", default=256),
            mlp_hidden_dim=_model_optional_int(cfg, "time_pair_mlp_hidden_dim"),
        )

    def build(self) -> None:
        self.base_model = self.adapter.load_pretrained_video_action_model()
        self.base_model = self.adapter.attach_action_time_pair_conditioner(
            self.base_model,
            self.time_pair_conditioner,
        )
        self.freeze_by_config()

    def freeze_by_config(self) -> None:
        if self.base_model is None:
            raise RuntimeError("Cannot freeze before build(): base_model is not set.")

        model_cfg = self.cfg.get("model")
        freeze_backbone = True
        trainable_modules: list[str] = []
        if isinstance(model_cfg, Mapping):
            freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
            raw_modules = model_cfg.get("trainable_modules", [])
            if isinstance(raw_modules, list):
                trainable_modules = [item for item in raw_modules if isinstance(item, str)]

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        for name, param in self.base_model.named_parameters():
            if any(module_name in name for module_name in trainable_modules):
                param.requires_grad = True

        for param in self.time_pair_conditioner.parameters():
            param.requires_grad = True

    def forward_action_flowmap(
        self,
        a_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
        z_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.base_model is None:
            raise RuntimeError("LingBotActionFlowMapStudent.build() must be called before forward.")

        pred_u = self.adapter.forward_action_flowmap(
            self.base_model,
            a_src=a_src,
            s_src=s_src,
            s_tgt=s_tgt,
            context=context,
            z_condition=z_condition,
        )
        if pred_u.shape != a_src.shape:
            raise ValueError(f"pred_u shape must match a_src shape: {pred_u.shape} != {a_src.shape}")
        return pred_u


def _model_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    value = default
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, Mapping) and key in model_cfg:
        value = model_cfg[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model.{key} must be an int.")
    return value


def _model_optional_int(cfg: Mapping[str, Any], key: str) -> int | None:
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, Mapping) or key not in model_cfg:
        return None
    value = model_cfg[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model.{key} must be an int when provided.")
    return value
