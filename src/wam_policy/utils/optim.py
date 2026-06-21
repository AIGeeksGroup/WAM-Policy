from __future__ import annotations

from typing import Iterable

import torch


def build_adamw_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
):
    param_list = list(params)
    try:
        return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)
    except ModuleNotFoundError as exc:
        if exc.name != "sympy":
            raise
        return _FallbackAdamW(param_list, lr=lr, weight_decay=weight_decay)


class _FallbackAdamW:
    """Small optimizer fallback for local Torch builds missing optional sympy."""

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
    ) -> None:
        self.param_groups = [{"params": params, "lr": lr, "weight_decay": weight_decay}]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()

    def step(self) -> None:
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if weight_decay:
                        param.mul_(1.0 - lr * weight_decay)
                    param.add_(param.grad, alpha=-lr)

    def state_dict(self) -> dict:
        return {
            "state": {},
            "param_groups": [
                {
                    "lr": group["lr"],
                    "weight_decay": group["weight_decay"],
                    "params": list(range(len(group["params"]))),
                }
                for group in self.param_groups
            ],
        }
