from __future__ import annotations

from typing import Mapping


def log_step_metrics(step: int, metrics: Mapping[str, float]) -> None:
    metric_text = " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
    print(f"step={step} {metric_text}")
