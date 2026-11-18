"""Rank-0 LoRA displacement diagnostics for W&B; never changes or stops training.

For each target layer W (out x in) with adapter s*B@A (B: out x r, A: r x in), the QR of
A^T = Q R gives the nonzero squared singular values of B@A as the eigenvalues of the r x r
matrix R B^T B R^T. Frobenius/spectral norms and the top-8 energy share follow exactly,
without materializing B@A. Base-weight norms and the initial A are cached once (A0 on CPU).
"""
import math
import statistics
import traceback

import torch

from arm_opsd.model.lora import LoRALinear


def lr_sum(config, steps):
    """Sum of the student lrs applied in the first `steps` updates (linear warm-up, then constant)."""
    warmup = config.experiment.student_lr_warmup_steps or 0
    base = config.method.student_lr
    return sum(base * (min(1., (s + 1) / warmup) if warmup else 1.) for s in range(steps))


@torch.no_grad()
def spectral_norm(weight, iterations=100):
    w = weight.float()
    v = torch.ones(w.shape[1], device=w.device) / math.sqrt(w.shape[1])
    for _ in range(iterations):
        v = w.T @ (w @ v)
        v = v / v.norm().clamp_min(1e-30)
    return float((w @ v).norm())


class AdapterMetrics:
    def __init__(self, model):
        self.layers = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    self.layers[name] = {"a0": module.lora_A.detach().cpu().clone(), "scale": module.scale,
                                         "w_fro": float(module.base.weight.float().norm()),
                                         "w_spec": spectral_norm(module.base.weight)}

    @torch.no_grad()
    def summary(self, adapters, applied_lr_sum):
        """adapters: layer name -> (A, B). Returns flat finite metrics (no prefix)."""
        fro, spec, top8 = [], [], []
        b_square = b_count = drift = a0_square = 0.
        for name, (a, b) in adapters.items():
            layer = self.layers[name]
            a, b = a.float(), b.float()
            b_square += float(b.square().sum()); b_count += b.numel()
            a0 = layer["a0"].to(a.device)
            drift += float((a - a0).square().sum()); a0_square += float(a0.square().sum())
            r = torch.linalg.qr(a.T, mode="r").R
            gram = (r @ (b.T @ b) @ r.T).double()
            eig = torch.linalg.eigvalsh((gram + gram.T) / 2).clamp_min(0).flip(0) * layer["scale"] ** 2
            energy = float(eig.sum())
            fro.append(math.sqrt(energy) / layer["w_fro"])
            spec.append(math.sqrt(float(eig[0])) / layer["w_spec"])
            if energy > 0:
                top8.append(float(eig[:8].sum()) / energy)
        result = {"b_rms": math.sqrt(b_square / b_count), "a_drift_rms": math.sqrt(drift / a0_square),
                  "rel_fro_median": statistics.median(fro), "rel_fro_max": max(fro),
                  "rel_spec_median": statistics.median(spec), "rel_spec_max": max(spec)}
        if applied_lr_sum > 0:
            result["b_rms_over_lr_sum"] = result["b_rms"] / applied_lr_sum
        if top8:
            result["top8_energy_mean"] = sum(top8) / len(top8)
        return {k: v for k, v in result.items() if math.isfinite(v)}

    def log_values(self, model, ema, config, steps):
        """W&B keys adapter/* (student) and adapter_ema/* (EMA, when present); {} on any failure."""
        try:
            applied = lr_sum(config, steps)
            modules = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
            data = {"adapter/" + k: v for k, v in self.summary(
                {n: (m.lora_A, m.lora_B) for n, m in modules.items()}, applied).items()}
            data["adapter/lr_sum"] = applied
            if ema is not None:
                pairs = {n: (ema[n + ".lora_A"], ema[n + ".lora_B"]) for n in modules}
                data.update({"adapter_ema/" + k: v for k, v in self.summary(pairs, applied).items()})
            return data
        except Exception:
            traceback.print_exc()
            return {}
