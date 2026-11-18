"""FP32 pseudo-Huber regression of the student's Euler endpoint onto the teacher ODE endpoint.

rho_c(d) = sqrt(d^2 + c^2) - c per element, masked mean; no critic, no
per-sample normalizer. The loss is a real distance with a floor of zero."""
import torch


def valid_mask(value, mask=None):
    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    if mask.shape != value.shape or mask.dtype != torch.bool:
        raise ValueError("Exact-shape boolean target mask required")
    if not torch.isfinite(value[mask]).all():
        raise FloatingPointError("Nonfinite valid target elements")
    return mask


def pseudo_huber(difference, c):
    if not 0 < float(c) < 1:
        raise ValueError("Pseudo-Huber c must lie in (0, 1)")
    d = difference.float()
    return torch.sqrt(d.square() + c * c) - c


def regression_loss(prediction, target, mask=None, *, c):
    """Masked mean of rho_c(prediction - target); the target is a constant.

    Returns (loss, metrics) with the valid element count and the masked L1 residual."""
    if prediction.shape != target.shape:
        raise ValueError("Regression prediction/target shape mismatch")
    mask = valid_mask(prediction, mask)
    valid_mask(target, mask)
    n = int(mask.sum())
    if not n:
        return prediction.reshape(-1)[:0].sum().float(), {"valid": 0, "residual_l1": 0.}
    difference = prediction.float()[mask] - target.detach().float()[mask]
    loss = pseudo_huber(difference, c).mean()
    return loss, {"valid": n, "residual_l1": float(difference.detach().abs().mean())}
