"""FP32 masked regression objectives; nonfinite values fail loudly."""
import torch


def masked_mean(value, mask):
    if value.shape != mask.shape:
        raise ValueError("Exact mask shape required")
    mask = mask.to(device=value.device, dtype=torch.float32)
    count = mask.flatten(1).sum(1)
    result = (value.float() * mask).flatten(1).sum(1) / count.clamp_min(1)
    return torch.where(count > 0, result, torch.zeros_like(result))


def finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {label}")
    return value


def pseudo_huber(residual, c=1e-3):
    """rho_c(d) = sqrt(d^2 + c^2) - c, computed as d^2 / (sqrt(d^2 + c^2) + c): exactly 0 at d = 0,
    L1 for |d| >> c and L2/(2c) for |d| << c."""
    if c <= 0:
        raise ValueError("pseudo-Huber requires c > 0")
    d = residual.float()
    square = d.square()
    return square / (torch.sqrt(square + c * c) + c)


def regression_loss(prediction, target, mask, c=1e-3):
    """Masked pseudo-Huber distance between a differentiable endpoint and its fixed target.

    Returns the batch-mean loss and detached diagnostics: the masked mean absolute residual
    per batch item and the valid element count."""
    if not (prediction.shape == target.shape == mask.shape):
        raise ValueError("Regression shape mismatch")
    residual = prediction.float() - target.detach().float()
    loss = masked_mean(pseudo_huber(residual, c), mask).mean()
    return finite(loss, "regression loss"), {"residual_l1": masked_mean(residual.abs().detach(), mask),
                                             "valid_count": mask.flatten(1).sum(1).detach()}
