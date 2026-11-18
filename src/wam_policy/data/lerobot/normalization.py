"""Train-split native9 action normalization: affine map of the 1%/99% quantiles to [-1, 1].

No clipping; constant dimensions keep scale 1; the three force dimensions stay offset 0 / scale 1.
"""
import numpy as np

from wam_policy.common.action_space import ActionNormalizer


def fit_training_normalizer(actions):
    """``actions``: iterable of [N,9] executed-action arrays from train episodes only."""
    arrays = []
    for a in actions:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 9 or not len(a) or not np.isfinite(a).all() or np.any(a[:, 6:] != 0):
            raise ValueError("Expected finite [N,9] actions with zero force dimensions")
        arrays.append(a)
    if not arrays:
        raise ValueError("No training actions")
    values = np.concatenate(arrays, axis=0)
    q01, q99 = np.quantile(values, [.01, .99], axis=0, method="linear")
    offset, scale = (q01 + q99) / 2, (q99 - q01) / 2
    constant = np.flatnonzero(scale <= 1e-8)
    scale[constant] = 1.
    offset[6:], scale[6:] = 0., 1.
    normalizer = ActionNormalizer(tuple(offset.tolist()), tuple(scale.tolist()))
    stats = {"episodes": len(arrays), "rows": len(values), "q01": q01.tolist(), "q99": q99.tolist(),
             "constant_dimensions": constant.tolist(), "method": "linear quantiles 1%/99%, affine, no clipping"}
    return normalizer, stats
