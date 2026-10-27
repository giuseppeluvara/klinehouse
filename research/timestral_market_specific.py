"""Fixed asset agreement and equal-weight local/common hourly signals."""

import numpy as np


def signals(x):
    if x.shape[-1] < 17 or not np.isfinite(x).all():
        raise ValueError("finite causal spot features required")
    common = np.clip(-(x[..., 3] + 2 * x[..., 16]) / 3, -3, 3)
    own = np.clip(-x[..., 3], -3, 3)
    return common, own, (common + own) / 2


def consensus(q, common, own):
    if (
        q.shape != common.shape
        or q.shape != own.shape
        or not np.isfinite(q).all()
        or np.any((q <= 0) | (q >= 1))
    ):
        raise ValueError("aligned finite probabilities required")
    return np.where(common * own > 0, q, 0.5)


def diagnose(y, q, flat, common, own):
    selected = (1 - flat) * np.maximum(q, 1 - q) >= 0.52
    moving = y != 1
    correct = (q >= 0.5) == (y == 2)
    own_correct = (own >= 0) == (y == 2)
    result = dict(
        n=len(y),
        flat_rate=float(np.mean(~moving)),
        flat_probability=float(flat.mean()),
        selected=int(selected.sum()),
        selected_flat_rate=float(np.mean(~moving[selected])),
        selected_binary_accuracy=float(correct[selected & moving].mean()),
        selected_all_accuracy=float((correct[selected] & moving[selected]).mean()),
        selected_mean_flat_probability=float(flat[selected].mean()),
        groups={},
    )
    product = common * own
    for name, mask in (
        ("agree", product > 0),
        ("disagree", product < 0),
        ("zero", product == 0),
    ):
        use = mask & moving
        chosen = use & selected
        result["groups"][name] = dict(
            all_n=int(mask.sum()),
            nonflat_n=int(use.sum()),
            selected_nonflat_n=int(chosen.sum()),
            common_accuracy=float(correct[use].mean()) if use.any() else None,
            own_accuracy=float(own_correct[use].mean()) if use.any() else None,
            selected_common_accuracy=float(correct[chosen].mean())
            if chosen.any()
            else None,
        )
    return result
