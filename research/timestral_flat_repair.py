"""Fixed shrinkage of flat probabilities without changing conditional direction."""

import numpy as np


def probabilities(up, flat, frequency):
    if not (up.shape == flat.shape == frequency.shape) or any(
        not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1))
        for p in (up, flat, frequency)
    ):
        raise ValueError("aligned strictly interior probabilities required")
    return {
        name: np.column_stack(((1 - f) * (1 - up), f, (1 - f) * up))
        for name, f in (
            ("old", flat),
            ("mixed", (flat + frequency) / 2),
            ("frequency", frequency),
        )
    }


def paired(y, p, control, anchors, mask):
    """Evaluate identical cases with full-day paired seven-day bootstrap blocks."""
    target = np.eye(3)[y]
    flat = y == 1
    own_ll = -np.log(p[np.arange(len(y)), y])
    control_ll = -np.log(control[np.arange(len(y)), y])
    own_flat_ll = -np.log(np.where(flat, p[:, 1], 1 - p[:, 1]))
    control_flat_ll = -np.log(np.where(flat, control[:, 1], 1 - control[:, 1]))
    values = np.column_stack(
        (
            np.square(control - target).sum(1) - np.square(p - target).sum(1),
            control_ll - own_ll,
            np.square(control[:, 1] - flat) - np.square(p[:, 1] - flat),
            control_flat_ll - own_flat_ll,
        )
    )
    days, inv = np.unique(anchors // 86400, return_inverse=True)
    if np.any(np.diff(days) != 1) or not mask.any():
        raise ValueError("complete days and nonempty fixed selection required")
    counts = np.bincount(inv, weights=mask)
    totals = np.column_stack(
        [np.bincount(inv, weights=mask * values[:, j]) for j in range(4)]
    )
    starts = np.random.default_rng(271032).integers(
        0, len(days) - 6, (4000, (len(days) + 6) // 7)
    )
    ids = (starts[:, :, None] + np.arange(7)).reshape(4000, -1)[:, : len(days)]
    den = counts[ids].sum(1)
    if np.any(den == 0):
        raise ValueError("empty bootstrap selection")
    boot = totals[ids].sum(1) / den[:, None]
    intervals = np.quantile(boot, [1 / 360, 359 / 360], axis=0)
    names = ("brier_gain", "logloss_gain", "flat_brier_gain", "flat_logloss_gain")
    result = {name: float(value) for name, value in zip(names, values[mask].mean(0))}
    result.update(
        {name + "_ci99_444": intervals[:, j].tolist() for j, name in enumerate(names)}
    )
    months = anchors.astype("datetime64[s]").astype("datetime64[M]")
    monthly = {
        str(m): float(values[mask & (months == m), 0].mean()) for m in np.unique(months)
    }
    result.update(
        n=int(mask.sum()),
        flats=int(np.sum(mask & flat)),
        monthly_brier_gain=monthly,
        improvement_gate=bool(
            np.all(intervals[0, :2] > 0) and sum(v > 0 for v in monthly.values()) >= 5
        ),
    )
    return result


def decomposition(y, p, baseline, mask):
    """Flat/moving plus moving-weighted conditional direction log loss gains."""
    moving = y != 1
    terms = []
    for q in (baseline, p):
        flat_loss = -np.log(np.where(moving, 1 - q[:, 1], q[:, 1]))
        conditional = q[:, 2] / (1 - q[:, 1])
        direction = -np.log(np.where(y == 2, conditional, 1 - conditional)) * moving
        terms.append([float(flat_loss[mask].mean()), float(direction[mask].mean())])
    gains = np.subtract(*terms)
    return dict(
        flat_gain=float(gains[0]),
        direction_gain=float(gains[1]),
        total_gain=float(gains.sum()),
        selected=int(mask.sum()),
        selected_flats=int(np.sum(mask & ~moving)),
    )
