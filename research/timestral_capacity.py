"""A fixed model-size/context ablation with an explicit historical screen."""

from pathlib import Path

import numpy as np

from . import timestral_experiment as shared

MODELS = {
    "5m": {
        "id": shared.MODEL_ID,
        "revision": shared.WEIGHT_REVISION,
        "sha256": shared.CHECKPOINT_SHA256["model.safetensors"],
        "config_sha256": shared.CHECKPOINT_SHA256["config.json"],
    },
    "28m": {
        "id": "mistralai/Timestral-28M",
        "revision": "49b26409ac87b5935539a5d8ffe7f397b720db47",
        "sha256": "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d",
        "config_sha256": "0cdf44b8ea981191de53078e0a5ecbbd437046bd939a546d5dbe17868f71281f",
    },
}
CONDITIONS = tuple(f"{size}_{context}" for size in MODELS for context in (256, 2048))


class Forecast:
    """Reuse exactly the established inference adapter with a verified checkpoint."""

    predict = shared.Timestral.predict

    def __init__(self, size, cache):
        import torch
        from timestral import TimestralForecaster

        shared.verify_model_installation()
        specification = MODELS[size]
        path = (
            Path(cache)
            / ("models--" + specification["id"].replace("/", "--"))
            / "snapshots"
            / specification["revision"]
        )
        if (
            shared.digest_file(path / "model.safetensors") != specification["sha256"]
            or shared.digest_file(path / "config.json")
            != specification["config_sha256"]
        ):
            raise ValueError("checkpoint checksum mismatch")
        self.checkpoint_files = {
            name: shared.digest_file(path / name)
            for name in ("config.json", "model.safetensors")
        }
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.forecaster = TimestralForecaster.from_pretrained(
            path, device="cpu", max_context=2048, local_files_only=True
        )


def screen(rows, name):
    """A preregistered screen, never sufficient to claim a functioning solution."""
    if not rows:
        raise ValueError("empty evaluation")
    actual = np.asarray([r["actual_log_return"] for r in rows])
    predicted = np.asarray([r["predictions"][name]["point"] for r in rows])
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("nonfinite outcome or forecast")
    error = predicted - actual
    differences = abs(error) - abs(actual)
    blocks = np.asarray(
        [(shared.utc(r["anchor"]) - shared.utc("2026-01-01")).days // 7 for r in rows]
    )
    n = int(blocks.max() + 1)
    sizes, sums = (
        np.bincount(blocks, minlength=n),
        np.bincount(blocks, weights=differences, minlength=n),
    )
    choices = np.random.default_rng(20261026).integers(0, n, (4000, n))
    interval = np.quantile(
        sums[choices].sum(axis=1) / sizes[choices].sum(axis=1), [0.003125, 0.996875]
    )
    months = sorted({r["anchor"][:7] for r in rows})
    wins = sum(
        float(
            differences[
                [i for i, r in enumerate(rows) if r["anchor"][:7] == month]
            ].mean()
        )
        < 0
        for month in months
    )
    improvement = (
        float(1 - abs(error).mean() / abs(actual).mean())
        if abs(actual).mean() > 0
        else None
    )
    checks = {
        "mae_gain_at_least_one_percent": improvement is not None
        and improvement >= 0.01,
        "rmse_no_worse": bool(np.mean(error**2) <= np.mean(actual**2)),
        "at_least_five_monthly_wins": wins >= 5,
        "adjusted_upper_below_zero": float(interval[1]) < 0,
    }
    return {
        "passes_replay_screen": all(checks.values()),
        "checks": checks,
        "monthly_wins": wins,
        "mae_improvement": improvement,
        "interval99_375": interval.tolist(),
        "solution_found": False,
        "independent_validation": "not performed; previously observed history",
    }
