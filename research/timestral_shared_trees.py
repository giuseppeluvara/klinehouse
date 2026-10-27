"""Bounded nonlinear shared classifier with portable numeric tree exports."""

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from . import timestral_universal as shared

PARAMETERS = dict(
    learning_rate=0.03,
    max_iter=120,
    max_leaf_nodes=7,
    min_samples_leaf=200,
    l2_regularization=10.0,
    max_bins=63,
    early_stopping=False,
    random_state=271030,
)


def raw_predict(fit, x):
    if not np.isfinite(x).all() or x.shape[1] != 23:
        raise ValueError("finite registered features required")
    logits = np.tile(fit["initial"], (len(x), 1))
    rows = np.arange(len(x))
    for iteration in fit["trees"]:
        for label, tree in enumerate(iteration):
            a = np.asarray(tree)
            idx = np.zeros(len(x), dtype=int)
            for _ in range(len(a)):
                node = a[idx]
                live = node[:, 5] == 0
                if not live.any():
                    break
                right = x[rows, node[:, 1].astype(int)] > node[:, 2]
                child = np.where(right, node[:, 4], node[:, 3]).astype(int)
                idx = np.where(live, child, idx)
            else:
                raise ValueError("cyclic or invalid tree")
            logits[:, label] += a[idx, 0]
    return softmax(logits, axis=1)


def fit(x, y, ends, cutoff):
    if len(y) < 100 or np.any(ends > cutoff) or not np.isfinite(x).all():
        raise ValueError("finite matured training examples required")
    a, b = shared.augment(x, y)
    estimator = HistGradientBoostingClassifier(**PARAMETERS).fit(a, b)
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("all three classes required")
    trees = []
    for iteration in estimator._predictors:
        group = []
        for predictor in iteration:
            nodes = predictor.nodes
            if np.any(nodes["is_categorical"]):
                raise ValueError("categorical tree export unsupported")
            group.append(
                [
                    [
                        float(n["value"]),
                        int(n["feature_idx"]),
                        float(n["num_threshold"]),
                        int(n["left"]),
                        int(n["right"]),
                        int(n["is_leaf"]),
                    ]
                    for n in nodes
                ]
            )
        trees.append(group)
    result = {
        "cutoff": int(cutoff),
        "n": len(y),
        "initial": estimator._baseline_prediction[0].tolist(),
        "trees": trees,
    }
    np.testing.assert_allclose(
        raw_predict(result, x[:128]),
        estimator.predict_proba(x[:128]),
        atol=1e-12,
        rtol=1e-12,
    )
    return result


def predict(fit, x, anchors):
    if np.any(anchors < fit["cutoff"]) or np.any(anchors + 360 > shared.BOUNDARY):
        raise ValueError("future fit or protected target")
    mirrored = x.copy()
    mirrored[:, shared.ODD] *= -1
    return (raw_predict(fit, x) + raw_predict(fit, mirrored)[:, ::-1]) / 2
