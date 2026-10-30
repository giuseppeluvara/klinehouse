"""Delayed Bayesian orientation of issued forecasts, separate from the live pilot.

This is an offline research policy. It never fits or changes the shadow models.
Each asset/cohort starts two equal-prior learners to handle the target delay.
"""

import numpy as np


EXPERTS = ("reversal", "continuation", "neutral")


def replay(probability, issue, label_at, label):
    """Mix q, 1-q and 1/2 using only earlier, available same-lane outcomes.

    ``label_at`` means actual outcome receipt in a prospective journal, and the
    registered target end in the historical conditional-availability replay.
    A negative label denotes an unobserved outcome. Flat labels carry no
    information about conditional up/down orientation and never change weights.
    """
    p = np.asarray(probability, dtype=float)
    issue = np.asarray(issue, dtype=float)
    label_at = np.asarray(label_at, dtype=float)
    label = np.asarray(label)
    n = len(p)
    if (
        p.shape != (n, 3)
        or issue.shape != (n,)
        or label_at.shape != (n,)
        or label.shape != (n,)
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or np.any(p >= 1)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-9)
        or not np.isfinite(issue).all()
        or np.any(np.diff(issue) <= 0)
        or not np.isin(label, [-1, 0, 1, 2]).all()
        or np.any(~np.isfinite(label_at[label >= 0]))
        or np.any(label_at[label >= 0] <= issue[label >= 0])
    ):
        raise ValueError(
            "aligned interior probabilities and causal outcome times required"
        )
    q = p[:, 2] / (p[:, 0] + p[:, 2])
    expert_up = np.column_stack((q, 1 - q, np.full(n, 0.5)))
    weights = np.empty((n, 3))
    counts = np.empty(n, dtype=np.int64)
    state = np.zeros((2, 3))
    learned = np.zeros(2, dtype=np.int64)
    pending = [[], []]
    for i in range(n):
        lane = i % 2
        keep = []
        for j in pending[lane]:
            if label[j] >= 0 and label_at[j] <= issue[i]:
                if label[j] != 1:
                    likelihood = expert_up[j] if label[j] == 2 else 1 - expert_up[j]
                    state[lane] += np.log(likelihood)
                    learned[lane] += 1
            else:
                keep.append(j)
        pending[lane] = keep
        value = np.exp(state[lane] - np.max(state[lane]))
        weights[i] = value / value.sum()
        counts[i] = learned[lane]
        pending[lane].append(i)
    # Equal orientation weights must yield exactly neutral probabilities.
    adaptive_up = 0.5 + (weights[:, 0] - weights[:, 1]) * (q - 0.5)

    def full(up):
        return np.column_stack(((1 - p[:, 1]) * (1 - up), p[:, 1], (1 - p[:, 1]) * up))

    return {
        "adaptive": full(adaptive_up),
        "neutral": full(np.full(n, 0.5)),
        "mirrored": p[:, [2, 1, 0]].copy(),
        "weights": weights,
        "feedback_count": counts,
    }
