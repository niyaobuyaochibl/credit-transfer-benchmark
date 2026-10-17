"""Step 5: per-person scores for binary choices, calibration and per-session effect sizes.

Arrays are people x trials. `probability` is the predicted probability of choice 1;
`choices` holds 0, 1, or -1 for a missing or invalid choice, which is ignored.
"""
import numpy as np


def _checked(probability, choices):
    p = np.asarray(probability, dtype=float)
    a = np.asarray(choices)
    assert p.ndim == 2 and p.shape == a.shape, 'expected people x trials arrays of equal shape'
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'probabilities must lie in [0, 1]'
    assert np.isin(a, [-1, 0, 1]).all(), 'choices must be 0, 1 or -1 (missing)'
    return p, a, a >= 0


def person_scores(probability, choices, eps=1e-9):
    """Mean negative log-likelihood (nats per valid choice), Brier score and accuracy per person."""
    p, a, valid = _checked(probability, choices)
    n = valid.sum(1)
    assert (n > 0).all(), 'every person needs at least one valid choice'
    q = np.clip(p, eps, 1 - eps)
    terms = {'nll': -np.where(a == 1, np.log(q), np.log1p(-q)),
             'brier': (p - (a == 1)) ** 2,
             'accuracy': ((p >= 0.5) == (a == 1)).astype(float)}
    out = {k: (v * valid).sum(1) / n for k, v in terms.items()}
    out['valid_count'] = n
    return out


def calibration(probability, choices, bins=10):
    """Fixed-width bins; each person carries total weight one regardless of missing choices."""
    p, a, valid = _checked(probability, choices)
    weight = valid / valid.sum(1, keepdims=True)
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        w = weight * ((index == b) & valid)
        total = w.sum()
        rows.append({'left': b / bins, 'right': (b + 1) / bins, 'valid_choices': int(((index == b) & valid).sum()),
                     'mean_probability': float((w * p).sum() / total) if total else None,
                     'observed_frequency': float((w * (a == 1)).sum() / total) if total else None})
    return rows


def per_session(delta_per_choice, valid_count):
    """Express a per-choice NLL difference per session: summed nats and the implied likelihood ratio."""
    d = np.asarray(delta_per_choice, dtype=float)
    n = np.asarray(valid_count, dtype=float)
    assert d.shape == n.shape
    total = d * n
    return {'valid_choices_per_session': {'min': float(n.min()), 'mean': float(n.mean()), 'max': float(n.max())},
            'mean_nats_per_session': float(total.mean()),
            'likelihood_ratio_at_mean': float(np.exp(abs(total.mean()))),
            'favours': 'candidate' if total.mean() > 0 else 'baseline' if total.mean() < 0 else 'neither'}
