"""Steps 2-3: fixed baseline selection and the primary paired transfer contrast.

Contrasts are baseline loss minus candidate loss, so positive values favour the candidate.
The unit of resampling is the person; people are resampled within strata (for example,
which format they saw first) so that every bootstrap sample keeps the design's balance.
"""
import numpy as np


def select_baseline(development_loss, exclude=()):
    """Lowest mean development loss wins; ties go to the alphabetically first name."""
    scores = {name: float(np.mean(v)) for name, v in development_loss.items() if name not in set(exclude)}
    assert scores, 'no eligible baseline'
    return min(scores, key=lambda n: (scores[n], n)), scores


def stratified_bootstrap(strata, draws=50000, seed=0):
    """Index matrix (draws x people) that resamples people with replacement within each stratum."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    columns = [rng.choice(np.flatnonzero(strata == s), (draws, int((strata == s).sum())), replace=True)
               for s in sorted(set(strata.tolist()))]
    return np.concatenate(columns, axis=1)


def describe(values, samples):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and np.isfinite(x).all()
    low, high = np.quantile(x[samples].mean(1), [0.025, 0.975])
    return {'mean': float(x.mean()), 'ci95': [float(low), float(high)],
            'people_favouring_candidate': int((x > 0).sum()), 'people_favouring_baseline': int((x < 0).sum()),
            'people': int(len(x))}


def transfer_contrast(baseline, candidate, strata, min_effect=0.0, draws=50000, seed=0):
    """Primary contrast averaged over transfer directions, plus each direction separately.

    `baseline` and `candidate` map a direction label (e.g. 'summed->component') to a
    vector of per-person held-out losses; positions must refer to the same people in
    every vector. The decision requires all three prespecified conditions.
    """
    assert set(baseline) == set(candidate) and baseline, 'directions must match'
    directions = sorted(baseline)
    diffs = {d: np.asarray(baseline[d], float) - np.asarray(candidate[d], float) for d in directions}
    assert len({len(v) for v in diffs.values()} | {len(strata)}) == 1, 'every direction needs every person'
    samples = stratified_bootstrap(strata, draws, seed)
    primary = describe(np.mean([diffs[d] for d in directions], axis=0), samples)
    per_direction = {d: describe(diffs[d], samples) for d in directions}
    checks = {'lower_bound_above_zero': primary['ci95'][0] > 0,
              'every_direction_nonnegative': all(r['mean'] >= 0 for r in per_direction.values()),
              f'effect_at_least_{min_effect}': primary['mean'] >= min_effect}
    return {'sign': 'baseline loss minus candidate loss; positive favours the candidate',
            'primary': primary, 'directions': per_direction, 'checks': checks,
            'candidate_supported': all(checks.values()),
            'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'person', 'strata': sorted(set(np.asarray(strata).tolist()))}}
