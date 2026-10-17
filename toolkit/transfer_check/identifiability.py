"""Step 4: can the candidate mechanisms be told apart under a given format?

Simulate learners whose mechanism is known, fit every candidate on part of the
simulated people, score each on the rest, and tabulate which candidate predicts best.
A mechanism is recovered if its own model wins. Exact ties (identical predictions,
e.g. two models that receive the same input under that format) are reported as ties.
"""
import numpy as np


def recoverability(generators, candidates, simulate, fit_and_score, seed=0, tol=1e-12):
    """Confusion table of mean held-out NLL.

    simulate(generator, rng) -> (train, test) data for one simulated cohort.
    fit_and_score(candidate, train, test) -> per-person NLL on `test`.
    """
    rng = np.random.default_rng(seed)
    nll = {}
    winners = {}
    for g in generators:
        train, test = simulate(g, rng)
        row = {c: float(np.mean(fit_and_score(c, train, test))) for c in candidates}
        nll[g] = row
        best = min(row.values())
        winners[g] = sorted(c for c, v in row.items() if v - best <= tol)
    recovered = {g: winners[g] == [g] for g in generators if g in candidates}
    return {'mean_nll': nll, 'best': winners, 'recovered': recovered,
            'all_recovered': bool(recovered) and all(recovered.values())}
