"""Worked example: the six checklist steps on simulated data (about a minute on a laptop CPU).

Task 1 has two options. Each choice yields an immediate outcome and a delayed outcome
that arrives on the next trial. Component-wise feedback reports the two parts
separately; summed feedback reports only their total, so part of it may belong to the
previous choice. Every simulated person completes one session in each format, in
counterbalanced order. Half of the simulated people credit delayed outcomes to the
choice that caused them (multi-credit); the other half ignore them (single-credit).

Run from the repository's toolkit folder:  python examples/two_format_bandit.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import transfer_check as tc  # noqa: E402

OUT = HERE / 'output'
FORMATS = {'component': np.eye(2), 'summed': np.ones((1, 2))}
SHARED = tc.common_observation(list(FORMATS.values()))
MODELS = ('multi_credit', 'single_credit', 'summed_only')
CANDIDATE = 'multi_credit'
ALPHAS, BETAS = np.linspace(0.05, 0.95, 19), np.geomspace(1, 20, 19)
TASK1 = {'delay': 1, 'p_now': (0.6, 0.3), 'p_later': (0.1, 0.6), 'trials': 150, 'missing': 0.01}
TASK2 = {'delay': 2, 'p_now': (0.5, 0.2), 'p_later': (0.2, 0.7), 'trials': 150, 'missing': 0.01}


def _get(q, a):
    return np.take_along_axis(q, a[..., None], -1)[..., 0]


def _add(q, a, x):
    np.put_along_axis(q, a[..., None], (_get(q, a) + x)[..., None], -1)


def update(model, fmt, qi, qd, a, a_prev, obs, alpha):
    """One learning step; a or a_prev of -1 means no choice was made on that trial."""
    now_ok, prev_ok = a >= 0, a_prev >= 0
    a, a_prev = np.where(now_ok, a, 0), np.where(prev_ok, a_prev, 0)
    rate = alpha * now_ok
    if model == 'summed_only':
        total = tc.project(obs, FORMATS[fmt], SHARED)[..., 0]
        _add(qi, a, rate * (total - _get(qi, a)))
    elif model == 'single_credit':
        _add(qi, a, rate * (obs[..., 0] - _get(qi, a)))
    elif fmt == 'component':
        _add(qi, a, rate * (obs[..., 0] - _get(qi, a)))
        _add(qd, a_prev, alpha * prev_ok * now_ok * (obs[..., 1] - _get(qd, a_prev)))
    else:
        error = obs[..., 0] - _get(qi, a) - prev_ok * _get(qd, a_prev)
        _add(qi, a, rate * error)
        _add(qd, a_prev, alpha * prev_ok * now_ok * error)


def choice_probability(qi, qd, beta):
    v = qi + qd
    return 1 / (1 + np.exp(-beta * (v[..., 1] - v[..., 0])))


def simulate(model, fmt, alpha, beta, task, rng):
    n, t_max, delay = len(alpha), task['trials'], task['delay']
    qi, qd = np.zeros((n, 2)), np.zeros((n, 2))
    choices = np.full((n, t_max), -1)
    latent = np.zeros((n, t_max, 2))
    arriving = np.zeros((n, t_max + delay))
    for t in range(t_max):
        a = (rng.random(n) < choice_probability(qi, qd, beta)).astype(int)
        a[rng.random(n) < task['missing']] = -1
        made = a >= 0
        now = made * (rng.random(n) < np.asarray(task['p_now'])[np.maximum(a, 0)])
        arriving[:, t + delay] = made * (rng.random(n) < np.asarray(task['p_later'])[np.maximum(a, 0)])
        latent[:, t] = np.stack([now, arriving[:, t] * made], -1)
        choices[:, t] = a
        a_prev = choices[:, t - delay] if t >= delay else np.full(n, -1)
        update(model, fmt, qi, qd, a, a_prev, latent[:, t] @ FORMATS[fmt].T, alpha)
    return choices, latent


def predict(model, fmt, alpha, beta, choices, latent, delay):
    """Probability of choice 1 for each parameter row (alpha, beta have shape (G, 1))."""
    g, (n, t_max) = len(alpha), choices.shape
    qi, qd = np.zeros((g, n, 2)), np.zeros((g, n, 2))
    out = np.zeros((g, n, t_max))
    obs = latent @ FORMATS[fmt].T
    for t in range(t_max):
        out[:, :, t] = choice_probability(qi, qd, beta)
        a = np.broadcast_to(choices[:, t], (g, n))
        a_prev = np.broadcast_to(choices[:, t - delay] if t >= delay else np.full(n, -1), (g, n))
        update(model, fmt, qi, qd, a, a_prev, np.broadcast_to(obs[:, t], (g, n, obs.shape[-1])), alpha)
    return out


def fit(model, fmt, choices, latent, delay):
    """Population maximum likelihood on a grid: one (alpha, beta) for all fitting people."""
    a, b = np.meshgrid(ALPHAS, BETAS, indexing='ij')
    p = predict(model, fmt, a.reshape(-1, 1), b.reshape(-1, 1), choices, latent, delay)
    q = np.clip(p, 1e-9, 1 - 1e-9)
    ll = np.where(choices == 1, np.log(q), np.where(choices == 0, np.log1p(-q), 0)).sum((1, 2))
    best = int(np.argmax(ll))
    return {'alpha': float(a.ravel()[best]), 'beta': float(b.ravel()[best])}


def score(model, fmt, params, choices, latent, delay):
    p = predict(model, fmt, np.array([[params['alpha']]]), np.array([[params['beta']]]), choices, latent, delay)[0]
    return p, tc.person_scores(p, choices)


def cohort(n, task, rng, fmt_order=None):
    """Mixed population; each person completes one session per format (or one format if fmt_order is a string)."""
    kind = np.where(np.arange(n) % 2 == 0, 'multi_credit', 'single_credit')
    alpha, beta = rng.uniform(0.1, 0.5, n), rng.uniform(3, 10, n)
    formats = [fmt_order] if isinstance(fmt_order, str) else list(FORMATS)
    sessions = {f: (np.zeros((n, task['trials']), int), np.zeros((n, task['trials'], 2))) for f in formats}
    for k in ('multi_credit', 'single_credit'):
        m = kind == k
        for f in formats:
            c, z = simulate(k, f, alpha[m], beta[m], task, rng)
            sessions[f][0][m], sessions[f][1][m] = c, z
    return sessions


def take(sessions, rows):
    return {f: (c[rows], z[rows]) for f, (c, z) in sessions.items()}


def main():
    rng = np.random.default_rng(20261017)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    report = {}

    # Step 1: define the shift. Formats are observation matrices; the shared feature is the total.
    report['step1_shared_feature'] = SHARED.tolist()
    directions = {'component->summed': ('component', 'summed'), 'summed->component': ('summed', 'component')}

    n = 90
    data = cohort(n, TASK1, rng)
    first = np.where(np.arange(n) % 2 == 0, 'component', 'summed')
    order = rng.permutation(n)
    split = {}
    for s in ('component', 'summed'):
        rows = order[first[order] == s]
        split.update({'fit': list(split.get('fit', [])) + list(rows[:20]),
                      'development': list(split.get('development', [])) + list(rows[20:30]),
                      'heldout': list(split.get('heldout', [])) + list(rows[30:])})
    split = {k: np.sort(np.asarray(v)) for k, v in split.items()}

    # Step 2: every model, including the shift-invariant control (summed_only), fitted on each source format.
    fits = {f'{m}__{src}': fit(m, src, *take(data, split['fit'])[src], TASK1['delay'])
            for m in MODELS for src in FORMATS}
    report['step2_fits'] = fits

    # Step 3: fixed selection on development people, then freeze before held-out access.
    dev = take(data, split['development'])
    dev_loss = {m: np.mean([score(m, tgt, fits[f'{m}__{src}'], *dev[tgt], TASK1['delay'])[1]['nll']
                            for src, tgt in directions.values()]) for m in MODELS}
    baseline, dev_scores = tc.select_baseline(dev_loss, exclude=[CANDIDATE])
    frozen = OUT / 'frozen'
    frozen.mkdir()
    (frozen / 'fits.json').write_text(json.dumps(fits, indent=1), encoding='utf-8')
    (frozen / 'selection.json').write_text(json.dumps({'baseline': baseline, 'development_nll': dev_scores}, indent=1),
                                           encoding='utf-8')
    gate = tc.Gate(OUT / 'gate')
    code = [Path(__file__)] + sorted((HERE.parent / 'transfer_check').glob('*.py'))
    gate.freeze(code + [frozen / 'fits.json', frozen / 'selection.json'],
                plan={'primary': f'{baseline} minus {CANDIDATE}, mean over both directions', 'min_effect': 0.005})
    gate.open_heldout('primary transfer contrast')
    report['step3_baseline'] = {'selected': baseline, 'development_nll': dev_scores, 'accesses': len(gate.accesses())}

    held = take(data, split['heldout'])
    losses, counts, calib = {m: {} for m in MODELS}, [], {}
    for name, (src, tgt) in directions.items():
        for m in MODELS:
            p, s = score(m, tgt, fits[f'{m}__{src}'], *held[tgt], TASK1['delay'])
            losses[m][name] = s['nll']
            if m == baseline:
                calib[name] = tc.calibration(p, held[tgt][0], bins=5)
        counts.append(s['valid_count'])
    strata = first[split['heldout']]
    primary = tc.transfer_contrast(losses[baseline], losses[CANDIDATE], strata, min_effect=0.005, draws=20000, seed=1)
    report['step3_primary'] = primary
    report['step1_candidate_vs_single_credit_by_direction'] = tc.transfer_contrast(
        losses['single_credit'], losses[CANDIDATE], strata, draws=20000, seed=2)['directions']

    # Step 4: identifiability under each format, with known generating mechanisms.
    def sim(fmt):
        def run(generator, r):
            alpha, beta = r.uniform(0.2, 0.4, 60), r.uniform(4, 8, 60)
            c, z = simulate(generator, fmt, alpha, beta, TASK1, r)
            return (c[:40], z[:40]), (c[40:], z[40:])
        return run

    def fit_and_score(fmt):
        return lambda m, train, test: score(m, fmt, fit(m, fmt, *train, TASK1['delay']), *test, TASK1['delay'])[1]['nll']
    report['step4_recoverability'] = {fmt: tc.recoverability(MODELS, MODELS, sim(fmt), fit_and_score(fmt), seed=3)
                                      for fmt in FORMATS}

    # Step 5: per-session scale, per-person counts and calibration.
    delta = np.mean([losses[baseline][d] - losses[CANDIDATE][d] for d in directions], axis=0)
    report['step5_per_session'] = tc.per_session(delta, np.mean(counts, axis=0))
    report['step5_baseline_calibration'] = calib

    # Step 6: refit every model on a second task (two-trial delay, attributable component-wise feedback)
    # and rank them on new people; tied models share a rank.
    task2 = cohort(60, TASK2, rng, fmt_order='component')
    train, test = take(task2, np.arange(30))['component'], take(task2, np.arange(30, 60))['component']
    nll2 = {m: float(score(m, 'component', fit(m, 'component', *train, TASK2['delay']), *test, TASK2['delay'])[1]['nll'].mean())
            for m in MODELS}
    report['step6_second_task'] = {'heldout_nll': nll2, 'rank_of_selected_baseline':
                                   1 + sum(v < nll2[baseline] for v in nll2.values())}

    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=float), encoding='utf-8')
    p = report['step3_primary']
    print(f"shared feature under both formats: {report['step1_shared_feature']}")
    print(f"selected baseline: {baseline}; development NLL {json.dumps({k: round(v, 4) for k, v in dev_scores.items()})}")
    print(f"primary contrast ({baseline} minus {CANDIDATE}): {p['primary']['mean']:+.4f} nats per choice, "
          f"95% CI {p['primary']['ci95'][0]:+.4f} to {p['primary']['ci95'][1]:+.4f}; "
          f"{p['primary']['people_favouring_candidate']} of {p['primary']['people']} people favour the candidate")
    for d, r in p['directions'].items():
        print(f"  {d}: {r['mean']:+.4f} ({r['ci95'][0]:+.4f} to {r['ci95'][1]:+.4f})")
    print(f"checks: {p['checks']}; candidate supported: {p['candidate_supported']}")
    for fmt, r in report['step4_recoverability'].items():
        print(f"recoverability under {fmt} feedback: best model per generator {r['best']}")
    s = report['step5_per_session']
    print(f"per session: {s['mean_nats_per_session']:+.2f} nats (likelihood ratio {s['likelihood_ratio_at_mean']:.3g}, "
          f"favours {s['favours']})")
    print(f"second task: held-out NLL {json.dumps({k: round(v, 4) for k, v in nll2.items()})}; "
          f"selected baseline ranks {report['step6_second_task']['rank_of_selected_baseline']} of {len(MODELS)}")
    print(f'full report: {OUT / "report.json"}')


if __name__ == '__main__':
    main()
