import json

import numpy as np
import pytest

import transfer_check as tc


def test_shared_feature_of_component_and_summed_feedback_is_the_total():
    shared = tc.common_observation([np.eye(2), np.ones((1, 2))])
    assert shared.tolist() == [[1.0, 1.0]]
    obs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert tc.project(obs, np.eye(2), shared).ravel().tolist() == [1.0, 1.0]
    assert tc.project(obs.sum(1, keepdims=True), np.ones((1, 2)), shared).ravel() == pytest.approx([1.0, 1.0])


def test_disjoint_formats_share_nothing():
    assert tc.common_observation([[[1, 0, 0]], [[0, 1, 0]]]).shape == (0, 3)


def test_person_scores_ignore_missing_choices():
    s = tc.person_scores([[0.5, 0.9, 0.2]], [[1, -1, 0]])
    assert s['valid_count'].tolist() == [2]
    assert s['nll'][0] == pytest.approx((np.log(2) - np.log(0.8)) / 2)


def test_per_session_scales_by_valid_choices():
    r = tc.per_session([-0.025, -0.025], [336, 332])
    assert r['mean_nats_per_session'] == pytest.approx(-0.025 * 334)
    assert r['favours'] == 'baseline'


def test_baseline_selection_excludes_candidate_and_breaks_ties_by_name():
    name, _ = tc.select_baseline({'cand': [0.1], 'b': [0.3], 'a': [0.3]}, exclude=['cand'])
    assert name == 'a'


def test_contrast_sign_and_decision():
    rng = np.random.default_rng(0)
    base = {'x->y': 0.6 + rng.normal(0, 0.01, 20), 'y->x': 0.6 + rng.normal(0, 0.01, 20)}
    cand = {k: v - 0.02 for k, v in base.items()}
    r = tc.transfer_contrast(base, cand, ['a', 'b'] * 10, min_effect=0.005, draws=2000)
    assert r['primary']['mean'] == pytest.approx(0.02)
    assert r['candidate_supported']
    r = tc.transfer_contrast(cand, base, ['a', 'b'] * 10, draws=2000)
    assert not r['candidate_supported'] and r['primary']['people_favouring_baseline'] == 20


def test_bootstrap_keeps_stratum_sizes():
    strata = np.array(['a'] * 3 + ['b'] * 5)
    idx = tc.stratified_bootstrap(strata, draws=50)
    assert ((strata[idx] == 'a').sum(1) == 3).all()


def test_gate_detects_changes_and_binds_manifest(tmp_path):
    f = tmp_path / 'fits.json'
    f.write_text('{"alpha": 0.3}')
    gate = tc.Gate(tmp_path / 'gate')
    with pytest.raises(RuntimeError):
        gate.open_heldout('too early')
    gate.freeze([f])
    gate.open_heldout('primary')
    assert len(gate.accesses()) == 1
    with pytest.raises(RuntimeError):
        gate.freeze([f])
    f.write_text('{"alpha": 0.4}')
    with pytest.raises(RuntimeError, match='changed'):
        gate.open_heldout('again')
    f.write_text('{"alpha": 0.3}')
    m = json.loads(gate.manifest.read_text())
    m['plan'] = 'edited'
    gate.manifest.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match='after first'):
        gate.open_heldout('again')


def test_recoverability_reports_ties():
    def simulate(g, rng):
        return g, g

    def fit_and_score(c, train, test):
        return np.array([0.0 if c == test or {c, test} == {'b', 'c'} else 1.0])
    r = tc.recoverability(['a', 'b', 'c'], ['a', 'b', 'c'], simulate, fit_and_score)
    assert r['best']['a'] == ['a'] and r['best']['b'] == ['b', 'c']
    assert r['recovered'] == {'a': True, 'b': False, 'c': False}
