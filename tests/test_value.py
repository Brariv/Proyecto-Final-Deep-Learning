"""Pruebas del modelo de valor: los umbrales óptimos deben coincidir con una búsqueda exhaustiva."""
import itertools

import numpy as np

from fraudscan.value import Costs, best_thresholds, decide, oracle_value, value_of_decisions


def _data(seed, n=60):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype("uint8")
    amt = rng.gamma(2.0, 60.0, n).astype("float32")
    score = np.clip(0.6 * y + rng.normal(0.3, 0.25, n), 0, 1).round(2)   # con empates
    return score, y, amt


def test_best_thresholds_matches_brute_force():
    costs = Costs()
    for seed in range(20):
        score, y, amt = _data(seed)
        best = best_thresholds(score, y, amt, costs)
        cands = np.r_[np.unique(score), score.max() + 1.0]
        brute = max(value_of_decisions(decide(score, tv, tb), y, amt, costs)
                    for tv, tb in itertools.product(cands, cands) if tb >= tv)
        got = value_of_decisions(decide(score, best["t_verify"], best["t_block"]), y, amt, costs)
        assert abs(got - brute) < 1e-6, (seed, got, brute)
        assert abs(best["value"] - got) < 1e-6
        assert best["t_block"] >= best["t_verify"]


def test_oracle_is_upper_bound_and_approve_all_is_zero():
    costs = Costs()
    score, y, amt = _data(1)
    best = best_thresholds(score, y, amt, costs)
    assert best["value"] <= oracle_value(y, amt, costs) + 1e-9
    assert value_of_decisions(decide(score, 2.0, 2.0), y, amt, costs) == 0.0


def test_weights_equal_repetition():
    costs = Costs()
    score, y, amt = _data(3, n=30)
    w = np.where(y == 1, 1.0, 3.0)
    rep = np.repeat(np.arange(30), w.astype(int))
    a = best_thresholds(score, y, amt, costs, weight=w)["value"]
    b = best_thresholds(score[rep], y[rep], amt[rep], costs)["value"]
    assert abs(a - b) < 1e-6
