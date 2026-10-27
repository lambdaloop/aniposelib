"""aniposelib.workflows._common: verdict rules, noise floor, result serialisation."""
import json

import numpy as np

import aniposelib
from aniposelib.workflows._common import (Rule, evaluate_rules, noise_floor, WorkflowResult,
                                          merge_options, MIN_NOISE_PX)


def test_rules_levels_and_overrides():
    rules = [Rule('loo_px', 2.0, warn_at=1.5, fail_at=3.0), Rule('coverage', 0.9, 0.8, 0.5, higher_is_worse=False),
             Rule('disconnected', False, fail_at=True)]
    v, reasons, table = evaluate_rules(rules)
    assert v == 'warn' and len(reasons) == 1 and 'loo_px' in reasons[0]
    v, reasons, _ = evaluate_rules(rules, {'loo_px': (5, 10)})
    assert v == 'ok' and not reasons
    v, _, _ = evaluate_rules(rules + [Rule('disconnected2', True, fail_at=True)])
    assert v == 'fail'
    v, _, _ = evaluate_rules([Rule('x', float('nan'), 1, 2)])
    assert v == 'fail'
    assert evaluate_rules([Rule('x', 5.0, 1, 2)], {'x': None})[0] == 'ok'


def test_noise_floor():
    assert noise_floor(np.zeros(100)) == MIN_NOISE_PX
    assert noise_floor([1.0] * 3) == MIN_NOISE_PX
    e = np.abs(np.random.default_rng(0).normal(0, 2, 1000))
    assert 1.0 < noise_floor(e) < 2.0
    assert noise_floor(e * 1000, diag=1000) == 10.0
    m = np.zeros(1000, bool); m[:500] = True
    assert np.isclose(noise_floor(e, m), np.median(e[:500]))


def test_result_serialises():
    r = WorkflowResult(report={'a': np.arange(3), 'b': np.float32(np.nan)}, verdict='warn', reasons=['x'])
    d = json.loads(r.to_json())
    assert d['report']['a'] == [0, 1, 2] and d['report']['b'] is None and d['verdict'] == 'warn'
    assert 'verdict warn' in r.summary() and r.ok


def test_merge_options():
    assert merge_options({'a': 1}, {'a': 2}) == {'a': 2}
    try:
        merge_options({'a': 1}, {'b': 2})
    except ValueError as e:
        assert 'b' in str(e)
    else:  # pragma: no cover
        raise AssertionError


def test_top_level_names():
    assert 'reconstruct_3d' in dir(aniposelib)
