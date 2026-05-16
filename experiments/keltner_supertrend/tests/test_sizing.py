"""Unit tests for sizing.py — instrument registry + contract-level sizing."""
import numpy as np
import pytest

from experiments.keltner_supertrend import sizing


# ── specs / instrument registry ───────────────────────────────────────────────

def test_specs_known_instruments():
    assert sizing.specs('ES') == (50.0, 0.25)
    assert sizing.specs('NQ') == (20.0, 0.25)
    assert sizing.specs('RTY') == (50.0, 0.10)
    assert sizing.specs('YM') == (5.0, 1.0)
    assert sizing.specs('GC') == (100.0, 0.10)
    assert sizing.specs('SI') == (5000.0, 0.005)


def test_specs_micros_present():
    for sym in ('MES', 'MNQ', 'M2K', 'MYM', 'MGC'):
        pv, tick = sizing.specs(sym)
        assert pv > 0 and tick > 0


def test_specs_unknown_raises():
    with pytest.raises(KeyError):
        sizing.specs('DOES_NOT_EXIST')


def test_account_config_defaults():
    assert sizing.ACCOUNT_SIZE == 150_000.0
    assert sizing.MAX_CONTRACTS == 10
    assert 0 < sizing.RISK_FRAC < 1


# ── position_size ─────────────────────────────────────────────────────────────

def test_position_size_basic_floor():
    # risk $450 (0.3% of 150k); stop 5 pts * $50 = $250/contract -> 1.8 -> floor 1
    n, capped, floored = sizing.position_size(
        150_000, stop_points=5.0, point_value=50.0, risk_frac=0.003, max_contracts=10)
    assert n == 1 and not capped and not floored


def test_position_size_scales_with_equity():
    kw = dict(stop_points=2.0, point_value=50.0, risk_frac=0.01, max_contracts=50)
    n_small, _, _ = sizing.position_size(100_000, **kw)
    n_big, _, _ = sizing.position_size(500_000, **kw)
    assert n_big > n_small


def test_position_size_caps():
    # huge risk budget -> wants many contracts -> capped
    n, capped, floored = sizing.position_size(
        10_000_000, stop_points=1.0, point_value=50.0, risk_frac=0.01, max_contracts=10)
    assert n == 10 and capped and not floored


def test_position_size_min_one_when_budget_too_small():
    # tiny budget vs wide stop -> wants < 1 contract -> floored up to 1
    n, capped, floored = sizing.position_size(
        1_000, stop_points=100.0, point_value=50.0, risk_frac=0.003, max_contracts=10)
    assert n == 1 and floored and not capped


def test_position_size_zero_stop_safe():
    n, capped, floored = sizing.position_size(
        150_000, stop_points=0.0, point_value=50.0)
    assert n == 1 and not capped and not floored


# ── simulate_account ──────────────────────────────────────────────────────────

def test_simulate_account_dollar_pnl():
    # 1 contract each (budget forces minimum); ES $50/pt
    pnl_points = np.array([2.0, -1.0, 3.0])
    stop_points = np.array([100.0, 100.0, 100.0])   # wide -> 1 contract each
    acct = sizing.simulate_account(pnl_points, stop_points, point_value=50.0,
                                   starting_equity=150_000, risk_frac=0.003)
    assert list(acct['contracts']) == [1, 1, 1]
    assert np.allclose(acct['trade_dollars'], [100.0, -50.0, 150.0])
    assert acct['n_min_floored'] == 3


def test_simulate_account_compounds_through_size():
    # tight stop + generous risk: contract count rises as equity grows
    pnl_points = np.full(40, 5.0)
    stop_points = np.full(40, 1.0)
    acct = sizing.simulate_account(pnl_points, stop_points, point_value=50.0,
                                   starting_equity=100_000, risk_frac=0.01,
                                   max_contracts=1000)
    assert acct['contracts'][-1] > acct['contracts'][0]


def test_simulate_account_respects_cap():
    pnl_points = np.full(20, 1.0)
    stop_points = np.full(20, 0.5)
    acct = sizing.simulate_account(pnl_points, stop_points, point_value=50.0,
                                   starting_equity=150_000, risk_frac=0.01,
                                   max_contracts=3)
    assert acct['contracts'].max() <= 3
    assert acct['n_capped'] > 0


def test_simulate_account_lengths_match():
    n = 15
    acct = sizing.simulate_account(np.ones(n), np.full(n, 10.0), point_value=50.0)
    assert len(acct['trade_dollars']) == n
    assert len(acct['contracts']) == n


def test_simulate_account_empty():
    acct = sizing.simulate_account(np.array([]), np.array([]), point_value=50.0)
    assert len(acct['trade_dollars']) == 0
    assert acct['n_capped'] == 0 and acct['n_min_floored'] == 0
