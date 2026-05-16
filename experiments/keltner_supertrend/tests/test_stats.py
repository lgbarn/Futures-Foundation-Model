"""Unit tests for stats.py — dollar-native backtest statistics."""
import numpy as np
import pandas as pd
import pytest

from experiments.keltner_supertrend.stats import (
    ROW_HEADER, compute_stats, format_row, format_stats,
)

START = 100_000.0


def _dts(n, start='2026-01-05 10:00'):
    """n trade timestamps, one per day, tz-aware NY (span must exceed a day
    so CAGR annualisation has a non-zero horizon)."""
    return pd.date_range(start, periods=n, freq='1D', tz='America/New_York')


# ── compute_stats — core aggregates ───────────────────────────────────────────

def test_pnl_and_total_return():
    tp = [100.0, -50.0, 200.0, -25.0]
    s = compute_stats('t', _dts(4), tp, START)
    assert s['pnl_dollars'] == pytest.approx(225.0)
    assert s['total_return'] == pytest.approx(225.0 / START)
    assert s['trades'] == 4


def test_win_rate():
    tp = [10.0, -5.0, 20.0, -1.0, 7.0]            # 3 wins / 5
    s = compute_stats('t', _dts(5), tp, START)
    assert s['win_rate'] == pytest.approx(0.6)


def test_biggest_win_and_largest_loss():
    tp = [10.0, -300.0, 450.0, -20.0]
    s = compute_stats('t', _dts(4), tp, START)
    assert s['biggest_win_dollars'] == pytest.approx(450.0)
    assert s['largest_loss_dollars'] == pytest.approx(-300.0)


def test_avg_win_and_avg_loss():
    tp = [100.0, 300.0, -40.0, -60.0]             # wins {100,300}, losses {-40,-60}
    s = compute_stats('t', _dts(4), tp, START)
    assert s['avg_win_dollars'] == pytest.approx(200.0)
    assert s['avg_loss_dollars'] == pytest.approx(-50.0)


def test_expectancy():
    tp = [100.0, -50.0, 200.0, -25.0]
    s = compute_stats('t', _dts(4), tp, START)
    assert s['expectancy_dollars'] == pytest.approx(225.0 / 4)


def test_profit_factor():
    tp = [100.0, 200.0, -100.0]                   # gross win 300 / gross loss 100
    s = compute_stats('t', _dts(3), tp, START)
    assert s['profit_factor'] == pytest.approx(3.0)


def test_profit_factor_all_wins_is_inf():
    s = compute_stats('t', _dts(3), [10.0, 20.0, 30.0], START)
    assert s['profit_factor'] == float('inf')
    assert s['avg_loss_dollars'] == 0.0


def test_max_drawdown_dollars():
    # equity: 100k -> 100.5k -> 99.5k -> 100.3k ; trough drop = -1000 from the 100.5k peak
    tp = [500.0, -1000.0, 800.0]
    s = compute_stats('t', _dts(3), tp, START)
    assert s['max_dd_dollars'] == pytest.approx(-1000.0)
    assert s['max_dd_pct'] < 0


def test_no_drawdown_when_monotonic_up():
    s = compute_stats('t', _dts(4), [10.0, 20.0, 30.0, 40.0], START)
    assert s['max_dd_dollars'] == pytest.approx(0.0)
    assert s['calmar'] == 0.0                     # no DD -> Calmar guarded to 0


def test_cagr_sign_follows_pnl():
    win = compute_stats('w', _dts(5), [500.0] * 5, START)
    lose = compute_stats('l', _dts(5), [-500.0] * 5, START)
    assert win['cagr'] > 0
    assert lose['cagr'] < 0


def test_blown_account_cagr_floored():
    s = compute_stats('t', _dts(3), [-40_000.0, -40_000.0, -40_000.0], START)
    assert s['cagr'] == -1.0                      # equity <= 0 -> -100%


def test_contracts_reporting():
    s = compute_stats('t', _dts(4), [10.0, -5.0, 10.0, 10.0], START,
                      contracts=np.array([1, 2, 3, 4]))
    assert s['avg_contracts'] == pytest.approx(2.5)
    assert s['max_contracts_used'] == 4


def test_contracts_optional():
    s = compute_stats('t', _dts(3), [1.0, 2.0, 3.0], START)
    assert s['avg_contracts'] == 0.0
    assert s['max_contracts_used'] == 0


def test_sharpe_sortino_finite():
    rng = np.random.default_rng(0)
    tp = (rng.standard_normal(60) * 200).tolist()
    s = compute_stats('t', _dts(60), tp, START)
    assert np.isfinite(s['sharpe'])
    assert np.isfinite(s['sortino'])


def test_daily_returns_series_returned():
    s = compute_stats('t', _dts(5), [10.0] * 5, START)
    assert isinstance(s['daily_returns'], pd.Series)


# ── formatters ────────────────────────────────────────────────────────────────

def test_format_stats_contains_required_metrics():
    s = compute_stats('variant_x', _dts(4), [100.0, -50.0, 200.0, -25.0], START,
                      contracts=np.array([1, 1, 2, 1]))
    block = format_stats(s)
    for token in ('variant_x', 'PnL', 'CAGR', 'Sortino', 'Calmar',
                  'max_drawdown', 'avg_win', 'avg_loss', 'profit_factor'):
        assert token in block


def test_format_row_and_header_aligned():
    s = compute_stats('p>=0.50', _dts(4), [100.0, -50.0, 200.0, -25.0], START,
                      contracts=np.array([1, 1, 1, 1]))
    row = format_row(s)
    assert 'p>=0.50' in row
    # header and row carry the same column count
    assert len(ROW_HEADER.split()) >= 10
    assert len(row.split()) >= 10
