"""Unit tests for backtest.py — exit simulators + the OOS entry pipeline."""
import numpy as np
import pytest

from experiments.keltner_supertrend import backtest as bt
from experiments.keltner_supertrend.backtest import (
    EOD_MIN, simulate_barrier_exit, simulate_hybrid_trail,
)

DAY = np.datetime64('2026-01-05')
NEXT_DAY = np.datetime64('2026-01-06')


def _arr(*vals):
    return np.array(vals, dtype=np.float64)


# ── simulate_barrier_exit ─────────────────────────────────────────────────────
# atr = 1.0 everywhere -> long: SL = entry-1.0, TP = entry+1.5.

def test_barrier_long_take_profit():
    high = _arr(100.0, 101.6)
    low = _arr(100.0, 100.5)
    close = _arr(100.0, 101.5)
    atr = _arr(1.0, 1.0)
    mod = np.array([600, 600])
    day = np.array([DAY, DAY])
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, 1) == pytest.approx(1.5)


def test_barrier_long_stop_loss():
    high = _arr(100.0, 98.5)
    low = _arr(100.0, 97.0)
    close = _arr(100.0, 98.0)
    atr = _arr(1.0, 1.0)
    mod = np.array([600, 600])
    day = np.array([DAY, DAY])
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, 1) == pytest.approx(-1.0)


def test_barrier_long_eod_exit():
    high = _arr(100.0, 100.4, 100.7)
    low = _arr(100.0, 100.2, 100.5)
    close = _arr(100.0, 100.3, 100.6)
    atr = _arr(1.0, 1.0, 1.0)
    mod = np.array([600, 600, EOD_MIN])
    day = np.array([DAY, DAY, DAY])
    # neither barrier; flat at the EOD bar -> exit at that close
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, 1) == pytest.approx(0.6)


def test_barrier_stop_checked_before_target():
    # one bar spans BOTH barriers — pessimistic: stop wins
    high = _arr(100.0, 102.0)
    low = _arr(100.0, 98.0)
    close = _arr(100.0, 101.0)
    atr = _arr(1.0, 1.0)
    mod = np.array([600, 600])
    day = np.array([DAY, DAY])
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, 1) == pytest.approx(-1.0)


def test_barrier_day_change_exit():
    high = _arr(100.0, 100.4, 100.8)
    low = _arr(100.0, 100.2, 100.6)
    close = _arr(100.0, 100.4, 100.8)
    atr = _arr(1.0, 1.0, 1.0)
    mod = np.array([600, 600, 600])
    day = np.array([DAY, DAY, NEXT_DAY])
    # next-day bar triggers exit at the prior bar's close
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, 1) == pytest.approx(0.4)


def test_barrier_short_take_profit():
    high = _arr(100.0, 99.0)
    low = _arr(100.0, 98.0)
    close = _arr(100.0, 98.5)
    atr = _arr(1.0, 1.0)
    mod = np.array([600, 600])
    day = np.array([DAY, DAY])
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, -1) == pytest.approx(1.5)


def test_barrier_short_stop_loss():
    high = _arr(100.0, 102.5)
    low = _arr(100.0, 101.5)
    close = _arr(100.0, 102.0)
    atr = _arr(1.0, 1.0)
    mod = np.array([600, 600])
    day = np.array([DAY, DAY])
    assert simulate_barrier_exit(high, low, close, atr, mod, day, 0, -1) == pytest.approx(-1.0)


# ── simulate_hybrid_trail ─────────────────────────────────────────────────────
# atr = 1.0 -> hard SL = entry-3.0; trail trigger/distance = 0.15.

def test_trail_hard_stop():
    close = _arr(100.0, 99.0, 98.0, 96.0)
    high = _arr(100.0, 99.5, 98.5, 96.5)
    low = _arr(100.0, 98.5, 97.5, 96.0)        # bar 3 low 96.0 <= SL 97.0
    atr = np.full(4, 1.0)
    mod = np.full(4, 600)
    assert simulate_hybrid_trail(high, low, close, atr, mod, 0, 1) == pytest.approx(-3.0)


def test_trail_ratchets_and_locks_gain():
    # price runs up (arms + ratchets the trail), then a pullback takes it out
    close = _arr(100.0, 100.5, 101.0, 101.5, 101.0)
    high = _arr(100.0, 100.6, 101.1, 101.6, 101.1)
    low = _arr(100.0, 100.4, 100.9, 101.4, 100.0)
    atr = np.full(5, 1.0)
    mod = np.full(5, 600)
    out = simulate_hybrid_trail(high, low, close, atr, mod, 0, 1)
    assert out == pytest.approx(1.45)          # hwm 101.6 - distance 0.15 - entry 100


def test_trail_eod_exit():
    close = _arr(100.0, 100.1, 100.2)
    high = _arr(100.0, 100.15, 100.25)
    low = _arr(100.0, 100.05, 100.15)
    atr = np.full(3, 1.0)
    mod = np.array([600, 600, EOD_MIN])
    assert simulate_hybrid_trail(high, low, close, atr, mod, 0, 1) == pytest.approx(0.2)


def test_trail_time_barrier_exit():
    n = 35
    close = np.full(n, 100.0)
    close[30] = 100.7
    high = close + 0.05
    low = close - 0.05
    atr = np.full(n, 1.0)
    mod = np.full(n, 600)                       # never EOD, never hard stop
    # no barrier within MAX_BARS (30) -> exit at close[i+30]
    assert simulate_hybrid_trail(high, low, close, atr, mod, 0, 1) == pytest.approx(0.7)


def test_trail_short_hard_stop():
    close = _arr(100.0, 101.0, 102.0, 104.0)
    high = _arr(100.0, 101.5, 102.5, 104.0)     # bar 3 high 104 >= SL 103
    low = _arr(100.0, 100.5, 101.5, 103.5)
    atr = np.full(4, 1.0)
    mod = np.full(4, 600)
    assert simulate_hybrid_trail(high, low, close, atr, mod, 0, -1) == pytest.approx(-3.0)


# ── compute_oos_trades — entry pipeline (integration; needs prepared data) ────

def _oos_data_ready():
    prep = bt.REPO_ROOT / 'data' / 'prep_input' / 'ES_3min.parquet'
    return prep.exists() and bt.BASELINE_TRADES.exists()


@pytest.mark.skipif(not _oos_data_ready(), reason='OOS data not prepared')
@pytest.mark.parametrize('exit_mode', ['barrier', 'trail'])
def test_compute_oos_trades_contract(exit_mode):
    oos = bt.compute_oos_trades('ES', exit_mode, cost_points=0.0)
    assert len(oos) > 0
    for col in ('datetime', 'direction', 'gross_pts', 'net_pts',
                'stop_pts', 'p_signal'):
        assert col in oos.columns
    assert (oos['stop_pts'] > 0).all()
    assert set(oos['direction'].unique()).issubset({-1, 1})
    assert oos['p_signal'].notna().all()          # unmatched entries dropped


@pytest.mark.skipif(not _oos_data_ready(), reason='OOS data not prepared')
def test_compute_oos_trades_costs_reduce_pnl():
    free = bt.compute_oos_trades('ES', 'barrier', cost_points=0.0)
    costed = bt.compute_oos_trades('ES', 'barrier', cost_points=0.58)
    # cost is subtracted per trade -> net_pts strictly lower, gross unchanged
    assert (costed['net_pts'] < free['net_pts']).all()
    assert np.allclose(costed['gross_pts'].to_numpy(), free['gross_pts'].to_numpy())


@pytest.mark.skipif(not _oos_data_ready(), reason='OOS data not prepared')
@pytest.mark.parametrize('argv', [
    ['backtest.py', '--exit', 'barrier'],
    ['backtest.py', '--exit', 'trail', '--costs'],
])
def test_main_end_to_end(monkeypatch, capsys, argv):
    pytest.importorskip('quantstats')
    monkeypatch.setattr('sys.argv', argv)
    bt.main()
    out = capsys.readouterr().out
    assert 'OOS BACKTEST' in out
    assert 'Sweep complete' in out
