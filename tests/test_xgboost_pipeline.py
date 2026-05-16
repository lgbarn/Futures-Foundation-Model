"""Unit tests for pipelines.xgboost — objective + V2 labeler.

No xgboost/optuna import here (deps not yet installed); these modules are
pure numpy/pandas and must be correct before the rest of the pipeline builds
on them. The V2 labeler is a from-spec port (no trading-research repo) so it
gets the most scrutiny.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from pipelines.xgboost.objective import (
    calc_cagr, calc_sortino_ratio, calc_max_drawdown, combined_objective,
)
from pipelines.xgboost.labeler import TripleBarrierV2Labeler


# ── objective ────────────────────────────────────────────────────────────────

def test_objective_empty_is_zero():
    assert combined_objective(pd.Series([], dtype=float), 252) == 0.0


def test_objective_no_trade_collapse_is_zero():
    # all-zero returns -> CAGR 0 -> score 0 (the anti-"learn to not trade")
    assert combined_objective(pd.Series([0.0] * 50), 252) == 0.0


def test_objective_positive_returns_positive_score():
    r = pd.Series([0.01, -0.005, 0.012, 0.008, -0.003, 0.015])
    assert combined_objective(r, 252) > 0.0


def test_objective_dd_penalty_applied():
    # build a series with a >20% drawdown but positive end
    r = pd.Series([0.3, -0.25, -0.10, 0.05, 0.4, 0.1])
    cagr = calc_cagr(r, 252); sortino = calc_sortino_ratio(r, 252)
    if cagr > 0 and sortino > 0:
        full = cagr * sortino ** 0.5
        assert calc_max_drawdown(r) < -0.20
        assert combined_objective(r, 252) == pytest.approx(full * 0.1)


def test_max_drawdown_sign_and_range():
    dd = calc_max_drawdown(pd.Series([0.1, -0.5, 0.1]))
    assert -1.0 <= dd <= 0.0


# ── V2 labeler ───────────────────────────────────────────────────────────────

def _bars(prices_hl, start='2024-01-02 09:30', atr=1.0):
    """prices_hl: list of (high, low, close). ET tz-aware, 5-min bars."""
    idx = pd.date_range(start, periods=len(prices_hl), freq='5min',
                         tz='America/New_York')
    h, l, c = zip(*prices_hl)
    return pd.DataFrame({'datetime': idx, 'high': h, 'low': l, 'close': c,
                         'atr': atr})


def test_bad_bar_minutes_raises():
    with pytest.raises(ValueError):
        TripleBarrierV2Labeler(bar_minutes=4)


def test_long_win_open_session():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    # event bar close=100, open session TP=2.0xATR(=1) -> long_tp=102,
    # long_sl=98.75. Next bar spikes to 103 (hits long TP, not SL).
    bars = [(100.1, 99.9, 100.0)] + [(103.0, 100.0, 102.5)] + \
           [(102.0, 101.0, 101.5)] * 12
    out = lab.label(_bars(bars))
    assert out.iloc[0] == 1


def test_short_win_open_session():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    # short_tp = 100 - 2 = 98 ; next bar drops to 97 (hits short TP, not SL)
    bars = [(100.1, 99.9, 100.0)] + [(100.0, 97.0, 97.5)] + \
           [(99.0, 98.0, 98.5)] * 12
    out = lab.label(_bars(bars))
    assert out.iloc[0] == -1


def test_timeout_is_zero():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    # price stays in a tight band -> neither TP nor SL within 12 bars -> 0
    bars = [(100.1, 99.9, 100.0)] + [(100.3, 99.7, 100.0)] * 15
    out = lab.label(_bars(bars))
    assert out.iloc[0] == 0


def test_outside_rth_is_zero():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    # 08:00 ET is pre-RTH -> no event
    bars = [(100.1, 99.9, 100.0)] + [(103.0, 100.0, 102.5)] * 13
    out = lab.label(_bars(bars, start='2024-01-02 08:00'))
    assert out.iloc[0] == 0


def test_same_bar_tp_and_sl_sl_first_not_a_win():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    # next bar straddles BOTH long_tp(102) and long_sl(98.75) -> SL-first
    # rule => long not a win; short_tp=98 also touched w/ short_sl=101.25
    # also touched same bar => short not a win => label 0
    bars = [(100.1, 99.9, 100.0)] + [(103.0, 97.0, 100.0)] + \
           [(100.1, 99.9, 100.0)] * 12
    out = lab.label(_bars(bars))
    assert out.iloc[0] == 0


def test_label_aligned_and_int():
    lab = TripleBarrierV2Labeler(bar_minutes=5)
    df = _bars([(100.1, 99.9, 100.0)] * 20)
    out = lab.label(df)
    assert len(out) == len(df)
    assert set(np.unique(out)).issubset({-1, 0, 1})


# ── trail ────────────────────────────────────────────────────────────────────

from pipelines.xgboost.trail import (
    rogers_satchell_atr, two_bar_fractals,
    update_trailing_stop_long, update_ms_hybrid_long,
)


def test_rs_atr_nonneg_and_length():
    n = 50
    o = np.full(n, 100.0); c = np.full(n, 100.5)
    h = np.full(n, 101.0); l = np.full(n, 99.5)
    atr = rogers_satchell_atr(o, h, l, c, n=10)
    assert len(atr) == n
    assert np.all(atr >= 0)


def test_rs_atr_zero_range_is_zero():
    n = 20
    o = h = l = c = np.full(n, 100.0)        # zero-range bars -> RS_i=0
    assert np.allclose(rogers_satchell_atr(o, h, l, c), 0.0)


def test_two_bar_fractals_causal_and_correct():
    # a clear swing-low at index 3 (l=90), 2 higher bars each side
    l = np.array([100, 98, 95, 90, 95, 98, 100, 101.0])
    h = np.array([101, 100, 99, 98, 99, 100, 101, 102.0])
    sl, sh = two_bar_fractals(h, l)
    # fractal centred at j is only visible from i = j+2 onward (causal)
    assert np.isnan(sl[4])             # not yet confirmed at bar 4
    assert sl[5] == 90.0               # confirmed at j+2 = 5


def test_trailing_stop_long_ratchets_only_up():
    bar = {"high": 110.0, "low": 105.0}
    hwm, sl, on = update_trailing_stop_long(
        bar, entry=100.0, hwm=100.0, sl=97.0, trail_on=False,
        trigger_pts=2.0, distance_pts=3.0)
    assert on is True and sl == 107.0          # 110 - 3
    # a lower bar must NOT loosen the stop
    bar2 = {"high": 108.0, "low": 104.0}
    _, sl2, _ = update_trailing_stop_long(
        bar2, 100.0, hwm, sl, on, 2.0, 3.0)
    assert sl2 == 107.0                          # ratchet holds


def test_ms_hybrid_long_picks_tighter():
    bar = {"high": 110.0, "low": 108.0, "atr_rs": 2.0,
           "last_swing_low": 109.0}
    cfg = {"atr_rs_entry": 2.0, "trail_trigger_atr": 0.5,
           "trail_distance_atr": 0.5, "ms_buffer_atr": 0.1}
    _, sl, on = update_ms_hybrid_long(bar, entry=100.0, hwm=100.0, sl=97.0,
                                      trail_on=False, bars_held=5,
                                      trail_min_bars=1, config=cfg)
    # structure stop 109-0.2=108.8 is tighter than atr stop 110-1=109? no:
    # higher = tighter for longs -> max(109, 108.8) = 109
    assert sl == pytest.approx(109.0)


# ── walkforward ──────────────────────────────────────────────────────────────

from pipelines.xgboost.walkforward import walk_forward_windows, optuna_holdout


def test_walk_forward_rolling_unanchored_3to1():
    idx = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    wins = list(walk_forward_windows(idx, 3, 1))
    # 12 months, 3 train + 1 test, stride 1 -> 9 windows (Jan..Sep starts)
    assert len(wins) == 9
    tr, te = wins[0]
    # unanchored: window 1 train != window 0 train (oldest month dropped)
    tr1, _ = wins[1]
    assert not np.array_equal(tr, tr1)
    assert tr.sum() > 0 and te.sum() > 0
    assert not (tr & te).any()                   # no train/test overlap


def test_optuna_holdout_is_temporal_tail():
    tr = np.zeros(100, bool); tr[:60] = True
    fit, val = optuna_holdout(tr, 0.15)
    assert fit.sum() + val.sum() == 60
    assert not (fit & val).any()
    # val is the most-recent slice -> its first index > fit's last index
    assert np.flatnonzero(val)[0] > np.flatnonzero(fit)[-1]


# ── backtest ─────────────────────────────────────────────────────────────────

from pipelines.xgboost.backtest import run_backtest


def _ohlcv(rows, start='2024-01-02 09:30'):
    idx = pd.date_range(start, periods=len(rows), freq='5min',
                         tz='America/New_York')
    o, h, l, c = zip(*rows)
    return pd.DataFrame({'datetime': idx, 'open': o, 'high': h,
                         'low': l, 'close': c})


def test_backtest_no_signal_no_trades():
    df = _ohlcv([(100, 101, 99, 100)] * 30)
    res = run_backtest(df, np.zeros(30))
    assert res['stats']['trades'] == 0


def test_backtest_long_trail_out_profit_single_position():
    # Need non-zero-range history so Rogers-Satchell vol > 0 at the entry bar
    # (the backtest correctly refuses to size a stop on zero volatility).
    rng = np.random.default_rng(0)
    warm = [(100 + d, 100 + d + 0.6, 100 + d - 0.6, 100 + d)
            for d in rng.normal(0, 0.3, 14)]            # ranged warm-up
    sig_bar = [(100.0, 100.6, 99.4, 100.0)]             # signal bar (idx 14)
    runup = [(c, c + 1.0, c - 1.0, c) for c in
             [101, 104, 108, 113, 118, 122]]            # entry@idx15 then run
    reversal = [(120, 121, 95, 96)]                     # sharp drop -> stop/trail
    tail = [(96, 96.6, 95.4, 96)] * 6
    rows = warm + sig_bar + runup + reversal + tail
    df = _ohlcv(rows)
    sig = np.zeros(len(rows)); sig[14] = 1              # signal at idx 14
    res = run_backtest(df, sig)
    assert res['stats']['trades'] == 1                  # one position at a time
    t = res['trades'][0]
    assert t['dir'] == 1 and t['entry_idx'] == 15


# ── plug-in contract (XGBStrategyLabeler / registry) ─────────────────────────

from pipelines.xgboost.base import (
    XGBStrategyLabeler, register, get_labeler, LABELERS,
)
import pipelines.xgboost.labeler  # noqa: F401 — registers v2_triple_barrier


def test_v2_is_registered_and_resolves():
    assert 'v2_triple_barrier' in LABELERS
    lab = get_labeler('v2_triple_barrier', bar_minutes=5)
    assert isinstance(lab, XGBStrategyLabeler)
    assert lab.name == 'v2_triple_barrier'
    assert len(lab.feature_cols()) == 68          # default = FFM 68
    assert 'version' in lab.config_dict()


def test_unknown_labeler_raises():
    with pytest.raises(KeyError):
        get_labeler('does_not_exist', bar_minutes=5)


def test_abc_requires_label():
    with pytest.raises(TypeError):
        XGBStrategyLabeler()                       # abstract: label() missing


def test_custom_labeler_plugs_in():
    @register("unit_test_dummy")
    class Dummy(XGBStrategyLabeler):
        name = "unit_test_dummy"
        def __init__(self, *, bar_minutes):
            self.bar_minutes = bar_minutes
        def label(self, df):
            # trivial alternating directional target
            import numpy as _np
            return pd.Series(_np.where(np.arange(len(df)) % 2, 1, -1),
                             index=df.index)
    lab = get_labeler("unit_test_dummy", bar_minutes=5)
    out = lab.label(pd.DataFrame({'datetime': pd.date_range(
        '2024-01-02 09:30', periods=6, freq='5min', tz='America/New_York'),
        'open': 1, 'high': 1, 'low': 1, 'close': 1, 'atr': 1.0}))
    assert set(np.unique(out)).issubset({-1, 0, 1})
    assert lab.feature_cols() == get_labeler(
        'v2_triple_barrier', bar_minutes=5).feature_cols()   # default 68
    del LABELERS["unit_test_dummy"]                # keep registry clean


def test_duplicate_register_raises():
    @register("dup_x")
    class _A(XGBStrategyLabeler):
        name = "dup_x"
        def label(self, df): return pd.Series([], dtype=int)
    with pytest.raises(ValueError):
        @register("dup_x")
        class _B(XGBStrategyLabeler):
            name = "dup_x"
            def label(self, df): return pd.Series([], dtype=int)
    del LABELERS["dup_x"]
