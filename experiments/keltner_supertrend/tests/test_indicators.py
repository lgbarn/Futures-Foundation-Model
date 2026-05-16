"""Unit tests for indicators.py — Keltner / SuperTrend / ATR / EMA.

Covers known-value math, length preservation, edge cases, and — most
importantly — causality: every indicator value at bar i must depend only on
bars <= i (no look-ahead).
"""
import numpy as np
import pytest

from experiments.keltner_supertrend.indicators import (
    calc_atr, calc_ema, calc_keltner, calc_supertrend,
)


# ── calc_ema ──────────────────────────────────────────────────────────────────

def test_ema_empty():
    assert len(calc_ema(np.array([]), 5)) == 0


def test_ema_seeds_with_first_value():
    out = calc_ema(np.array([10.0, 20.0, 30.0]), 5)
    assert out[0] == 10.0


def test_ema_constant_input_is_constant():
    out = calc_ema(np.full(20, 7.0), 5)
    assert np.allclose(out, 7.0)


def test_ema_known_value():
    # period=3 -> k = 2/(3+1) = 0.5;  ema[1] = 20*0.5 + 10*0.5 = 15
    out = calc_ema(np.array([10.0, 20.0]), 3)
    assert out[1] == pytest.approx(15.0)


def test_ema_length_preserved():
    assert len(calc_ema(np.arange(50.0), 14)) == 50


# ── calc_atr ──────────────────────────────────────────────────────────────────

def test_atr_first_value_is_first_range():
    high = np.array([102.0, 103.0, 104.0])
    low = np.array([100.0, 101.0, 102.0])
    close = np.array([101.0, 102.0, 103.0])
    atr = calc_atr(high, low, close, 14)
    assert atr[0] == pytest.approx(2.0)            # high[0] - low[0]


def test_atr_constant_range_converges():
    n = 200
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    close = np.full(n, 100.0)
    atr = calc_atr(high, low, close, 14)
    assert atr[-1] == pytest.approx(1.0, abs=1e-6)  # true range is always 1.0


def test_atr_nonnegative_and_length():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(120))
    high = close + rng.random(120)
    low = close - rng.random(120)
    atr = calc_atr(high, low, close, 14)
    assert len(atr) == 120
    assert (atr >= 0).all()


# ── calc_keltner ──────────────────────────────────────────────────────────────

def test_keltner_band_structure():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.standard_normal(80))
    high = close + 1.0
    low = close - 1.0
    middle, upper, lower = calc_keltner(high, low, close, 20, 14, 1.5)
    assert len(middle) == len(upper) == len(lower) == 80
    assert (upper >= middle).all()
    assert (lower <= middle).all()
    # bands are symmetric about the middle
    assert np.allclose(upper - middle, middle - lower)


def test_keltner_middle_is_ema_of_close():
    close = np.linspace(100, 120, 60)
    high = close + 0.5
    low = close - 0.5
    middle, _, _ = calc_keltner(high, low, close, 20, 14, 1.5)
    assert np.allclose(middle, calc_ema(close, 20))


# ── calc_supertrend ───────────────────────────────────────────────────────────

def test_supertrend_direction_domain():
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.standard_normal(150))
    high = close + rng.random(150)
    low = close - rng.random(150)
    value, direction = calc_supertrend(high, low, close, 10, 3.0)
    assert len(value) == len(direction) == 150
    assert set(np.unique(direction)).issubset({-1, 1})


def test_supertrend_uptrend_is_bullish():
    close = np.linspace(100, 200, 120)         # strong, steady uptrend
    high = close + 0.5
    low = close - 0.5
    _, direction = calc_supertrend(high, low, close, 10, 3.0)
    assert direction[-1] == 1                   # ends bullish


def test_supertrend_downtrend_is_bearish():
    close = np.linspace(200, 100, 120)
    high = close + 0.5
    low = close - 0.5
    _, direction = calc_supertrend(high, low, close, 10, 3.0)
    assert direction[-1] == -1


# ── Causality — no look-ahead ─────────────────────────────────────────────────
# Each indicator value at bar i must equal the value computed on just bars
# 0..i. We verify the full series matches every truncated-prefix recomputation.

@pytest.fixture
def ohlc():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.standard_normal(140))
    high = close + rng.random(140) + 0.1
    low = close - rng.random(140) - 0.1
    return high, low, close


def test_ema_causal(ohlc):
    _, _, close = ohlc
    full = calc_ema(close, 14)
    for k in (30, 70, 110, 140):
        assert calc_ema(close[:k], 14)[-1] == pytest.approx(full[k - 1])


def test_atr_causal(ohlc):
    high, low, close = ohlc
    full = calc_atr(high, low, close, 14)
    for k in (30, 70, 110, 140):
        prefix = calc_atr(high[:k], low[:k], close[:k], 14)
        assert prefix[-1] == pytest.approx(full[k - 1])


def test_keltner_causal(ohlc):
    high, low, close = ohlc
    fm, fu, fl = calc_keltner(high, low, close, 22, 20, 1.25)
    for k in (40, 90, 140):
        pm, pu, pl = calc_keltner(high[:k], low[:k], close[:k], 22, 20, 1.25)
        assert pm[-1] == pytest.approx(fm[k - 1])
        assert pu[-1] == pytest.approx(fu[k - 1])
        assert pl[-1] == pytest.approx(fl[k - 1])


def test_supertrend_causal(ohlc):
    high, low, close = ohlc
    fv, fd = calc_supertrend(high, low, close, 10, 3.0)
    for k in (40, 90, 140):
        pv, pd_ = calc_supertrend(high[:k], low[:k], close[:k], 10, 3.0)
        assert pv[-1] == pytest.approx(fv[k - 1])
        assert pd_[-1] == fd[k - 1]
