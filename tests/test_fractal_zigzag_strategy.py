"""Fractal-zigzag RLStrategy plug-in — acceptance-criteria seams (issue #7).

One red→green seam per acceptance criterion:
  1. entries truncation-invariant (the lookahead proof, prior art:
     tests/test_fractal_pivots.py)
  2. every candidate carries a 1x ATR initial stop at detection time
  3. all six 3-min Parquet symbols load and produce plausible entry counts
  4. observation features causal — same truncation proof on the vectors
"""
import numpy as np
import pandas as pd

from futures_foundation.rl.causal import assert_causal
from futures_foundation.rl.fractal_zigzag import FractalZigzagStrategy


def _df(n=3000, seed=11):
    """Synthetic 3-min OHLCV random walk with a tz-aware DatetimeIndex."""
    rng = np.random.default_rng(seed)
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n))
    v = rng.integers(1, 1000, n).astype(float)
    idx = pd.date_range("2021-04-05 13:30", periods=n, freq="3min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l,
                         "close": c, "volume": v}, index=idx)


# ------------------------------------------------- AC 1: truncation invariance
def test_entries_truncation_invariant():
    """Entries on data[:t+1] == full-series entries with bar_idx <= t, for
    every cut t — the causal-parity contract detect_entries must satisfy."""
    df = _df()
    strat = FractalZigzagStrategy()
    detector = lambda d: strat.detect_entries(d, d, "NQ")
    assert_causal(detector, df,
                  cols=("bar_idx", "direction", "entry_price",
                        "sl_distance", "tp_rr"))
    # stronger, both directions: the prefix run emits EXACTLY the full-series
    # entries up to the cut (live_edge semantics — no drift at the data edge)
    full = detector(df)
    assert len(full) > 10
    for t in (500, 1400, 2600):
        pref = detector(df.iloc[:t + 1]).reset_index(drop=True)
        want = full[full["bar_idx"] <= t].reset_index(drop=True)
        pd.testing.assert_frame_equal(pref, want)
