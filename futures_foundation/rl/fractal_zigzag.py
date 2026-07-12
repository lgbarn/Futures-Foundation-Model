"""Fractal-zigzag entry plug-in — the first concrete RLStrategy entry side.

detect_entries is driven by detect_fractal_zigzag_pivots (the trigger-scan
causal winner) with live_edge=True: a pivot confirming on the NEWEST bar is
emitted, so bar-by-bar consumers can fire and the detector is truncation-
invariant at the data edge (proven in tests/test_fractal_pivots.py; re-proven
on this plug-in in tests/test_fractal_zigzag_strategy.py).

Every candidate carries direction, the detection-time entry reference price
(close of the confirm bar — the actual fill is the NEXT bar's open, enforced
centrally by the pipeline), and a 1x ATR initial stop attached at detection
time. Everything is strictly causal: row bar_idx uses only bars <= bar_idx.
"""
import numpy as np
import pandas as pd

from futures_foundation.pipeline._primitives import compute_atr
from futures_foundation.primitives.detection import detect_fractal_zigzag_pivots

from .base import RLStrategy, register

ENTRY_COLUMNS = ["bar_idx", "direction", "entry_price",
                 "sl_distance", "tp_rr", "datetime"]


@register("fractal_zigzag")
class FractalZigzagStrategy(RLStrategy):
    """Mechanical entries at confirmed fractal-zigzag pivots, 1x ATR stop."""
    name = "fractal_zigzag"
    entry_filter = True

    def __init__(self, k=2, min_leg_atr=1.25, atr_period=20,
                 stop_atr=1.0, tp_rr=3.0):
        self.k = int(k)
        self.min_leg_atr = float(min_leg_atr)
        self.atr_period = int(atr_period)
        self.stop_atr = float(stop_atr)     # issue #7 contract: 1x ATR stop
        self.tp_rr = float(tp_rr)

    def config_dict(self) -> dict:
        return {"version": 1, "k": self.k, "min_leg_atr": self.min_leg_atr,
                "atr_period": self.atr_period, "stop_atr": self.stop_atr,
                "tp_rr": self.tp_rr}

    def detect_entries(self, df_raw: pd.DataFrame, ctx_df: pd.DataFrame,
                       ticker: str) -> pd.DataFrame:
        o = df_raw["open"].to_numpy(float)
        h = df_raw["high"].to_numpy(float)
        l = df_raw["low"].to_numpy(float)
        c = df_raw["close"].to_numpy(float)
        piv = detect_fractal_zigzag_pivots(
            o, h, l, c, k=self.k, min_leg_atr=self.min_leg_atr,
            atr_period=self.atr_period, live_edge=True)
        atr = compute_atr(h, l, c, self.atr_period)
        rows = []
        for p in piv:
            bi = p["confirm"]                 # signal bar (entry = next open)
            a = atr[bi]
            if not (np.isfinite(a) and a > 0):
                continue                      # ATR warm-up: no stop, no entry
            rows.append({"bar_idx": bi,
                         "direction": p["direction"],
                         "entry_price": c[bi],
                         "sl_distance": self.stop_atr * a,
                         "tp_rr": self.tp_rr,
                         "datetime": df_raw.index[bi]})
        return pd.DataFrame(rows, columns=ENTRY_COLUMNS)
