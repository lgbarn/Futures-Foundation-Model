"""Hybrid ATR/structure trailing stop (spec section 7).

Verbatim port of trading-research Python/backtest.py helpers. The model picks
DIRECTION only; this module is the exit. Uses Rogers-Satchell volatility (not
FFM's Wilder ATR) for the trail's own ATR — RS is drift-independent and stays
low in clean trends so the stop tightens aggressively.

ATR-freeze rule (spec 7.5 note): the trail TRIGGER uses ATR(RS) frozen at
entry; the trail DISTANCE uses the live (per-bar) ATR(RS).
"""
import numpy as np


def rogers_satchell_atr(o, h, l, c, n: int = 10) -> np.ndarray:
    """ATR(RS) per bar = sqrt(rolling_mean(RS, n)) * close.

    RS_i = ln(H/C)ln(H/O) + ln(L/C)ln(L/O).  Zero-range bars -> RS_i=0
    (ln(1)=0), no special-casing. N=10 (RS uses all four OHLC prices, so a
    shorter window still has enough info and reacts faster).
    """
    o = np.asarray(o, np.float64); h = np.asarray(h, np.float64)
    l = np.asarray(l, np.float64); c = np.asarray(c, np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = (np.log(h / c) * np.log(h / o) +
              np.log(l / c) * np.log(l / o))
    rs = np.nan_to_num(rs, nan=0.0, posinf=0.0, neginf=0.0)
    # causal rolling mean (window of last n, min 1)
    csum = np.cumsum(np.insert(rs, 0, 0.0))
    out = np.empty(len(rs))
    for i in range(len(rs)):
        a = max(0, i - n + 1)
        out[i] = (csum[i + 1] - csum[a]) / (i - a + 1)
    return np.sqrt(np.clip(out, 0.0, None)) * c


def two_bar_fractals(h, l):
    """Confirmed 2-bar fractals (a swing needs 2 lower/higher bars each side).

    Returns (last_swing_low, last_swing_high) arrays: at bar i, the most
    recent CONFIRMED swing as of i (a fractal at j is confirmed at j+2, so
    only fractals with j+2 <= i are visible -> causal).
    """
    h = np.asarray(h, np.float64); l = np.asarray(l, np.float64)
    n = len(h)
    sl = np.full(n, np.nan); sh = np.full(n, np.nan)
    last_lo = last_hi = np.nan
    for i in range(n):
        j = i - 2                       # fractal centred at j confirmed now
        if j >= 2:
            if l[j] < l[j-1] and l[j] < l[j-2] and l[j] < l[j+1] and l[j] < l[j+2]:
                last_lo = l[j]
            if h[j] > h[j-1] and h[j] > h[j-2] and h[j] > h[j+1] and h[j] > h[j+2]:
                last_hi = h[j]
        sl[i] = last_lo
        sh[i] = last_hi
    return sl, sh


def update_trailing_stop_long(bar, entry, hwm, sl, trail_on,
                              trigger_pts, distance_pts,
                              bars_held=None, trail_min_bars=1):
    new_hwm, new_sl, new_trail_on = hwm, sl, trail_on
    if bar["high"] > new_hwm:
        new_hwm = bar["high"]
    can_activate = bars_held is None or bars_held >= trail_min_bars
    if not new_trail_on and can_activate and (new_hwm - entry) >= trigger_pts:
        new_trail_on = True
    if new_trail_on:
        candidate_sl = new_hwm - distance_pts          # ratchet:
        if candidate_sl > new_sl:                      # only move up
            new_sl = candidate_sl
    return new_hwm, new_sl, new_trail_on


def update_trailing_stop_short(bar, entry, lwm, sl, trail_on,
                               trigger_pts, distance_pts,
                               bars_held=None, trail_min_bars=1):
    new_lwm, new_sl, new_trail_on = lwm, sl, trail_on
    if bar["low"] < new_lwm:
        new_lwm = bar["low"]
    can_activate = bars_held is None or bars_held >= trail_min_bars
    if not new_trail_on and can_activate and (entry - new_lwm) >= trigger_pts:
        new_trail_on = True
    if new_trail_on:
        candidate_sl = new_lwm + distance_pts
        if candidate_sl < new_sl:                      # ratchet down
            new_sl = candidate_sl
    return new_lwm, new_sl, new_trail_on


def update_ms_hybrid_long(bar, entry, hwm, sl, trail_on,
                          bars_held, trail_min_bars, config):
    """Tightest of ATR trail and market-structure swing (higher=tighter)."""
    new_hwm = max(hwm, bar["high"])
    # trigger uses entry-frozen ATR(RS); distance uses live ATR(RS)
    tt_pts = config["atr_rs_entry"] * config["trail_trigger_atr"]
    td_pts = bar["atr_rs"] * config["trail_distance_atr"]
    _, atr_sl, atr_trail = update_trailing_stop_long(
        bar, entry, new_hwm, sl, trail_on, tt_pts, td_pts,
        bars_held, trail_min_bars)
    swing_low = bar.get("last_swing_low")
    ms_sl = sl
    if swing_low is not None and not np.isnan(swing_low) and swing_low < bar["low"]:
        candidate = swing_low - bar["atr_rs"] * config.get("ms_buffer_atr", 0.1)
        if candidate > ms_sl:
            ms_sl = candidate
    if atr_sl >= ms_sl:
        return new_hwm, atr_sl, atr_trail
    return new_hwm, ms_sl, True


def update_ms_hybrid_short(bar, entry, lwm, sl, trail_on,
                           bars_held, trail_min_bars, config):
    """Mirror: tightest of ATR trail and structure swing (lower=tighter)."""
    new_lwm = min(lwm, bar["low"])
    tt_pts = config["atr_rs_entry"] * config["trail_trigger_atr"]
    td_pts = bar["atr_rs"] * config["trail_distance_atr"]
    _, atr_sl, atr_trail = update_trailing_stop_short(
        bar, entry, new_lwm, sl, trail_on, tt_pts, td_pts,
        bars_held, trail_min_bars)
    swing_high = bar.get("last_swing_high")
    ms_sl = sl
    if swing_high is not None and not np.isnan(swing_high) and swing_high > bar["high"]:
        candidate = swing_high + bar["atr_rs"] * config.get("ms_buffer_atr", 0.1)
        if candidate < ms_sl:
            ms_sl = candidate
    if atr_sl <= ms_sl:
        return new_lwm, atr_sl, atr_trail
    return new_lwm, ms_sl, True


# Spec 7.2 defaults
TRAIL_CONFIG = {
    "init_stop_atr":      3.0,   # initial stop = 3.0 x ATR(RS)_entry
    "trail_trigger_atr":  0.5,   # activate when move >= 0.5 x ATR(RS)_entry
    "trail_distance_atr": 0.5,   # trail distance = 0.5 x ATR(RS)_live
    "ms_buffer_atr":      0.1,
}
