"""Trade simulation -> per-trade returns (spec section 7.6).

Model picks direction; exit = hybrid ATR/structure trail (trail.py). One
position at a time, causal entry at the NEXT bar's open after a signal.

Exit priority per bar (strict order, spec 7.6):
  1. EOD 15:55 ET  -> forced close at bar close
  2. Daily max loss -> forced close (optional; default off — needs contract $
     specifics not present in FFM OHLCV data)
  3. Trail stop hit (if active) / initial stop hit (if not) -> fill at the
     EXACT stop level (no gap-through; intraday stops are stop-limit)
  4. Take profit (unused — direction model, exit via trail)
Same bar SL & TP -> assume SL first (pessimistic).

Per-trade return = dir * (exit - entry) / entry  (fractional price move on
notional). Documented assumption: contract multiplier / account sizing is not
in FFM data, so we score the objective on fractional returns — consistent
across runs, which is all the Optuna objective needs.
"""
import numpy as np
import pandas as pd

from .trail import (rogers_satchell_atr, two_bar_fractals,
                     update_ms_hybrid_long, update_ms_hybrid_short,
                     TRAIL_CONFIG)

_EOD_H, _EOD_M = 15, 55


def _et(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt)
    if dt.dt.tz is None:
        return dt.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    return dt.dt.tz_convert('America/New_York')


def run_backtest(df: pd.DataFrame, signals: np.ndarray,
                 config: dict | None = None) -> dict:
    """df: datetime, open, high, low, close (ascending). signals: array of
    {-1,0,+1} aligned to df rows (already confidence-thresholded).

    Returns dict: 'returns' (pd.Series per-trade fractional), 'trades' (list
    of dicts), 'stats' (summary block)."""
    cfg = dict(TRAIL_CONFIG)
    if config:
        cfg.update(config)

    o = df['open'].to_numpy(np.float64)
    h = df['high'].to_numpy(np.float64)
    l = df['low'].to_numpy(np.float64)
    c = df['close'].to_numpy(np.float64)
    et = _et(df['datetime'])
    eod = ((et.dt.hour > _EOD_H) |
           ((et.dt.hour == _EOD_H) & (et.dt.minute >= _EOD_M))).to_numpy()
    n = len(df)

    atr_rs = rogers_satchell_atr(o, h, l, c, n=10)
    sw_lo, sw_hi = two_bar_fractals(h, l)

    trades = []
    i = 0
    while i < n - 1:
        sig = int(signals[i]) if np.isfinite(signals[i]) else 0
        if sig == 0:
            i += 1
            continue
        # causal entry: next bar open
        e_idx = i + 1
        entry = o[e_idx]
        atr_e = atr_rs[e_idx]
        if not np.isfinite(atr_e) or atr_e <= 0:
            i += 1
            continue
        is_long = sig == 1
        sl = (entry - cfg["init_stop_atr"] * atr_e if is_long
              else entry + cfg["init_stop_atr"] * atr_e)
        hwm = lwm = entry
        trail_on = False
        cfg_run = dict(cfg, atr_rs_entry=atr_e)

        exit_px = exit_idx = None
        held = 0
        for j in range(e_idx, n):
            held = j - e_idx
            bar = {"high": h[j], "low": l[j], "atr_rs": atr_rs[j],
                   "last_swing_low": sw_lo[j], "last_swing_high": sw_hi[j]}
            # 1) EOD forced close at this bar's close
            if eod[j]:
                exit_px, exit_idx = c[j], j
                break
            # 3) stop check BEFORE trail update (pessimistic intrabar)
            if is_long:
                if l[j] <= sl:
                    exit_px, exit_idx = sl, j
                    break
                hwm, sl, trail_on = update_ms_hybrid_long(
                    bar, entry, hwm, sl, trail_on, held, 1, cfg_run)
            else:
                if h[j] >= sl:
                    exit_px, exit_idx = sl, j
                    break
                lwm, sl, trail_on = update_ms_hybrid_short(
                    bar, entry, lwm, sl, trail_on, held, 1, cfg_run)
        if exit_px is None:                       # ran out of data
            exit_px, exit_idx = c[n - 1], n - 1

        dirn = 1 if is_long else -1
        ret = dirn * (exit_px - entry) / entry
        trades.append({'entry_idx': e_idx, 'exit_idx': exit_idx,
                       'dir': dirn, 'entry': entry, 'exit': exit_px,
                       'ret': ret, 'bars_held': exit_idx - e_idx})
        i = exit_idx + 1                          # one position at a time

    returns = pd.Series([t['ret'] for t in trades], dtype=float)
    return {'returns': returns, 'trades': trades,
            'stats': _stats(returns, trades)}


def _stats(returns: pd.Series, trades: list) -> dict:
    if len(returns) == 0:
        return {'trades': 0, 'win_rate': 0.0, 'pnl': 0.0,
                'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'max_dd': 0.0}
    w = returns[returns > 0]
    lo = returns[returns < 0]
    gp = float(w.sum()); gl = float(-lo.sum())
    cum = (1 + returns).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {
        'trades': len(returns),
        'win_rate': float((returns > 0).mean()),
        'pnl': float(returns.sum()),
        'profit_factor': (gp / gl) if gl > 0 else float('inf'),
        'avg_win': float(w.mean()) if len(w) else 0.0,
        'avg_loss': float(lo.mean()) if len(lo) else 0.0,
        'max_consec_win': _max_streak(returns > 0),
        'max_consec_loss': _max_streak(returns < 0),
        'max_dd': dd,
    }


def _max_streak(mask) -> int:
    best = cur = 0
    for v in np.asarray(mask):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
