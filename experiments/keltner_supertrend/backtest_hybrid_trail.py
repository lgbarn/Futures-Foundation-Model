# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "pyarrow>=12.0",
#   "quantstats>=0.0.62",
# ]
# ///
"""Re-score the OOS backtest with the hybrid trailing stop — cost + path realistic.

The ML model gates *entries* on FFM + strategy features (exit-agnostic), so the
entry set and p_signal from run 4 are reused unchanged — "rerun just the backtest".
Exits use the hybrid trail ported from trading-research labeler_trail.py
(TrailAwareLabeler), with two realism layers added:

  COST MODEL — per round-turn: commission + slippage on entry and exit fills,
  subtracted from gross P&L in price points.

  PATH REALISM — the OHLC trail sim is PESSIMISTIC: within each bar the adverse
  extreme is assumed to print first (low-before-high for longs), and the trailing
  stop is checked at its pre-bar level before that bar's high can ratchet it up.
  This removes the "ratchet up then fill at the higher stop in the same bar"
  optimism of a naive OHLC simulation.

The confidence gate is swept across levels so the trade-count / stats trade-off
is visible end to end.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.keltner_supertrend.indicators import calc_atr, calc_keltner, calc_supertrend
from experiments.keltner_supertrend.stats import ROW_HEADER, compute_stats, format_row
from experiments.keltner_supertrend import sizing

# ── Strategy params (must match the labeler so entries line up with run 4) ──
KC_EMA, KC_ATR, KC_MULT = 22, 20, 1.25
ST_PERIOD, ST_MULT = 10, 3.0
WARMUP_BARS = 50
RTH_OPEN_MIN, ENTRY_CLOSE_MIN, EOD_MIN = 9 * 60 + 30, 15 * 60 + 51, 15 * 60 + 55

# ── Hybrid trail params (TrailAwareLabeler defaults — Luther Barnum) ──
TRAIL_TRIGGER_ATR = 0.15
TRAIL_DISTANCE_ATR = 0.15
SL_ATR_MULT = 3.0
MAX_BARS = 30
TRAIL_MIN_BARS = 2

# ── Instrument (drives the data file + contract specs from the registry) ──
INSTRUMENT = 'ES'
POINT_VALUE, TICK_SIZE = sizing.specs(INSTRUMENT)

# ── Cost model (per contract) ──
COMMISSION_USD = 4.00       # round-turn commission per contract
SLIPPAGE_TICKS = 1.0        # slippage per fill (entry + exit = 2 fills)
COST_POINTS = 2 * SLIPPAGE_TICKS * TICK_SIZE + COMMISSION_USD / POINT_VALUE

# ── OOS test fold + confidence sweep ──
VAL_END = pd.Timestamp('2026-01-04', tz='America/New_York')
TEST_END = pd.Timestamp('2026-04-04', tz='America/New_York')
GATE_SWEEP = [0.00, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52]

PREP_PARQUET = REPO_ROOT / 'data' / 'prep_input' / f'{INSTRUMENT}_3min.parquet'
OUT_DIR = REPO_ROOT / 'models' / 'keltner_supertrend'
BASELINE_TRADES = OUT_DIR / 'trades_baseline_ungated.csv'


def simulate_hybrid_trail(high, low, close, atr, minute_of_day, i, direction):
    """Pessimistic OHLC simulation of the hybrid trail. Returns gross P&L in points.

    Pessimistic ordering: each bar's adverse extreme is assumed to print first,
    and the stop is tested at its pre-bar level before this bar's favourable
    extreme can ratchet it. The favourable extreme only tightens the stop for
    *subsequent* bars.
    """
    entry = close[i]
    a = atr[i]
    trigger = a * TRAIL_TRIGGER_ATR
    distance = a * TRAIL_DISTANCE_ATR
    sl_pts = a * SL_ATR_MULT
    n = len(close)
    end = min(i + MAX_BARS + 1, n)

    if direction == 1:
        sl = entry - sl_pts
        hwm = high[i]
        trail_on = False
        for j in range(i + 1, end):
            if low[j] <= sl:                       # adverse extreme first
                return (sl - entry)
            if high[j] > hwm:                      # then ratchet for next bar
                hwm = high[j]
            if not trail_on and (j - i) >= TRAIL_MIN_BARS and (hwm - entry) >= trigger:
                trail_on = True
            if trail_on:
                sl = max(sl, hwm - distance)
            if minute_of_day[j] >= EOD_MIN:
                return (close[j] - entry)
        return (close[min(i + MAX_BARS, n - 1)] - entry)
    else:
        sl = entry + sl_pts
        lwm = low[i]
        trail_on = False
        for j in range(i + 1, end):
            if high[j] >= sl:                      # adverse extreme first
                return (entry - sl)
            if low[j] < lwm:
                lwm = low[j]
            if not trail_on and (j - i) >= TRAIL_MIN_BARS and (entry - lwm) >= trigger:
                trail_on = True
            if trail_on:
                sl = min(sl, lwm + distance)
            if minute_of_day[j] >= EOD_MIN:
                return (entry - close[j])
        return (entry - close[min(i + MAX_BARS, n - 1)])


def compute_oos_trades():
    """Recompute Keltner+SuperTrend entries, simulate the cost-adjusted hybrid
    trail, attach model p_signal. net_pts is per-contract P&L net of costs;
    stop_pts is the hard-stop distance (used for contract sizing)."""
    df = pd.read_parquet(PREP_PARQUET).sort_values('datetime').reset_index(drop=True)
    high = df['high'].to_numpy(np.float64)
    low = df['low'].to_numpy(np.float64)
    close = df['close'].to_numpy(np.float64)
    n = len(df)

    _, upper, lower = calc_keltner(high, low, close, KC_EMA, KC_ATR, KC_MULT)
    _, st_dir = calc_supertrend(high, low, close, ST_PERIOD, ST_MULT)
    atr = np.maximum(calc_atr(high, low, close, KC_ATR), 1e-6)

    dt = pd.DatetimeIndex(df['datetime'])
    minute_of_day = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()

    rows = []
    for i in range(max(WARMUP_BARS, 1), n):
        if not (RTH_OPEN_MIN <= minute_of_day[i] <= ENTRY_CLOSE_MIN):
            continue
        long_break = close[i - 1] <= upper[i - 1] and close[i] > upper[i] and st_dir[i] == 1
        short_break = close[i - 1] >= lower[i - 1] and close[i] < lower[i] and st_dir[i] == -1
        if not (long_break or short_break):
            continue
        direction = 1 if long_break else -1
        gross = simulate_hybrid_trail(high, low, close, atr, minute_of_day, i, direction)
        net = gross - COST_POINTS                  # apply round-turn cost
        rows.append({
            'datetime': dt[i],
            'direction': direction,
            'gross_pts': gross,
            'net_pts': net,
            'stop_pts': SL_ATR_MULT * atr[i],
        })

    trades = pd.DataFrame(rows)
    trades['utc'] = pd.to_datetime(trades['datetime'], utc=True)
    oos = trades[(trades['datetime'] >= VAL_END) & (trades['datetime'] < TEST_END)].copy()

    base = pd.read_csv(BASELINE_TRADES)
    base['utc'] = pd.to_datetime(base['datetime'], utc=True)
    oos['p_signal'] = oos['utc'].map(base.set_index('utc')['p_signal'])
    matched = int(oos['p_signal'].notna().sum())
    print(f'  OOS entries recomputed: {len(oos)}  |  matched to model p_signal: '
          f'{matched} / {len(base)}')
    return oos.dropna(subset=['p_signal']).reset_index(drop=True)


def main():
    import quantstats as qs

    print(f'\n{"="*94}')
    print(f'  HYBRID-TRAIL BACKTEST (cost + path realistic) — ES 3m OOS '
          f'({VAL_END.date()} → {TEST_END.date()})')
    print(f'  trail: trigger {TRAIL_TRIGGER_ATR} / distance {TRAIL_DISTANCE_ATR} / '
          f'hard SL {SL_ATR_MULT} ATR / max {MAX_BARS} bars  |  path: pessimistic OHLC')
    print(f'  cost: {COMMISSION_USD:.0f}$ commission + {SLIPPAGE_TICKS:.0f} tick/fill '
          f'slippage = {COST_POINTS:.2f} pts/round-turn')
    print(f'  account: ${sizing.ACCOUNT_SIZE:,.0f}  |  {INSTRUMENT} '
          f'(${POINT_VALUE:.0f}/pt)  |  risk {sizing.RISK_FRAC:.2%}/trade  |  '
          f'cap {sizing.MAX_CONTRACTS} contracts')
    print(f'{"="*94}')

    oos = compute_oos_trades()
    print(f'  p_signal range: {oos.p_signal.min():.3f} – {oos.p_signal.max():.3f}\n')
    print(ROW_HEADER)
    print('  ' + '-' * (len(ROW_HEADER) - 2))

    summary = []
    for thr in GATE_SWEEP:
        sub = oos[oos['p_signal'] >= thr].sort_values('datetime').reset_index(drop=True)
        label = f'p>={thr:.2f}' + ('  (all)' if thr == 0.0 else '')
        if len(sub) < 10:
            print(f'  {label:<14}{len(sub):>7}   — too few trades')
            continue
        acct = sizing.simulate_account(sub['net_pts'].to_numpy(),
                                       sub['stop_pts'].to_numpy(), POINT_VALUE)
        sub['contracts'] = acct['contracts']
        sub['pnl_dollars'] = acct['trade_dollars']
        s = compute_stats(f'p>={thr:.2f}', sub['datetime'], acct['trade_dollars'],
                          sizing.ACCOUNT_SIZE, contracts=acct['contracts'])
        print(format_row(s))
        row = {k: v for k, v in s.items() if k != 'daily_returns'}
        row['n_capped'] = acct['n_capped']
        row['n_min_floored'] = acct['n_min_floored']
        summary.append(row)

        tag = f'thr{int(round(thr*100)):02d}'
        sub.drop(columns=['utc']).to_csv(OUT_DIR / f'trades_hybridtrail_{tag}.csv', index=False)
        html = str(OUT_DIR / f'quantstats_hybridtrail_{tag}.html')
        try:
            qs.reports.html(s['daily_returns'], output=html, download_filename=html,
                            title=f'Keltner+SuperTrend ES 3m hybrid-trail — {s["name"]}')
        except Exception as exc:  # noqa: BLE001
            print(f'    ⚠ tearsheet failed for {tag}: {exc}')

    pd.DataFrame(summary).to_csv(OUT_DIR / 'hybridtrail_sweep_summary.csv', index=False)
    print(f'\n  ✅ Sweep complete — per-level tearsheets + hybridtrail_sweep_summary.csv'
          f' in {OUT_DIR}')


if __name__ == '__main__':
    main()
