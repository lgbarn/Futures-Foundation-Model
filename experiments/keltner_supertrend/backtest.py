# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "pyarrow>=12.0",
#   "quantstats>=0.0.62",
# ]
# ///
"""Out-of-sample backtest for the Keltner+SuperTrend FFM strategy.

Re-scores the OOS test fold: recomputes the mechanical entries, reuses the
model's p_signal (from trades_baseline_ungated.csv), simulates exits, and
sweeps the confidence gate.

The DEFAULT is the standard fixed triple-barrier exit, frictionless. Every
non-standard behaviour is an explicit opt-in switch:

  --exit {barrier,trail}   exit model — barrier (default) or the hybrid
                           trailing stop (TrailAwareLabeler, Luther Barnum)
  --costs                  apply commission + slippage (default: off)
  --instrument SYM         traded instrument, from the sizing registry (default ES)
"""

import argparse
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

# ── Entry params (must match the labeler so entries line up with the model) ──
KC_EMA, KC_ATR, KC_MULT = 22, 20, 1.25
ST_PERIOD, ST_MULT = 10, 3.0
WARMUP_BARS = 50
RTH_OPEN_MIN, ENTRY_CLOSE_MIN, EOD_MIN = 9 * 60 + 30, 15 * 60 + 51, 15 * 60 + 55

# ── Fixed triple-barrier exit (DEFAULT) — matches the labeler ──
BARRIER_SL_ATR = 1.0        # hard stop = 1.0 ATR
BARRIER_TP_ATR = 1.5        # take-profit = 1.5 ATR (1.5R)

# ── Hybrid trail exit (opt-in: --exit trail) — TrailAwareLabeler, Luther Barnum ──
TRAIL_TRIGGER_ATR = 0.15
TRAIL_DISTANCE_ATR = 0.15
TRAIL_SL_ATR = 3.0
TRAIL_MAX_BARS = 30
TRAIL_MIN_BARS = 2

EXIT_BARRIER, EXIT_TRAIL = 'barrier', 'trail'

# ── Cost model (opt-in: --costs) ──
COMMISSION_USD = 4.00       # round-turn commission per contract
SLIPPAGE_TICKS = 1.0        # slippage per fill (entry + exit = 2 fills)

# ── OOS test fold + confidence sweep ──
VAL_END = pd.Timestamp('2026-01-04', tz='America/New_York')
TEST_END = pd.Timestamp('2026-04-04', tz='America/New_York')
GATE_SWEEP = [0.00, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52]

OUT_DIR = REPO_ROOT / 'models' / 'keltner_supertrend'
BASELINE_TRADES = OUT_DIR / 'trades_baseline_ungated.csv'


def simulate_barrier_exit(high, low, close, atr, minute_of_day, day, i, direction):
    """Fixed triple-barrier exit: 1.0-ATR stop, 1.5-ATR target, EOD flat.
    Returns gross P&L in price points. Stop checked before target (pessimistic)."""
    entry = close[i]
    sl_pts = atr[i] * BARRIER_SL_ATR
    tp_pts = atr[i] * BARRIER_TP_ATR
    n = len(close)
    if direction == 1:
        sl, tp = entry - sl_pts, entry + tp_pts
        for j in range(i + 1, n):
            if day[j] != day[i]:
                return close[j - 1] - entry
            if low[j] <= sl:
                return sl - entry
            if high[j] >= tp:
                return tp - entry
            if minute_of_day[j] >= EOD_MIN:
                return close[j] - entry
        return close[n - 1] - entry
    else:
        sl, tp = entry + sl_pts, entry - tp_pts
        for j in range(i + 1, n):
            if day[j] != day[i]:
                return entry - close[j - 1]
            if high[j] >= sl:
                return entry - sl
            if low[j] <= tp:
                return entry - tp
            if minute_of_day[j] >= EOD_MIN:
                return entry - close[j]
        return entry - close[n - 1]


def simulate_hybrid_trail(high, low, close, atr, minute_of_day, i, direction):
    """Pessimistic OHLC simulation of the hybrid trail (opt-in). Gross P&L in points.

    Each bar's adverse extreme is assumed to print first, and the trailing stop
    is tested at its pre-bar level before this bar's favourable extreme can
    ratchet it — the favourable extreme only tightens the stop for later bars.
    """
    entry = close[i]
    a = atr[i]
    trigger = a * TRAIL_TRIGGER_ATR
    distance = a * TRAIL_DISTANCE_ATR
    sl_pts = a * TRAIL_SL_ATR
    n = len(close)
    end = min(i + TRAIL_MAX_BARS + 1, n)

    if direction == 1:
        sl = entry - sl_pts
        hwm = high[i]
        trail_on = False
        for j in range(i + 1, end):
            if low[j] <= sl:
                return sl - entry
            if high[j] > hwm:
                hwm = high[j]
            if not trail_on and (j - i) >= TRAIL_MIN_BARS and (hwm - entry) >= trigger:
                trail_on = True
            if trail_on:
                sl = max(sl, hwm - distance)
            if minute_of_day[j] >= EOD_MIN:
                return close[j] - entry
        return close[min(i + TRAIL_MAX_BARS, n - 1)] - entry
    else:
        sl = entry + sl_pts
        lwm = low[i]
        trail_on = False
        for j in range(i + 1, end):
            if high[j] >= sl:
                return entry - sl
            if low[j] < lwm:
                lwm = low[j]
            if not trail_on and (j - i) >= TRAIL_MIN_BARS and (entry - lwm) >= trigger:
                trail_on = True
            if trail_on:
                sl = min(sl, lwm + distance)
            if minute_of_day[j] >= EOD_MIN:
                return entry - close[j]
        return entry - close[min(i + TRAIL_MAX_BARS, n - 1)]


def compute_oos_trades(instrument, exit_mode, cost_points):
    """Recompute Keltner+SuperTrend entries, simulate the chosen exit, attach
    model p_signal. net_pts = per-contract P&L net of costs; stop_pts = the
    hard-stop distance used for contract sizing."""
    prep = REPO_ROOT / 'data' / 'prep_input' / f'{instrument}_3min.parquet'
    df = pd.read_parquet(prep).sort_values('datetime').reset_index(drop=True)
    high = df['high'].to_numpy(np.float64)
    low = df['low'].to_numpy(np.float64)
    close = df['close'].to_numpy(np.float64)
    n = len(df)

    _, upper, lower = calc_keltner(high, low, close, KC_EMA, KC_ATR, KC_MULT)
    _, st_dir = calc_supertrend(high, low, close, ST_PERIOD, ST_MULT)
    atr = np.maximum(calc_atr(high, low, close, KC_ATR), 1e-6)

    dt = pd.DatetimeIndex(df['datetime'])
    minute_of_day = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
    day = dt.normalize().to_numpy()
    stop_atr = TRAIL_SL_ATR if exit_mode == EXIT_TRAIL else BARRIER_SL_ATR

    rows = []
    for i in range(max(WARMUP_BARS, 1), n):
        if not (RTH_OPEN_MIN <= minute_of_day[i] <= ENTRY_CLOSE_MIN):
            continue
        long_break = close[i - 1] <= upper[i - 1] and close[i] > upper[i] and st_dir[i] == 1
        short_break = close[i - 1] >= lower[i - 1] and close[i] < lower[i] and st_dir[i] == -1
        if not (long_break or short_break):
            continue
        direction = 1 if long_break else -1
        if exit_mode == EXIT_TRAIL:
            gross = simulate_hybrid_trail(high, low, close, atr, minute_of_day, i, direction)
        else:
            gross = simulate_barrier_exit(high, low, close, atr, minute_of_day, day, i, direction)
        rows.append({
            'datetime': dt[i],
            'direction': direction,
            'gross_pts': gross,
            'net_pts': gross - cost_points,
            'stop_pts': stop_atr * atr[i],
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

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exit', choices=(EXIT_BARRIER, EXIT_TRAIL), default=EXIT_BARRIER,
                    dest='exit_mode', help='exit model (default: barrier)')
    ap.add_argument('--costs', action='store_true',
                    help='apply commission + slippage (default: off)')
    ap.add_argument('--instrument', default='ES', help='traded instrument (default ES)')
    args = ap.parse_args()

    point_value, tick = sizing.specs(args.instrument)
    cost_points = (2 * SLIPPAGE_TICKS * tick + COMMISSION_USD / point_value
                   if args.costs else 0.0)

    print(f'\n{"="*108}')
    print(f'  OOS BACKTEST — {args.instrument} 3m  ({VAL_END.date()} → {TEST_END.date()})')
    if args.exit_mode == EXIT_TRAIL:
        print(f'  exit: HYBRID TRAIL (opt-in) — trigger {TRAIL_TRIGGER_ATR} / '
              f'distance {TRAIL_DISTANCE_ATR} / hard SL {TRAIL_SL_ATR} ATR / '
              f'max {TRAIL_MAX_BARS} bars  |  pessimistic OHLC path')
    else:
        print(f'  exit: fixed triple barrier (default) — {BARRIER_SL_ATR} ATR stop / '
              f'{BARRIER_TP_ATR} ATR target / EOD flat')
    print(f'  costs: {"ON — " if args.costs else "OFF (frictionless)"}'
          f'{f"{cost_points:.2f} pts/round-turn" if args.costs else ""}')
    print(f'  account: ${sizing.ACCOUNT_SIZE:,.0f}  |  {args.instrument} '
          f'(${point_value:.0f}/pt)  |  risk {sizing.RISK_FRAC:.2%}/trade  |  '
          f'cap {sizing.MAX_CONTRACTS} contracts')
    print(f'{"="*108}')

    oos = compute_oos_trades(args.instrument, args.exit_mode, cost_points)
    print(f'  p_signal range: {oos.p_signal.min():.3f} – {oos.p_signal.max():.3f}\n')
    print(ROW_HEADER)
    print('  ' + '-' * (len(ROW_HEADER) - 2))

    summary = []
    for thr in GATE_SWEEP:
        sub = oos[oos['p_signal'] >= thr].sort_values('datetime').reset_index(drop=True)
        if len(sub) < 10:
            print(f'  p>={thr:.2f}{"":<8}{len(sub):>7}   — too few trades')
            continue
        acct = sizing.simulate_account(sub['net_pts'].to_numpy(),
                                       sub['stop_pts'].to_numpy(), point_value)
        sub['contracts'] = acct['contracts']
        sub['pnl_dollars'] = acct['trade_dollars']
        s = compute_stats(f'p>={thr:.2f}', sub['datetime'], acct['trade_dollars'],
                          sizing.ACCOUNT_SIZE, contracts=acct['contracts'])
        print(format_row(s))
        row = {k: v for k, v in s.items() if k != 'daily_returns'}
        row['n_capped'] = acct['n_capped']
        row['n_min_floored'] = acct['n_min_floored']
        summary.append(row)

        tag = f'{args.exit_mode}_thr{int(round(thr*100)):02d}'
        sub.drop(columns=['utc']).to_csv(OUT_DIR / f'trades_{tag}.csv', index=False)
        html = str(OUT_DIR / f'quantstats_{tag}.html')
        try:
            qs.reports.html(s['daily_returns'], output=html, download_filename=html,
                            title=f'Keltner+SuperTrend {args.instrument} 3m '
                                  f'[{args.exit_mode}] — {s["name"]}')
        except Exception as exc:  # noqa: BLE001
            print(f'    ⚠ tearsheet failed for {tag}: {exc}')

    pd.DataFrame(summary).to_csv(OUT_DIR / f'{args.exit_mode}_sweep_summary.csv', index=False)
    print(f'\n  ✅ Sweep complete ({args.exit_mode} exit) — per-level tearsheets + '
          f'{args.exit_mode}_sweep_summary.csv in {OUT_DIR}')


if __name__ == '__main__':
    main()
