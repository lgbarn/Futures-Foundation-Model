"""Phase D — the verdict run: full multi-year, multi-ticker walk-forward
+ label-shuffle robustness control.

Runs run_pipeline per ticker (full walk-forward, spec 3:1 folds by default)
and, unless disabled, a SHUFFLED-train-label control per ticker. The control
is the audit that killed CRT's false positive: a model that still passes the
every-OOS-month-PF>1 gate with shuffled train labels has a leakage/overfit
artifact, not a real edge.

Local CPU only (XGBoost/Optuna — no GPU). Use --max-windows / --trials for a
fast bounded probe before committing to the full 300-trial run.

  python -m pipelines.xgboost.phase_d --timeframe 5m \
      --tickers ES,NQ,RTY,YM,GC,SI --trials 300

PASS (real edge) requires, for EVERY ticker:
  • real: every-OOS-month-PF>1 gate PASS  AND  aggregate PF > 1.2
  • robustness: shuffled aggregate PF < 1.10  AND  real PF clearly > shuffled
Anything else => FAIL (no credible edge / leakage).
"""
import argparse
import sys

from .base import get_labeler
from . import labeler as _v2          # noqa: F401 — registers v2_triple_barrier
from .train import run_pipeline, _TF


def _row(tag, agg, gate, nм):
    pf = agg['profit_factor']
    return (f"  {tag:18s} months={nм:3d} gate={'PASS' if gate else 'FAIL':4s} "
            f"PF={pf:6.2f} WR={agg['win_rate']:.1%} PnL={agg['pnl']:+.4f} "
            f"maxDD={agg['max_dd']:+.2%}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeframe', choices=['5m', '3m'], default='5m')
    ap.add_argument('--tickers', default='ES,NQ,RTY,YM,GC,SI')
    ap.add_argument('--labeler', default='v2_triple_barrier')
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--max-windows', type=int, default=None,
                    help='bound walk-forward for a fast probe (e.g. 12)')
    ap.add_argument('--train-months', type=int, default=3)
    ap.add_argument('--test-months', type=int, default=1)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--no-shuffle-control', action='store_true',
                    help='skip the leakage audit (NOT recommended)')
    a = ap.parse_args(argv)

    if a.train_months != 3 or a.test_months != 1:
        print(f'⚠ DEVIATION: folds {a.train_months}/{a.test_months} differ '
              f'from the spec-validated 3/1 ratio — results not comparable to '
              f'the authoritative reference.', flush=True)

    bar_min = _TF[a.timeframe][2]
    tickers = [t.strip().upper() for t in a.tickers.split(',') if t.strip()]
    shuffle = not a.no_shuffle_control
    print(f'== PHASE D | {a.timeframe} | labeler={a.labeler} | '
          f'tickers={tickers} | trials={a.trials} | '
          f'folds={a.train_months}/{a.test_months} | '
          f'shuffle_control={shuffle} | '
          f'{"PROBE max_windows="+str(a.max_windows) if a.max_windows else "FULL"} ==',
          flush=True)

    results = {}
    for tk in tickers:
        print(f'\n———————— {tk} (REAL) ————————', flush=True)
        real = run_pipeline(
            get_labeler(a.labeler, bar_minutes=bar_min), a.timeframe, tk,
            trials=a.trials, max_windows=a.max_windows,
            train_months=a.train_months, test_months=a.test_months,
            val_frac=a.val_frac, shuffle_train_labels=False,
            save_artifact=False)
        shuf = None
        if shuffle:
            print(f'\n———————— {tk} (SHUFFLE control) ————————', flush=True)
            shuf = run_pipeline(
                get_labeler(a.labeler, bar_minutes=bar_min), a.timeframe, tk,
                trials=a.trials, max_windows=a.max_windows,
                train_months=a.train_months, test_months=a.test_months,
                val_frac=a.val_frac, shuffle_train_labels=True,
                save_artifact=False)
        results[tk] = (real, shuf)

    # ── consolidated verdict ──
    print('\n' + '=' * 64)
    print('  PHASE D — CONSOLIDATED VERDICT')
    print('=' * 64)
    overall = True
    reasons = []
    for tk, (real, shuf) in results.items():
        print(f'\n{tk}:')
        print(_row('REAL', real['aggregate'], real['gate_pass'],
                   real['n_months']))
        ra = real['aggregate']
        tk_ok = real['gate_pass'] and ra['profit_factor'] > 1.2
        if not tk_ok:
            reasons.append(f"{tk}: real gate/{ra['profit_factor']:.2f}PF "
                           f"below bar")
        if shuf is not None:
            print(_row('SHUFFLED', shuf['aggregate'], shuf['gate_pass'],
                       shuf['n_months']))
            sa = shuf['aggregate']
            robust = sa['profit_factor'] < 1.10 and \
                ra['profit_factor'] > sa['profit_factor'] + 0.30
            if not robust:
                tk_ok = False
                reasons.append(f"{tk}: shuffled PF {sa['profit_factor']:.2f} "
                               f"too high / real not clearly above (leakage)")
        overall &= tk_ok
        print(f'  -> {tk}: {"PASS" if tk_ok else "FAIL"}')

    print('\n' + '=' * 64)
    print(f'  PHASE D VERDICT: {"PASS — credible edge" if overall else "FAIL"}')
    if not overall:
        for r in reasons:
            print(f'   - {r}')
        print('   (a real edge survives shuffle; an artifact does not — '
              'same audit that killed CRT)')
    print('=' * 64, flush=True)
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
