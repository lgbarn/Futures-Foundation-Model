"""End-to-end training CLI (spec section 9).

  python -m pipelines.xgboost.train --timeframe 5m --instrument ES --trials 300

features (FFM 68) -> V2 labels -> rolling walk-forward [Optuna(combined obj,
hybrid-trail backtest) -> refit -> OOS backtest] -> aggregate -> save joblib
-> full stat block (every OOS month printed). xgboost/joblib imported lazily.

Optional --rf-gate / --hmm (spec 10/11) are NOT implemented (out of the
primary 1-9 scope); the flags are accepted and ignored with a notice so a
caller is never silently misled.
"""
import argparse
import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

from futures_foundation.features import derive_features
from .base import get_labeler, XGBStrategyLabeler
from . import labeler as _v2          # noqa: F401 — registers v2_triple_barrier
from .walkforward import walk_forward_windows, optuna_holdout
from .tuner import tune, _fit_xgb, _signals_from_proba, CONF_THRESHOLD
from .backtest import run_backtest
from .objective import PERIODS_PER_YEAR

_TF = {'5m': ('5min', 14, 5), '3m': ('3min', 20, 3)}   # file, atr_period, bar_min
_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _print_stats(tag: str, st: dict):
    pf = st['profit_factor']
    print(f"  [{tag}] trades={st['trades']} WR={st['win_rate']:.1%} "
          f"PnL={st['pnl']:+.4f} PF={pf:.2f} "
          f"avgW={st['avg_win']:+.4f} avgL={st['avg_loss']:+.4f} "
          f"maxDD={st['max_dd']:+.2%} "
          f"maxWcons={st.get('max_consec_win',0)} "
          f"maxLcons={st.get('max_consec_loss',0)}")


def run_pipeline(labeler: XGBStrategyLabeler, timeframe: str,
                 instrument: str = 'ES', trials: int = 300,
                 max_windows: int | None = None,
                 train_months: int = 3, test_months: int = 1,
                 val_frac: float = 0.15,
                 shuffle_train_labels: bool = False,
                 save_artifact: bool = True, seed: int = 42) -> dict:
    """End-to-end run for ANY strategy labeler (finetune-parity API):

        run_pipeline(MyLabeler(bar_minutes=5), '5m', 'ES', trials=300)

    The harness owns features/walk-forward/Optuna/trail/gate/artifact; the
    labeler owns only the {-1,0,+1} target + (optionally) feature_cols.

    train_months/test_months: walk-forward window (spec-validated = 3/1).
    val_frac: Optuna validation = last `val_frac` of each train window.
    shuffle_train_labels: ROBUSTNESS CONTROL — permute the label vector
        within each TRAIN window only (OOS untouched). A model that still
        passes the gate with shuffled train labels has a leakage/overfit
        artifact, not a real edge (the audit that killed CRT)."""
    period, atr_p, bar_min = _TF[timeframe]
    csv = os.path.join(_DATA, f'{instrument}_{period}.csv')
    if not os.path.exists(csv):
        sys.exit(f'data file not found: {csv} (run databento/build_continuous.py '
                 f'{period})')

    print(f'== XGBoost pipeline | {instrument} {timeframe} | '
          f'labeler={labeler.name} | trials={trials} ==', flush=True)
    df = pd.read_csv(csv)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    feat = derive_features(df, instrument=instrument, atr_period=atr_p)
    FCOLS = labeler.feature_cols()
    X = feat[FCOLS].reset_index(drop=True)             # NO nan-fill (xgb native)

    lab_df = pd.DataFrame({'datetime': df['datetime'].values,
                           'open': df['open'].values, 'high': df['high'].values,
                           'low': df['low'].values, 'close': df['close'].values,
                           'atr': feat['vty_atr_raw'].values})
    y = np.asarray(labeler.label(lab_df))

    ohlcv = df[['datetime', 'open', 'high', 'low', 'close']].reset_index(drop=True)
    idx = pd.DatetimeIndex(df['datetime'])
    ppy = PERIODS_PER_YEAR[timeframe]

    if shuffle_train_labels:
        print('  ⚠ SHUFFLE CONTROL: train labels permuted per window '
              '(OOS untouched) — a PASS here means leakage/overfit, not edge',
              flush=True)
    rng = np.random.default_rng(seed)

    oos_returns, month_rows, last_model = [], [], None
    for w, (tr_m, te_m) in enumerate(
            walk_forward_windows(idx, train_months, test_months), 1):
        if max_windows is not None and w > max_windows:
            print(f'  (stopping at --max-windows={max_windows})', flush=True)
            break
        fit_m, val_m = optuna_holdout(tr_m, val_frac)
        if val_m.sum() < 20 or te_m.sum() < 10:
            continue
        yw = y.copy()
        if shuffle_train_labels:
            tr_idx = np.flatnonzero(tr_m)        # permute TRAIN labels only
            yw[tr_idx] = rng.permutation(yw[tr_idx])
        print(f'  window {w}: train={int(tr_m.sum())} val={int(val_m.sum())} '
              f'test={int(te_m.sum())} bars — tuning...', flush=True)
        Xf, yf = X[fit_m].to_numpy(), yw[fit_m]
        Xv = X[val_m].to_numpy()
        dfv = ohlcv[val_m].reset_index(drop=True)

        best = tune(Xf, yf, Xv, dfv, timeframe, n_trials=trials)

        # refit on the FULL train window, early-stopping on the val fold
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            tree_method='hist', n_jobs=-1, early_stopping_rounds=50, **best)
        from .tuner import _TO_XGB
        ytr_all = np.array([_TO_XGB[v] for v in yw[tr_m]])
        yval_x = np.array([_TO_XGB[v] for v in yw[val_m]])
        model.fit(X[tr_m].to_numpy(), ytr_all,
                  eval_set=[(Xv, yval_x)], verbose=False)
        last_model = model

        proba = model.predict_proba(X[te_m].to_numpy())
        sig = _signals_from_proba(proba, CONF_THRESHOLD)
        res = run_backtest(ohlcv[te_m].reset_index(drop=True), sig)
        st = res['stats']
        mlabel = idx[te_m][0].strftime('%Y-%m')
        month_rows.append((mlabel, st))
        oos_returns.append(res['returns'])
        _print_stats(f'OOS {mlabel}', st)

    if not oos_returns:
        sys.exit('No completed walk-forward windows (need >=4 months data).')

    agg = pd.concat(oos_returns, ignore_index=True)
    from .backtest import _stats
    print('\n=== AGGREGATE OOS ===')
    _print_stats('AGG', _stats(agg, []))
    print('\n=== PER-OOS-MONTH ===')
    pf_floor_ok = True
    for m, st in month_rows:
        _print_stats(m, st)
        if not (st['profit_factor'] > 1.0):
            pf_floor_ok = False
    print(f"\nGATE (every OOS month PF>1): "
          f"{'PASS' if pf_floor_ok else 'FAIL'}")

    out = None
    if save_artifact and not shuffle_train_labels:
        date_str = _dt.date.today().strftime('%Y%m%d')
        out = (f'xgb_{instrument.lower()}_{timeframe}_{labeler.name}_'
               f'{date_str}.joblib')
        import joblib
        joblib.dump({'model': last_model, 'feature_names': FCOLS,
                     'classes': [-1, 0, 1],
                     'confidence_threshold': CONF_THRESHOLD,
                     'timeframe': timeframe, 'instrument': instrument,
                     'atr_period': atr_p, 'labeler': labeler.name,
                     'labeler_config': labeler.config_dict()}, out)
        print(f'\nsaved model -> {out}', flush=True)
    print(f'  data: {csv} | span {idx[0].date()}..{idx[-1].date()}',
          flush=True)
    return {'gate_pass': pf_floor_ok, 'months': month_rows,
            'aggregate': _stats(agg, []), 'artifact': out,
            'n_months': len(month_rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeframe', choices=['5m', '3m'], default='5m')
    ap.add_argument('--instrument', default='ES')
    ap.add_argument('--labeler', default='v2_triple_barrier',
                    help='registered strategy labeler name (see base.LABELERS)')
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--max-windows', type=int, default=None,
                    help='cap walk-forward windows (smoke: e.g. 3). '
                         'trials only bounds Optuna; this bounds the run.')
    ap.add_argument('--rf-gate', action='store_true')
    ap.add_argument('--hmm', action='store_true')
    a = ap.parse_args(argv)
    if a.rf_gate or a.hmm:
        print('NOTE: --rf-gate/--hmm are optional spec sections 10/11 and are '
              'NOT implemented (primary path = sections 1-9). Ignoring.')
    bar_min = _TF[a.timeframe][2]
    labeler = get_labeler(a.labeler, bar_minutes=bar_min)
    run_pipeline(labeler, a.timeframe, a.instrument, a.trials, a.max_windows)


if __name__ == '__main__':
    main()
