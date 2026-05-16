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

from futures_foundation.features import derive_features, get_model_feature_columns
from .labeler import TripleBarrierV2Labeler
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeframe', choices=['5m', '3m'], default='5m')
    ap.add_argument('--instrument', default='ES')
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--rf-gate', action='store_true')
    ap.add_argument('--hmm', action='store_true')
    a = ap.parse_args(argv)
    if a.rf_gate or a.hmm:
        print('NOTE: --rf-gate/--hmm are optional spec sections 10/11 and are '
              'NOT implemented in this build (primary path = sections 1-9). '
              'Ignoring.')

    period, atr_p, bar_min = _TF[a.timeframe]
    csv = os.path.join(_DATA, f'{a.instrument}_{period}.csv')
    if not os.path.exists(csv):
        sys.exit(f'data file not found: {csv} (run databento/build_continuous.py '
                 f'{period})')

    print(f'== XGBoost pipeline | {a.instrument} {a.timeframe} | '
          f'trials={a.trials} ==')
    df = pd.read_csv(csv)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    feat = derive_features(df, instrument=a.instrument, atr_period=atr_p)
    FCOLS = get_model_feature_columns()
    X = feat[FCOLS].reset_index(drop=True)             # NO nan-fill (xgb native)

    lab_df = pd.DataFrame({'datetime': df['datetime'].values,
                           'high': df['high'].values, 'low': df['low'].values,
                           'close': df['close'].values,
                           'atr': feat['vty_atr_raw'].values})
    y = TripleBarrierV2Labeler(bar_minutes=bar_min).label(lab_df).to_numpy()

    ohlcv = df[['datetime', 'open', 'high', 'low', 'close']].reset_index(drop=True)
    idx = pd.DatetimeIndex(df['datetime'])
    ppy = PERIODS_PER_YEAR[a.timeframe]

    oos_returns, month_rows, last_model = [], [], None
    for w, (tr_m, te_m) in enumerate(walk_forward_windows(idx), 1):
        fit_m, val_m = optuna_holdout(tr_m, 0.15)
        if val_m.sum() < 20 or te_m.sum() < 10:
            continue
        Xf, yf = X[fit_m].to_numpy(), y[fit_m]
        Xv = X[val_m].to_numpy()
        dfv = ohlcv[val_m].reset_index(drop=True)

        best = tune(Xf, yf, Xv, dfv, a.timeframe, n_trials=a.trials)

        # refit on the FULL train window, early-stopping on the val fold
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            tree_method='hist', n_jobs=-1, early_stopping_rounds=50, **best)
        from .tuner import _TO_XGB
        ytr_all = np.array([_TO_XGB[v] for v in y[tr_m]])
        yval_x = np.array([_TO_XGB[v] for v in y[val_m]])
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

    date_str = _dt.date.today().strftime('%Y%m%d')
    out = f'xgb_{a.instrument.lower()}_{a.timeframe}_combined_{date_str}.joblib'
    import joblib
    joblib.dump({'model': last_model, 'feature_names': FCOLS,
                 'classes': [-1, 0, 1], 'confidence_threshold': CONF_THRESHOLD,
                 'timeframe': a.timeframe, 'instrument': a.instrument,
                 'atr_period': atr_p}, out)
    print(f'\nsaved model -> {out}  | data: {csv}  '
          f'| span {idx[0].date()}..{idx[-1].date()}')


if __name__ == '__main__':
    main()
