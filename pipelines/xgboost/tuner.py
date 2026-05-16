"""Optuna study (spec sections 6 + 8). TPE sampler, MedianPruner, 300 trials.

Each trial: sample XGB params within the section-6 anti-overfit bounds, fit on
the train-fit slice, build signals on the Optuna validation fold, run the
hybrid-trail backtest -> per-trade returns -> combined objective. Optuna
maximizes the combined score.

xgboost/optuna are imported lazily so the rest of the pipeline (and its
tests) import fine before the owner installs the deps.
"""
import numpy as np
import pandas as pd

from .objective import combined_objective, PERIODS_PER_YEAR
from .backtest import run_backtest

# spec section 6 — tight anti-overfit bounds (do NOT widen)
_BOUNDS = dict(
    max_depth=(3, 6), learning_rate=(0.01, 0.1),
    subsample=(0.6, 0.85), colsample_bytree=(0.7, 1.0),
    reg_lambda=(1, 10), min_child_weight=(5, 50),
    n_estimators=(200, 800),
)
CONF_THRESHOLD = 0.4
_CLASSES = [-1, 0, 1]
_TO_XGB = {-1: 0, 0: 1, 1: 2}            # contiguous for XGBoost
_FROM_XGB = {0: -1, 1: 0, 2: 1}


def _signals_from_proba(proba: np.ndarray, thr: float) -> np.ndarray:
    idx = proba.argmax(axis=1)
    conf = proba[np.arange(len(proba)), idx]
    sig = np.array([_FROM_XGB[i] for i in idx])
    sig[(sig == 0) | (conf < thr)] = 0
    return sig


def _fit_xgb(params, Xf, yf):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", n_jobs=-1, **params)
    m.fit(Xf, np.array([_TO_XGB[v] for v in yf]))
    return m


def tune(Xf, yf, Xv, df_val, timeframe: str, n_trials: int = 300,
         seed: int = 42) -> dict:
    """Xf/yf: train-fit features/labels. Xv: val-fold features. df_val:
    val-fold OHLCV+datetime (row-aligned to Xv) for the backtest. Returns the
    best XGB param dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ppy = PERIODS_PER_YEAR[timeframe]

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", *_BOUNDS["max_depth"]),
            learning_rate=trial.suggest_float("learning_rate",
                                              *_BOUNDS["learning_rate"], log=True),
            subsample=trial.suggest_float("subsample", *_BOUNDS["subsample"]),
            colsample_bytree=trial.suggest_float("colsample_bytree",
                                                 *_BOUNDS["colsample_bytree"]),
            reg_lambda=trial.suggest_float("reg_lambda", *_BOUNDS["reg_lambda"]),
            min_child_weight=trial.suggest_int("min_child_weight",
                                               *_BOUNDS["min_child_weight"]),
            n_estimators=trial.suggest_int("n_estimators",
                                           *_BOUNDS["n_estimators"]),
        )
        model = _fit_xgb(params, Xf, yf)
        proba = model.predict_proba(Xv)
        sig = _signals_from_proba(proba, CONF_THRESHOLD)
        res = run_backtest(df_val, sig)
        return combined_objective(res['returns'], ppy)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
