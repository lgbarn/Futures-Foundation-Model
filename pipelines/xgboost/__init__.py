"""Standalone XGBoost trading pipeline.

Independent of the FFM transformer (model.py / pretrain / finetune). Borrows
only feature engineering from futures_foundation.features. Build spec:
docs/xgboost-pipeline.md (sections 1-9 = primary path; 10 RF-gate / 11 HMM are
optional, default-off).

Pipeline: derive_features (68) -> V2 session triple-barrier labels (-1/0/+1)
-> rolling walk-forward -> Optuna(TPE, combined CAGR*sqrt(Sortino) objective,
scored on a hybrid ATR/structure-trail backtest) -> XGBClassifier -> joblib.
"""
