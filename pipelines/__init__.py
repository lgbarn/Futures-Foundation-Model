"""Standalone trading pipelines.

Each subpackage is an independent, self-contained pipeline that may borrow
FFM feature engineering (futures_foundation.features) but does NOT depend on
the transformer backbone, pretraining, or the fine-tune framework.

Pipelines:
  - xgboost/  : gradient-boosted direction classifier + hybrid trail
                (spec: docs/xgboost-pipeline.md)
"""
