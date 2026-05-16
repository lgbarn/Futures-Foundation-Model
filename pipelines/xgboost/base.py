"""Strategy-customization contract for the XGBoost pipeline.

Mirrors the DEVELOPER EXPERIENCE of futures_foundation.finetune.StrategyLabeler
(subclass one class, implement the strategy's target, register it — the
harness does features/walk-forward/Optuna/trail/gate/artifact) WITHOUT reusing
its contract: finetune's run() returns strategy-specific features + a binary
signal_label/max_rr (backbone-amplification paradigm). XGBoost predicts a
3-class DIRECTION from the 68 FFM features, so the contract here is a single
`label(df) -> Series[{-1,0,+1}]`. Same ease of use, correct semantics.

Add a strategy:
    @register("my_strategy")
    class MyLabeler(XGBStrategyLabeler):
        name = "my_strategy"
        def label(self, df): ...        # -> pd.Series of {-1,0,+1}
then:  --labeler my_strategy   (CLI)   or   run_pipeline(labeler=MyLabeler())
"""
from abc import ABC, abstractmethod
import pandas as pd

from futures_foundation.features import get_model_feature_columns


class XGBStrategyLabeler(ABC):
    #: short slug used in logs / artifact metadata (override)
    name: str = "unnamed"

    def feature_cols(self) -> list:
        """Model input columns. Default = the 68 FFM features. Override only
        to restrict/extend (must be columns present in derive_features)."""
        return get_model_feature_columns()

    def config_dict(self) -> dict:
        """JSON-serialisable params that affect label output. Same role as
        finetune's config_dict (cache-hash / provenance). Override; include a
        version + every threshold that changes labels."""
        return {}

    @abstractmethod
    def label(self, df: pd.DataFrame) -> pd.Series:
        """df: row-aligned to the feature matrix, columns datetime (tz-aware),
        open, high, low, close, atr (raw Wilder = vty_atr_raw). Return a
        pd.Series of {-1,0,+1} (short / no-trade / long) aligned to df.index.

        Causality contract: features the model sees are <= bar; the LABEL may
        use the bar's future (it is a label, like triple-barrier outcomes)."""
        ...


# ── registry: name -> class/factory, instantiated as (bar_minutes=...) ───────
# Strategy labelers take a keyword-only `bar_minutes` (the V2 default does;
# strategies that don't need it should accept `**_` or bar_minutes=None).
LABELERS: dict[str, callable] = {}


def register(name: str):
    def deco(cls_or_factory):
        if name in LABELERS:
            raise ValueError(f"labeler '{name}' already registered")
        LABELERS[name] = cls_or_factory
        return cls_or_factory
    return deco


def get_labeler(name: str, bar_minutes: int) -> XGBStrategyLabeler:
    if name not in LABELERS:
        raise KeyError(f"unknown labeler '{name}'. registered: "
                       f"{sorted(LABELERS)}")
    lab = LABELERS[name](bar_minutes=bar_minutes)
    if not isinstance(lab, XGBStrategyLabeler):
        raise TypeError(f"labeler '{name}' must be an XGBStrategyLabeler")
    return lab
