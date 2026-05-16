"""Inference wrapper (spec section 9.4). Loads the joblib artifact, predicts
(-1/0/+1, confidence). Auto-applies an optional RF gate / HMM features ONLY if
their companion artifacts are present next to the model (sections 10/11 —
default-off; absence => plain model, still complete)."""
import numpy as np


class XGBPredictor:
    def __init__(self, art: dict):
        self.model = art["model"]
        self.feature_names = art["feature_names"]
        self.classes = art["classes"]                  # [-1, 0, 1]
        self.confidence_threshold = art.get("confidence_threshold", 0.4)
        self.timeframe = art.get("timeframe")
        self.instrument = art.get("instrument")
        self.atr_period = art.get("atr_period")

    @classmethod
    def load(cls, path: str) -> "XGBPredictor":
        import joblib                       # lazy: only needed at load time
        return cls(joblib.load(path))

    def predict(self, feature_row) -> tuple[int, float]:
        """feature_row: a 1-row DataFrame (or 2-D array col-aligned to
        feature_names). Returns (signal, confidence)."""
        X = feature_row[self.feature_names] if hasattr(feature_row, 'columns') \
            else np.asarray(feature_row).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        cls_idx = int(proba.argmax())
        signal = self.classes[cls_idx]                 # -1 / 0 / +1
        conf = float(proba[cls_idx])
        if signal == 0 or conf < self.confidence_threshold:
            return 0, conf
        return signal, conf


def artifact_path(instrument: str, tf: str, date_str: str) -> str:
    return f"xgb_{instrument.lower()}_{tf}_combined_{date_str}.joblib"
