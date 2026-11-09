"""Mô hình phục vụ (MLflow pyfunc), được đăng ký vào Model Registry để API nạp.

Gói estimator, danh sách đặc trưng và ngưỡng. Đầu vào: DataFrame theo features.json (cột cờ *_win_missing tự tính).
Đầu ra: score, prediction (0/1), label ("Benign"/"Attack"). Thiếu cột hoặc có NaN/inf thì báo ValueError.

    model = mlflow.pyfunc.load_model("models:/cic-ids-binary@champion")
"""
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.models.estimators import scores

WINDOW_FLAGS = {"fwd_win_missing": "Init Fwd Win Byts", "bwd_win_missing": "Init Bwd Win Byts"}
OUTPUT_LABELS = np.array(["Benign", "Attack"])


class IDSModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, features, threshold, model_type):
        self.estimator = estimator
        self.features = list(features)
        self.threshold = float(threshold)
        self.model_type = model_type

    def prepare(self, df):
        df = pd.DataFrame(df)
        df = df.assign(**{flag: (df[src] == -1).astype(np.int8) for flag, src in WINDOW_FLAGS.items()
                          if flag in self.features and src in df.columns})
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu {len(missing)} cột đặc trưng: {missing}")
        X = df[self.features].astype(np.float32)
        bad = ~np.isfinite(X.to_numpy()).all(axis=1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} dòng có NaN/inf (dòng đầu tiên: {int(np.flatnonzero(bad)[0])})")
        return X

    def predict(self, context, model_input, params=None):
        s = scores(self.estimator, self.prepare(model_input))
        pred = (s >= self.threshold).astype(np.int64)
        return pd.DataFrame({"score": s.astype(np.float64), "prediction": pred, "label": OUTPUT_LABELS[pred]})


def signature(features):
    """Schema đầu vào / đầu ra; cột cờ không bắt buộc."""
    inputs = Schema([ColSpec("double", c, required=c not in WINDOW_FLAGS) for c in features])
    outputs = Schema([ColSpec("double", "score"), ColSpec("long", "prediction"), ColSpec("string", "label")])
    return ModelSignature(inputs=inputs, outputs=outputs)
