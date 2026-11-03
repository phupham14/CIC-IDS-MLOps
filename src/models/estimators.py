"""Khởi tạo và huấn luyện các mô hình: Logistic Regression, Linear SVM, Random Forest, XGBoost.

- Mất cân bằng lớp: dùng trọng số lớp (class_weight / scale_pos_weight), không SMOTE.
- Mô hình tuyến tính: log có dấu + RobustScaler trong Pipeline. Mô hình cây không cần scale.
"""
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

MODEL_TYPES = ("lr", "svm", "rf", "xgb")
N_JOBS = int(os.environ.get("CIC_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def _linear(clf):
    return Pipeline([("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                     ("scale", RobustScaler()),
                     ("clf", clf)])


def build(name, params, scale_pos_weight, seed=42):
    if name == "lr":
        return _linear(LogisticRegression(class_weight="balanced", random_state=seed, **params))
    if name == "svm":
        return _linear(LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000, **params))
    if name == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=seed, **params)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", eval_metric="aucpr", scale_pos_weight=scale_pos_weight,
                             n_jobs=N_JOBS, random_state=seed, **params)
    raise ValueError(f"Không có mô hình '{name}', chọn trong {MODEL_TYPES}")


def fit(name, model, train, val):
    if name == "xgb":
        # Truyền numpy để XGBoost không chép thêm dữ liệu; gán tên cột để kiểm tra khi dự đoán
        model.fit(train.X.to_numpy(), train.y, eval_set=[(val.X.to_numpy(), val.y)], verbose=False)
        model.get_booster().feature_names = list(train.X.columns)
    else:
        model.fit(train.X, train.y)
    return model


def scores(model, X):
    """Điểm Attack trong [0, 1]. LinearSVC dùng sigmoid(decision_function), không phải xác suất."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


def importances(name, model, features):
    """Độ quan trọng đặc trưng (cây: feature_importances_, tuyến tính: |hệ số|)."""
    if name in ("rf", "xgb"):
        values = model.feature_importances_
    else:
        values = np.abs(model.named_steps["clf"].coef_).ravel()
    return dict(sorted(zip(features, map(float, values)), key=lambda kv: -kv[1]))


def best_iteration(name, model):
    return int(model.best_iteration) if name == "xgb" and hasattr(model, "best_iteration") else None
