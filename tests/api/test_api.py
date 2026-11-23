"""Kiểm thử API phục vụ mô hình (src/api/) theo hợp đồng INTERFACE.md.

Chạy: python -m pytest tests/api
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.api.app import create_app
from src.api.replay import make_timestamps, replay_batches
from src.models import estimators as E
from src.models import registry as R
from src.models import train as T
from src.models.pyfunc import WINDOW_FLAGS
from src.monitoring.scenarios import Batch, to_split
from tests.models.test_models import FEATURES, make_frame

RAW_FEATURES = [c for c in FEATURES if c not in WINDOW_FLAGS]


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    """Tạo MLflow SQLite tạm, đăng ký 2 phiên bản model và khởi tạo TestClient."""
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)

    cfg = yaml.safe_load(Path("models.yaml").read_text(encoding="utf-8"))
    cfg["mlflow"]["tracking_uri"] = db_uri
    cfg["mlflow"]["artifact_root"] = str(tmp_path / "artifacts")
    R.setup(cfg)

    df = make_frame(200, seed=42)
    from src.models.data import Split
    sp = Split(df[FEATURES], df["y_binary"].to_numpy(), df["Label"].to_numpy(), df["meta_day"].to_numpy())

    # Đăng ký version 1 (lr) làm @champion
    with mlflow.start_run(run_name="v1-lr"):
        est1 = E.fit("lr", E.build("lr", {"C": 1.0, "max_iter": 100}, 1.0), sp, sp)
        info1 = T.log_model("lr", est1, FEATURES, 0.45, "data/preprocessing", {"feature_set": "all"})
    R.register(info1.model_uri, cfg["registry"]["model_name"], cfg["registry"]["champion_alias"])

    # Đăng ký version 2 (xgb) làm @challenger
    with mlflow.start_run(run_name="v2-xgb"):
        est2 = E.fit("xgb", E.build("xgb", {"n_estimators": 10, "max_depth": 3}, 1.0), sp, sp)
        info2 = T.log_model("xgb", est2, FEATURES, 0.65, "data/preprocessing", {"feature_set": "all"})
    R.register(info2.model_uri, cfg["registry"]["model_name"], cfg["registry"]["challenger_alias"])

    log_dir = tmp_path / "logs" / "predictions"
    app = create_app(
        cfg=cfg,
        log_dir=str(log_dir),
        reload_interval_sec=3600,
        log_flush_max_rows=10,
        log_flush_interval_sec=3600,
    )
    with TestClient(app) as client:
        yield client, cfg, log_dir, df


def test_model_info_and_single_predict(api_env):
    client, _, _, df = api_env

    # 1. Kiểm tra GET /model
    res_model = client.get("/model")
    assert res_model.status_code == 200
    info = res_model.json()
    assert info["loaded"] is True
    assert info["version"] == "1"
    assert info["model_type"] == "lr"
    assert info["threshold"] == pytest.approx(0.45)
    assert info["required_features_count"] == 70

    # 2. Gửi 1 luồng gồm 70 cột đặc trưng (không gửi 2 cột cờ *_win_missing, giá trị kiểu int/float lẫn lộn)
    row = df[RAW_FEATURES].iloc[0].to_dict()
    row["Dst Port"] = int(row["Dst Port"])
    row["Protocol"] = int(row["Protocol"])
    res_pred = client.post("/predict", json=row)
    assert res_pred.status_code == 200
    body = res_pred.json()
    assert set(body.keys()) >= {"request_id", "score", "prediction", "label", "model_version", "threshold"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Benign", "Attack")
    assert body["model_version"] == "1"


def test_validation_returns_422_on_missing_or_nan_inf(api_env):
    client, _, _, df = api_env
    row = df[RAW_FEATURES].iloc[0].to_dict()

    # Thiếu cột đặc trưng -> 422
    incomplete = dict(list(row.items())[:15])
    res_missing = client.post("/predict", json=incomplete)
    assert res_missing.status_code == 422
    assert "Thiếu" in res_missing.json()["detail"]

    # Giá trị chuỗi không phải số -> 422
    bad_type = dict(row)
    bad_type["Flow Duration"] = "not-a-number"
    res_bad = client.post("/predict", json=bad_type)
    assert res_bad.status_code == 422


def test_batch_predict_and_parquet_logging(api_env):
    client, _, log_dir, df = api_env

    # Gửi 12 dòng (> log_flush_max_rows=10) để tự động flush ra Parquet
    flows = df[RAW_FEATURES].iloc[:12].to_dict(orient="records")
    req_ids = [f"req-{i:03d}" for i in range(12)]
    timestamps = ["2026-03-01T10:15:00Z"] * 12

    res = client.post(
        "/predict/batch",
        json={"flows": flows, "request_ids": req_ids, "timestamps": timestamps},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 12
    assert data["model_version"] == "1"
    assert len(data["predictions"]) == 12
    assert [p["request_id"] for p in data["predictions"]] == req_ids

    # Kiểm tra file Parquet được ghi đúng cấu trúc logs/predictions/date=2026-03-01/part-*.parquet
    day_dir = log_dir / "date=2026-03-01"
    assert day_dir.exists()
    files = list(day_dir.glob("part-*.parquet"))
    assert len(files) == 1

    logged = pd.read_parquet(files[0])
    assert len(logged) == 12
    assert list(logged["request_id"]) == req_ids
    assert (logged["model_version"] == "1").all()
    assert logged["score"].dtype == np.float64
    assert logged["prediction"].dtype == np.int8
    assert logged["latency_ms"].dtype == np.float32
    for col in RAW_FEATURES:
        assert col in logged.columns
        assert logged[col].dtype == np.float32


def test_reload_switches_champion_without_restart_and_exposes_metrics(api_env):
    client, cfg, _, df = api_env

    # Promote version 2 lên @champion
    R.promote(cfg["registry"]["model_name"], "2", cfg["registry"])

    # Gọi POST /reload
    res_reload = client.post("/reload")
    assert res_reload.status_code == 200
    r_info = res_reload.json()
    assert r_info["reloaded"] is True
    assert r_info["previous_version"] == "1"
    assert r_info["current_version"] == "2"
    assert r_info["model_type"] == "xgb"
    assert r_info["threshold"] == pytest.approx(0.65)

    # Dự đoán sau khi reload phải ghi nhận model_version == "2"
    row = df[RAW_FEATURES].iloc[0].to_dict()
    res_pred = client.post("/predict", json=row)
    assert res_pred.status_code == 200
    assert res_pred.json()["model_version"] == "2"

    # Kiểm tra GET /metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    text = res_metrics.text
    assert "ids_requests_total" in text
    assert "ids_predictions_total" in text
    assert "ids_score_bucket" in text
    assert "ids_inference_latency_seconds" in text
    assert 'ids_model_info{model_type="xgb",threshold="0.650000",version="2"} 1.0' in text
    assert "ids_model_version 2.0" in text


def test_replay_batches_delays_labels_by_one_batch(api_env, tmp_path):
    client, _, log_dir, df = api_env
    labels_dir = tmp_path / "logs" / "labels"

    # Tạo 3 lô (batch 0, 1 ngày 2018-02-22; batch 2 ngày 2018-02-23), mỗi lô 8 dòng
    b0 = Batch(0, "2018-02-22", to_split(df.iloc[:8].assign(meta_day="2018-02-22"), FEATURES))
    b1 = Batch(1, "2018-02-22", to_split(df.iloc[8:16].assign(meta_day="2018-02-22"), FEATURES))
    b2 = Batch(2, "2018-02-23", to_split(df.iloc[16:24].assign(meta_day="2018-02-23"), FEATURES))

    stats = replay_batches(
        batches=[b0, b1, b2],
        raw_features=RAW_FEATURES,
        client=client,
        labels_dir=labels_dir,
        delay_batches=1,
        chunk_size=5,
        flush_remaining_labels=False,
    )
    assert stats["batches_sent"] == 3
    assert stats["rows_sent"] == 24
    # Với delay_batches=1, sau khi gửi 3 lô thì chỉ có nhãn của lô 0 và lô 1 được ghi
    assert stats["label_batches_written"] == 2
    assert (labels_dir / "date=2018-02-22" / "batch-0000.parquet").exists()
    assert (labels_dir / "date=2018-02-22" / "batch-0001.parquet").exists()
    assert not (labels_dir / "date=2018-02-23" / "batch-0002.parquet").exists()

    # Kiểm tra request_id trong file nhãn khớp với request_id trong file dự đoán
    pred_files = list((log_dir / "date=2018-02-22").glob("part-*.parquet"))
    assert len(pred_files) >= 2
    pred_df = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    lab0_df = pd.read_parquet(labels_dir / "date=2018-02-22" / "batch-0000.parquet")
    assert set(lab0_df.columns) >= {"request_id", "Label"}
    assert set(lab0_df["request_id"]).issubset(set(pred_df["request_id"]))



def test_timestamps_use_real_time_or_never_overlap_within_a_day(api_env):
    _, _, _, df = api_env
    sp = to_split(df.iloc[:4].assign(meta_day="2018-02-22"), FEATURES)
    t = pd.to_datetime(["2018-02-22 10:00:00", "2018-02-22 10:00:01", "2018-02-22 10:00:01", "2018-02-22 23:59:59"])
    stamps, off = make_timestamps(Batch(30, "2018-02-22", sp, t.to_numpy()))
    assert stamps == ["2018-02-22T10:00:00.000Z", "2018-02-22T10:00:01.000Z",
                      "2018-02-22T10:00:01.000Z", "2018-02-22T23:59:59.000Z"] and off == 0

    # Không có time: lô 0 và lô 24 cùng ngày từng bị trùng giờ (batch_id % 24); giờ đây nối tiếp nhau
    s0, off = make_timestamps(Batch(0, "2018-02-22", sp))
    s24, _ = make_timestamps(Batch(24, "2018-02-22", sp), off)
    assert not set(s0) & set(s24) and max(s0) < min(s24)
    assert all(s.startswith("2018-02-22T") for s in s0 + s24)
