"""Test src/monitoring/live.py và push.py trên log tổng hợp, ghi bằng chính PredictionLogger của API (INTERFACE.md)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from prometheus_client import generate_latest

from src.api.logger import PredictionLogger
from src.api.replay import write_label_batch
from src.monitoring import live as L
from src.monitoring.push import Pusher

FEATURES = json.loads(Path("data/preprocessing/features.json").read_text(encoding="utf-8"))
RAW = [c for c in FEATURES if not c.endswith("_win_missing")]
RETRAIN = {"min_support": 30, "min_class_recall": 0.5, "max_recall_drop": 0.1}


class StubDetector:
    def check(self, X, scores):
        assert list(X.columns) == FEATURES and len(X) == len(scores)
        share = float(np.mean(scores))                 # đủ để phân biệt các lô trong test
        return {"n_drifted": 0, "drift_share": share, "dataset_drift": share > 0.5, "score_drift": 0.0,
                "score_drifted": False, "top_columns": []}


_LOGGERS = {}


def write_log(log_dir, day, start_sec, n, version="1", attack_every=0, seed=0):
    """Ghi n luồng dự đoán của một ngày qua PredictionLogger; trả về (request_ids, nhãn thật).

    Dùng chung một logger cho mỗi thư mục, như một tiến trình API (logger mới đặt lại số thứ tự file)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2.0, 100.0, (n, len(RAW))), columns=RAW)
    labels = np.array(["Benign"] * n, dtype=object)
    if attack_every:
        labels[::attack_every] = "Bot"
    score = np.where(labels == "Bot", 0.1, 0.0)        # mô hình bỏ sót mọi Bot
    out = pd.DataFrame({"score": score, "prediction": (score >= 0.5).astype(int)})
    ts = (pd.Timestamp(day, tz="UTC") + pd.to_timedelta(start_sec + np.arange(n), unit="s"))
    ids = [f"{day}-{start_sec + i}" for i in range(n)]
    lg = _LOGGERS.setdefault(str(log_dir), PredictionLogger(log_dir, FEATURES, max_rows=10**9, flush_interval_sec=1e9))
    lg.record(X, out, version, 1.0, request_ids=ids, timestamps=ts.strftime("%Y-%m-%dT%H:%M:%SZ").tolist())
    lg.flush()
    return ids, labels


@pytest.fixture()
def env(tmp_path):
    cfg = {"live": {"predictions_dir": str(tmp_path / "pred"), "labels_dir": str(tmp_path / "labels"),
                    "state_dir": str(tmp_path / "state"), "interval_sec": 1, "min_label_frac": 1.0},
           "stream": {"batch_size": 1000}, "retrain": RETRAIN}
    built = []

    def context_fn(version):
        built.append(version)
        return L.ModelContext(StubDetector(), threshold=0.5, baseline_recall=0.99)

    return cfg, context_fn, built, tmp_path


def test_batches_wait_for_full_size_or_next_day_and_resume_from_state(env):
    cfg, context_fn, built, tmp = env
    write_log(tmp / "pred", "2018-02-22", 0, 1500)
    mon = L.LiveMonitor(cfg, FEATURES, context_fn)
    rows = mon.tick()
    assert [r["rows"] for r in rows] == [1000] and len(mon.buffer) == 500      # 500 dòng chờ thêm dữ liệu

    write_log(tmp / "pred", "2018-02-22", 1500, 1000, seed=1)
    write_log(tmp / "pred", "2018-02-23", 0, 1200, seed=2)
    rows = L.LiveMonitor(cfg, FEATURES, context_fn).tick()                     # tiến trình mới đọc lại trạng thái
    assert [(r["batch"], r["day"], r["rows"]) for r in rows] == [
        (1, "2018-02-22", 1000), (2, "2018-02-22", 500), (3, "2018-02-23", 1000)]
    assert all(r["time_start"] <= r["time_end"] for r in rows)

    mon = L.LiveMonitor(cfg, FEATURES, context_fn)
    assert mon.tick() == []                                                     # không có file mới
    rows = mon.tick(final=True)
    assert [(r["batch"], r["rows"]) for r in rows] == [(4, 200)]
    assert built == ["1", "1", "1"]                                             # mỗi tiến trình dựng context một lần
    lines = (tmp / "state" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["batch"] for x in lines] == [0, 1, 2, 3, 4]


def test_labels_are_joined_by_request_id_and_trigger_retrain(env):
    cfg, context_fn, _, tmp = env
    ids, labels = write_log(tmp / "pred", "2018-02-22", 0, 2000, attack_every=20)
    write_log(tmp / "pred", "2018-02-23", 0, 10, seed=1)                      # để lô cuối ngày 22 được cắt
    mon = L.LiveMonitor(cfg, FEATURES, context_fn)
    rows = mon.tick()
    assert [r["type"] for r in rows] == ["batch", "batch"] and len(mon.state["pending"]) == 2

    write_label_batch(tmp / "labels", 0, "2018-02-22", ids[:1000], labels[:1000])
    write_label_batch(tmp / "labels", 1, "2018-02-22", ids[1000:1500], labels[1000:1500])   # lô 1 mới có nửa nhãn
    rows = mon.tick()
    assert [(r["type"], r["labeled_batch"]) for r in rows] == [("labels", 0)]
    r = rows[0]
    assert r["labeled_rows"] == 1000 and r["recall"] == 0.0 and r["fpr"] == 0.0
    assert r["retrain_trigger"] and any("recall[Bot]" in x for x in r["retrain_reasons"])
    assert mon.state["cum"]["1"]["Bot"] == [50, 0] and [p["batch"] for p in mon.state["pending"]] == [1]
    assert mon.state["events"]["first_retrain_trigger"]["labeled_batch"] == 0

    write_label_batch(tmp / "labels", 2, "2018-02-22", ids[1500:], labels[1500:])
    rows = L.LiveMonitor(cfg, FEATURES, context_fn).tick()
    assert [r["labeled_batch"] for r in rows] == [1] and rows[0]["cum_recall"] == {"Bot": 0.0}


def test_each_model_version_has_its_own_context_and_cumulative_counts(env):
    cfg, context_fn, built, tmp = env
    ids1, lab1 = write_log(tmp / "pred", "2018-02-22", 0, 1000, version="1", attack_every=10)
    ids2, lab2 = write_log(tmp / "pred", "2018-02-22", 1000, 1000, version="2", attack_every=10, seed=1)
    write_label_batch(tmp / "labels", 0, "2018-02-22", ids1 + ids2, np.concatenate([lab1, lab2]))
    mon = L.LiveMonitor(cfg, FEATURES, context_fn)
    rows = mon.tick(final=True)
    assert [(r["type"], r["model_version"]) for r in rows] == [
        ("batch", "1"), ("labels", "1"), ("batch", "2"), ("labels", "2")]
    assert built == ["1", "2"] and set(mon.state["cum"]) == {"1", "2"}


def test_pusher_keeps_last_labeled_values_and_survives_push_errors():
    sent = []

    def fake_push(url, job, registry, grouping_key, timeout):
        sent.append(generate_latest(registry).decode())
        if len(sent) == 2:
            raise OSError("pushgateway down")

    p = Pusher("http://pg:9091", grouping_key={"scenario": "live"}, push_fn=fake_push, retry_sec=0.0)
    p.push({"batch": 0, "drift_share": 0.4, "dataset_drift": True, "score_drift": 0.1, "alert_rate": 0.01}, "3")
    assert not p.push({"labeled_batch": 0, "fpr": 0.002, "retrain_trigger": True,
                       "cum_recall": {"Bot": 0.0, "Web": 0.25}}, "3")             # lỗi: lần sau vẫn mang giá trị này
    assert p.push({"batch": 1, "drift_share": 0.1, "dataset_drift": False, "score_drift": 0.0, "alert_rate": 0.0}, "3")
    assert p.failures == 1
    text = sent[-1]
    assert 'ids_drift_share{model_version="3"} 0.1' in text and 'ids_dataset_drift{model_version="3"} 0.0' in text
    assert 'ids_retrain_trigger{model_version="3"} 1.0' in text                 # giữ giá trị lần có nhãn gần nhất
    assert 'ids_recall{label="Web",model_version="3"} 0.25' in text
    assert 'ids_monitor_batch{model_version="3"} 1.0' in text

    p.push({"batch": 2, "drift_share": 0.2, "dataset_drift": False, "score_drift": 0.0, "alert_rate": 0.0}, "4")
    assert 'model_version="3"' not in sent[-1] and 'ids_drift_share{model_version="4"} 0.2' in sent[-1]

    down = Pusher("http://pg:9091", push_fn=lambda *a, **k: (_ for _ in ()).throw(OSError("down")), retry_sec=60)
    for i in range(5):
        down.push({"batch": i, "drift_share": 0.0}, "1")
    assert down.failures == 1                                                   # chỉ thử 1 lần trong 60 giây
