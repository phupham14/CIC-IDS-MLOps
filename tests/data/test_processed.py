"""Kiểm tra dữ liệu sau tiền xử lý (chạy sau khi đã chạy src/data/preprocess.py).

pytest tests/data
Mỗi test ứng với một kết luận của nhóm trong notebooks/QUYET_DINH.md (số #).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

PROC = Path("data/processed")
ART = Path("data/preprocessing")
MODEL_SPLITS = ["train", "val", "test"]
ALL = MODEL_SPLITS + ["production_sim"]
META = ["Label", "Label_raw", "y_binary", "y_multi", "meta_day", "meta_time", "meta_key"]
TRAIN_LAST_DAY = "2018-02-21"

pytestmark = pytest.mark.skipif(not all((PROC / f"{n}.parquet").exists() for n in ALL),
                                reason="chưa chạy preprocess.py")


@pytest.fixture(scope="module")
def features():
    return json.loads((ART / "features.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meta():
    """Các cột nhãn/thời gian của cả 4 tệp."""
    return {n: pq.read_table(PROC / f"{n}.parquet", columns=["Label", "Label_raw", "y_binary", "y_multi",
                                                              "meta_day", "meta_time", "meta_key"]).to_pandas()
            for n in ALL}


def test_same_schema(features):
    schemas = [pq.ParquetFile(PROC / f"{n}.parquet").schema_arrow for n in ALL]
    assert all(s.equals(schemas[0]) for s in schemas)
    assert schemas[0].names == features + META


def test_no_leaky_or_dropped_columns(features):                      # #11 và chống rò rỉ
    for bad in ["Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port", "Bwd PSH Flags", "Fwd Pkts/b Avg"]:
        assert bad not in features


def test_no_missing_or_inf_in_features(features):                    # #5
    for n in ALL:
        for b in pq.ParquetFile(PROC / f"{n}.parquet").iter_batches(batch_size=500000, columns=features):
            arr = np.column_stack([b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)])
            assert np.isfinite(arr).all(), f"{n} có NaN/inf"


def test_window_flags_keep_minus_one(features):                       # #6
    assert "fwd_win_missing" in features and "bwd_win_missing" in features
    t = pq.read_table(PROC / "train.parquet",
                      columns=["Init Fwd Win Byts", "Init Bwd Win Byts", "fwd_win_missing", "bwd_win_missing"]).to_pandas()
    assert (t["Init Fwd Win Byts"] == -1).any() and (t["Init Bwd Win Byts"] == -1).any()   # giữ nguyên giá trị -1
    assert (t["fwd_win_missing"] == (t["Init Fwd Win Byts"] == -1)).all()
    assert (t["bwd_win_missing"] == (t["Init Bwd Win Byts"] == -1)).all()


def test_days_per_split(meta):                                         # #2
    for n in MODEL_SPLITS:
        assert meta[n]["meta_day"].max() <= TRAIN_LAST_DAY, f"{n} có ngày sau {TRAIN_LAST_DAY}"
    assert meta["production_sim"]["meta_day"].min() > TRAIN_LAST_DAY


def test_group_split_ratio_and_time_order(meta):                       # #3
    df = pd.concat([m.assign(split=n) for n, m in meta.items() if n in MODEL_SPLITS], ignore_index=True)
    for (day, label), g in df.groupby(["meta_day", "Label"]):
        m = len(g)
        share = g["split"].value_counts(normalize=True)
        if m >= 100:
            assert abs(share.get("train", 0) - 0.8) < 0.02, (day, label)
            assert abs(share.get("val", 0) - 0.1) < 0.02, (day, label)
        tmax = g.groupby("split")["meta_time"].max()
        tmin = g.groupby("split")["meta_time"].min()
        if "val" in tmin:
            assert tmax["train"] <= tmin["val"], (day, label)      # train sớm hơn val
        if "val" in tmax and "test" in tmin:
            assert tmax["val"] <= tmin["test"], (day, label)       # val sớm hơn test


def test_dedup_on_features_and_label(meta):                            # #1
    keys = [np.sort(meta[n]["meta_key"].to_numpy()) for n in ALL]
    for k in keys:
        assert len(np.unique(k)) == len(k), "còn dòng trùng trong một tập"
    for i in range(len(ALL)):
        for j in range(i + 1, len(ALL)):
            assert len(np.intersect1d(keys[i], keys[j])) == 0, f"{ALL[i]} và {ALL[j]} có dòng trùng nhau"


def test_labels_consistent(meta):                                      # #9, #10
    label_map = json.loads((ART / "label_map.json").read_text(encoding="utf-8"))
    for n, t in meta.items():
        assert set(t["Label"]) <= set(label_map)
        assert (t["y_multi"] == t["Label"].map(label_map)).all()
        assert (t["y_binary"] == (t["Label"] != "Benign").astype(int)).all()
        assert not t["Label"].isin(["Brute Force -Web", "Brute Force -XSS", "SQL Injection"]).any()   # đã gộp
    assert {"Brute Force -Web", "Brute Force -XSS", "SQL Injection"} <= set(meta["production_sim"]["Label_raw"])
    assert "Web Attack" in set(meta["production_sim"]["Label"])


def test_every_train_label_in_every_split(meta):
    train_labels = set(meta["train"]["Label"])
    for n in ("val", "test"):
        assert set(meta[n]["Label"]) == train_labels, f"{n} thiếu hoặc thừa nhãn so với train"
