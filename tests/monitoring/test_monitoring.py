"""Test src/monitoring trên dữ liệu tổng hợp cùng schema với data/processed. Chạy: python -m pytest tests/monitoring"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import data as D
from src.monitoring import drift as DR
from src.monitoring import scenarios as S

FEATURES = json.loads(Path("data/preprocessing/features.json").read_text(encoding="utf-8"))
DAYS = ["2018-02-22", "2018-02-28", "2018-03-02"]
DRIFT_CFG = {"method": "wasserstein", "threshold": 0.1, "drift_share": 0.3, "score_threshold": 0.1, "top_columns": 3}
RETRAIN_CFG = {"min_support": 30, "min_class_recall": 0.5, "max_recall_drop": 0.1}


def make_production(n_per_day=3000, seed=0):
    """Mỗi ngày: Benign + một loại tấn công riêng; dòng bị xáo trộn (như production_sim gốc)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, day in enumerate(DAYS):
        labels = rng.choice(["Benign"] * 9 + [["Web Attack", "Infiltration", "Bot"][i]], size=n_per_day)
        df = pd.DataFrame(rng.gamma(2.0, 100.0, size=(n_per_day, len(FEATURES))).astype(np.float32), columns=FEATURES)
        df["Label"] = labels
        df["y_binary"] = (labels != "Benign").astype(np.int8)
        df["meta_day"] = day
        df["meta_time"] = pd.Timestamp(day) + pd.to_timedelta(rng.integers(8 * 3600, 17 * 3600, n_per_day), unit="s")
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def production(tmp_path_factory):
    path = tmp_path_factory.mktemp("mon") / "production_sim.parquet"
    make_production().to_parquet(path, index=False)
    return S.load_frame(path, FEATURES)


def test_load_frame_sorts_by_day_then_time(production):
    assert list(production.columns) == FEATURES + S.META_COLS
    assert production[FEATURES].dtypes.eq(np.float32).all()
    key = production["meta_day"].astype(str) + production["meta_time"].astype(str)
    assert key.is_monotonic_increasing


def test_holdout_is_the_latest_part_of_every_day_label_group(production):
    stream, hold = S.split_holdout(production, 0.2)
    assert len(stream) + len(hold) == len(production)
    for (day, lab), g in production.groupby(["meta_day", "Label"]):
        st = stream[(stream.meta_day == day) & (stream.Label == lab)]
        ho = hold[(hold.meta_day == day) & (hold.Label == lab)]
        assert len(ho) == round(len(g) * 0.2)
        assert st["meta_time"].max() <= ho["meta_time"].min()            # holdout nằm sau luồng phát lại


def test_batches_stay_within_one_day_and_cover_the_stream(production):
    stream, _ = S.split_holdout(production, 0.2)
    batches = S.make_batches(stream, FEATURES, batch_size=1000)
    assert sum(len(b) for b in batches) == len(stream)
    assert [b.id for b in batches] == list(range(len(batches)))
    assert [b.day for b in batches] == sorted(b.day for b in batches)
    for b in batches:
        assert set(b.data.day) == {b.day} and 500 <= len(b) < 1500
        assert list(b.data.X.columns) == FEATURES


def test_benign_scenario_keeps_only_benign(production):
    sc = {"days": None, "labels": ["Benign"]}
    batches = S.make_batches(S.select(production, sc), FEATURES, 1000)
    assert all(set(b.data.label) == {"Benign"} for b in batches)
    assert {b.day for b in batches} == set(DAYS)


def test_detector_flags_shifted_columns_but_not_same_distribution():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(rng.gamma(2.0, 100.0, (4000, len(FEATURES))).astype(np.float32), columns=FEATURES)
    det = DR.DriftDetector(ref, rng.beta(1, 9, 4000), DRIFT_CFG)
    same = pd.DataFrame(rng.gamma(2.0, 100.0, (2000, len(FEATURES))).astype(np.float32), columns=FEATURES)
    r = det.check(same, rng.beta(1, 9, 2000))
    assert not r["dataset_drift"] and not r["score_drifted"] and r["drift_share"] < 0.1

    shifted = same.copy()
    cols = FEATURES[: len(FEATURES) // 2]
    shifted[cols] *= 3
    r = det.check(shifted, rng.beta(5, 2, 2000))
    assert r["dataset_drift"] and r["score_drifted"] and r["drift_share"] >= 0.45
    assert all(c in cols for c, _, _ in r["top_columns"])


def test_calibration_raises_thresholds_of_noisy_columns_only():
    rng = np.random.default_rng(2)
    n = len(FEATURES)
    noisy = FEATURES[:5]

    def frame(size, day_shift=0.0):
        df = pd.DataFrame(rng.gamma(2.0, 100.0, (size, n)).astype(np.float32), columns=FEATURES)
        df[noisy] *= 1 + day_shift                                    # 5 cột dao động mạnh giữa các "ngày"
        return df

    det = DR.DriftDetector(frame(4000), rng.beta(1, 9, 4000), DRIFT_CFG)
    calib = [S.Batch(i, "d", D.Split(frame(2000, s), None, None, None)) for i, s in enumerate([-0.3, 0.3, 0.2, -0.2])]
    score_fn = lambda X: rng.beta(1, 9, len(X))                         # noqa: E731
    dist = det.calibrate(calib, score_fn, quantile=1.0, min_threshold=0.1)
    assert dist.shape == (4, n + 1) and det.calibrated
    thr = det.threshold_map()
    assert all(thr[c] > 0.2 for c in noisy)                            # cột dao động: ngưỡng cao hơn
    assert sum(v == 0.1 for v in thr.values()) >= n - 5                # cột ổn định: giữ ngưỡng sàn
    share, drifted, _ = det.flags(dist)
    assert not drifted.any() and share.max() < 0.1                     # lô hiệu chỉnh không bị coi là drift

    r = det.check(frame(2000, 0.15), score_fn(np.zeros(2000)))           # dao động như hiệu chỉnh: Evidently không báo
    assert r["n_drifted"] == 0 and not r["dataset_drift"]
    shifted = frame(2000)
    shifted[FEATURES[10:40]] *= 3                                      # drift thật trên 30 cột
    r = det.check(shifted, score_fn(np.zeros(2000)))
    assert r["dataset_drift"] and r["n_drifted"] >= 30
    assert all(c in FEATURES[10:40] for c, _, _ in r["top_columns"])


def test_reference_keeps_only_configured_labels(tmp_path):
    df = make_production(1000)
    df["meta_day"], df["meta_time"] = "2018-02-14", 0
    df.to_parquet(tmp_path / "train.parquet", index=False)
    cfg = {"models": {"data": {"processed_dir": str(tmp_path), "benign_frac_by_day": {}, "seed": 0}},
           "reference": {"rows": 1500, "labels": ["Benign"]}}
    ref = S.reference(cfg, FEATURES)
    assert set(ref.label) == {"Benign"} and abs(len(ref) - 1500) <= 2


def test_check_retrain_needs_enough_labeled_samples():
    assert DR.check_retrain({"Benign": [1000, 1], "Bot": [10, 0]}, 0.99, RETRAIN_CFG) == []      # quá ít mẫu Bot
    reasons = DR.check_retrain({"Benign": [1000, 1], "Bot": [40, 2]}, 0.99, RETRAIN_CFG)
    assert any("recall[Bot]" in r for r in reasons) and any("tổng thể" in r for r in reasons)
    assert DR.check_retrain({"Benign": [1000, 1], "Bot": [40, 39]}, 0.99, RETRAIN_CFG) == []


def test_monitor_uses_delayed_labels_and_records_trigger(production):
    stream, _ = S.split_holdout(production, 0.2)
    batches = S.make_batches(stream, FEATURES, batch_size=1200)
    ref = D.Split(stream[FEATURES].iloc[:2000].reset_index(drop=True), None, None, None)
    score_fn = lambda X: np.full(len(X), 0.05)                                  # noqa: E731  (bỏ sót mọi tấn công)
    det = DR.DriftDetector(ref.X, score_fn(ref.X), DRIFT_CFG)
    cfg = {"labels": {"delay_batches": 1}, "retrain": RETRAIN_CFG}
    rows, events, cum = DR.monitor(batches, score_fn, 0.5, det, cfg, baseline_recall=0.99)
    assert len(rows) == len(batches)
    assert "labeled_batch" not in rows[0]                                       # nhãn lô 0 chưa đến
    assert [r["labeled_batch"] for r in rows[1:]] == [b.id for b in batches[:-1]]
    trig = events["first_retrain_trigger"]
    assert trig is not None and trig["step"] == trig["labeled_batch"] + 1
    assert all(v["rate"] == 0.0 for k, v in cum.items())
