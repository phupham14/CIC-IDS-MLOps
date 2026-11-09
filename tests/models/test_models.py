"""Test src/models trên dữ liệu tổng hợp cùng schema với data/processed. Chạy: python -m pytest tests/models"""
import json
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models import compare as C
from src.models import data as D
from src.models import estimators as E
from src.models import metrics as M
from src.models import train as T
from src.models.pyfunc import IDSModel

ART = Path("data/preprocessing")
FEATURES = json.loads((ART / "features.json").read_text(encoding="utf-8"))
ATTACKS = ["DoS attacks-Hulk", "SSH-BruteForce"]


def make_frame(n, seed, days=("2018-02-14", "2018-02-20")):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Benign"] * 8 + ATTACKS, size=n)
    atk = labels != "Benign"
    X = rng.gamma(2.0, 100.0, size=(n, len(FEATURES))).astype(np.float32)
    X[atk, 2:12] *= rng.uniform(1.5, 4.0, size=(int(atk.sum()), 10))      # Attack lệch ở vài đặc trưng
    df = pd.DataFrame(X, columns=FEATURES)
    for flag, src in [("fwd_win_missing", "Init Fwd Win Byts"), ("bwd_win_missing", "Init Bwd Win Byts")]:
        df[src] = np.where(rng.random(n) < 0.3, -1.0, df[src]).astype(np.float32)
        df[flag] = (df[src] == -1).astype(np.int8)
    df["Label"] = labels
    df["y_binary"] = atk.astype(np.int8)
    df["meta_day"] = rng.choice(days, size=n)
    return df


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("cic")
    (root / "processed").mkdir()
    (root / "art").mkdir()
    for i, (split, n) in enumerate({"train": 6000, "val": 2000, "test": 2000}.items()):
        make_frame(n, seed=i).to_parquet(root / "processed" / f"{split}.parquet", index=False)
    for f in ("features.json", "label_map.json"):
        shutil.copy(ART / f, root / "art" / f)
    return root


def test_load_split_samples_benign_only_on_given_day(processed):
    full = D.load_split(processed / "processed" / "train.parquet", FEATURES)
    sp = D.load_split(processed / "processed" / "train.parquet", FEATURES, {"2018-02-20": 0.0})
    assert list(sp.X.columns) == FEATURES and sp.X.dtypes.eq(np.float32).all()
    assert not ((sp.label == "Benign") & (sp.day == "2018-02-20")).any()
    assert (sp.y == 1).sum() == (full.y == 1).sum()                                    # giữ mọi dòng Attack
    assert ((sp.label == "Benign") & (sp.day == "2018-02-14")).sum() == \
        ((full.label == "Benign") & (full.day == "2018-02-14")).sum()


def test_stratified_sample_keeps_every_label(processed):
    sp = D.load_split(processed / "processed" / "train.parquet", FEATURES)
    sub = D.stratified_sample(sp, 600)
    assert len(sub) == pytest.approx(600, abs=3)
    assert set(sub.label) == set(sp.label)
    loaded = D.load_split(processed / "processed" / "train.parquet", FEATURES, {"2018-02-20": 0.5}, max_rows=600)
    assert len(loaded) == pytest.approx(600, abs=3) and set(loaded.label) == set(sp.label)
    assert not np.isnan(loaded.X.to_numpy()).any()


def test_binary_metrics_and_per_label():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.4, 0.95])
    m = M.binary_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["recall"] == 0.75 and m["fpr"] == 0.25 and m["precision"] == 0.75
    labels = np.array(["Benign"] * 4 + ["Bot", "Bot", "Infiltration", "Infiltration"])
    pl = M.per_label(labels, (s >= 0.5).astype(int))
    assert pl["Benign"] == {"support": 4, "predicted_attack": 1, "rate": 0.25, "metric": "fpr"}
    assert pl["Bot"]["rate"] == 1.0 and pl["Infiltration"]["rate"] == 0.5


def test_metrics_with_single_class_batch():
    m = M.binary_metrics(np.zeros(5, dtype=int), np.array([0.1, 0.2, 0.6, 0.3, 0.4]), 0.5)
    assert m["fpr"] == 0.2 and np.isnan(m["roc_auc"]) and np.isnan(m["recall"])


def test_select_threshold_respects_max_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(5000), np.ones(500)].astype(int)
    s = np.r_[rng.beta(2, 5, 5000), rng.beta(5, 2, 500)]
    thr = M.select_threshold(y, s, max_fpr=0.01)
    assert M.binary_metrics(y, s, thr)["fpr"] <= 0.01
    assert M.select_threshold(y, s, strategy="fixed", fixed=0.3) == 0.3


@pytest.mark.parametrize("name", E.MODEL_TYPES)
def test_every_model_trains_and_scores(processed, name):
    tr = D.load_split(processed / "processed" / "train.parquet", FEATURES)
    va = D.load_split(processed / "processed" / "val.parquet", FEATURES)
    params = {"lr": {}, "svm": {}, "rf": {"n_estimators": 10}, "xgb": {"n_estimators": 20}}[name]
    model = E.fit(name, E.build(name, params, scale_pos_weight=4.0), tr, va)
    s = E.scores(model, va.X)
    assert s.shape == (len(va),) and (s >= 0).all() and (s <= 1).all()
    assert M.binary_metrics(va.y, s, 0.5)["roc_auc"] > 0.8
    assert set(E.importances(name, model, FEATURES)) == set(FEATURES)


def test_pyfunc_reorders_columns_and_derives_window_flags(processed):
    tr = D.load_split(processed / "processed" / "train.parquet", FEATURES)
    model = E.fit("xgb", E.build("xgb", {"n_estimators": 10}, 4.0), tr, tr)
    ids = IDSModel(model, FEATURES, 0.5, "xgb")
    raw = make_frame(50, seed=9)
    expected = ids.predict(None, raw[FEATURES])
    shuffled = raw.drop(columns=["fwd_win_missing", "bwd_win_missing"])[FEATURES[::-1][2:]]   # đảo cột, bỏ cờ
    shuffled["extra"] = 1
    got = ids.predict(None, shuffled)
    assert list(got.columns) == ["score", "prediction", "label"]
    np.testing.assert_allclose(got["score"], expected["score"])
    assert set(got["label"]) <= {"Benign", "Attack"}
    with pytest.raises(ValueError, match="Thiếu"):
        ids.predict(None, raw[FEATURES[:10]])
    bad = raw[FEATURES].copy()
    bad.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/inf"):
        ids.predict(None, bad)


def ev(fpr, rates, pr_auc=0.99, macro_f1=0.98, latency=10.0, support=100):
    pl = {"Benign": {"support": 1000, "rate": fpr}}
    pl.update({k: {"support": support, "rate": v} for k, v in rates.items()})
    return {"overall": {"fpr": fpr, "pr_auc": pr_auc, "macro_f1": macro_f1}, "per_label": pl,
            "latency_ms_1000rows": latency}


PROMO = {"min_support": 30, "max_fpr": 0.01, "max_fpr_increase": 0.002, "max_recall_drop": 0.02,
         "min_recall_gain": 0.01, "max_metric_drop": {"pr_auc": 0.005, "macro_f1": 0.005}, "max_latency_ratio": 2.0}


def test_decide_first_model_only_checks_absolute_fpr():
    assert C.decide(ev(0.005, {"Bot": 0.1}), None, PROMO)["accepted"]
    assert not C.decide(ev(0.02, {"Bot": 0.9}), None, PROMO)["accepted"]


def test_decide_rejects_regressions_and_accepts_improvement():
    champ = ev(0.004, {"DoS": 0.99, "Bot": 0.05})
    assert C.decide(ev(0.004, {"DoS": 0.99, "Bot": 0.90}), champ, PROMO, ("Bot",))["accepted"]
    r = C.decide(ev(0.004, {"DoS": 0.90, "Bot": 0.90}), champ, PROMO, ("Bot",))           # quên lớp cũ
    assert not r["accepted"] and [c["check"] for c in r["checks"] if not c["passed"]] == ["recall[DoS]"]
    assert not C.decide(ev(0.008, {"DoS": 0.99, "Bot": 0.9}), champ, PROMO, ("Bot",))["accepted"]   # FPR tăng
    assert not C.decide(ev(0.004, {"DoS": 0.99, "Bot": 0.055}), champ, PROMO, ("Bot",))["accepted"]  # không cải thiện
    assert not C.decide(ev(0.004, {"DoS": 0.99, "Bot": 0.9}, latency=30), champ, PROMO, ("Bot",))["accepted"]


def test_decide_skips_classes_with_too_few_samples():
    champ = ev(0.004, {"FTP-BruteForce": 1.0}, support=5)
    r = C.decide(ev(0.004, {"FTP-BruteForce": 0.0}, support=5), champ, PROMO)
    assert r["accepted"] and any("FTP-BruteForce" in n for n in r["notes"])


def test_end_to_end_train_register_compare_serve(processed, tmp_path, monkeypatch):
    """train --register -> compare --promote (lần đầu) -> train lại -> compare -> API nạp @champion."""
    cfg = yaml.safe_load(open("models.yaml", encoding="utf-8"))
    cfg["data"].update(processed_dir=str(processed / "processed"), artifacts_dir=str(processed / "art"))
    cfg["mlflow"]["artifact_root"] = str(tmp_path / "artifacts")
    cfg["reports_dir"] = str(tmp_path / "reports")
    cfg["models"]["xgb"]["params"]["n_estimators"] = 20
    cfg["promotion"].update(holdout=str(processed / "processed" / "test.parquet"), max_fpr=0.05)
    cfg_path = tmp_path / "models.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), "--models", "lr,xgb", "--register"])
    T.main()
    first = C.run_comparison(cfg, promote=True)
    assert first["champion"] is None and first["accepted"] and first["promoted"]["champion"] == "1"

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), "--models", "xgb", "--register",
                                      "--feature-set", "no_port_proto"])
    T.main()
    second = C.run_comparison(cfg, promote=True)
    assert second["champion"]["version"] == "1" and second["challenger"]["version"] == "2"
    assert {c["check"] for c in second["checks"]} >= {"fpr", "fpr_increase", "pr_auc", "latency_ms_1000rows"}
    assert (tmp_path / "reports" / "train_all_summary.json").exists()

    served = mlflow.pyfunc.load_model(f"models:/{cfg['registry']['model_name']}@champion")
    req = make_frame(5, seed=11)[FEATURES[:-2]].round()                          # như request từ API: không gửi cờ
    out = served.predict(req.astype("float64"))
    assert list(out.columns) == ["score", "prediction", "label"] and len(out) == 5
    with pytest.raises(mlflow.exceptions.MlflowException, match="int64"):         # API phải đổi sang float (Pydantic float)
        served.predict(req.astype("int64"))
