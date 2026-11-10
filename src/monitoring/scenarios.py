"""Kịch bản Drift: tách production_sim thành luồng phát lại và holdout, rồi cắt luồng thành các lô theo thời gian.

- production_sim.parquet được ghi theo thứ tự dòng gốc, nên ở đây sắp lại theo (meta_day, meta_time), sắp ổn định.
- Holdout (MODEL.md, M9): trong từng nhóm (ngày, nhãn), phần cuối theo thời gian (stream.holdout_frac) không bao giờ
  được phát lại hay dùng để tái huấn luyện. Gộp với test.parquet thành holdout có cả tấn công cũ và mới.
- Kịch bản (monitoring.yaml, mục scenarios) lọc luồng theo ngày và nhãn; mỗi lô chỉ thuộc một ngày.

Chạy: python -m src.monitoring.scenarios   -> data/processed/production_holdout.parquet, reports/monitoring/scenarios.json
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.models import data as D
from src.models.train import load_cfg as load_models_cfg, log

META_COLS = ["Label", "y_binary", "meta_day", "meta_time"]


@dataclass
class Batch:
    id: int
    day: str
    data: D.Split

    def __len__(self):
        return len(self.data)


def load_cfg(path="monitoring.yaml", smoke=False):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cfg["models"] = load_models_cfg(cfg["models_config"], smoke)
    if smoke:                                              # dữ liệu nhỏ (preprocess.py --smoke)
        cfg["reports_dir"] = "data/interim/smoke/reports/monitoring"
        cfg["experiment"] += "-smoke"
        cfg["reference"]["rows"] = 2000
        cfg["stream"]["batch_size"] = 2000
    return cfg


def processed_dir(cfg):
    return Path(cfg["models"]["data"]["processed_dir"])


def load_frame(path, features):
    """Đọc file Parquet có nhãn, sắp theo (meta_day, meta_time); đặc trưng là float32."""
    t = pq.read_table(path, columns=features + META_COLS)
    df = pd.DataFrame({c: t.column(c).to_numpy().astype(np.float32) for c in features})
    for c in META_COLS:
        df[c] = t.column(c).to_numpy()
    return df.sort_values(["meta_day", "meta_time"], kind="stable", ignore_index=True)


def split_holdout(df, frac):
    """Phần cuối (theo thời gian) của mỗi nhóm (ngày, nhãn) là holdout; trả về (luồng, holdout), đều giữ thứ tự."""
    g = df.groupby(["meta_day", "Label"], sort=False)
    pos, size = g.cumcount(), g["Label"].transform("size")
    hold = pos >= size - np.round(size * frac).astype(int)
    return df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)


def select(df, scenario):
    """Lọc luồng theo ngày và nhãn của kịch bản (null = không lọc)."""
    keep = np.ones(len(df), dtype=bool)
    if scenario.get("days"):
        keep &= df["meta_day"].isin(scenario["days"]).to_numpy()
    if scenario.get("labels"):
        keep &= df["Label"].isin(scenario["labels"]).to_numpy()
    return df[keep].reset_index(drop=True)


def to_split(df, features):
    return D.Split(df[features].reset_index(drop=True), df["y_binary"].to_numpy(np.int8),
                   df["Label"].to_numpy(dtype=object), df["meta_day"].to_numpy(dtype=object))


def make_batches(df, features, batch_size):
    """Cắt theo từng ngày thành lô batch_size dòng; phần dư nhỏ hơn nửa lô được gộp vào lô trước cùng ngày."""
    batches = []
    for day, g in df.groupby("meta_day", sort=True):
        bounds = list(range(0, len(g), batch_size)) + [len(g)]
        if len(bounds) > 2 and bounds[-1] - bounds[-2] < batch_size // 2:
            del bounds[-2]
        for a, b in zip(bounds[:-1], bounds[1:]):
            batches.append(Batch(len(batches), str(day), to_split(g.iloc[a:b], features)))
    return batches


def reference(cfg, features):
    """Mẫu tham chiếu từ train, cùng cách lấy mẫu Benign như khi huấn luyện, phân tầng theo Label.

    reference.labels (vd [Benign]) giới hạn các nhãn; mặc định giữ mọi nhãn."""
    m, r = cfg["models"]["data"], cfg["reference"]
    labels = r.get("labels")
    split = D.load_split(processed_dir(cfg) / "train.parquet", features, m["benign_frac_by_day"], m["seed"],
                         r["rows"] * 2 if labels else r["rows"])
    if labels:
        split = D.stratified_sample(split.subset(np.flatnonzero(np.isin(split.label, labels))), r["rows"], m["seed"])
    return split


def stream_and_holdout(cfg, features):
    df = load_frame(processed_dir(cfg) / "production_sim.parquet", features)
    return split_holdout(df, cfg["stream"]["holdout_frac"])


def file_batches(cfg, source, features, scenario=None):
    """Các lô từ một file khác trong processed_dir (vd val.parquet, test.parquet), lọc như một kịch bản."""
    df = load_frame(processed_dir(cfg) / source, features)
    return make_batches(select(df, scenario or {}), features, cfg["stream"]["batch_size"])


def calibration_batches(cfg, features):
    c = cfg["calibration"]
    return file_batches(cfg, c["source"], features, {"labels": c.get("labels")})


def scenario_batches(cfg, name, features, stream=None):
    sc = cfg["scenarios"][name]
    if sc.get("source"):
        return file_batches(cfg, sc["source"], features, sc)
    stream = stream if stream is not None else stream_and_holdout(cfg, features)[0]
    return make_batches(select(stream, sc), features, cfg["stream"]["batch_size"])


def describe(df):
    return {"rows": int(len(df)),
            "by_day": {d: {k: int(v) for k, v in g["Label"].value_counts().items()}
                       for d, g in df.groupby("meta_day", sort=True)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="monitoring.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.smoke)
    features = D.load_features(cfg["models"]["data"]["artifacts_dir"])
    stream, holdout = stream_and_holdout(cfg, features)
    out = processed_dir(cfg) / cfg["stream"]["holdout_file"]
    holdout.to_parquet(out, index=False, compression="zstd")
    log(f"Luồng phát lại {len(stream):,} dòng, holdout {len(holdout):,} dòng -> {out}")

    report = {"holdout_frac": cfg["stream"]["holdout_frac"], "batch_size": cfg["stream"]["batch_size"],
              "stream": describe(stream), "holdout": describe(holdout), "scenarios": {}}
    for name, sc in cfg["scenarios"].items():
        batches = scenario_batches(cfg, name, features, stream)
        report["scenarios"][name] = {"description": sc["description"], "source": sc.get("source", "production_sim"),
                                     "batches": len(batches),
                                     "rows": int(sum(len(b) for b in batches)),
                                     "batches_by_day": pd.Series([b.day for b in batches]).value_counts()
                                     .sort_index().astype(int).to_dict()}
        log(f"{name}: {len(batches)} lô, {report['scenarios'][name]['rows']:,} dòng")
    rep = Path(cfg["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "scenarios.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
