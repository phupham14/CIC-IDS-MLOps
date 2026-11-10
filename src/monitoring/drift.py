"""Giám sát mô hình trên các lô của một kịch bản: Data Drift (Evidently) và hiệu năng khi nhãn đến trễ.

Mỗi lô k (monitoring.yaml):
  1. Dự đoán bằng mô hình hiện hành (mặc định @champion).
  2. Evidently so phân phối đặc trưng và điểm dự đoán của lô với reference (mẫu Benign từ train):
     khoảng cách Wasserstein chuẩn hóa; cột drift khi vượt ngưỡng riêng của cột; lô drift khi >= drift.drift_share
     số cột. Ngưỡng từng cột được hiệu chỉnh trước trên các lô Benign của val (không drift): phân vị
     calibration.quantile của khoảng cách. Ngưỡng chung 0.1 làm cả lô từ test cũng bị báo drift.
  3. Nhãn của lô k chỉ đến sau labels.delay_batches lô: khi đó mới tính Precision/Recall/FPR, cộng dồn recall
     từng lớp tấn công và kiểm tra điều kiện tái huấn luyện (retrain). Drift chỉ là cảnh báo, không kích hoạt retrain.
Ở đây chỉ ghi nhận điều kiện retrain, không tái huấn luyện (mô hình giữ nguyên, như Kịch bản 2).

Chạy: python -m src.monitoring.drift --scenario s1_new_attacks
      python -m src.monitoring.drift --scenario s2_benign_shift --model 3
Kết quả: reports/monitoring/<kịch bản>_<phiên bản>.json, .csv, .png và một run trong experiment cic-ids-monitoring.
"""
import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from scipy.stats import wasserstein_distance

from src.models import data as D
from src.models import metrics as M
from src.models import registry as R
from src.models.train import log
from src.monitoring import scenarios as S


class DriftDetector:
    """So phân phối một lô với reference bằng Evidently; thêm khoảng cách từng cột để biết cột nào drift mạnh nhất.

    Mặc định mọi cột dùng chung ngưỡng drift.threshold (và drift.score_threshold cho score). Sau calibrate(),
    mỗi cột có ngưỡng riêng, truyền vào Evidently qua per_column_threshold."""

    def __init__(self, reference_X, reference_scores, cfg):
        self.cfg = cfg
        self.features = list(reference_X.columns)
        self.columns = self.features + ["score"]
        self.definition = DataDefinition(numerical_columns=self.columns)
        ref = reference_X.assign(score=np.asarray(reference_scores, dtype=np.float64))
        self.reference = Dataset.from_pandas(ref, data_definition=self.definition)
        self.ref_values = ref.to_numpy(np.float64)
        self.ref_norm = np.maximum(self.ref_values.std(axis=0), 1e-3)          # chuẩn hóa như Evidently
        self.thresholds = np.array([cfg["threshold"]] * len(self.features) + [cfg["score_threshold"]])
        self.calibrated = False

    def distances(self, X, scores):
        """Khoảng cách Wasserstein chuẩn hóa của từng cột (đặc trưng + score), cùng công thức với Evidently."""
        cur = X.assign(score=np.asarray(scores, dtype=np.float64)).to_numpy(np.float64)
        return np.array([wasserstein_distance(self.ref_values[:, j], cur[:, j])
                         for j in range(cur.shape[1])]) / self.ref_norm

    def calibrate(self, batches, score_fn, quantile, min_threshold):
        """Ngưỡng từng cột = phân vị `quantile` của khoảng cách trên các lô không drift (>= min_threshold).

        Trả về ma trận khoảng cách (lô x cột) để thống kê tỉ lệ báo động giả."""
        dist = np.vstack([self.distances(b.data.X, score_fn(b.data.X)) for b in batches])
        self.thresholds = np.maximum(np.quantile(dist, quantile, axis=0), min_threshold)
        self.calibrated = True
        return dist

    def flags(self, dist):
        """Từ khoảng cách (một lô hoặc ma trận lô x cột): tỉ lệ cột drift, lô drift, score drift."""
        dist = np.atleast_2d(dist)
        over = dist >= self.thresholds
        share = over[:, :-1].mean(axis=1)
        return share, share >= self.cfg["drift_share"], over[:, -1]

    def threshold_map(self):
        return {c: round(float(t), 4) for c, t in zip(self.columns, self.thresholds)}

    def report(self):
        c, thr = self.cfg, self.threshold_map()
        per_col = {k: thr[k] for k in self.features} if self.calibrated else None
        return Report([DriftedColumnsCount(columns=self.features, method=c["method"], threshold=c["threshold"],
                                           per_column_threshold=per_col, drift_share=c["drift_share"]),
                       ValueDrift(column="score", method=c["method"], threshold=float(self.thresholds[-1]))])

    def check(self, X, scores, html_path=None):
        current = Dataset.from_pandas(X.assign(score=np.asarray(scores, dtype=np.float64)),
                                      data_definition=self.definition)
        snap = self.report().run(current, self.reference)
        if html_path:
            snap.save_html(str(html_path))
        values = {m["config"]["type"].rsplit(":", 1)[-1]: m["value"] for m in snap.dict()["metrics"]}
        cnt, score_drift = values["DriftedColumnsCount"], float(values["ValueDrift"])
        dist = self.distances(X, scores)[:-1]
        ratio = dist / self.thresholds[:-1]                                    # > 1 = vượt ngưỡng của cột
        top = np.argsort(-ratio)[:self.cfg["top_columns"]]
        return {"n_drifted": int(cnt["count"]), "drift_share": float(cnt["share"]),
                "dataset_drift": bool(cnt["share"] >= self.cfg["drift_share"]),
                "score_drift": score_drift, "score_drifted": bool(score_drift >= self.thresholds[-1]),
                "top_columns": [[self.features[j], round(float(dist[j]), 4), round(float(self.thresholds[j]), 4)]
                                for j in top]}


def check_retrain(cum, baseline_recall, cfg):
    """Điều kiện tái huấn luyện từ hiệu năng đã có nhãn, cộng dồn từ khi mô hình hiện hành bắt đầu chạy.

    cum: {nhãn: [số mẫu, số dự đoán Attack]}. Trả về danh sách lý do (rỗng = không kích hoạt)."""
    reasons = []
    attacks = {k: v for k, v in cum.items() if k != M.BENIGN and v[0] >= cfg["min_support"]}
    for k, (n, hit) in sorted(attacks.items()):
        if hit / n < cfg["min_class_recall"]:
            reasons.append(f"recall[{k}] = {hit / n:.3f} < {cfg['min_class_recall']} ({n} mẫu)")
    n_att = sum(v[0] for v in attacks.values())
    if n_att:
        recall = sum(v[1] for v in attacks.values()) / n_att
        if recall < baseline_recall - cfg["max_recall_drop"]:
            reasons.append(f"recall tổng thể = {recall:.3f} < mốc {baseline_recall:.3f} - {cfg['max_recall_drop']}")
    return reasons


def monitor(batches, score_fn, threshold, detector, cfg, baseline_recall, html_dir=None):
    """Chạy qua các lô theo thứ tự; trả về một dòng cho mỗi lô và các mốc sự kiện (cảnh báo drift, kích hoạt retrain)."""
    delay = cfg["labels"]["delay_batches"]
    rows, pending, cum = [], [], defaultdict(lambda: [0, 0])
    events = {"first_drift_alert": None, "first_retrain_trigger": None}
    seen_days = set()
    for step, b in enumerate(batches):
        t0 = time.time()
        s = np.asarray(score_fn(b.data.X), dtype=np.float64)
        pred = (s >= threshold).astype(np.int8)
        html = None
        if html_dir is not None and b.day not in seen_days:              # lưu báo cáo Evidently lô đầu mỗi ngày
            html = Path(html_dir) / f"batch{b.id:04d}_{b.day}.html"
        seen_days.add(b.day)
        drift = detector.check(b.data.X, s, html)
        row = {"batch": b.id, "day": b.day, "rows": len(b), "n_attack": int(b.data.y.sum()),
               "alert_rate": float(pred.mean()), **drift}
        pending.append((b, s))

        # nhãn của lô (step - delay) đến sau khi xử lý xong lô hiện tại
        if len(pending) > delay:
            lb, ls = pending.pop(0)
            res = M.evaluate(lb.data.y, ls, lb.data.label, threshold)
            o = res["overall"]
            for lab, v in res["per_label"].items():
                cum[lab][0] += v["support"]
                cum[lab][1] += v["predicted_attack"]
            reasons = check_retrain(cum, baseline_recall, cfg["retrain"])
            row.update(labeled_batch=lb.id, recall=o["recall"], precision=o["precision"], fpr=o["fpr"], f1=o["f1"],
                       per_label={k: {"support": v["support"], "rate": v["rate"]} for k, v in res["per_label"].items()},
                       retrain_trigger=bool(reasons), retrain_reasons=reasons)
            if reasons and events["first_retrain_trigger"] is None:
                events["first_retrain_trigger"] = {"step": step, "labeled_batch": lb.id, "day": lb.day,
                                                   "reasons": reasons}
        if drift["dataset_drift"] and events["first_drift_alert"] is None:
            events["first_drift_alert"] = {"step": step, "batch": b.id, "day": b.day,
                                           "drift_share": drift["drift_share"]}
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        log(f"lô {b.id:3d} {b.day} | drift {drift['drift_share']:.2f}{' !' if drift['dataset_drift'] else '  '} "
            f"score {drift['score_drift']:.3f} | cảnh báo {row['alert_rate']:.4f}"
            + (f" | nhãn lô {row['labeled_batch']}: recall {row['recall']:.3f} fpr {row['fpr']:.5f}"
               if "labeled_batch" in row else "")
            + (" | RETRAIN" if row.get("retrain_trigger") else ""))
    cum_recall = {k: {"support": n, "rate": hit / n} for k, (n, hit) in sorted(cum.items()) if n}
    return rows, events, cum_recall


def describe_labels(labels):
    return ", ".join(f"{k} {v:,}" for k, v in pd.Series(labels).value_counts().items())


def summarize(rows):
    """Tóm tắt theo ngày: số lô, số lô drift, tỉ lệ cột drift TB, score drift TB, tỉ lệ cảnh báo TB."""
    df = pd.DataFrame(rows)
    g = df.groupby("day")
    by_day = pd.DataFrame({"batches": g.size(), "drift_batches": g["dataset_drift"].sum(),
                           "drift_share_mean": g["drift_share"].mean(), "score_drift_batches": g["score_drifted"].sum(),
                           "alert_rate_mean": g["alert_rate"].mean()})
    return {"batches": len(df), "drift_batches": int(df["dataset_drift"].sum()),
            "score_drift_batches": int(df["score_drifted"].sum()),
            "by_day": {d: {k: (int(v) if k.endswith("batches") else round(float(v), 5)) for k, v in r.items()}
                       for d, r in by_day.iterrows()}}


def plot(rows, drift_share, score_threshold, title, path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(df["batch"], df["drift_share"], marker=".", label="Tỉ lệ cột drift")
    ax[0].plot(df["batch"], df["score_drift"], marker=".", label="Drift điểm dự đoán")
    ax[0].axhline(drift_share, ls="--", c="C0", lw=1, label="Ngưỡng lô drift")
    ax[0].axhline(score_threshold, ls=":", c="C1", lw=1, label="Ngưỡng drift điểm")
    ax[0].set(ylabel="Drift", title=title)
    if "labeled_batch" in df:                                          # hiệu năng vẽ tại lô mà nhãn thuộc về
        lab = df.dropna(subset=["labeled_batch"])
        ax[1].plot(lab["labeled_batch"], lab["recall"], marker=".", label="Recall (khi nhãn đến)")
        ax[1].plot(lab["labeled_batch"], lab["fpr"] * 100, marker=".", label="FPR (%)")
        trig = lab[lab["retrain_trigger"] == True]                     # noqa: E712 (cột có NaN)
        if len(trig):
            ax[1].axvline(trig["batch"].iloc[0], c="red", lw=1.5, label="Kích hoạt retrain (lần đầu)")
    ax[1].plot(df["batch"], df["alert_rate"], marker=".", label="Tỉ lệ cảnh báo Attack")
    ax[1].set(ylim=(-0.03, 1.03), xlabel="Lô", ylabel="Hiệu năng")
    for a in ax:
        for x in df.groupby("day")["batch"].min().iloc[1:]:
            a.axvline(x - 0.5, c="grey", lw=0.6)
        a.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    for day, x in df.groupby("day")["batch"].min().items():
        ax[0].text(x, 1.02, day[5:], transform=ax[0].get_xaxis_transform(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="monitoring.yaml")
    ap.add_argument("--scenario", required=True, help="tên trong scenarios của monitoring.yaml")
    ap.add_argument("--model", default="champion", help="số phiên bản hoặc alias trong Model Registry")
    ap.add_argument("--no-calibrate", action="store_true", help="dùng ngưỡng chung drift.threshold cho mọi cột")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = S.load_cfg(args.config, args.smoke)
    mcfg = cfg["models"]
    R.setup(mcfg, cfg["experiment"])
    name = mcfg["registry"]["model_name"]
    mv = R.get_version(name, args.model)
    assert mv is not None, f"Không tìm thấy '{args.model}' của {name}"
    model = mlflow.pyfunc.load_model(R.model_uri(name, mv.version))
    thr = float(model.metadata.metadata["threshold"])
    score_fn = lambda X: model.predict(X)["score"].to_numpy()               # noqa: E731
    features = D.load_features(mcfg["data"]["artifacts_dir"])

    log(f"Mô hình {name} v{mv.version} ({model.metadata.metadata['model_type']}), ngưỡng {thr:.4f}")
    ref = S.reference(cfg, features)
    y_te, s_te, lab_te, _ = D.score_file(S.processed_dir(cfg) / "test.parquet", features, score_fn)
    baseline = M.evaluate(y_te, s_te, lab_te, thr)
    log(f"Reference {len(ref):,} dòng ({describe_labels(ref.label)}); "
        f"mốc recall trên test {baseline['overall']['recall']:.4f}")
    detector = DriftDetector(ref.X, score_fn(ref.X), cfg["drift"])
    calib = {"enabled": not args.no_calibrate}
    if not args.no_calibrate:
        c = cfg["calibration"]
        cb = S.calibration_batches(cfg, features)
        dist = detector.calibrate(cb, score_fn, c["quantile"], c["min_threshold"])
        share, drifted, score_drifted = detector.flags(dist)
        alert = [float((score_fn(b.data.X) >= thr).mean()) for b in cb]
        calib.update(source=c["source"], labels=c.get("labels"), quantile=c["quantile"], batches=len(cb),
                     drift_share_mean=float(share.mean()), drift_share_max=float(share.max()),
                     dataset_drift_batches=int(drifted.sum()), score_drift_batches=int(score_drifted.sum()),
                     alert_rate_mean=float(np.mean(alert)), alert_rate_max=float(np.max(alert)),
                     thresholds=detector.threshold_map())
        log(f"Hiệu chỉnh trên {len(cb)} lô {c['source']}: tỉ lệ cột drift TB {share.mean():.3f} (max {share.max():.3f}), "
            f"ngưỡng score {detector.thresholds[-1]:.4f}, tỉ lệ cảnh báo TB {np.mean(alert):.5f}")
    batches = S.scenario_batches(cfg, args.scenario, features)
    log(f"{args.scenario}: {len(batches)} lô")

    tag = f"{args.scenario}_v{mv.version}" + ("_uncalibrated" if args.no_calibrate else "")
    out = Path(cfg["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=tag), tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({"scenario": args.scenario, "model_version": mv.version, "stage": "monitoring"})
        mlflow.log_params({"model": f"{name}/{mv.version}", "threshold": thr, "batches": len(batches),
                           "reference_rows": len(ref), "reference_labels": cfg["reference"].get("labels"),
                           "calibrated": calib["enabled"], "batch_size": cfg["stream"]["batch_size"],
                           "label_delay_batches": cfg["labels"]["delay_batches"],
                           **{f"drift_{k}": v for k, v in cfg["drift"].items()},
                           **{f"retrain_{k}": v for k, v in cfg["retrain"].items()}})
        rows, events, cum = monitor(batches, score_fn, thr, detector, cfg, baseline["overall"]["recall"], tmp)
        for r in rows:
            mlflow.log_metrics({k: r[k] for k in ("drift_share", "score_drift", "alert_rate") if k in r}
                               | {k: r[k] for k in ("recall", "fpr") if k in r and np.isfinite(r[k])},
                               step=r["batch"])
        report = {"scenario": args.scenario, "description": cfg["scenarios"][args.scenario]["description"],
                  "model": {"name": name, "version": str(mv.version), "threshold": thr,
                            "model_type": model.metadata.metadata["model_type"]},
                  "baseline_test": baseline, "calibration": calib, "events": events,
                  "summary": summarize(rows), "cumulative_by_label": cum,
                  "config": {k: cfg[k] for k in ("reference", "calibration", "stream", "drift", "labels", "retrain")},
                  "batches": rows}
        (out / f"{tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float),
                                         encoding="utf-8")
        flat = pd.DataFrame(rows).drop(columns=["per_label", "top_columns", "retrain_reasons"], errors="ignore")
        flat.to_csv(out / f"{tag}.csv", index=False)
        plot(rows, cfg["drift"]["drift_share"], float(detector.thresholds[-1]),
             f"{name} v{mv.version} – {args.scenario}" + ("" if calib["enabled"] else " (chưa hiệu chỉnh)"),
             out / f"{tag}.png")
        for f in (f"{tag}.json", f"{tag}.csv", f"{tag}.png"):
            mlflow.log_artifact(str(out / f))
        mlflow.log_artifacts(tmp, "evidently")
        mlflow.set_tags({k: json.dumps(v, ensure_ascii=False, default=str) for k, v in events.items()})
    log(f"Xong. Cảnh báo drift đầu tiên: {events['first_drift_alert']}; "
        f"kích hoạt retrain đầu tiên: {events['first_retrain_trigger']}. Kết quả: {out / tag}.*")


if __name__ == "__main__":
    main()
