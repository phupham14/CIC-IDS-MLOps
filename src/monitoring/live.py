"""Giám sát trực tiếp từ log của API (INTERFACE.md, mục 3) và đẩy chỉ số lên Pushgateway (mục 4).

Mỗi lần chạy (tick):
  1. Đọc các file mới trong logs/predictions/date=*/ (API ghi), gộp vào bộ đệm, sắp theo timestamp.
  2. Cắt bộ đệm thành lô stream.batch_size dòng, không vượt qua ranh giới ngày. Lô cuối của một ngày chỉ được cắt
     khi đã có dữ liệu ngày sau (hoặc --final); phần chưa đủ một lô ở lại bộ đệm cho lần sau.
  3. Mỗi lô: dùng score/prediction đã ghi trong log (không chấm lại), kiểm tra drift bằng DriftDetector của đúng
     phiên bản mô hình đã dự đoán (model_version), đẩy chỉ số lên Pushgateway.
  4. Đọc nhãn mới trong logs/labels/date=*/, ghép theo request_id với các lô đang chờ nhãn. Lô đủ nhãn
     (live.min_label_frac) thì tính Recall/FPR, cộng dồn theo phiên bản mô hình và kiểm tra điều kiện retrain.

Trạng thái (file đã đọc, bộ đệm, lô chờ nhãn, số liệu cộng dồn) nằm trong live.state_dir, nên chạy lại sẽ tiếp tục
từ chỗ dừng; chạy một lần (Airflow gọi định kỳ) hoặc --follow (vòng lặp mỗi live.interval_sec giây).

Chạy: python -m src.monitoring.live --push-url http://localhost:9091 --follow
      python -m src.monitoring.live --final          # xử lý nốt phần còn trong bộ đệm rồi dừng
Kết quả: <state_dir>/rows.jsonl (một dòng cho mỗi lô và mỗi lần nhãn đến), <state_dir>/state.json.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import data as D
from src.models import metrics as M
from src.models import registry as R
from src.models.train import log
from src.monitoring import drift as DR
from src.monitoring import scenarios as S

LOG_COLS = ["request_id", "timestamp", "model_version", "score", "prediction"]


class ModelContext:
    """Những gì cần cho một phiên bản mô hình: detector (reference + hiệu chỉnh), ngưỡng, mốc recall trên test."""

    def __init__(self, detector, threshold, baseline_recall):
        self.detector, self.threshold, self.baseline_recall = detector, float(threshold), float(baseline_recall)


def registry_context(cfg, features):
    """Tạo ModelContext từ Model Registry (chậm: nạp mô hình, chấm test, hiệu chỉnh trên val)."""
    def build(version):
        model = DR.load_model(cfg, version)
        log(f"Chuẩn bị giám sát cho {model.name} v{model.version} ({model.model_type}), ngưỡng {model.threshold:.4f}")
        detector, _ = DR.build_detector(cfg, features, model)
        baseline = DR.baseline_on_test(cfg, features, model)["overall"]["recall"]
        return ModelContext(detector, model.threshold, baseline)
    return build


class LiveMonitor:
    def __init__(self, cfg, features, context_fn, pusher=None):
        live = cfg["live"]
        self.cfg, self.features, self.context_fn, self.pusher = cfg, list(features), context_fn, pusher
        self.pred_dir, self.label_dir = Path(live["predictions_dir"]), Path(live["labels_dir"])
        self.dir = Path(live["state_dir"])
        (self.dir / "pending").mkdir(parents=True, exist_ok=True)
        self.batch_size = int(cfg["stream"]["batch_size"])
        self.min_label_frac = float(live.get("min_label_frac", 1.0))
        self.contexts = {}
        self.state = self._load_json("state.json") or {
            "pred_files": [], "label_files": [], "next_batch": 0, "pending": [], "cum": {}, "events": {}}
        self.buffer = self._load_frame("buffer.parquet")
        self.labels = self._load_frame("labels.parquet")

    # ------------------------------------------------------------------ trạng thái
    def _load_json(self, name):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _load_frame(self, name):
        p = self.dir / name
        return pd.read_parquet(p) if p.exists() else None

    def _save(self):
        for name, df in (("buffer.parquet", self.buffer), ("labels.parquet", self.labels)):
            p = self.dir / name
            if df is None or df.empty:
                p.unlink(missing_ok=True)
            else:
                df.to_parquet(p, index=False)
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        tmp.replace(self.dir / "state.json")

    def _emit(self, row):
        with open(self.dir / "rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        if self.pusher is not None:
            self.pusher.push(row, row["model_version"])

    def context(self, version):
        if version not in self.contexts:
            self.contexts[version] = self.context_fn(version)
        return self.contexts[version]

    # ------------------------------------------------------------------ đọc log
    def _new_files(self, root, key):
        done = set(self.state[key])
        files = sorted(p for p in root.glob("date=*/*.parquet") if p.relative_to(root).as_posix() not in done)
        return files

    def read_predictions(self):
        files = self._new_files(self.pred_dir, "pred_files")
        if files:
            cols = LOG_COLS + self.features
            new = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
            new["model_version"] = new["model_version"].astype(str)
            if self.state.get("last_cut_time") and len(new):
                late = int((new["timestamp"] < pd.Timestamp(self.state["last_cut_time"])).sum())
                if late:
                    log(f"{late} dòng có timestamp sớm hơn lô đã cắt (đến muộn), sẽ vào lô tiếp theo")
            self.buffer = pd.concat([d for d in (self.buffer, new) if d is not None], ignore_index=True)
            self.buffer = self.buffer.sort_values("timestamp", kind="stable", ignore_index=True)
            self.state["pred_files"] += [p.relative_to(self.pred_dir).as_posix() for p in files]
        return len(files)

    def read_labels(self):
        files = self._new_files(self.label_dir, "label_files")
        if files:
            new = pd.concat([pd.read_parquet(p, columns=["request_id", "Label"]) for p in files], ignore_index=True)
            self.labels = pd.concat([d for d in (self.labels, new) if d is not None], ignore_index=True)
            self.labels = self.labels.drop_duplicates("request_id", keep="last", ignore_index=True)
            self.state["label_files"] += [p.relative_to(self.label_dir).as_posix() for p in files]
        return len(files)

    # ------------------------------------------------------------------ cắt lô
    def cut_batches(self, final=False):
        """Lấy các lô đã đủ ra khỏi bộ đệm (theo thứ tự thời gian)."""
        out = []
        while self.buffer is not None and len(self.buffer):
            day = self.buffer["timestamp"].dt.strftime("%Y-%m-%d")
            first = day.iloc[0]
            n_day = int((day == first).sum())                 # bộ đệm đã sắp nên các dòng cùng ngày liền nhau
            later_day = n_day < len(self.buffer)
            if n_day >= self.batch_size:
                n = self.batch_size
                if n_day - n < self.batch_size // 2 and (later_day or final):
                    n = n_day                                  # phần dư nhỏ của ngày gộp vào lô này (như make_batches)
            elif later_day or final:
                n = n_day
            else:
                break
            out.append((first, self.buffer.iloc[:n].reset_index(drop=True)))
            self.state["last_cut_time"] = str(self.buffer["timestamp"].iloc[n - 1])
            self.buffer = self.buffer.iloc[n:].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------ xử lý
    def process_batch(self, day, df):
        bid = self.state["next_batch"]
        self.state["next_batch"] += 1
        versions = df["model_version"].value_counts()
        version = str(versions.index[0])                       # lô có thể chứa lúc đổi mô hình: lấy bản chiếm đa số
        ctx = self.context(version)
        X = df[self.features].astype(np.float32)
        s = df["score"].to_numpy(np.float64)
        drift = ctx.detector.check(X, s)
        row = {"type": "batch", "batch": bid, "day": day, "rows": len(df), "model_version": version,
               "model_versions": {str(k): int(v) for k, v in versions.items()},
               "time_start": str(df["timestamp"].iloc[0]), "time_end": str(df["timestamp"].iloc[-1]),
               "alert_rate": float(df["prediction"].mean()), **drift}
        df[["request_id", "score", "prediction"]].to_parquet(self.dir / "pending" / f"batch-{bid:05d}.parquet",
                                                             index=False)
        self.state["pending"].append({"batch": bid, "day": day, "model_version": version})
        ev = self.state["events"]
        if drift["dataset_drift"] and "first_drift_alert" not in ev:
            ev["first_drift_alert"] = {"batch": bid, "day": day, "drift_share": drift["drift_share"]}
        log(f"lô {bid:4d} {day} v{version} | {len(df):,} dòng | drift {drift['drift_share']:.2f}"
            f"{' !' if drift['dataset_drift'] else '  '} score {drift['score_drift']:.3f} | "
            f"cảnh báo {row['alert_rate']:.4f}")
        return row

    def process_labels(self):
        """Đánh giá các lô đang chờ đã đủ nhãn, theo thứ tự lô."""
        rows, still = [], []
        for p in self.state["pending"]:
            path = self.dir / "pending" / f"batch-{p['batch']:05d}.parquet"
            if self.labels is None or self.labels.empty:
                still.append(p)
                continue
            df = pd.read_parquet(path).merge(self.labels, on="request_id", how="left")
            known = df["Label"].notna()
            if known.mean() < self.min_label_frac:
                still.append(p)
                continue
            df = df[known]
            ctx = self.context(p["model_version"])
            cum = defaultdict(lambda: [0, 0], self.state["cum"].get(p["model_version"], {}))
            res = DR.evaluate_labeled((df["Label"] != M.BENIGN).to_numpy(np.int8), df["score"], df["Label"],
                                      ctx.threshold, cum, ctx.baseline_recall, self.cfg["retrain"])
            self.state["cum"][p["model_version"]] = {k: list(v) for k, v in cum.items()}
            row = {"type": "labels", "labeled_batch": p["batch"], "day": p["day"],
                   "model_version": p["model_version"], "labeled_rows": int(len(df)), **res}
            ev = self.state["events"]
            if res["retrain_trigger"] and "first_retrain_trigger" not in ev:
                ev["first_retrain_trigger"] = {"labeled_batch": p["batch"], "day": p["day"],
                                               "model_version": p["model_version"], "reasons": res["retrain_reasons"]}
            self.labels = self.labels[~self.labels["request_id"].isin(df["request_id"])].reset_index(drop=True)
            path.unlink(missing_ok=True)
            log(f"nhãn lô {p['batch']:4d} {p['day']} | recall {res['recall']:.3f} fpr {res['fpr']:.5f}"
                + (f" | RETRAIN: {'; '.join(res['retrain_reasons'])}" if res["retrain_trigger"] else ""))
            rows.append(row)
        self.state["pending"] = still
        return rows

    def tick(self, final=False):
        """Một lượt: đọc log mới, xử lý các lô đủ, rồi đánh giá các lô đủ nhãn. Trả về các dòng đã ghi."""
        n_pred, n_lab = self.read_predictions(), self.read_labels()
        rows = []
        for day, df in self.cut_batches(final):
            rows.append(self.process_batch(day, df))
            self._emit(rows[-1])
            self._save()
            for r in self.process_labels():                  # nhãn có thể đã đến trước khi lô sau được cắt
                rows.append(r)
                self._emit(r)
        for r in self.process_labels():
            rows.append(r)
            self._emit(r)
        self._save()
        if n_pred or n_lab or rows:
            log(f"tick: {n_pred} file dự đoán, {n_lab} file nhãn mới; {sum(r['type'] == 'batch' for r in rows)} lô, "
                f"{sum(r['type'] == 'labels' for r in rows)} lần nhãn đến; bộ đệm còn "
                f"{0 if self.buffer is None else len(self.buffer):,} dòng, {len(self.state['pending'])} lô chờ nhãn")
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="monitoring.yaml")
    ap.add_argument("--push-url", help="Pushgateway, vd http://localhost:9091 (mặc định: push.url / PUSHGATEWAY_URL)")
    ap.add_argument("--follow", action="store_true", help="chạy liên tục, mỗi live.interval_sec giây một lượt")
    ap.add_argument("--final", action="store_true", help="xử lý cả phần chưa đủ một lô còn trong bộ đệm")
    ap.add_argument("--reset", action="store_true", help="xóa trạng thái cũ, đọc lại log từ đầu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = S.load_cfg(args.config, args.smoke)
    R.setup(cfg["models"], cfg["experiment"])
    features = D.load_features(cfg["models"]["data"]["artifacts_dir"])
    state_dir = Path(cfg["live"]["state_dir"])
    if args.reset and state_dir.exists():
        for p in [*state_dir.glob("pending/*.parquet"), *state_dir.glob("*.*")]:
            p.unlink()
        log(f"Đã xóa trạng thái trong {state_dir}")
    pusher = DR.make_pusher(cfg, args.push_url, {"scenario": "live"})
    mon = LiveMonitor(cfg, features, registry_context(cfg, features), pusher)
    log(f"Giám sát {mon.pred_dir} và {mon.label_dir}; trạng thái ở {state_dir} "
        f"(đã xử lý {mon.state['next_batch']} lô, {len(mon.state['pending'])} lô chờ nhãn)")
    while True:
        mon.tick(final=args.final and not args.follow)
        if not args.follow:
            break
        time.sleep(float(cfg["live"]["interval_sec"]))


if __name__ == "__main__":
    main()
