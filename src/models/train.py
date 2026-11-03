"""Huấn luyện LR, Linear SVM, Random Forest, XGBoost (Benign / Attack) và log lên MLflow.

Mỗi mô hình: (tùy chọn) tìm siêu tham số trên val, huấn luyện, chọn ngưỡng trên val, đánh giá val/test,
log tham số, chỉ số, biểu đồ và mô hình. --register đăng ký mô hình tốt nhất với alias @challenger.

Chạy (từ thư mục gốc repo):
  python -m src.models.train                        # cả 4 mô hình
  python -m src.models.train --tune --register      # tìm siêu tham số, đăng ký bản tốt nhất
  python -m src.models.train --models xgb --feature-set no_port_proto
  python -m src.models.train --smoke                # dữ liệu nhỏ (preprocess.py --smoke)
"""
import argparse
import json
import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from src.models import data as D
from src.models import estimators as E
from src.models import metrics as M
from src.models import registry as R
from src.models.pyfunc import IDSModel, signature


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path, smoke=False):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    if smoke:                                              # dữ liệu nhỏ, chạy nhanh
        cfg["reports_dir"] = "data/interim/smoke/reports"
        cfg["data"]["processed_dir"] = "data/interim/smoke/processed"
        cfg["data"]["artifacts_dir"] = "data/interim/smoke/artifacts"
        cfg["mlflow"]["experiment"] += "-smoke"
        cfg["mlflow"]["compare_experiment"] += "-smoke"
        cfg["registry"]["model_name"] += "-smoke"
        cfg["tuning"].update(n_trials=2, max_train_rows=20000)
        cfg["models"]["xgb"]["params"]["n_estimators"] = 50
        cfg["models"]["rf"]["params"]["n_estimators"] = 20
    return cfg


def pos_weight(y):
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def tune(name, cfg, train, val):
    """Random search trên tập con của train, chấm trên val; mỗi lần thử là một run con trên MLflow."""
    t, seed = cfg["tuning"], cfg["data"]["seed"]
    space = cfg["models"][name]["search"]
    base = cfg["models"][name]["params"]
    sub = D.stratified_sample(train, t["max_train_rows"], seed)
    n_grid = int(np.prod([len(v) for v in space.values()]))
    rows = []
    for i, cand in enumerate(ParameterSampler(space, n_iter=min(t["n_trials"], n_grid), random_state=seed)):
        params = {**base, **cand}
        with mlflow.start_run(run_name=f"{name}-trial-{i}", nested=True):
            mlflow.set_tags({"model_type": name, "stage": "tuning"})
            t0 = time.time()
            model = E.fit(name, E.build(name, params, pos_weight(sub.y), seed), sub, val)
            fit_time = time.time() - t0
            s = E.scores(model, val.X)
            m = M.binary_metrics(val.y, s, M.select_threshold(val.y, s, **cfg["threshold"]))
            row = {**cand, "val_pr_auc": float(average_precision_score(val.y, s)), "val_errors": m["fp"] + m["fn"],
                   "val_fpr": m["fpr"], "val_recall": m["recall"], "fit_time_s": fit_time,
                   "best_iteration": E.best_iteration(name, model)}
            mlflow.log_params({**params, "train_rows": len(sub)})
            mlflow.log_metrics({k: v for k, v in row.items() if k not in cand and v is not None})
        rows.append(row)
        log(f"  {name} thử {i}: {cand} -> val PR-AUC {row['val_pr_auc']:.6f}, lỗi {row['val_errors']:,} "
            f"({row['fit_time_s']:.0f}s)")
    # PR-AUC gần bằng nhau (chênh <= tie_tol) thì chọn lần ít lỗi (FP + FN) hơn, rồi nhanh hơn
    key = f"val_{t['metric']}"
    top = max(r[key] for r in rows)
    best = min((r for r in rows if r[key] >= top - t["tie_tol"]), key=lambda r: (r["val_errors"], r["fit_time_s"]))
    return {**base, **{k: best[k] for k in space}}, rows


def log_model(name, model, features, threshold, artifacts_dir, extra_meta):
    """Log mô hình pyfunc kèm features.json, label_map.json."""
    reqs = [f"scikit-learn=={sklearn.__version__}", f"xgboost=={xgboost.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}"]
    with tempfile.TemporaryDirectory() as tmp:
        fpath = Path(tmp) / "features.json"
        fpath.write_text(json.dumps(features, ensure_ascii=False, indent=1), encoding="utf-8")
        ids = IDSModel(model, features, threshold, name)
        example = pd.DataFrame(np.zeros((2, len(features)), dtype=np.float64), columns=features)
        return mlflow.pyfunc.log_model(
            name="model", python_model=ids, code_paths=["src"], signature=signature(features),
            input_example=example, pip_requirements=reqs,
            artifacts={"features": str(fpath), "label_map": str(Path(artifacts_dir) / "label_map.json")},
            metadata={"model_type": name, "threshold": threshold, "n_features": len(features),
                      "target": "y_binary", **extra_meta})


def train_one(name, cfg, paths, features, run_tags, tune_on, parent_run_id):
    """Huấn luyện và đánh giá một mô hình (chạy trong tiến trình con)."""
    mcfg, seed = cfg["models"][name], cfg["data"]["seed"]
    R.setup(cfg)
    val = D.load_split(paths["val"], features)
    tr = D.load_split(paths["train"], features, cfg["data"]["benign_frac_by_day"], seed, mcfg.get("max_train_rows"))
    with mlflow.start_run(run_name=f"{name}-{run_tags['feature_set']}", parent_run_id=parent_run_id) as run:
        mlflow.set_tags({**run_tags, "model_type": name, "stage": "train"})
        params, trials = (tune(name, cfg, tr, val) if tune_on and mcfg.get("search")
                          else (dict(mcfg["params"]), None))
        if trials:
            mlflow.log_table(pd.DataFrame(trials), "tuning/trials.json")
        spw = pos_weight(tr.y)
        mlflow.log_params({**params, "train_rows": len(tr), "n_features": len(features), "scale_pos_weight": spw,
                           "class_weight": "scale_pos_weight" if name == "xgb" else "balanced",
                           "benign_frac_by_day": json.dumps(cfg["data"]["benign_frac_by_day"]),
                           "tuned": bool(trials)})
        mlflow.log_dict(D.describe(tr), "data/train_used.json")

        log(f"{name}: huấn luyện trên {len(tr):,} dòng, {len(features)} đặc trưng")
        t0 = time.time()
        model = E.fit(name, E.build(name, params, spw, seed), tr, val)
        fit_time = time.time() - t0
        del tr

        s_val = E.scores(model, val.X)
        thr = M.select_threshold(val.y, s_val, **cfg["threshold"])
        res_val = M.evaluate(val.y, s_val, val.label, thr)
        y_te, s_te, lab_te, _ = D.score_file(paths["test"], features, lambda X: E.scores(model, X))
        res_test = M.evaluate(y_te, s_te, lab_te, thr)
        lat = M.latency_ms(lambda X: E.scores(model, X), val.X)

        metrics = {**M.flat(res_val, "val"), **M.flat(res_test, "test"), **lat,
                   "fit_time_s": fit_time, "threshold": thr}
        if E.best_iteration(name, model) is not None:
            metrics["best_iteration"] = E.best_iteration(name, model)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(res_val, "eval/val.json")
        mlflow.log_dict(res_test, "eval/test.json")
        mlflow.log_dict(E.importances(name, model, features), "eval/feature_importance.json")
        with tempfile.TemporaryDirectory() as tmp:
            for split, (y, s) in {"val": (val.y, s_val), "test": (y_te, s_te)}.items():
                p = Path(tmp) / f"{split}_curves.png"
                M.plot_eval(y, s, thr, f"{name} – {split}", p)
                mlflow.log_artifact(str(p), "eval")
        info = log_model(name, model, features, thr, cfg["data"]["artifacts_dir"],
                         {"feature_set": run_tags["feature_set"], "train_run_id": run.info.run_id})
        o, t = res_val["overall"], res_test["overall"]
        log(f"{name}: ngưỡng {thr:.4f} | val PR-AUC {o['pr_auc']:.5f} F1 {o['f1']:.4f} FPR {o['fpr']:.5f} | "
            f"test Recall {t['recall']:.4f} FPR {t['fpr']:.5f} ({fit_time:.0f}s)")
        return {"run_id": run.info.run_id, "model_uri": info.model_uri, "params": params, "threshold": thr,
                "fit_time_s": fit_time, **lat, "val": res_val, "test": res_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--models", default=",".join(E.MODEL_TYPES), help="danh sách, ví dụ lr,svm,xgb")
    ap.add_argument("--feature-set", default="all", help="tên trong feature_sets của models.yaml")
    ap.add_argument("--tune", action="store_true", help="tìm siêu tham số trên val trước khi huấn luyện")
    ap.add_argument("--register", action="store_true", help="đăng ký mô hình tốt nhất với alias @challenger")
    ap.add_argument("--smoke", action="store_true", help="chạy thử trên dữ liệu nhỏ (data/interim/smoke)")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.smoke)
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    assert set(names) <= set(E.MODEL_TYPES), f"Mô hình hợp lệ: {E.MODEL_TYPES}"
    d = cfg["data"]
    proc, art = Path(d["processed_dir"]), Path(d["artifacts_dir"])
    features = D.load_features(art, exclude=cfg["feature_sets"][args.feature_set])
    paths = {s: proc / f"{s}.parquet" for s in ("train", "val", "test")}
    R.setup(cfg)

    t0 = time.time()
    log(f"Dữ liệu: {proc}; lấy mẫu Benign khi huấn luyện: {d['benign_frac_by_day']}")

    session = time.strftime("%Y%m%d-%H%M%S")
    tags = {"feature_set": args.feature_set, "session": session, "target": d["target"]}
    results = {}
    with mlflow.start_run(run_name=f"train-{args.feature_set}-{session}") as parent:
        mlflow.set_tags({**tags, "stage": "session"})
        mlflow.log_params({"models": ",".join(names), "feature_set": args.feature_set, "tune": args.tune,
                           "excluded_features": ",".join(cfg["feature_sets"][args.feature_set]) or "-"})
        for f in [args.config, "params.yaml", art / "features.json", art / "label_map.json",
                  "reports/data/split_report.json", "data/raw/MANIFEST.json"]:
            if Path(f).exists():
                mlflow.log_artifact(str(f), "config")
        mlflow.log_dict(D.fingerprint(paths.values()), "data/fingerprint.json")

        # Mỗi mô hình một tiến trình con để RAM được giải phóng sau khi xong
        for name in names:
            with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
                results[name] = ex.submit(train_one, name, cfg, paths, features, tags, args.tune,
                                          parent.info.run_id).result()

        best = max(results, key=lambda n: results[n]["val"]["overall"]["pr_auc"])
        summary = summarize(results, best, args, cfg, session)
        out = Path(cfg["reports_dir"])
        out.mkdir(parents=True, exist_ok=True)
        stem = f"train_{args.feature_set}"
        M.plot_per_label({n: r["test"]["per_label"] for n, r in results.items()},
                         f"Tỉ lệ dự đoán Attack theo nhãn trên test ({args.feature_set})", out / f"{stem}_per_label.png")
        mlflow.log_artifact(str(out / f"{stem}_per_label.png"))
        mlflow.set_tag("best_model", best)

        if args.register:
            reg = cfg["registry"]
            mv = R.register(results[best]["model_uri"], reg["model_name"], reg["challenger_alias"],
                            tags={"model_type": best, "feature_set": args.feature_set, "session": session,
                                  "train_run_id": results[best]["run_id"]})
            summary["registered"] = {"name": reg["model_name"], "version": str(mv.version), "alias": reg["challenger_alias"]}
            mlflow.set_tag("registered_version", mv.version)
            log(f"Đã đăng ký {reg['model_name']} v{mv.version} ({best}) với alias @{reg['challenger_alias']}")
        mlflow.log_dict(summary, "summary.json")
        (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"Xong sau {time.time() - t0:.0f}s. Mô hình tốt nhất (PR-AUC trên val): {best}. "
        f"Xem: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")


def summarize(results, best, args, cfg, session):
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "fpr"]
    return {
        "session": session, "feature_set": args.feature_set, "tuned": args.tune, "best_model": best,
        "selection_metric": "val pr_auc", "benign_frac_by_day": cfg["data"]["benign_frac_by_day"],
        "threshold_rule": cfg["threshold"],
        "models": {n: {"run_id": r["run_id"], "params": r["params"], "threshold": r["threshold"],
                       "fit_time_s": round(r["fit_time_s"], 1),
                       "latency_ms_1000rows": round(r["latency_ms_1000rows"], 2),
                       "val": {k: r["val"]["overall"][k] for k in keys},
                       "test": {k: r["test"]["overall"][k] for k in keys},
                       "test_per_label": {k: {"support": v["support"], "rate": v["rate"]}
                                          for k, v in r["test"]["per_label"].items()}}
                   for n, r in results.items()},
    }


if __name__ == "__main__":
    main()
