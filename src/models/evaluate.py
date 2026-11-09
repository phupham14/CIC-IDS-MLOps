"""Đánh giá offline một mô hình trên file Parquet có nhãn, tổng thể và theo từng ngày (vd production_sim).

Chạy:
  python -m src.models.evaluate --model champion --data data/processed/production_sim.parquet
  python -m src.models.evaluate --model runs:/<run_id>/model --data data/processed/test.parquet
Kết quả ghi vào reports/models/eval_*.json và .png.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.models import data as D
from src.models import metrics as M
from src.models import registry as R
from src.models.train import load_cfg, log


def by_day(y, s, labels, days, threshold):
    out = {}
    for day in sorted(set(days)):
        m = days == day
        out[day] = M.evaluate(y[m], s[m], labels[m], threshold)
    return out


def plot_by_day(days_res, title, path):
    rates = pd.DataFrame({d: {k: v["rate"] for k, v in r["per_label"].items()} for d, r in days_res.items()}).T
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in rates.columns:
        style = "--" if col == M.BENIGN else "-"
        ax.plot(rates.index, rates[col], style, marker="o", label=f"{col} (FPR)" if col == M.BENIGN else col)
    ax.set(ylim=(-0.03, 1.03), ylabel="Recall (tấn công) / FPR (Benign)", title=title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--model", default="champion",
                    help="số phiên bản hoặc alias trong Model Registry, hoặc URI runs:/<run_id>/model")
    ap.add_argument("--data", required=True, help="file Parquet có nhãn, ví dụ data/processed/production_sim.parquet")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.smoke)
    R.setup(cfg)
    if args.model.startswith("runs:/"):
        model = mlflow.pyfunc.load_model(args.model)
        meta = model.metadata.metadata
        name, tag, info = meta["model_type"], f"{meta['model_type']}_{args.model.split('/')[1][:8]}", {"uri": args.model}
    else:
        name = cfg["registry"]["model_name"]
        mv = R.get_version(name, args.model)
        assert mv is not None, f"Không tìm thấy '{args.model}' của {name}"
        model = mlflow.pyfunc.load_model(R.model_uri(name, mv.version))
        tag, info = f"v{mv.version}", {"name": name, "version": str(mv.version), "tags": mv.tags}
    thr = float(model.metadata.metadata["threshold"])
    features = D.load_features(cfg["data"]["artifacts_dir"])
    path = Path(args.data)

    log(f"Chấm {name} {tag} ({model.metadata.metadata['model_type']}) trên {path}")
    y, s, labels, days = D.score_file(path, features, lambda X: model.predict(X)["score"].to_numpy(), 200_000)
    report = {"model": {**info, "model_type": model.metadata.metadata["model_type"], "threshold": thr},
              "data": str(path), "overall": M.evaluate(y, s, labels, thr), "by_day": by_day(y, s, labels, days, thr)}

    out = Path(cfg["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"eval_{path.stem}_{tag}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    plot_by_day(report["by_day"], f"{name} {tag} trên {path.stem}: tỉ lệ dự đoán Attack theo ngày",
                stem.with_suffix(".png"))

    o = report["overall"]["overall"]
    print(f"\nTổng thể: Recall {o['recall']:.4f}  Precision {o['precision']:.4f}  FPR {o['fpr']:.5f}  "
          f"PR-AUC {o['pr_auc']:.4f}")
    print(f"{'Ngày':12s} {'Nhãn':28s} {'Số mẫu':>10s} {'Tỉ lệ dự đoán Attack':>22s}")
    for day, r in report["by_day"].items():
        for lab, v in sorted(r["per_label"].items()):
            print(f"{day:12s} {lab:28s} {v['support']:>10,d} {v['rate']:>22.4f}  ({v['metric']})")
    log(f"Đã ghi {stem}.json, {stem}.png")


if __name__ == "__main__":
    main()
