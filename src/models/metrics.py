"""Chỉ số đánh giá: Precision, Recall, F1, FPR, ROC-AUC, PR-AUC và recall theo từng loại tấn công."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

BENIGN = "Benign"


def binary_metrics(y, s, threshold):
    """y: 0/1, s: điểm Attack. AUC là NaN nếu chỉ có một lớp."""
    pred = (s >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = 0 < y.sum() < len(y)
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def per_label(labels, pred):
    """Tỉ lệ dự đoán Attack theo từng nhãn (lớp tấn công: recall, Benign: FPR)."""
    df = pd.DataFrame({"label": labels, "pred": pred})
    g = df.groupby("label")["pred"].agg(support="size", predicted_attack="sum")
    g["rate"] = g["predicted_attack"] / g["support"]
    g["metric"] = np.where(g.index == BENIGN, "fpr", "recall")
    return {k: {"support": int(r.support), "predicted_attack": int(r.predicted_attack), "rate": float(r.rate),
                "metric": r.metric} for k, r in g.iterrows()}


def select_threshold(y, s, strategy="max_f1_at_fpr", max_fpr=0.01, fixed=0.5):
    """Ngưỡng có F1 lớn nhất trong các ngưỡng có FPR <= max_fpr."""
    if strategy == "fixed":
        return float(fixed)
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    tp, fp = tpr * n_pos, fpr * n_neg
    f1 = np.divide(2 * tp, tp + fp + n_pos, out=np.zeros_like(tp), where=(tp + fp + n_pos) > 0)
    ok = (fpr <= max_fpr) & np.isfinite(thr)
    i = int(np.flatnonzero(ok)[np.argmax(f1[ok])]) if ok.any() else int(np.argmin(np.where(np.isfinite(thr), fpr, 2)))
    return float(min(thr[i], 1.0))


def evaluate(y, s, labels, threshold):
    pred = (s >= threshold).astype(np.int8)
    return {"overall": binary_metrics(y, s, threshold), "per_label": per_label(labels, pred)}


def flat(result, prefix, min_support=0):
    """Phẳng hóa kết quả để log lên MLflow (vd val_recall, test_rate__Bot)."""
    out = {f"{prefix}_{k}": v for k, v in result["overall"].items() if np.isfinite(v)}
    for lab, r in result["per_label"].items():
        if r["support"] >= min_support:
            out[f"{prefix}_rate__{lab.replace(' ', '_')}"] = r["rate"]
    return out


def latency_ms(predict, X, n_batch=1000, repeats=5):
    """Trung vị thời gian dự đoán (ms) cho 1 dòng và cho n_batch dòng."""
    def med(rows):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            predict(rows)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))
    return {"latency_ms_1row": med(X.iloc[:1]), f"latency_ms_{n_batch}rows": med(X.iloc[:n_batch])}


def plot_eval(y, s, threshold, title, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    pred = (s >= threshold).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"],
              xlabel="Dự đoán", ylabel="Thực tế", title=f"Ma trận nhầm lẫn (ngưỡng {threshold:.3f})")
    if 0 < y.sum() < len(y):
        p, r, _ = precision_recall_curve(y, s)
        ax[1].plot(r, p)
        ax[1].set(xlabel="Recall", ylabel="Precision", title=f"PR (AP={average_precision_score(y, s):.4f})")
        f, t, _ = roc_curve(y, s)
        ax[2].plot(f, t)
        ax[2].set(xscale="log", xlim=(1e-5, 1), xlabel="FPR (log)", ylabel="TPR",
                  title=f"ROC (AUC={roc_auc_score(y, s):.4f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_label(per_label_by_model, title, path):
    """Biểu đồ recall / FPR theo nhãn, mỗi mô hình một màu."""
    df = pd.DataFrame({m: {k: v["rate"] for k, v in pl.items()} for m, pl in per_label_by_model.items()})
    df = df.sort_index()
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(df) + 1.5))
    df.plot.barh(ax=ax, width=0.8)
    ax.set(xlim=(0, 1.05), xlabel="Recall (lớp tấn công) / FPR (Benign)", title=title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
