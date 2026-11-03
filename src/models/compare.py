"""Champion–Challenger: so sánh mô hình mới (@challenger) với mô hình đang dùng (@champion) trên cùng holdout.

Challenger chỉ được nhận khi đạt mọi tiêu chí trong promotion (models.yaml): FPR, recall từng lớp tấn công,
PR-AUC, Macro-F1, thời gian suy luận. Chưa có champion thì chỉ xét FPR.

Chạy:
  python -m src.models.compare --promote
  python -m src.models.compare --holdout a.parquet b.parquet --target-labels "Bot,Infiltration" --promote
Gọi từ pipeline tái huấn luyện: run_comparison(cfg, ...)["accepted"]
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np

from src.models import data as D
from src.models import metrics as M
from src.models import registry as R
from src.models.train import load_cfg, log


def evaluate_model(uri, holdouts, features, batch_size=200_000):
    """Chấm điểm bằng đúng artifact pyfunc mà API nạp."""
    model = mlflow.pyfunc.load_model(uri)
    thr = float(model.metadata.metadata["threshold"])
    parts = [D.score_file(h, features, lambda X: model.predict(X)["score"].to_numpy(), batch_size) for h in holdouts]
    y, s, lab = (np.concatenate([p[i] for p in parts]) for i in range(3))
    res = M.evaluate(y, s, lab, thr)
    sample = next(D.iter_batches(holdouts[0], features, batch_size=1000)).X
    res.update(threshold=thr, **M.latency_ms(lambda X: model.predict(X), sample))
    return res


def decide(chall, champ, p, targets=()):
    checks, notes = [], []

    def check(name, passed, challenger, champion=None, limit=""):
        checks.append({"check": name, "passed": bool(passed), "challenger": challenger, "champion": champion,
                       "limit": limit})

    co, cl = chall["overall"], chall["per_label"]
    check("fpr", co["fpr"] <= p["max_fpr"], co["fpr"], champ and champ["overall"]["fpr"], f"<= {p['max_fpr']}")
    if champ is None:
        notes.append("Chưa có champion: chỉ xét FPR tuyệt đối.")
        return {"accepted": all(c["passed"] for c in checks), "checks": checks, "notes": notes}

    ho, hl = champ["overall"], champ["per_label"]
    check("fpr_increase", co["fpr"] - ho["fpr"] <= p["max_fpr_increase"], co["fpr"], ho["fpr"],
          f"tăng <= {p['max_fpr_increase']}")
    for m, tol in p["max_metric_drop"].items():
        check(m, co[m] >= ho[m] - tol, co[m], ho[m], f">= champion - {tol}")

    attacks = sorted(k for k in cl if k != M.BENIGN)
    small = [k for k in attacks if cl[k]["support"] < p["min_support"]]
    if small:
        notes.append(f"Không xét riêng (ít hơn {p['min_support']} mẫu): {small}")
    big = [k for k in attacks if k not in small]
    missing = [t for t in targets if t not in cl]
    if missing:
        notes.append(f"Holdout không có lớp mục tiêu: {missing}")
    for k in big:
        drop = p["max_recall_drop"]
        name = f"target_no_drop[{k}]" if k in targets else f"recall[{k}]"
        check(name, cl[k]["rate"] >= hl[k]["rate"] - drop, cl[k]["rate"], hl[k]["rate"], f">= champion - {drop}")
    tgt = [k for k in targets if k in big]
    if targets:
        gains = {k: cl[k]["rate"] - hl[k]["rate"] for k in tgt}
        best = max(gains, key=gains.get) if gains else None
        check("target_gain", best is not None and gains[best] >= p["min_recall_gain"],
              cl[best]["rate"] if best else None, hl[best]["rate"] if best else None,
              f"ít nhất một lớp mục tiêu tăng >= {p['min_recall_gain']}" + (f" (tốt nhất: {best})" if best else ""))
    check("latency_ms_1000rows", chall["latency_ms_1000rows"] <= p["max_latency_ratio"] * champ["latency_ms_1000rows"],
          chall["latency_ms_1000rows"], champ["latency_ms_1000rows"], f"<= {p['max_latency_ratio']} x champion")
    return {"accepted": all(c["passed"] for c in checks), "checks": checks, "notes": notes}


def run_comparison(cfg, challenger=None, champion=None, holdouts=None, targets=(), promote=False):
    reg, p = cfg["registry"], cfg["promotion"]
    name = reg["model_name"]
    R.setup(cfg, cfg["mlflow"]["compare_experiment"])
    ch = R.get_version(name, challenger or reg["challenger_alias"])
    assert ch is not None, f"Không tìm thấy challenger '{challenger or reg['challenger_alias']}' của {name}"
    cp = R.get_version(name, champion or reg["champion_alias"])
    ch_v, cp_v = str(ch.version), (str(cp.version) if cp else None)
    assert ch_v != cp_v, f"Challenger và champion cùng là v{ch_v}"
    holdouts = [Path(h) for h in (holdouts or [p["holdout"]])]
    features = D.load_features(cfg["data"]["artifacts_dir"])

    log(f"Chấm challenger v{ch_v}" + (f" và champion v{cp_v}" if cp else " (chưa có champion)")
        + f" trên {[h.name for h in holdouts]}")
    res_ch = evaluate_model(R.model_uri(name, ch_v), holdouts, features)
    res_cp = evaluate_model(R.model_uri(name, cp_v), holdouts, features) if cp else None
    decision = decide(res_ch, res_cp, p, tuple(targets))
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model_name": name,
              "challenger": {"version": ch_v, "tags": ch.tags, **res_ch},
              "champion": {"version": cp_v, "tags": cp.tags, **res_cp} if cp else None,
              "holdouts": [str(h) for h in holdouts], "target_labels": list(targets), "criteria": p, **decision}

    tag = f"v{cp_v or 'none'}-vs-v{ch_v}"
    with mlflow.start_run(run_name=tag):
        mlflow.set_tags({"decision": "accepted" if decision["accepted"] else "rejected",
                         "challenger_version": ch_v, "champion_version": cp_v or "none"})
        mlflow.log_params({"holdouts": ",".join(h.name for h in holdouts), "target_labels": ",".join(targets) or "-"})
        for who, res in (("challenger", res_ch), ("champion", res_cp)):
            if res:
                mlflow.log_metrics(M.flat(res, who))
        mlflow.log_dict(report, "comparison.json")

    client = mlflow.MlflowClient()
    client.set_model_version_tag(name, ch_v, "comparison", f"{tag}: "
                                 + ("accepted" if decision["accepted"] else "rejected"))
    if promote and decision["accepted"]:
        old = R.promote(name, ch_v, reg, tags={"promoted_at": report["time"]})
        report["promoted"] = {"champion": ch_v, "previous": old}
        log(f"Đã chuyển @{reg['champion_alias']} -> v{ch_v}" + (f" (cũ: v{old} -> @{reg['previous_alias']})"
                                                                      if old else ""))
    out = Path(cfg["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    return report


def _fmt(v):
    return "-" if v is None else f"{v:.5f}"


def print_report(r):
    print(f"\n{'Tiêu chí':34s} {'Challenger':>12s} {'Champion':>12s}  Đạt  Giới hạn")
    for c in r["checks"]:
        print(f"{c['check']:34s} {_fmt(c['challenger']):>12s} {_fmt(c['champion']):>12s}  "
              f"{'✓' if c['passed'] else '✗':^3s}  {c['limit']}")
    for n in r["notes"]:
        print(f"- {n}")
    print(f"\nKẾT LUẬN: {'NHẬN challenger' if r['accepted'] else 'GIỮ champion'} (challenger v{r['challenger']['version']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models.yaml")
    ap.add_argument("--challenger", help="số phiên bản hoặc alias (mặc định @challenger)")
    ap.add_argument("--champion", help="số phiên bản hoặc alias (mặc định @champion)")
    ap.add_argument("--holdout", nargs="+", help="một hoặc nhiều file Parquet có nhãn (mặc định promotion.holdout)")
    ap.add_argument("--target-labels", default="", help='lớp tấn công mới cần cải thiện, ví dụ "Bot,Infiltration"')
    ap.add_argument("--promote", action="store_true", help="chuyển @champion nếu challenger đạt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.smoke)
    if args.smoke and not args.holdout:
        args.holdout = [str(Path(cfg["data"]["processed_dir"]) / "test.parquet")]
    targets = [t.strip() for t in args.target_labels.split(",") if t.strip()]
    print_report(run_comparison(cfg, args.challenger, args.champion, args.holdout, targets, args.promote))


if __name__ == "__main__":
    main()
