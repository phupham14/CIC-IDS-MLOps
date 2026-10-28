"""Tiền xử lý CSE-CIC-IDS2018, chia train/val/test 80/10/10 và xuất Parquet.

Làm theo kết luận của nhóm trong notebooks/QUYET_DINH.md (số # là số điểm trong file đó):
  1. Làm sạch từng chunk (#5): header lặp, Timestamp sai năm, NaN/inf, giá trị âm; bỏ cột không cần
     thiết (định danh, Timestamp, hằng số, #11); giữ giá trị -1 và thêm cờ fwd/bwd_win_missing (#6);
     chuẩn hóa tên nhãn (#10) và gộp lớp Web (#9); suy luận AM/PM của Timestamp (#4).
  2. Loại dòng trùng trên các cột đặc trưng + nhãn, bỏ qua Timestamp, trước khi chia (#1).
  3. Chia: các ngày đến 21/02 -> train/val/test; các ngày sau -> production_sim (#2).
     Train/val/test chia theo thời gian 80/10/10 trong từng nhóm (ngày, nhãn) (#3).
  4. (tùy chọn, mặc định tắt vì ngưỡng chưa chốt, #11) bỏ đặc trưng tương quan cao, chọn trên train.
  5. Ghi báo cáo (reports/data/) và các file dùng lại khi huấn luyện/phục vụ (data/preprocessing/).
     Dữ liệu đã chia nằm ở data/processed/: train.parquet, val.parquet, test.parquet, production_sim.parquet.

Chạy:  python src/data/preprocess.py            (toàn bộ dữ liệu, khoảng 5 phút)
       python src/data/preprocess.py --smoke    (chạy thử nhỏ, ghi vào data/interim/smoke/)
Tham số: params.yaml. Thư mục dữ liệu thô: data.raw_dir, hoặc biến môi trường CIC_RAW_DIR.
"""
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

TS_FMT = "%d/%m/%Y %H:%M:%S"
SPLIT_NAMES = ["train", "val", "test", "production_sim"]
PRIME = np.uint64(1000003)


def day_of(path):
    return re.match(r"(\w+-\d\d-\d\d-\d{4})", path.name).group(1)


def day_date(day):
    return pd.to_datetime(day.split("-", 1)[1], format="%d-%m-%Y")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Bước 1: làm sạch theo chunk
# ----------------------------------------------------------------------------------------------
def clean_chunk(chunk, feature_cols, cfg, stats):
    c = cfg["clean"]
    label_col, time_col = cfg["data"]["label_col"], cfg["data"]["time_col"]
    chunk.columns = chunk.columns.str.strip()
    stats["rows_in"] += len(chunk)

    # 1) dòng header bị lặp trong dữ liệu
    hdr = chunk[label_col].astype(str).str.strip() == label_col
    stats["header_rows"] += int(hdr.sum())
    chunk = chunk[~hdr]

    # 2) Timestamp không parse được hoặc sai năm (ví dụ 1970)
    ts = pd.to_datetime(chunk[time_col], format=TS_FMT, errors="coerce")
    bad_ts = ts.isna() | (ts.dt.year < c["min_year"])
    stats["bad_timestamp_rows"] += int(bad_ts.sum())
    chunk, ts = chunk[~bad_ts], ts[~bad_ts]

    # 3) suy luận AM/PM (#4): giờ trong [lo, hi] được hiểu là buổi chiều
    lo, hi = cfg["time"]["pm_hours"]
    pm = (ts.dt.hour >= lo) & (ts.dt.hour <= hi)
    ts = ts + pd.to_timedelta(pm.astype(int) * 12, unit="h")

    raw = chunk[label_col].astype(str).str.strip().replace(cfg["label"]["rename"])
    merged = raw.replace(cfg["label"]["merge"])
    num = chunk[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    num = num.replace([np.inf, -np.inf], np.nan)

    # 4) NaN / inf (#5)
    if c["drop_missing"]:
        miss = num.isna().any(axis=1)
        stats["nan_inf_rows"] += int(miss.sum())
        keep = ~miss
        num, raw, merged, ts = num[keep], raw[keep], merged[keep], ts[keep]

    # 5) giá trị âm ở các cột không cho phép (giữ -1 của Init Win Byts, #6)
    if c["drop_negative"]:
        neg_cols = [x for x in feature_cols if x not in c["allow_negative_cols"]]
        neg = (num[neg_cols] < 0).any(axis=1)
        stats["negative_rows"] += int(neg.sum())
        keep = ~neg
        num, raw, merged, ts = num[keep], raw[keep], merged[keep], ts[keep]

    stats["rows_after_clean"] += len(num)
    tsec = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64").to_numpy()
    return num.reset_index(drop=True), raw.reset_index(drop=True), merged.reset_index(drop=True), tsec


def stage_clean(cfg, paths, args):
    d, cl = cfg["data"], cfg["clean"]
    raw_dir = Path(os.environ.get("CIC_RAW_DIR", d["raw_dir"]))
    files = sorted(raw_dir.glob(d["file_glob"]), key=lambda p: day_date(day_of(p)))
    assert files, f"Không thấy file nào ở {raw_dir}"
    log(f"Đọc {len(files)} file từ {raw_dir}")
    shutil.rmtree(paths["parts"], ignore_errors=True)
    paths["parts"].mkdir(parents=True, exist_ok=True)

    drop = set(cl["drop_id_cols"]) | set(cl["drop_constant_cols"]) | {d["label_col"], d["time_col"]}
    feature_cols, dropped_cols, per_file, parts = None, {}, {}, []

    for f in files:
        day, t0 = day_of(f), time.time()
        date = day_date(day)
        st = dict(rows_in=0, header_rows=0, bad_timestamp_rows=0, nan_inf_rows=0,
                  negative_rows=0, rows_after_clean=0)
        for i, chunk in enumerate(pd.read_csv(f, chunksize=args.chunksize, low_memory=False)):
            if args.max_chunks and i >= args.max_chunks:
                break
            chunk.columns = chunk.columns.str.strip()
            if feature_cols is None:
                feature_cols = [c for c in chunk.columns if c not in drop]
                for c in chunk.columns:
                    if c in cl["drop_id_cols"]:
                        dropped_cols[c] = "cột định danh (rò rỉ)"
                    elif c in cl["drop_constant_cols"]:
                        dropped_cols[c] = "cột hằng số (phương sai 0)"
                    elif c == d["time_col"]:
                        dropped_cols[c] = "Timestamp (chỉ dùng để sắp thứ tự khi chia, không làm đặc trưng)"
                log(f"{len(feature_cols)} đặc trưng gốc sau khi bỏ cột không cần thiết")
            missing = [c for c in feature_cols if c not in chunk.columns]
            assert not missing, f"{f.name} thiếu cột {missing}"

            num, raw, merged, tsec = clean_chunk(chunk, feature_cols, cfg, st)
            if num.empty:
                continue
            fh = pd.util.hash_pandas_object(num, index=False).to_numpy()   # hash đặc trưng (float64)
            out = num.astype("float32")
            out["Label"] = merged.to_numpy(dtype=object)
            out["Label_raw"] = raw.to_numpy(dtype=object)
            out["meta_day"] = f"{date:%Y-%m-%d}"
            out["meta_time"] = tsec
            out["meta_fkey"] = fh
            p = paths["parts"] / f"{date:%Y%m%d}_{i:04d}.parquet"
            out.to_parquet(p, index=False, compression="zstd")
            parts.append(p)
        per_file[day] = st
        log(f"{day}: {st['rows_in']:,} -> {st['rows_after_clean']:,} dòng ({time.time() - t0:.0f}s)")
    return sorted(parts), feature_cols, dropped_cols, per_file


# ----------------------------------------------------------------------------------------------
# Bước 2 và 3: loại trùng toàn cục, chia
# ----------------------------------------------------------------------------------------------
def stage_dedup_split(cfg, parts):
    d, sp = cfg["data"], cfg["split"]
    log("Đọc khóa, nhãn và thời gian của các phần")
    fkeys, tsecs, labs, sizes = [], [], [], []
    for p in parts:
        t = pq.read_table(p, columns=["Label", "meta_time", "meta_fkey"]).to_pandas()
        fkeys.append(t["meta_fkey"].to_numpy())
        tsecs.append(t["meta_time"].to_numpy())
        labs.append(t["Label"].astype(str).to_numpy(dtype=object))
        sizes.append(len(t))
    day_ints = np.repeat([int(Path(p).stem.split("_")[0]) for p in parts], sizes)
    fkey, tsec, lab = np.concatenate(fkeys), np.concatenate(tsecs), np.concatenate(labs)
    del fkeys, tsecs, labs
    n = len(fkey)

    labels_sorted = [d["benign_label"]] + sorted(set(np.unique(lab).tolist()) - {d["benign_label"]})
    label_map = {name: i for i, name in enumerate(labels_sorted)}
    codes = pd.Categorical(lab, categories=labels_sorted).codes.astype(np.int8)
    del lab

    # thứ tự theo (ngày, thời gian đã suy luận AM/PM, thứ tự dòng gốc): sắp ổn định (#4)
    ord_ = np.lexsort((np.arange(n), tsec, day_ints))

    with np.errstate(over="ignore"):
        fkl = fkey * PRIME + codes.astype(np.uint64)             # đặc trưng + nhãn
        exact = fkl * PRIME + tsec.astype(np.uint64)             # đặc trưng + nhãn + thời điểm

    def first_mask(keys):                                        # giữ dòng xuất hiện sớm nhất
        dup = pd.Series(keys[ord_]).duplicated(keep="first").to_numpy()
        m = np.empty(n, dtype=bool)
        m[ord_] = ~dup
        return m

    keep_feat, keep_exact = first_mask(fkl), first_mask(exact)
    mode = cfg["clean"]["dedup"]
    kept = {"features": keep_feat, "exact": keep_exact, "none": np.ones(n, dtype=bool)}[mode]

    before = np.bincount(codes, minlength=len(labels_sorted))
    after = np.bincount(codes[kept], minlength=len(labels_sorted))
    uniq_feat = np.bincount(codes[keep_feat], minlength=len(labels_sorted))
    dedup_report = {labels_sorted[i]: {
        "before": int(before[i]), "after": int(after[i]), "removed": int(before[i] - after[i]),
        "unique_feature_vectors": int(uniq_feat[i])} for i in range(len(labels_sorted))}
    dedup_summary = {"mode": mode,
                     "duplicates_exact_row": int((~keep_exact).sum()),
                     "duplicates_features_and_label": int((~keep_feat).sum())}
    log(f"Loại trùng (chế độ '{mode}'): {n:,} -> {int(kept.sum()):,} dòng")

    # vector đặc trưng giống hệt nhau nhưng khác nhãn (nhiễu nhãn)
    pairs = pd.DataFrame({"f": fkey[kept], "l": codes[kept]}).drop_duplicates()
    amb = pairs.loc[pairs["f"].duplicated(keep=False), "f"].unique()
    ambiguity = {"feature_vectors_with_multiple_labels": int(len(amb)),
                 "rows_affected": int(np.isin(fkey[kept], amb).sum())}

    # chia: ngày <= train_last_day -> train/val/test theo thời gian trong từng nhóm (ngày, nhãn); sau đó -> production_sim
    last = int(pd.Timestamp(sp["train_last_day"]).strftime("%Y%m%d"))
    split = np.full(n, -1, dtype=np.int8)                        # -1 đã bị loại; 0 train; 1 val; 2 test; 3 production_sim
    split[kept & (day_ints > last)] = 3
    day_s, code_s, kept_s = day_ints[ord_], codes[ord_], kept[ord_]
    groups = []
    for dd in np.unique(day_ints[day_ints <= last]):
        for c in range(len(labels_sorted)):
            rows = ord_[np.flatnonzero(kept_s & (day_s == dd) & (code_s == c))]   # đã sắp theo thời gian
            m = len(rows)
            if m == 0:
                continue
            n_val, n_test = int(round(m * sp["val"])), int(round(m * sp["test"]))
            if m >= 3:                                           # nhóm nhỏ vẫn có ít nhất 1 mẫu ở val và test
                n_val, n_test = max(n_val, 1), max(n_test, 1)
            n_tr = m - n_val - n_test
            split[rows[:n_tr]] = 0
            split[rows[n_tr:n_tr + n_val]] = 1
            split[rows[n_tr + n_val:]] = 2
            groups.append({"day": f"{pd.Timestamp(str(dd)):%Y-%m-%d}", "label": labels_sorted[c],
                           "rows": m, "train": n_tr, "val": n_val, "test": n_test})
    return sizes, codes, split, label_map, dedup_report, dedup_summary, ambiguity, groups


def stage_write(cfg, paths, parts, sizes, codes, split, feature_cols):
    add_flags = cfg["clean"]["add_window_flags"]
    base_features = list(feature_cols)
    all_features = base_features + (["fwd_win_missing", "bwd_win_missing"] if add_flags else [])
    schema = pa.schema(
        [(c, pa.float32()) for c in base_features]
        + ([("fwd_win_missing", pa.int8()), ("bwd_win_missing", pa.int8())] if add_flags else [])
        + [("Label", pa.string()), ("Label_raw", pa.string()), ("y_binary", pa.int8()), ("y_multi", pa.int8()),
           ("meta_day", pa.string()), ("meta_time", pa.timestamp("s")), ("meta_key", pa.uint64())])
    paths["processed"].mkdir(parents=True, exist_ok=True)
    writers = {s: pq.ParquetWriter(paths["processed"] / f"{SPLIT_NAMES[s]}.parquet", schema, compression="zstd")
               for s in range(4)}
    stats = {s: {"rows": 0, "fwd_win_missing": 0, "bwd_win_missing": 0, "tmin": None, "tmax": None}
             for s in range(4)}
    benign = cfg["data"]["benign_label"]
    off = 0
    for p, size in zip(parts, sizes):
        df = pq.read_table(p).to_pandas()
        sl = split[off:off + size]
        cd = codes[off:off + size]
        off += size
        df["y_binary"] = (df["Label"] != benign).astype(np.int8)
        df["y_multi"] = cd
        with np.errstate(over="ignore"):
            df["meta_key"] = df["meta_fkey"].to_numpy() * PRIME + cd.astype(np.uint64)   # khóa đặc trưng + nhãn
        if add_flags:
            df["fwd_win_missing"] = (df["Init Fwd Win Byts"] == -1).astype(np.int8)
            df["bwd_win_missing"] = (df["Init Bwd Win Byts"] == -1).astype(np.int8)
        df["meta_time"] = pd.to_datetime(df["meta_time"], unit="s")
        for s in range(4):
            sub = df.loc[sl == s, [f.name for f in schema]]
            if len(sub) == 0:
                continue
            writers[s].write_table(pa.Table.from_pandas(sub, schema=schema, preserve_index=False))
            st = stats[s]
            st["rows"] += len(sub)
            if add_flags:
                st["fwd_win_missing"] += int(sub["fwd_win_missing"].sum())
                st["bwd_win_missing"] += int(sub["bwd_win_missing"].sum())
            lo, hi = sub["meta_time"].min(), sub["meta_time"].max()
            st["tmin"] = lo if st["tmin"] is None else min(st["tmin"], lo)
            st["tmax"] = hi if st["tmax"] is None else max(st["tmax"], hi)
    for w in writers.values():
        w.close()
    log("Đã ghi train / val / test / production_sim")
    return all_features, stats


# ----------------------------------------------------------------------------------------------
# Bước 4 (tùy chọn): bỏ đặc trưng tương quan cao, chọn trên train
# ----------------------------------------------------------------------------------------------
def stage_correlated(cfg, paths, features, base_features):
    f = cfg["features"]
    if not f["drop_correlated"]:
        return features, {}
    rng = np.random.default_rng(cfg["data"]["seed"])
    pf = pq.ParquetFile(paths["processed"] / "train.parquet")
    samples = []
    for b in pf.iter_batches(batch_size=500000, columns=base_features):
        df = b.to_pandas()
        samples.append(df[rng.random(len(df)) < f["corr_sample_frac"]])
    corr = pd.concat(samples, ignore_index=True).corr().abs()
    kept, dropped = [], {}
    for c in base_features:
        hit = next((k for k in kept if corr.loc[c, k] >= f["corr_threshold"]), None)
        if hit is None:
            kept.append(c)
        else:
            dropped[c] = f"tương quan |r|={corr.loc[c, hit]:.4f} với '{hit}'"
    log(f"Bỏ {len(dropped)} đặc trưng tương quan >= {f['corr_threshold']}")
    for name in SPLIT_NAMES:
        path = paths["processed"] / f"{name}.parquet"
        tmp = path.with_suffix(".tmp")
        src = pq.ParquetFile(path)
        keep_cols = [c for c in src.schema_arrow.names if c not in dropped]
        writer = pq.ParquetWriter(tmp, pa.schema([src.schema_arrow.field(c) for c in keep_cols]), compression="zstd")
        for b in src.iter_batches(batch_size=500000, columns=keep_cols):
            writer.write_batch(b)
        writer.close()
        os.replace(tmp, path)
    return [c for c in features if c not in dropped], dropped


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhỏ, ghi vào data/interim/smoke/")
    ap.add_argument("--keep-parts", action="store_true", help="giữ lại các phần trung gian")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    d = cfg["data"]
    args.chunksize = 50_000 if args.smoke else d["chunksize"]
    args.max_chunks = 2 if args.smoke else 0
    base = Path(d["interim_dir"]) / "smoke" if args.smoke else None
    paths = {
        "parts": Path(d["interim_dir"]) / ("smoke/parts" if args.smoke else "clean_parts"),
        "processed": base / "processed" if args.smoke else Path(d["processed_dir"]),
        "artifacts": base / "artifacts" if args.smoke else Path(d["artifacts_dir"]),
        "reports": base / "reports" if args.smoke else Path(d["reports_dir"]),
    }
    for k in ("artifacts", "reports"):
        paths[k].mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    parts, feature_cols, dropped_cols, per_file = stage_clean(cfg, paths, args)
    sizes, codes, split, label_map, dedup_report, dedup_summary, ambiguity, groups = stage_dedup_split(cfg, parts)
    features, stats = stage_write(cfg, paths, parts, sizes, codes, split, feature_cols)
    features, dropped_corr = stage_correlated(cfg, paths, features, feature_cols)

    # ---- báo cáo ----
    tot = {k: sum(v[k] for v in per_file.values()) for k in next(iter(per_file.values()))}
    rows_final = int((split >= 0).sum())
    cleaning = {
        "per_file": per_file, "total": tot, "rows_after_clean": tot["rows_after_clean"],
        "dedup": dedup_summary, "duplicates_removed": tot["rows_after_clean"] - rows_final,
        "rows_final": rows_final, "dedup_by_label": dedup_report,
        "ambiguous_feature_vectors": ambiguity,
        "dropped_columns": {**dropped_cols, **dropped_corr}, "n_features_final": len(features)}
    write_json(paths["reports"] / "cleaning_report.json", cleaning)

    report = {"seed": d["seed"], "train_last_day": cfg["split"]["train_last_day"],
              "ratio": [cfg["split"][s] for s in ("train", "val", "test")],
              "time_assumption": f"giờ {cfg['time']['pm_hours'][0]} đến {cfg['time']['pm_hours'][1]} được hiểu là buổi chiều (+12h)",
              "splits": {}, "groups": groups}
    for s, name in enumerate(SPLIT_NAMES):
        path = paths["processed"] / f"{name}.parquet"
        t = pq.read_table(path, columns=["Label", "meta_day"]).to_pandas()
        st = stats[s]
        report["splits"][name] = {
            "rows": int(len(t)), "share_of_final": float(len(t) / rows_final),
            "attack_share": float(1 - (t["Label"] == d["benign_label"]).mean()),
            "label_counts": {k: int(v) for k, v in t["Label"].value_counts().items()},
            "day_counts": {k: int(v) for k, v in t["meta_day"].value_counts().sort_index().items()},
            "fwd_win_missing_share": float(st["fwd_win_missing"] / max(st["rows"], 1)),
            "bwd_win_missing_share": float(st["bwd_win_missing"] / max(st["rows"], 1)),
            "time_min": str(st["tmin"]), "time_max": str(st["tmax"]),
            "file_mb": round(path.stat().st_size / 1e6, 1)}
    write_json(paths["reports"] / "split_report.json", report)

    write_json(paths["artifacts"] / "label_map.json", label_map)
    write_json(paths["artifacts"] / "features.json", features)
    write_json(paths["artifacts"] / "dropped_features.json", cleaning["dropped_columns"])

    if not args.keep_parts:
        shutil.rmtree(paths["parts"], ignore_errors=True)
    log(f"Xong sau {time.time() - t0:.0f}s: {rows_final:,} dòng, {len(features)} đặc trưng")


if __name__ == "__main__":
    main()
