"""EDA dạng streaming: đọc từng file theo chunk, gom thống kê vào reports/data/eda_stats.json
và lưu một mẫu phân tầng (stratified) ra data/interim/eda_sample.parquet để vẽ biểu đồ.

Chạy: python src/data/eda_stats.py
"""
import json, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT_STATS = Path("reports/data/eda_stats.json")
OUT_SAMPLE = Path("data/interim/eda_sample.parquet")
CHUNK = 500_000
PER_CHUNK_LABEL_CAP = 4_000          # tối đa số dòng lấy mẫu / (chunk, nhãn)
ID_COLS = ["Flow ID", "Src IP", "Dst IP"]
TS_FMT = "%d/%m/%Y %H:%M:%S"
SEED = 42

FILES = sorted(RAW.glob("*_TrafficForML_CICFlowMeter.csv"))   # bỏ qua cic.csv (bản trùng)


def day_of(path):
    return re.match(r"(\w+-\d\d-\d\d-\d{4})", path.name).group(1)


def main():
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    per_file = {}
    label_by_file = defaultdict(lambda: defaultdict(int))
    label_by_hour = defaultdict(int)          # "day|hour|label" -> count
    col_stats = defaultdict(lambda: {"nan": 0, "inf": 0, "neg": 0, "n": 0,
                                     "min": np.inf, "max": -np.inf})
    columns_by_file = {}
    samples = []

    for path in FILES:
        t0 = time.time()
        day = day_of(path)
        hashes = set()
        n_rows = n_hdr = n_bad_ts = n_dup = 0
        ts_min, ts_max = None, None
        n_chunks = 0

        for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
            n_chunks += 1
            chunk.columns = chunk.columns.str.strip()
            if n_chunks == 1:
                columns_by_file[day] = list(chunk.columns)

            # dòng header bị lặp trong dữ liệu
            hdr = chunk["Label"].astype(str).str.strip() == "Label"
            n_hdr += int(hdr.sum())
            chunk = chunk[~hdr].copy()
            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            # trùng lặp trong cùng một file (so sánh nguyên dòng)
            h = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
            seen = np.fromiter((x in hashes for x in h), dtype=bool, count=len(h))
            dup_inside = pd.Series(h).duplicated().to_numpy()
            n_dup += int((seen | dup_inside).sum())
            hashes.update(h.tolist())

            # thời gian
            ts = pd.to_datetime(chunk["Timestamp"], format=TS_FMT, errors="coerce")
            n_bad_ts += int(ts.isna().sum())
            if ts.notna().any():
                lo, hi = ts.min(), ts.max()
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)

            # nhãn theo file và theo giờ (chỉ tính dòng có Timestamp trong ngày chính)
            for lab, c in chunk["Label"].value_counts().items():
                label_by_file[day][lab] += int(c)
            tmp = pd.DataFrame({"hour": ts.dt.hour, "Label": chunk["Label"]}).dropna()
            for (hr, lab), c in tmp.groupby(["hour", "Label"]).size().items():
                label_by_hour[f"{day}|{int(hr)}|{lab}"] += int(c)

            # lỗi theo cột
            feat_cols = [c for c in chunk.columns
                         if c not in ("Label", "Timestamp", *ID_COLS)]
            num = chunk[feat_cols].apply(pd.to_numeric, errors="coerce")
            isinf = np.isinf(num)
            for c in feat_cols:
                s = col_stats[c]
                v = num[c]
                s["n"] += len(v)
                s["nan"] += int(v.isna().sum())
                s["inf"] += int(isinf[c].sum())
                s["neg"] += int((v < 0).sum())
                finite = v[np.isfinite(v)]
                if len(finite):
                    s["min"] = min(s["min"], float(finite.min()))
                    s["max"] = max(s["max"], float(finite.max()))

            n_rows += len(chunk)

            # mẫu phân tầng theo nhãn
            keep = []
            for lab, idx in chunk.groupby("Label").indices.items():
                k = min(len(idx), PER_CHUNK_LABEL_CAP)
                keep.append(rng.choice(idx, size=k, replace=False))
            keep = np.concatenate(keep)
            smp = num.iloc[keep].astype("float32")
            smp["Label"] = chunk["Label"].iloc[keep].to_numpy()
            smp["Timestamp"] = ts.iloc[keep].to_numpy()
            smp["day"] = day
            samples.append(smp)

        per_file[day] = {
            "file": path.name, "bytes": path.stat().st_size, "rows": n_rows,
            "repeated_header_rows": n_hdr, "bad_timestamps": n_bad_ts,
            "duplicate_rows_within_file": n_dup,
            "ts_min": str(ts_min), "ts_max": str(ts_max),
            "n_columns": len(columns_by_file[day]),
        }
        print(f"{day}: {n_rows:,} rows, dup={n_dup:,}, hdr={n_hdr}, badTS={n_bad_ts}, "
              f"{time.time() - t0:.0f}s", flush=True)

    # so sánh tập cột giữa các file
    all_sets = {d: set(c) for d, c in columns_by_file.items()}
    common = set.intersection(*all_sets.values())
    extra = {d: sorted(s - common) for d, s in all_sets.items() if s - common}

    stats = {
        "files": per_file,
        "n_columns_common": len(common),
        "extra_columns_by_file": extra,
        "label_by_file": {d: dict(v) for d, v in label_by_file.items()},
        "label_by_hour": dict(label_by_hour),
        "column_stats": {c: {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                             for k, v in s.items()} for c, s in col_stats.items()},
    }
    OUT_STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False))

    sample = pd.concat(samples, ignore_index=True)
    sample.to_parquet(OUT_SAMPLE, index=False)
    print("sample:", sample.shape, "->", OUT_SAMPLE)
    print("stats ->", OUT_STATS)


if __name__ == "__main__":
    main()
