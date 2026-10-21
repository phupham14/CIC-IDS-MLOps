"""Kiểm tra Timestamp: dòng lệch ngày, các giờ xuất hiện, thứ tự thời gian trong file.
Chạy: python src/data/eda_timestamps.py  -> reports/data/timestamp_check.json
"""
import json, re
from pathlib import Path
import pandas as pd

FMT = "%d/%m/%Y %H:%M:%S"
out = {}
for f in sorted(Path("data/raw").glob("*_TrafficForML_CICFlowMeter.csv")):
    day = re.match(r"(\w+-\d\d-\d\d-\d{4})", f.name).group(1)
    expected = pd.to_datetime(day.split("-", 1)[1], format="%d-%m-%Y").date()
    df = pd.read_csv(f, usecols=["Timestamp", "Label"], low_memory=False)
    df = df[df["Timestamp"] != "Timestamp"]
    ts = pd.to_datetime(df["Timestamp"], format=FMT, errors="coerce")
    off = ts.dt.date != expected
    hours = sorted(int(h) for h in ts.dt.hour.dropna().unique())
    out[day] = {
        "expected_date": str(expected),
        "rows": int(len(df)),
        "rows_other_date": int(off.sum()),
        "other_dates": {str(k): int(v) for k, v in ts[off].dt.date.value_counts().head(5).items()},
        "labels_of_other_date_rows": df.loc[off, "Label"].value_counts().to_dict(),
        "hours_present": hours,
        "monotonic_in_file": bool(ts[~off].is_monotonic_increasing),
    }
    print(day, out[day]["rows_other_date"], hours, out[day]["monotonic_in_file"], flush=True)
Path("reports/data/timestamp_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
