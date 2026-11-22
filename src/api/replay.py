"""Script phát lại dữ liệu vận hành gửi tới API (Bước 5 - Demo End-to-End).

Theo INTERFACE.md (mục 3 & Phân việc bước 5):
  1. Đọc các lô của kịch bản (mặc định s1_new_attacks từ production_sim.parquet, đã tách 20% holdout
     theo đúng src/monitoring/scenarios.py).
  2. Gửi các luồng mạng (70 cột đặc trưng gốc) tới POST /predict/batch kèm request_id và timestamp
     thật của luồng (meta_time, coi là UTC), rồi gọi POST /flush để API ghi ngay file Parquet vào
     logs/predictions/date=YYYY-MM-DD/part-*.parquet.
  3. Giả lập nhãn đến trễ (labels.delay_batches, mặc định chậm 1 lô): sau khi gửi xong lô k+1 mới
     ghi nhãn thật của lô k vào logs/labels/date=YYYY-MM-DD/batch-<id>.parquet (gồm request_id, Label).

Chạy:
  python -m src.api.replay --scenario s1_new_attacks --api-url http://localhost:8000
  python -m src.api.replay --scenario s1_new_attacks --max-batches 5 --sleep 1.0
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import uuid

import httpx
import pandas as pd

from src.models import data as D
from src.models.pyfunc import WINDOW_FLAGS
from src.models.train import log
from src.monitoring import scenarios as S


def make_timestamps(batch, day_offset_ms=0):
    """Timestamp UTC (ISO 8601, đuôi Z) cho từng dòng của lô.

    Dùng meta_time thật của luồng (Batch.time). Lô không có thời gian (vd tạo tay trong test) thì sinh giờ giả
    tăng dần trong ngày, bắt đầu từ day_offset_ms, để các lô cùng ngày không bao giờ trùng nhau.
    Trả về (danh sách timestamp, offset cho lô tiếp theo cùng ngày)."""
    n = len(batch)
    if batch.time is not None:
        ts = pd.to_datetime(pd.Series(batch.time)).dt.tz_localize(None)
        return ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3].add("Z").tolist(), day_offset_ms
    base = datetime.strptime(batch.day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    stamps = [(base + timedelta(milliseconds=day_offset_ms + i)).isoformat(timespec="milliseconds")
              .replace("+00:00", "Z") for i in range(n)]
    return stamps, day_offset_ms + n


def write_label_batch(labels_dir: Path, batch_id: int, day_str: str, request_ids: list[str], labels, y_binary=None):
    """Ghi nhãn thật của một lô vào logs/labels/date=YYYY-MM-DD/batch-<id>.parquet."""
    day_dir = Path(labels_dir) / f"date={day_str}"
    day_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "request_id": pd.Series(request_ids, dtype="string"),
        "Label": pd.Series(labels, dtype="string"),
    })
    if y_binary is not None:
        df["y_binary"] = pd.Series(y_binary, dtype="int8")
    out_path = day_dir / f"batch-{batch_id:04d}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def replay_batches(batches, raw_features: list[str], client, api_url: str = "",
                   labels_dir: Path = Path("logs/labels"), delay_batches: int = 1,
                   chunk_size: int = 1000, sleep_sec: float = 0.0,
                   flush_remaining_labels: bool = False):
    """Phát lại danh sách các Batch qua HTTP client (httpx.Client hoặc FastAPI TestClient)."""
    base_url = api_url.rstrip("/")
    pending_labels = []
    day_offsets = {}                                   # giờ giả tiếp theo của từng ngày (chỉ khi lô không có time)
    stats = {
        "batches_sent": 0,
        "rows_sent": 0,
        "attack_predicted": 0,
        "label_batches_written": 0,
    }

    for step, b in enumerate(batches):
        t0 = time.perf_counter()
        n = len(b)
        req_ids = [str(uuid.uuid4()) for _ in range(n)]
        timestamps, day_offsets[b.day] = make_timestamps(b, day_offsets.get(b.day, 0))

        X_raw = b.data.X[raw_features]
        batch_attacks = 0
        last_version = "unknown"

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            flows = X_raw.iloc[start:end].to_dict(orient="records")
            res = client.post(
                f"{base_url}/predict/batch",
                json={
                    "flows": flows,
                    "request_ids": req_ids[start:end],
                    "timestamps": timestamps[start:end],
                },
            )
            res.raise_for_status()
            body = res.json()
            batch_attacks += int(body["attack_count"])
            last_version = body["model_version"]

        # Ép API flush bộ đệm ra Parquet ngay sau mỗi lô kịch bản để TV4 đọc được theo từng lô
        client.post(f"{base_url}/flush").raise_for_status()

        stats["batches_sent"] += 1
        stats["rows_sent"] += n
        stats["attack_predicted"] += batch_attacks

        pending_labels.append((b.id, b.day, req_ids, b.data.label, b.data.y))
        label_msg = "chờ nhãn"
        if len(pending_labels) > delay_batches:
            lb_id, lb_day, lb_ids, lb_labels, lb_y = pending_labels.pop(0)
            lpath = write_label_batch(labels_dir, lb_id, lb_day, lb_ids, lb_labels, lb_y)
            stats["label_batches_written"] += 1
            label_msg = f"đã ghi nhãn lô {lb_id} -> {lpath.name}"

        elapsed = time.perf_counter() - t0
        log(
            f"[Replay] lô {b.id:3d} ({b.day}) | {n:,} dòng | model v{last_version} | "
            f"dự đoán Attack: {batch_attacks:,} ({batch_attacks / max(n, 1):.4f}) | "
            f"{label_msg} ({elapsed:.2f}s)"
        )
        if sleep_sec > 0 and step + 1 < len(batches):
            time.sleep(sleep_sec)

    if flush_remaining_labels:
        while pending_labels:
            lb_id, lb_day, lb_ids, lb_labels, lb_y = pending_labels.pop(0)
            write_label_batch(labels_dir, lb_id, lb_day, lb_ids, lb_labels, lb_y)
            stats["label_batches_written"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description="Phát lại dữ liệu production_sim gửi vào FastAPI và ghi nhãn trễ.")
    ap.add_argument("--config", default="monitoring.yaml", help="Đường dẫn file cấu hình monitoring.yaml")
    ap.add_argument("--scenario", default="s1_new_attacks", help="Tên kịch bản trong monitoring.yaml")
    ap.add_argument("--api-url", default="http://localhost:8000", help="URL của FastAPI server")
    ap.add_argument("--labels-dir", default="logs/labels", help="Thư mục ghi nhãn thật đến trễ")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Số dòng mỗi HTTP request tới /predict/batch")
    ap.add_argument("--max-batches", type=int, default=None, help="Giới hạn số lô phát lại khi chạy thử")
    ap.add_argument("--delay-batches", type=int, default=None, help="Số lô nhãn đến trễ (mặc định theo monitoring.yaml)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Thời gian nghỉ (giây) giữa các lô")
    ap.add_argument("--flush-final-labels", action="store_true", help="Ghi nốt nhãn của lô cuối khi kết thúc")
    ap.add_argument("--smoke", action="store_true", help="Dùng cấu hình smoke nhỏ")
    args = ap.parse_args()

    cfg = S.load_cfg(args.config, args.smoke)
    features = D.load_features(cfg["models"]["data"]["artifacts_dir"])
    raw_features = [c for c in features if c not in WINDOW_FLAGS]
    delay = args.delay_batches if args.delay_batches is not None else int(cfg["labels"]["delay_batches"])

    batches = S.scenario_batches(cfg, args.scenario, features)
    if args.max_batches is not None:
        batches = batches[: args.max_batches]

    log(
        f"Bắt đầu phát lại kịch bản '{args.scenario}': {len(batches)} lô "
        f"({sum(len(b) for b in batches):,} dòng) -> {args.api_url} (trễ nhãn: {delay} lô)"
    )
    with httpx.Client(timeout=120.0) as client:
        stats = replay_batches(
            batches=batches,
            raw_features=raw_features,
            client=client,
            api_url=args.api_url,
            labels_dir=Path(args.labels_dir),
            delay_batches=delay,
            chunk_size=args.chunk_size,
            sleep_sec=args.sleep,
            flush_remaining_labels=args.flush_final_labels,
        )
    log(f"Hoàn tất phát lại: {stats}")


if __name__ == "__main__":
    main()
