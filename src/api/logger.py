"""Ghi log dự đoán ra file Parquet theo đợt (mục 3 của INTERFACE.md).

- Vị trí: logs/predictions/date=YYYY-MM-DD/part-<giờ>-<seq>.parquet (UTC)
- Ghi theo đợt: đủ max_rows (mặc định 1.000 dòng) hoặc định kỳ mỗi flush_interval_sec (mặc định 60 giây).
- Schema:
    request_id (string), timestamp (timestamp UTC), model_version (string),
    các cột đặc trưng trong features.json (float32, giá trị gốc chưa scale),
    score (float64), prediction (int8), latency_ms (float32).
"""
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pyfunc import WINDOW_FLAGS


class PredictionLogger:
    def __init__(self, log_dir="logs/predictions", features=(), max_rows=1000, flush_interval_sec=60.0):
        self.log_dir = Path(log_dir)
        self.features = list(features)
        self.raw_features = [c for c in self.features if c not in WINDOW_FLAGS]
        self.max_rows = int(max_rows)
        self.flush_interval_sec = float(flush_interval_sec)

        self._lock = threading.Lock()
        self._buffer = []
        self._buffered_rows = 0
        self._last_flush = time.monotonic()
        self._seq = 0

    def record(self, features_df: pd.DataFrame, out_df: pd.DataFrame, model_version: str,
               latency_ms: float, request_ids=None, timestamps=None):
        """Đưa một lô kết quả dự đoán vào bộ đệm; tự flush nếu đủ max_rows hoặc quá thời gian."""
        n = len(out_df)
        if n == 0:
            return []

        now_utc = datetime.now(timezone.utc)
        if request_ids is None:
            req_ids = [str(uuid.uuid4()) for _ in range(n)]
        else:
            req_ids = [str(r) for r in request_ids]

        if timestamps is None:
            ts_series = pd.Series([now_utc] * n, dtype="datetime64[us, UTC]")
        else:
            ts_series = pd.to_datetime(pd.Series(timestamps), format="ISO8601", utc=True).astype("datetime64[us, UTC]")

        # Chuẩn bị các cột đặc trưng float32 (giá trị gốc, tự tính thêm 2 cột cờ nếu có trong features.json)
        feat_df = features_df.reset_index(drop=True).copy()
        for flag, src in WINDOW_FLAGS.items():
            if flag in self.features and flag not in feat_df.columns and src in feat_df.columns:
                feat_df[flag] = (feat_df[src] == -1).astype(np.float32)

        cols_to_keep = [c for c in self.features if c in feat_df.columns]
        feat_f32 = feat_df[cols_to_keep].astype(np.float32)

        batch_df = pd.DataFrame({
            "request_id": pd.Series(req_ids, dtype="string"),
            "timestamp": ts_series.reset_index(drop=True),
            "model_version": pd.Series([str(model_version)] * n, dtype="string"),
        })
        batch_df = pd.concat([batch_df, feat_f32], axis=1)
        batch_df["score"] = out_df["score"].to_numpy(dtype=np.float64)
        batch_df["prediction"] = out_df["prediction"].to_numpy(dtype=np.int8)
        batch_df["latency_ms"] = np.float32(latency_ms / max(n, 1))

        to_write = None
        with self._lock:
            self._buffer.append(batch_df)
            self._buffered_rows += n
            elapsed = time.monotonic() - self._last_flush
            if self._buffered_rows >= self.max_rows or elapsed >= self.flush_interval_sec:
                to_write = self._drain_locked()

        if to_write is not None:
            self._write_parquet(to_write)
        return req_ids

    def flush(self) -> int:
        """Ghi toàn bộ dữ liệu đang chờ trong bộ đệm ra file Parquet ngay lập tức."""
        with self._lock:
            to_write = self._drain_locked()
        if to_write is None:
            return 0
        self._write_parquet(to_write)
        return len(to_write)

    def flush_if_due(self) -> int:
        """Được gọi định kỳ từ background task để flush nếu đã quá flush_interval_sec."""
        with self._lock:
            if not self._buffer or (time.monotonic() - self._last_flush) < self.flush_interval_sec:
                return 0
            to_write = self._drain_locked()
        if to_write is None:
            return 0
        self._write_parquet(to_write)
        return len(to_write)

    def _drain_locked(self):
        if not self._buffer:
            return None
        combined = pd.concat(self._buffer, ignore_index=True)
        self._buffer.clear()
        self._buffered_rows = 0
        self._last_flush = time.monotonic()
        return combined

    def _write_parquet(self, df: pd.DataFrame):
        # Tách theo ngày UTC của cột timestamp để luôn đúng cấu trúc date=YYYY-MM-DD/part-<giờ>.parquet
        dates = df["timestamp"].dt.strftime("%Y-%m-%d").rename("date_part")
        hours = df["timestamp"].dt.strftime("%H").rename("hour_part")
        for (date_str, hour_str), group in df.groupby([dates, hours], sort=False):
            day_dir = self.log_dir / f"date={date_str}"
            day_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._seq += 1
                seq = self._seq
            stamp = datetime.now(timezone.utc).strftime("%M%S")
            file_path = day_dir / f"part-{hour_str}h-{stamp}-{seq:04d}.parquet"
            group.reset_index(drop=True).to_parquet(file_path, index=False)
