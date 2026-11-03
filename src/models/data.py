"""Nạp dữ liệu đã tiền xử lý (data/processed/*.parquet) cho huấn luyện và đánh giá.

Chỉ các cột trong features.json được đưa vào mô hình. Đọc theo batch và lấy mẫu ngay khi đọc để tiết kiệm RAM.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LABEL_COLS = ["y_binary", "Label", "meta_day"]


@dataclass
class Split:
    X: pd.DataFrame          # float32, đúng thứ tự features
    y: np.ndarray            # int8, 0 = Benign, 1 = Attack
    label: np.ndarray        # nhãn chi tiết, để tính recall từng loại tấn công
    day: np.ndarray          # meta_day

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return Split(self.X.iloc[idx].reset_index(drop=True), self.y[idx], self.label[idx], self.day[idx])


def load_features(artifacts_dir, exclude=()):
    features = json.loads((Path(artifacts_dir) / "features.json").read_text(encoding="utf-8"))
    missing = [c for c in exclude if c not in features]
    assert not missing, f"Không có đặc trưng {missing} trong features.json"
    return [c for c in features if c not in exclude]


def _strings(col):
    """Cột chuỗi -> mảng object dùng chung vài chuỗi duy nhất (tiết kiệm RAM)."""
    enc = col.combine_chunks().dictionary_encode()
    return np.array(enc.dictionary.to_pylist(), dtype=object)[enc.indices.to_numpy()]


def _row_groups(path, columns, batch_size):
    """Đọc từng row group rồi cắt thành batch (ParquetFile.iter_batches giữ đệm, tốn RAM)."""
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=columns)
        for start in range(0, t.num_rows, batch_size):
            yield t.slice(start, batch_size)


def iter_batches(path, features, batch_size=250_000):
    """Duyệt file theo batch; X là float32, đúng thứ tự features."""
    for b in _row_groups(path, features + LABEL_COLS, batch_size):
        X = np.empty((b.num_rows, len(features)), dtype=np.float32)
        for j, c in enumerate(features):
            X[:, j] = b.column(c).to_numpy()
        yield Split(pd.DataFrame(X, columns=features, copy=False), b.column("y_binary").to_numpy().astype(np.int8),
                    _strings(b.column("Label")), _strings(b.column("meta_day")))


def load_split(path, features, benign_frac_by_day=None, seed=42, max_rows=None, batch_size=250_000):
    """Đọc một file, có thể lấy mẫu Benign theo ngày (vd {"2018-02-20": 0.2}) và lấy mẫu phân tầng còn max_rows dòng.

    Đọc 2 lượt: lượt 1 chọn dòng, lượt 2 ghi thẳng vào mảng cấp phát sẵn (chỉ một bản X trong RAM)."""
    rng = np.random.default_rng(seed)
    masks, labels = [], []
    for b in _row_groups(path, ["y_binary", "Label", "meta_day"], batch_size):
        if benign_frac_by_day:
            frac = pd.Series(_strings(b.column("meta_day"))).map(benign_frac_by_day).fillna(1.0).to_numpy()
            masks.append((b.column("y_binary").to_numpy() == 1) | (rng.random(b.num_rows) < frac))
        else:
            masks.append(np.ones(b.num_rows, dtype=bool))
        labels.append(_strings(b.column("Label")))
    keep = np.concatenate(masks)
    if max_rows and keep.sum() > max_rows:
        lab, frac = np.concatenate(labels), max_rows / keep.sum()
        sampled = np.zeros_like(keep)
        for name in np.unique(lab[keep]):
            rows = np.flatnonzero(keep & (lab == name))
            sampled[rng.choice(rows, size=max(1, int(round(len(rows) * frac))), replace=False)] = True
        keep = sampled
    bounds = np.cumsum([0] + [len(m) for m in masks])
    masks = [keep[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    n = int(keep.sum())
    X = np.empty((n, len(features)), dtype=np.float32)
    y, label, day = np.empty(n, dtype=np.int8), np.empty(n, dtype=object), np.empty(n, dtype=object)
    pos = 0
    for sp, m in zip(iter_batches(path, features, batch_size), masks):
        k = int(m.sum())
        X[pos:pos + k] = sp.X.to_numpy()[m]
        y[pos:pos + k], label[pos:pos + k], day[pos:pos + k] = sp.y[m], sp.label[m], sp.day[m]
        pos += k
    return Split(pd.DataFrame(X, columns=features, copy=False), y, label, day)


def score_file(path, features, score_fn, batch_size=250_000):
    """Chấm điểm cả file theo batch, trả về y, điểm, nhãn, ngày (không giữ X)."""
    ys, ss, labs, days = [], [], [], []
    for sp in iter_batches(path, features, batch_size):
        ss.append(np.asarray(score_fn(sp.X), dtype=np.float64))
        ys.append(sp.y)
        labs.append(sp.label)
        days.append(sp.day)
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(labs), np.concatenate(days)


def stratified_sample(split, max_rows, seed=42):
    """Lấy mẫu tối đa max_rows dòng, giữ tỉ lệ từng Label."""
    if not max_rows or len(split) <= max_rows:
        return split
    rng = np.random.default_rng(seed)
    frac = max_rows / len(split)
    idx = []
    for lab in np.unique(split.label):
        rows = np.flatnonzero(split.label == lab)
        k = max(1, int(round(len(rows) * frac)))
        idx.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return split.subset(np.sort(np.concatenate(idx)))


def fingerprint(paths):
    """SHA-256 và dung lượng từng file (log lên MLflow)."""
    out = {}
    for p in map(Path, paths):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while block := f.read(1 << 22):
                h.update(block)
        out[p.name] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    return out


def describe(split):
    return {"rows": int(len(split)), "attack_share": float(split.y.mean()) if len(split) else 0.0,
            "label_counts": {k: int(v) for k, v in pd.Series(split.label).value_counts().items()}}
