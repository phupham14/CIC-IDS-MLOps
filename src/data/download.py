"""Tạo MANIFEST.json (SHA-256 + dung lượng) cho các file CSV trong data/raw."""
import hashlib, json
from pathlib import Path

RAW = Path("data/raw")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    manifest = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(RAW.glob("*.csv"))
    }
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} files")
    for k, v in manifest.items():
        print(f"{k:55s} {v['bytes']:>12,d}  {v['sha256'][:12]}")
