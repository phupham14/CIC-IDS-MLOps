"""Chuyển docs/DATA.md thành docs/DATA_report.html tự chứa (ảnh nhúng base64) để xem ở mọi trình duyệt."""
import base64, re
from pathlib import Path
import markdown

DOCS = Path("docs")
md = (DOCS / "DATA.md").read_text()

def embed(m):
    alt, src = m.group(1), m.group(2)
    data = base64.b64encode((DOCS / src).read_bytes()).decode()
    return f"![{alt}](data:image/png;base64,{data})"

md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md)
# mermaid -> khối code thường (trình duyệt không tự vẽ mermaid)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
css = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#0b0b0b}
img{max-width:100%;height:auto;border:1px solid #e3e2de;border-radius:6px;margin:.5rem 0}
table{border-collapse:collapse;margin:1rem 0;font-size:.92rem}th,td{border:1px solid #d0cfca;padding:.35rem .6rem;text-align:left}
th{background:#f0efec}code{background:#f0efec;padding:.1rem .3rem;border-radius:3px}pre{background:#f6f5f2;padding:1rem;overflow:auto;border-radius:6px}
blockquote{border-left:4px solid #2a78d6;margin:1rem 0;padding:.2rem 1rem;background:#f4f8fd}"""
(DOCS / "DATA_report.html").write_text(f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Báo cáo Data - CyberMLOps</title><style>{css}</style></head><body>{body}</body></html>")
print("ok")
