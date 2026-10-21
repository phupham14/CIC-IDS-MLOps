"""Chuyển file Markdown thành notebook .ipynb để mở bằng Colab/Jupyter.
Mỗi mục cấp 2 (##) là một ô văn bản; ảnh PNG cục bộ được nhúng base64.
Dùng: python src/data/md_to_ipynb.py FILE.md [FILE2.md ...]  ->  colab/<tên>.ipynb (DATA.md -> DATA_report.ipynb)
"""
import base64, re, sys
from pathlib import Path
import nbformat as nbf

OUT = Path("colab")


def convert(src):
    src = Path(src)
    md = src.read_text()

    def embed(m):
        p = src.parent / m.group(2)
        if not p.exists():
            return m.group(0)
        return f"![{m.group(1)}](data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md)
    parts = [p.strip() for p in re.split(r"(?m)^(?=## )", md) if p.strip()]
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(p) for p in parts]
    OUT.mkdir(exist_ok=True)
    out = OUT / (src.stem + ("_report" if src.stem == "DATA" else "") + ".ipynb")
    nb.metadata["colab"] = {"name": out.name}
    nbf.write(nb, out)
    print(src.name, "->", out.name, f"({len(nb.cells)} cells)")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        convert(f)
