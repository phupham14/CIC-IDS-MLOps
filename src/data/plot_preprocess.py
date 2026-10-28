"""Biểu đồ cho bước tiền xử lý: 08 số dòng bị loại ở từng bước, 09 nhãn ở train/val/test/production_sim,
10 chia 80/10/10 trong từng nhóm (ngày, nhãn).
Chạy sau preprocess.py:  python src/data/plot_preprocess.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("reports/data")
FIG = R / "figures"
CL = json.loads((R / "cleaning_report.json").read_text(encoding="utf-8"))
SP = json.loads((R / "split_report.json").read_text(encoding="utf-8"))

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a", "production_sim": "#eda100"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


# ---- 08: số dòng bị loại ở từng bước ----
t = CL["total"]
mode = CL["dedup"]["mode"]
steps = [("Dòng header bị lặp", t["header_rows"]),
         ("Timestamp sai năm", t["bad_timestamp_rows"]),
         ("Có NaN hoặc inf", t["nan_inf_rows"]),
         ("Có giá trị âm không hợp lệ", t["negative_rows"]),
         (f"Dòng trùng (chế độ '{mode}')", CL["duplicates_removed"])]
vals = np.array([s[1] for s in steps], dtype=float)
fig, ax = plt.subplots(figsize=(9.5, 3.8))
y = np.arange(len(steps))
ax.barh(y, np.maximum(vals, 0.5), color=COLOR["train"], height=0.6)
ax.set_xscale("log"); ax.set_xlim(0.4, vals.max() * 12)
for i, v in enumerate(vals):
    ax.text(max(v, 0.5) * 1.15, i, f"{fmt(v)}  ({v / t['rows_in']:.3%})", va="center", color=INK2, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Số dòng bị loại (thang log)")
ax.set_title(f"Làm sạch: {fmt(t['rows_in'])} dòng ban đầu còn {fmt(CL['rows_final'])} dòng ({CL['rows_final'] / t['rows_in']:.1%})")
fig.savefig(FIG / "08_cleaning_steps.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 09: nhãn theo tập ----
names = ["train", "val", "test", "production_sim"]
labs = sorted({l for n in names for l in SP["splits"][n]["label_counts"]},
              key=lambda k: -sum(SP["splits"][n]["label_counts"].get(k, 0) for n in names))
fig, ax = plt.subplots(figsize=(10.5, 0.7 * len(labs) + 1.8))
h = 0.2
y = np.arange(len(labs))
for k, name in enumerate(names):
    c = [SP["splits"][name]["label_counts"].get(l, 0) for l in labs]
    ax.barh(y + (k - 1.5) * h, np.maximum(c, 0.4), height=h * 0.9, color=COLOR[name],
            label=f"{name} ({fmt(SP['splits'][name]['rows'])} dòng)")
    for yi, v in zip(y + (k - 1.5) * h, c):
        if v:
            ax.text(v * 1.08, yi, fmt(v), va="center", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
ax.set_xlim(0.4, ax.get_xlim()[1] * 6); ax.grid(axis="y", visible=False)
ax.set_xlabel("Số luồng (thang log)"); ax.legend(frameon=False, loc="lower right")
ax.set_title("Phân bố nhãn: train / val / test (ngày 14/02 đến 21/02) và production_sim (22/02 đến 02/03)")
fig.savefig(FIG / "09_split_distribution.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 10: chia 80/10/10 trong từng nhóm (ngày, nhãn) ----
G = SP["groups"]
fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(G) + 1.6))
y = np.arange(len(G))
left = np.zeros(len(G))
for name in ("train", "val", "test"):
    share = np.array([g[name] / g["rows"] for g in G])
    ax.barh(y, share * 100, left=left, color=COLOR[name], height=0.7, label=name,
            edgecolor=SURFACE, linewidth=1.5)
    left += share * 100
for i, g in enumerate(G):
    ax.text(101, i, f"{fmt(g['rows'])} dòng", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([f"{g['day'][8:]}/{g['day'][5:7]}  {g['label']}" for g in G])
ax.invert_yaxis(); ax.set_xlim(0, 118); ax.grid(axis="y", visible=False)
ax.set_xlabel("% số dòng của nhóm (sắp theo thời gian: train trước, rồi val, rồi test)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
ax.set_title("Chia 80/10/10 theo thời gian trong từng nhóm (ngày, nhãn)")
fig.savefig(FIG / "10_split_by_group.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved 08, 09, 10")
