"""Vẽ biểu đồ EDA từ reports/data/eda_stats.json và data/interim/eda_sample.parquet.

Chạy: python src/data/eda_plots.py   (sau khi chạy eda_stats.py)
Ảnh lưu ở reports/data/figures/, số liệu tóm tắt ở reports/data/eda_summary.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

STATS = json.loads(Path("reports/data/eda_stats.json").read_text())
SAMPLE = pd.read_parquet("data/interim/eda_sample.parquet")
FIG = Path("reports/data/figures"); FIG.mkdir(parents=True, exist_ok=True)

# --- bảng màu tham chiếu (light), dùng theo đúng thứ tự slot, không chu kỳ ---
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
FAMILIES = ["Benign", "DoS", "DDoS", "Brute Force", "Botnet", "Web Attack", "Infiltration", "Other"]
FAM_COLOR = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("bwr_gray", ["#256abf", "#f0efec", "#e34948"])
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def family(label):
    s = label.lower()
    if s == "benign":
        return "Benign"
    if any(k in s for k in ("xss", "sql", "web")):
        return "Web Attack"
    if "ddos" in s:
        return "DDoS"
    if "dos" in s:
        return "DoS"
    if "brute" in s:
        return "Brute Force"
    if s.startswith("bot"):
        return "Botnet"
    if "infilt" in s:
        return "Infiltration"
    return "Other"


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


days = list(STATS["files"].keys())
order = sorted(days, key=lambda d: pd.to_datetime(d.split("-", 1)[1], format="%d-%m-%Y"))
short = {d: d.split("-", 1)[1][:5] for d in order}          # dd-mm
_dt = {d: pd.to_datetime(d.split("-", 1)[1], format="%d-%m-%Y") for d in order}
short_wd = {d: f"{_dt[d].strftime('%a')} {short[d].replace('-', '/')}" for d in order}

summary = {}

# ---------------- 1. Phân bố nhãn ----------------
tot = {}
for d in order:
    for lab, c in STATS["label_by_file"][d].items():
        tot[lab] = tot.get(lab, 0) + c
tot = pd.Series(tot).sort_values()
total_rows = int(tot.sum())
summary.update(total_rows=total_rows, n_labels=int(len(tot)),
               label_counts=tot.sort_values(ascending=False).astype(int).to_dict(),
               benign_share=float(tot.get("Benign", 0) / total_rows))
fig, ax = plt.subplots(figsize=(9, 0.42 * len(tot) + 1.4))
ax.barh(tot.index, tot.values, color=[FAM_COLOR[family(l)] for l in tot.index], height=0.62)
ax.set_xscale("log"); ax.set_xlim(left=max(1, tot.min() * 0.4), right=tot.max() * 6)
for y, v in enumerate(tot.values):
    ax.text(v * 1.12, y, f"{fmt(v)}  ({v / total_rows:.2%})", va="center", color=INK2, fontsize=9)
ax.set_xlabel("Số luồng (thang log)"); ax.grid(axis="y", visible=False)
ax.set_title(f"Phân bố nhãn trên toàn bộ dữ liệu ({fmt(total_rows)} luồng)")
handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOR[f]) for f in FAMILIES if f in {family(l) for l in tot.index}]
ax.legend(handles, [f for f in FAMILIES if f in {family(l) for l in tot.index}],
          loc="lower right", frameon=False, title="Nhóm")
save(fig, "01_label_dist.png")

# ---------------- 2. Nhãn theo ngày (heatmap có số) ----------------
tab = pd.DataFrame(STATS["label_by_file"]).T.reindex(order).fillna(0).astype(int)
tab = tab[sorted(tab.columns, key=lambda l: (FAMILIES.index(family(l)), l))]
fig, ax = plt.subplots(figsize=(1.0 * len(tab.columns) + 2.5, 0.55 * len(tab) + 2.2))
with np.errstate(divide="ignore"):
    z = np.log10(tab.values + 1)
ax.imshow(z, cmap=SEQ, aspect="auto")
ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels(tab.columns, rotation=40, ha="right")
ax.set_yticks(range(len(tab))); ax.set_yticklabels([short_wd[d] for d in tab.index])
ax.grid(False)
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        v = tab.values[i, j]
        if v:
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                    color="white" if z[i, j] > 0.5 * z.max() else INK)
ax.set_title("Số luồng theo ngày và nhãn (màu theo thang log, ô trống = không có)")
save(fig, "02_label_by_day.png")
summary["label_by_day"] = {short[d]: {k: int(v) for k, v in tab.loc[d].items() if v} for d in order}

# ---------------- 3. Giá trị lỗi theo cột ----------------
cs = pd.DataFrame(STATS["column_stats"]).T
for c in ("nan", "inf", "neg", "n"):
    cs[c] = cs[c].astype(float)
rate = pd.DataFrame({"NaN": cs["nan"] / cs["n"], "inf": cs["inf"] / cs["n"], "Giá trị âm": cs["neg"] / cs["n"]})
bad = rate[(rate > 0).any(axis=1)]
bad = bad.loc[bad.max(axis=1).sort_values(ascending=False).index]
summary["cols_with_nan"] = {c: int(cs.loc[c, "nan"]) for c in cs.index if cs.loc[c, "nan"] > 0}
summary["cols_with_inf"] = {c: int(cs.loc[c, "inf"]) for c in cs.index if cs.loc[c, "inf"] > 0}
summary["cols_with_negative"] = {c: int(cs.loc[c, "neg"]) for c in cs.index if cs.loc[c, "neg"] > 0}
summary["constant_columns"] = [c for c in cs.index if cs.loc[c, "min"] is not None
                               and not pd.isna(cs.loc[c, "min"]) and cs.loc[c, "min"] == cs.loc[c, "max"]]
top = bad.head(14)
if len(top):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(top) + 1.6))
    y = np.arange(len(top)); w = 0.26
    for k, (col, color) in enumerate(zip(["NaN", "inf", "Giá trị âm"], [FAM_COLOR["Benign"], FAM_COLOR["DoS"], FAM_COLOR["DDoS"]])):
        ax.barh(y + (k - 1) * w, top[col] * 100, height=w * 0.9, color=color, label=col)
    ax.set_yticks(y); ax.set_yticklabels(top.index); ax.invert_yaxis()
    xmax = max(top.max().max() * 100, 1)
    for i, col_name in enumerate(top.index):
        txt = " · ".join(f"{lab} {fmt(cs.loc[col_name, key])}" for lab, key in
                         (("NaN", "nan"), ("inf", "inf"), ("âm", "neg")) if cs.loc[col_name, key] > 0)
        ax.text(xmax * 0.58, i, txt, va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("% số dòng của cột bị lỗi (số dòng thực tế ghi bên phải)"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Các cột có giá trị lỗi ({len(bad)} trên {len(cs)} cột bị ảnh hưởng)")
    save(fig, "03_bad_values.png")

# ---------------- 4. Trùng lặp theo ngày ----------------
dup = pd.Series({d: STATS["files"][d]["duplicate_rows_within_file"] / STATS["files"][d]["rows"] for d in order})
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar([short_wd[d] for d in dup.index], dup.values * 100, color=BLUE, width=0.6)
for i, v in enumerate(dup.values):
    ax.text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", color=INK2, fontsize=9)
ax.set_ylabel("% dòng trùng trong cùng file"); ax.grid(axis="x", visible=False)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_title("Tỉ lệ dòng trùng lặp theo ngày")
save(fig, "04_duplicates.png")
summary["rows_by_day"] = {short[d]: STATS["files"][d]["rows"] for d in order}
summary["duplicates_by_day"] = {short[d]: STATS["files"][d]["duplicate_rows_within_file"] for d in order}
summary["total_duplicates"] = int(sum(summary["duplicates_by_day"].values()))
summary["extra_columns_by_file"] = STATS["extra_columns_by_file"]
summary["n_columns_common"] = STATS["n_columns_common"]
summary["bad_timestamps"] = {short[d]: STATS["files"][d]["bad_timestamps"] for d in order}
summary["timestamp_check"] = json.loads(Path("reports/data/timestamp_check.json").read_text())

# ---------------- 5. Tương quan ----------------
meta = {"Label", "Timestamp", "day"}
feat = [c for c in SAMPLE.columns if c not in meta]
X = SAMPLE[feat].replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.notna().mean() > 0.9]
X = X.loc[:, X.nunique() > 1]
corr = X.corr(min_periods=1000)
fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("Hệ số tương quan Pearson")
ax.set_title(f"Tương quan giữa {corr.shape[0]} đặc trưng (trên mẫu phân tầng {fmt(len(SAMPLE))} dòng)")
save(fig, "05_correlation.png")
iu = np.triu_indices_from(corr, k=1)
pairs = pd.DataFrame({"a": corr.index[iu[0]], "b": corr.columns[iu[1]], "r": corr.values[iu]}).dropna()
hi = pairs[pairs["r"].abs() >= 0.99].sort_values("r", key=abs, ascending=False)
summary["n_features_in_corr"] = int(corr.shape[0])
summary["n_pairs_abs_corr_ge_0.99"] = int(len(hi))
summary["top_corr_pairs"] = hi.head(15).round(4).to_dict("records")

# ---------------- 6. Benign vs Attack trên vài đặc trưng ----------------
is_att = (SAMPLE["Label"] != "Benign")
wanted = ["Flow Duration", "Tot Fwd Pkts", "Fwd Pkt Len Mean", "Flow Pkts/s", "Init Fwd Win Byts", "Pkt Len Mean"]
cols = [c for c in wanted if c in SAMPLE.columns][:6]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
for ax, c in zip(axes.ravel(), cols):
    v = SAMPLE[c].replace([np.inf, -np.inf], np.nan)
    data = [np.sign(v[~is_att].dropna()) * np.log1p(np.abs(v[~is_att].dropna())),
            np.sign(v[is_att].dropna()) * np.log1p(np.abs(v[is_att].dropna()))]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.5))
    for patch, col in zip(bp["boxes"], [FAM_COLOR["Benign"], FAM_COLOR["DoS"]]):
        patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(0.85)
    ax.set_xticklabels(["Benign", "Attack"]); ax.set_title(c, fontsize=10); ax.grid(axis="x", visible=False)
    ax.set_ylabel("sign·log(1+|x|)", fontsize=8)
fig.suptitle("Benign và Attack trên một số đặc trưng (mẫu phân tầng, thang log đối xứng)",
             x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
save(fig, "06_benign_vs_attack.png")

# ---------------- 7. Lưu lượng theo giờ ----------------
# Timestamp chỉ có giờ 1..12 (đồng hồ 12h, mất AM/PM). Suy luận: giờ 1-7 là buổi chiều (+12), 8-12 giữ nguyên.
H = pd.DataFrame([(k.split("|")[0], int(k.split("|")[1]), k.split("|")[2], v)
                  for k, v in STATS["label_by_hour"].items()], columns=["day", "hour", "label", "n"])
H["hour24"] = np.where(H["hour"] <= 7, H["hour"] + 12, H["hour"])
H["grp"] = np.where(H["label"] == "Benign", "Benign", "Attack")
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
HRS = list(range(8, 20))
for ax, d in zip(axes.ravel(), order):
    g = H[H.day == d].groupby(["hour24", "grp"])["n"].sum().unstack(fill_value=0).reindex(HRS, fill_value=0)
    for grp, color in (("Benign", FAM_COLOR["Benign"]), ("Attack", FAM_COLOR["DoS"])):
        if grp in g:
            ax.plot(g.index, g[grp].replace(0, np.nan), color=color, linewidth=1.6,
                    marker="o", markersize=4, label=grp)
    ax.set_yscale("log"); ax.set_title(short_wd[d], fontsize=10); ax.set_xticks([8, 12, 16, 19])
for ax in axes[1]:
    ax.set_xlabel("Giờ (suy luận 24h)")
for ax in axes[:, 0]:
    ax.set_ylabel("Số luồng (log)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Số luồng theo giờ: Benign và Attack (giờ 1-7 trong dữ liệu được hiểu là buổi chiều, +12)",
             x=0.01, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
save(fig, "07_traffic_by_hour.png")

# attack theo giờ: giờ đầu/cuối có tấn công mỗi ngày
att = H[H.grp == "Attack"].groupby("day")["hour24"].agg(["min", "max"])
summary["attack_hours_by_day"] = {short[d]: [int(att.loc[d, "min"]), int(att.loc[d, "max"])] for d in order if d in att.index}

Path("reports/data/eda_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
print("summary saved")
