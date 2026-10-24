"""Draw the README charts from saved results (no model requests).

python scripts/make_charts.py  ->  docs/images/*.png
Reads results/version_cleanup/measurement_20261012.json and results/no_evidence/live_*.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/images"
IMG.mkdir(parents=True, exist_ok=True)

for path in ("C:/Windows/Fonts/msyh.ttc", "/System/Library/Fonts/PingFang.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
        break

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, MUTED = "#2a78d6", "#b9b8b1"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# 1. Chunks kept in Chroma after each upload of a new version of the same file
cleanup = json.loads((ROOT / "results/version_cleanup/measurement_20261012.json").read_text(encoding="utf-8"))["results"]
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
series = [("before", "改进前", MUTED), ("after", "版本化入库", ACCENT)]
for phase, label, color in series:
    rows = cleanup[phase]["records"]
    x = [row["version"] for row in rows]
    y = [row["stored_chunks"] for row in rows]
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    stale = cleanup[phase]["stale_chunks"]
    ax.annotate(f"{label}：库里 {y[-1]} 个片段，其中过期 {stale} 个", (x[-1], y[-1]), xytext=(-6, 8 if phase == "before" else 8),
                textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK if phase == "after" else INK2)
ax.set_xticks(range(1, 11))
ax.set_xlim(0.6, 10.4)
ax.set_ylim(0, 11.5)
ax.set_xlabel("同名文件上传到第几个版本")
ax.set_ylabel("向量库中该文件的片段数")
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
tidy(ax)
ax.set_title("同一文件更新 10 次后，旧版本片段还留在库里吗", loc="left", color=INK, fontsize=11.5, pad=10)
fig.text(0.01, 0.01, "本地 Chroma + 离线哈希向量，每个版本 1 个片段。数据：results/version_cleanup/measurement_20261012.json",
         color=INK2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(IMG / "version_cleanup.png")
plt.close(fig)

# 2. Model calls with and without evidence, before and after the short circuit
runs = {name: json.loads((ROOT / f"results/no_evidence/live_{name}.json").read_text(encoding="utf-8")) for name in ("before", "after")}
count = lambda run, kind: sum(len(c["calls"]) for c in run["cases"] if c["kind"] == kind)
labels = ["改进前", "无证据短路"]
empty = [count(runs[n], "no_evidence") for n in ("before", "after")]
grounded = [count(runs[n], "with_evidence") for n in ("before", "after")]
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
y = [0, 1]
ax.barh(y, grounded, height=0.5, color=MUTED, label="有证据的 4 例")
ax.barh(y, empty, left=grounded, height=0.5, color=ACCENT, label="无证据的 6 例（空检索或空白片段）",
        edgecolor=SURFACE, linewidth=2)
for i in y:
    ax.text(grounded[i] / 2, i, str(grounded[i]), ha="center", va="center", color=INK, fontsize=9.5)
    if empty[i]:
        ax.text(grounded[i] + empty[i] / 2, i, str(empty[i]), ha="center", va="center", color="#ffffff", fontsize=9.5)
    ax.text(grounded[i] + empty[i] + 0.15, i, f"共 {grounded[i] + empty[i]} 次", va="center", color=INK, fontsize=9.5)
ax.set_yticks(y, labels, color=INK)
ax.set_xlim(0, 11.5)
ax.set_xticks([])  # every segment carries its own count
ax.invert_yaxis()
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
tidy(ax)
ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncols=2, bbox_to_anchor=(0, -0.02))
ax.set_title("10 例固定检索输入中，回答模型被调用的次数", loc="left", color=INK, fontsize=11.5, pad=10)
fig.text(0.01, 0.01, "qwen-plus，temperature 0，2026-10-09。数据：results/no_evidence/live_before.json、live_after.json",
         color=INK2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(IMG / "no_evidence_calls.png")
plt.close(fig)
print("charts ->", IMG)
