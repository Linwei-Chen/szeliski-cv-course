#!/usr/bin/env python3
"""E127 数据集/基准/统计口径 lab：划分泄漏、均值陷阱、不确定度、数据配比。"""
import json
import pathlib
import hashlib
import datetime
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"
INK = "#2B2B2B"
ACCENT = "#C24E2A"
BLUE = "#3B6EA5"
GREEN = "#4F7A4A"
GRID = "#D8CDBB"

for cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = PAPER
plt.rcParams["axes.facecolor"] = PAPER
plt.rcParams["savefig.facecolor"] = PAPER
plt.rcParams["text.color"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.edgecolor"] = INK

print(f"[lab] E127 数据集/基准 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成数据/合成分数；不是真实 benchmark 复现；教材只给评测原则")
checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))

# ── 实验一：随机划分 vs 按场景划分 ─────────────────────────────
rng = np.random.default_rng(127)
n_scenes, n_per = 8, 40
offsets = np.linspace(-2.1, 2.1, n_scenes)
X, Y, S = [], [], []
for s, mu in enumerate(offsets):
    for k in range(n_per):
        y = k % 2
        X.append(mu + (2 * y - 1) * 1.2 + rng.normal(0, 0.40))
        Y.append(y); S.append(s)
X = np.asarray(X); Y = np.asarray(Y); S = np.asarray(S)

# 每场景阈值 = 训练集中两类均值的平均；未见场景回退到全局阈值
def fit_thresholds(mask):
    per = {}
    for s in np.unique(S[mask]):
        m = mask & (S == s)
        if len(np.unique(Y[m])) == 2:
            per[int(s)] = (X[m & (Y == 0)].mean() + X[m & (Y == 1)].mean()) / 2
    glob = (X[mask & (Y == 0)].mean() + X[mask & (Y == 1)].mean()) / 2
    return per, glob

def predict(mask_train, mask_test):
    per, glob = fit_thresholds(mask_train)
    th = np.array([per.get(int(s), glob) for s in S[mask_test]])
    return (X[mask_test] > th).astype(int)

# 随机 70/30（同场景内也随机）
idx = rng.permutation(len(X)); cut = int(0.7 * len(X))
tr = np.zeros(len(X), bool); tr[idx[:cut]] = True; te = ~tr
acc_random = float((predict(tr, te) == Y[te]).mean())
# 按场景留出：前 6 场景训练，后 2 场景测试
tr2 = S < 6; te2 = ~tr2
acc_disjoint = float((predict(tr2, te2) == Y[te2]).mean())
leak_gap = acc_random - acc_disjoint
check("随机划分准确率 ≥ 0.95", acc_random >= 0.95, f"{acc_random:.3f}")
check("留出场景准确率 ≤ 0.60", acc_disjoint <= 0.60, f"{acc_disjoint:.3f}")
check("泄漏差值 ≥ 0.35", leak_gap >= 0.35, f"{leak_gap:.3f}")

# ── 实验二：均值陷阱与自助置信区间 ───────────────────────────
A = np.array([0.99, 0.99, 0.99, 0.55, 0.55, 0.55, 0.55, 0.55])
B = np.array([0.70] * 8)
mean_A, mean_B = float(A.mean()), float(B.mean())
med_A, med_B = float(np.median(A)), float(np.median(B))
worst_A, worst_B = float(A.min()), float(B.min())
wins_A = int((A > B).sum()); wins_B = int((B > A).sum())
n_boot = 4000
boot = np.empty(n_boot)
for i in range(n_boot):
    pick = rng.integers(0, len(A), len(A))
    boot[i] = (A[pick] - B[pick]).mean()
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
check("方法 A 的平均分更高", mean_A > mean_B, f"{mean_A:.3f} > {mean_B:.3f}")
check("方法 B 的中位数更高", med_B > med_A, f"{med_B:.3f} > {med_A:.3f}")
check("方法 B 的最差场景更好", worst_B > worst_A, f"{worst_B:.3f} > {worst_A:.3f}")
check("A−B 均值差的 95% CI 含 0", ci_lo <= 0 <= ci_hi, f"[{ci_lo:+.3f}, {ci_hi:+.3f}]")
check("B 赢更多场景（≥5/8）", wins_B >= 5, f"B {wins_B} vs A {wins_A}")

# ── 实验三：数据配比改变冠军 ─────────────────────────────────
indoor_A, outdoor_A = 0.90, 0.60
indoor_B, outdoor_B = 0.75, 0.80
def overall(w_indoor, a_in, a_out, b_in, b_out):
    return (w_indoor * a_in + (1 - w_indoor) * a_out,
            w_indoor * b_in + (1 - w_indoor) * b_out)
score_A_50, score_B_50 = overall(0.50, indoor_A, outdoor_A, indoor_B, outdoor_B)
score_A_75, score_B_75 = overall(0.75, indoor_A, outdoor_A, indoor_B, outdoor_B)
cross_w = (outdoor_B - outdoor_A) / ((indoor_A - outdoor_A) - (indoor_B - outdoor_B))
check("五五配比冠军 = B", score_B_50 > score_A_50, f"B {score_B_50:.3f} > A {score_A_50:.3f}")
check("七成室内冠军 = A", score_A_75 > score_B_75, f"A {score_A_75:.3f} > B {score_B_75:.3f}")
check("冠军翻转点落在 0.5–0.75", 0.5 < cross_w < 0.75, f"{cross_w:.3f}")
protocol = dict(split_by_scene=True, test_scenes=[6, 7], n_scenes=n_scenes,
                n_boot=n_boot, metrics=["accuracy"], primary="mean+median+worst")
check("协议字段齐全且按场景留出", protocol["split_by_scene"] and protocol["test_scenes"] == [6, 7], str(protocol))

# ── 图1：划分泄漏 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
ax = axes[0]
colors = [BLUE if s < 6 else ACCENT for s in S]
ax.scatter(X[Y == 0], S[Y == 0] + rng.normal(0, 0.05, (Y == 0).sum()), s=18, c=[c for c, y in zip(colors, Y) if y == 0], alpha=0.65, edgecolors="none")
ax.scatter(X[Y == 1], S[Y == 1] + rng.normal(0, 0.05, (Y == 1).sum()), s=18, c=[c for c, y in zip(colors, Y) if y == 1], alpha=0.65, edgecolors="none", marker="^")
for s in range(n_scenes):
    ax.axhline(s, color=GRID, lw=0.6, zorder=0)
ax.set_yticks(range(n_scenes)); ax.set_yticklabels([f"场景 {s+1}" for s in range(n_scenes)], fontsize=17)
ax.set_xlabel("特征值 x（圆=类 0，三角=类 1）", fontsize=18)
ax.set_title("前 6 场景训练，后 2 场景留出", fontsize=19)
ax.text(0.98, 0.02, "红=留出场景", transform=ax.transAxes, ha="right", va="bottom", fontsize=17, color=ACCENT)
ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6)
ax = axes[1]
bars = ax.bar(["随机划分", "按场景留出"], [acc_random, acc_disjoint], color=[BLUE, ACCENT], width=0.55)
for b, v in zip(bars, [acc_random, acc_disjoint]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=19)
ax.set_ylim(0, 1.12); ax.set_ylabel("测试准确率", fontsize=18)
ax.set_title("同一模型，两种划分", fontsize=19)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.suptitle("随机划分把同一场景泄漏进测试集", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e127_split.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ── 图2：均值/中位数/最差 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
ax = axes[0]
xs = np.arange(len(A))
ax.plot(xs, A, "o-", color=ACCENT, lw=2.2, ms=8, label="方法 A（三次大胜）")
ax.plot(xs, B, "s-", color=BLUE, lw=2.2, ms=7, label="方法 B（八次稳定）")
ax.set_xticks(xs); ax.set_xticklabels([f"S{i+1}" for i in xs], fontsize=17)
ax.set_ylim(0.45, 1.04); ax.set_ylabel("场景得分", fontsize=18)
ax.set_title("逐场景：A 大起大落，B 平稳", fontsize=19)
ax.legend(fontsize=17, frameon=False, loc="lower left")
ax.grid(color=GRID, lw=0.6, alpha=0.6)
ax = axes[1]
labels = ["平均", "中位数", "最差"]
a_vals = [mean_A, med_A, worst_A]; b_vals = [mean_B, med_B, worst_B]
xp = np.arange(len(labels)); w = 0.34
b1 = ax.bar(xp - w/2, a_vals, w, color=ACCENT, label="方法 A")
b2 = ax.bar(xp + w/2, b_vals, w, color=BLUE, label="方法 B")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=17)
ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=17)
ax.set_ylim(0, 1.05); ax.set_ylabel("得分", fontsize=18)
ax.set_title("只看平均，会漏掉灾难场景", fontsize=19)
ax.legend(fontsize=17, frameon=False, loc="upper right")
ax.text(0.02, 0.98, f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}] 含 0", transform=ax.transAxes, va="top", fontsize=17, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.suptitle("A−B 均值差 95% CI 含 0；中位数与最差都站在 B", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e127_uncertainty.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ── 图3：数据配比改变冠军 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11.6, 4.8))
ws = np.linspace(0, 1, 101)
a_scores = np.array([overall(w, indoor_A, outdoor_A, indoor_B, outdoor_B)[0] for w in ws])
b_scores = np.array([overall(w, indoor_A, outdoor_A, indoor_B, outdoor_B)[1] for w in ws])
ax.plot(ws * 100, a_scores, color=ACCENT, lw=2.6, label="方法 A：室内强、室外弱")
ax.plot(ws * 100, b_scores, color=BLUE, lw=2.6, label="方法 B：室内弱、室外强")
ax.axvline(cross_w * 100, color=GREEN, ls="--", lw=1.8)
ax.text(cross_w * 100 + 2, 0.845, f"翻转点 {cross_w*100:.1f}%", color=GREEN, fontsize=17)
for w, tag in [(50, "五五"), (75, "七成室内")]:
    ax.axvline(w, color=GRID, lw=1.0, ls=":")
    ax.text(w, 0.585, tag, ha="center", fontsize=17, color=INK)
ax.set_xlabel("测试集中室内场景的比例（%）", fontsize=18)
ax.set_ylabel("总体得分", fontsize=18)
ax.set_title("同一对方法，换数据配比就换冠军", fontsize=20)
ax.set_xlim(0, 100); ax.set_ylim(0.58, 0.93)
ax.legend(fontsize=17, frameon=False, loc="upper left")
ax.grid(color=GRID, lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(FIG / "e127_composition.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ── 汇总 ────────────────────────────────────────────────────
n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
readings = dict(
    n_pass=n_pass, n_total=len(checks),
    acc_random=acc_random, acc_disjoint=acc_disjoint, leak_gap=leak_gap,
    mean_A=mean_A, mean_B=mean_B, med_A=med_A, med_B=med_B,
    worst_A=worst_A, worst_B=worst_B, wins_A=wins_A, wins_B=wins_B,
    ci_lo=float(ci_lo), ci_hi=float(ci_hi),
    score_A_50=score_A_50, score_B_50=score_B_50,
    score_A_75=score_A_75, score_B_75=score_B_75,
    cross_w=float(cross_w), protocol=protocol,
    fig_sizes={"split": [12.6, 4.7], "uncertainty": [12.6, 4.8], "composition": [11.6, 4.8]},
)
(ROOT / "outputs" / "e127_benchmark_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
for f in ("e127_split.png", "e127_uncertainty.png", "e127_composition.png"):
    print(f"[lab] PNG sha256({f})={hashlib.sha256((FIG/f).read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((ROOT/'outputs'/'e127_benchmark_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
