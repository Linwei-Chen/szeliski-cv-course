#!/usr/bin/env python3
"""E118 榜单阅读数值沙盘：同一张误差图，换指标/掩码/数据划分就换名次。

边界声明：本 lab 只用合成误差图演示评测协议，不冒充 Middlebury/KITTI 官方统计；
Table 12.1 的训练/测试规模是书页事实，真值覆盖率是讲者沙盘假设。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"

H, W = 120, 160
N = H * W
yy, xx = np.mgrid[0:H, 0:W]
border = (yy < 8) | (yy >= H - 8) | (xx < 8) | (xx >= W - 8)
occlusion = (xx >= 80) & (xx < 96)
interior = ~(border | occlusion)

def pick(mask, k):
    idx = np.flatnonzero(mask.ravel())
    return idx[np.linspace(0, len(idx) - 1, k).astype(int)]

# A：均匀小错；B：少量边界/遮挡大错；C：稀疏有效区
err_A = np.full((H, W), 0.2)
for k, m in ((1920, border), (1920, occlusion), (1920, interior)):
    err_A.ravel()[pick(m, k)] = 1.2
err_B = np.full((H, W), 0.1)
for k, m in ((1152, border), (768, occlusion)):
    err_B.ravel()[pick(m, k)] = 5.0
rng = np.random.default_rng(118)
valid_C = np.zeros(N, dtype=bool); valid_C[rng.choice(N, 5760, replace=False)] = True
valid_C = valid_C.reshape(H, W)
err_C = np.where(valid_C, 0.1, np.nan)

def bad(err, mask, t): return float(np.mean(err[mask] > t))
def rmse(err, mask): return float(np.sqrt(np.mean(err[mask] ** 2)))
def pct(x): return f"{100 * x:.1f}%"

coverage_dense = 1.0
coverage_sparse = float(valid_C.mean())
bad1_A, bad1_B = bad(err_A, np.ones_like(border), 1.0), bad(err_B, np.ones_like(border), 1.0)
bad2_A, bad2_B = bad(err_A, np.ones_like(border), 2.0), bad(err_B, np.ones_like(border), 2.0)
rmse_A, rmse_B = rmse(err_A, np.ones_like(border)), rmse(err_B, np.ones_like(border))
border_bad1_A, border_bad1_B = bad(err_A, ~border, 1.0), bad(err_B, ~border, 1.0)
occl_bad1_A, occl_bad1_B = bad(err_A, ~occlusion, 1.0), bad(err_B, ~occlusion, 1.0)
sparse_valid_bad1 = bad(err_C, valid_C, 1.0)
sparse_missing_full = float(np.mean(~valid_C))
# 同一阈值 1.5：测试集上 A 赢，换域后 B 赢
err_A_test = np.where(err_A > 1.1, 1.2, 0.2)
err_A_hold = np.where(err_A > 1.1, 2.2, 0.2)
test_A_bad, test_B_bad = bad(err_A_test, np.ones_like(border), 1.5), bad(err_B, np.ones_like(border), 1.5)
hold_A_bad, hold_B_bad = bad(err_A_hold, np.ones_like(border), 1.5), bad(err_B, np.ones_like(border), 1.5)
domain_before, domain_after = 0.20, 1.20
domain_factor = domain_after / domain_before

checks = [
 ("稠密沙盘真值覆盖 100.0%", coverage_dense == 1.0, "dense"),
 ("稀疏沙盘真值覆盖 30.0%", abs(coverage_sparse - 0.3) < 1e-12, "sparse"),
 (f"阈值 1 像素：B {pct(bad1_B)} 优于 A {pct(bad1_A)}", bad1_B < bad1_A, "bad1"),
 (f"阈值 2 像素：A {pct(bad2_A)} 优于 B {pct(bad2_B)}（排名翻转）", bad2_A < bad2_B, "bad2"),
 (f"RMSE：A {rmse_A:.3f} 优于 B {rmse_B:.3f}", rmse_A < rmse_B, "rmse"),
 (f"去边界：B {pct(border_bad1_B)} 优于 A {pct(border_bad1_A)}", border_bad1_B < border_bad1_A, "border"),
 (f"去遮挡：B {pct(occl_bad1_B)} 优于 A {pct(occl_bad1_A)}", occl_bad1_B < occl_bad1_A, "occl"),
 (f"稀疏预测 C：有效区 bad1 {pct(sparse_valid_bad1)}，覆盖 {pct(coverage_sparse)}，全图缺失 {pct(sparse_missing_full)}", sparse_valid_bad1 == 0.0 and abs(coverage_sparse - 0.3) < 1e-12 and abs(sparse_missing_full - 0.7) < 1e-12, "sparseC"),
 (f"测试集调阈值 1.5：A {pct(test_A_bad)} vs B {pct(test_B_bad)}；换域后 B {pct(hold_B_bad)} vs A {pct(hold_A_bad)}", test_A_bad < test_B_bad and hold_B_bad < hold_A_bad, "tune"),
 (f"跨域误差放大 {domain_factor:.1f} 倍 | {domain_before:.2f} → {domain_after:.2f}", abs(domain_factor - 6.0) < 1e-12, "domain"),
]

print(f"[lab] E118 榜单沙盘开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成误差图；Table 12.1 规模=书页事实，真值覆盖率=沙盘假设")
n_pass = 0
for text, ok, _key in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    n_pass += bool(ok)

# 图1 五个代表基准的规模
names = ["Middlebury stereo", "KITTI 2015", "DTU", "ETH3D", "Freiburg Scene Flow"]
vals = [33, 400, 124, 25, 39000]
fig, ax = plt.subplots(figsize=(11.2, 5.8))
ypos = np.arange(len(names))[::-1]
bars = ax.barh(ypos, vals, color=[BLUE, ACCENT, GREEN, "#6b3fa0", GRAY], height=0.58)
ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=17)
ax.set_xscale("log"); ax.set_xlim(20, 80000)
ax.set_xlabel("图像对/场景数（对数轴）", fontsize=17)
ax.tick_params(axis="x", labelsize=14.5)
for y, v in zip(ypos, vals):
    ax.text(v * 1.12, y, f"{v:,}", va="center", fontsize=16, color=INK)
ax.set_title("表 12.1 里挑五个代表：规模差三个数量级", fontsize=20, color=INK, pad=12)
ax.text(0.0, -0.18, "KITTI 2015 = 200 训练 + 200 测试；ETH3D = 13 + 12；DTU = 124 场景 × 49–64 图；Scene Flow = 39k 合成对",
        transform=ax.transAxes, fontsize=14.5, color=GRAY)
ax.grid(axis="x", alpha=0.25, which="both")
fig.tight_layout(); fig.savefig(FIG / "e118_benchmarks.png", dpi=160, facecolor=PAPER, bbox_inches="tight", pad_inches=0.22); plt.close(fig)

# 图2 误差分布 + 阈值排名翻转
fig = plt.figure(figsize=(11.6, 5.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
for err, color, lab, marker in [(err_A, BLUE, "A：均匀小错", "o"), (err_B, ACCENT, "B：边界/遮挡大错", "s")]:
    vals, counts = np.unique(np.round(err.ravel(), 3), return_counts=True)
    frac = counts / N
    ax.vlines(vals, 1e-3, frac, color=color, lw=2.0, alpha=0.85)
    ax.plot(vals, frac, marker, color=color, ms=10, label=lab)
    for xv, yv in zip(vals, frac):
        if yv < 0.5:
            if xv > 4:
                ax.text(xv - 0.10, yv, f"{100*yv:.1f}%", ha="right", va="center", fontsize=14, color=color)
            else:
                ax.text(xv + 0.10, yv, f"{100*yv:.1f}%", ha="left", va="center", fontsize=14, color=color)
ax.axvline(1.0, color=GRAY, ls="--", lw=1.2); ax.axvline(2.0, color=GRAY, ls=":", lw=1.2)
ax.text(1.0, 2.2e-3, "阈值 1", ha="center", va="bottom", fontsize=12.5, color=GRAY)
ax.text(2.0, 2.2e-3, "阈值 2", ha="center", va="bottom", fontsize=12.5, color=GRAY)
ax.set_yscale("log"); ax.set_ylim(1e-3, 1.15)
ax.set_xlim(0, 5.4); ax.set_xlabel("绝对误差 / 像素", fontsize=16)
ax.set_ylabel("像素比例（对数）", fontsize=16)
ax.tick_params(labelsize=14); ax.grid(axis="y", alpha=0.22, which="both")
ax.legend(fontsize=13.5, frameon=False, loc="lower right")
ax.set_title("两条误差分布：A 多中等错，B 少但大错", fontsize=16.5, color=INK)

ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(2); w = 0.34
ax2.bar(x - w/2, [bad1_A, bad1_B], w, color=[BLUE, ACCENT], label="bad-1")
ax2.bar(x + w/2, [bad2_A, bad2_B], w, color=[BLUE, ACCENT], alpha=0.45, label="bad-2")
ax2.set_xticks(x); ax2.set_xticklabels(["A 均匀小错", "B 边界/遮挡大错"], fontsize=14)
ax2.set_ylabel("坏像素比例", fontsize=16); ax2.set_ylim(0, 0.40)
ax2.tick_params(axis="y", labelsize=14)
ax2.text(0.02, 0.96, f"bad-1：B {100*bad1_B:.1f}% < A {100*bad1_A:.1f}%", transform=ax2.transAxes, fontsize=13.5, color=ACCENT, weight="bold")
ax2.text(0.02, 0.88, f"bad-2：A {100*bad2_A:.1f}% < B {100*bad2_B:.1f}%", transform=ax2.transAxes, fontsize=13.5, color=BLUE, weight="bold")
ax2.text(0.02, 0.80, f"RMSE：A {rmse_A:.2f} < B {rmse_B:.2f}", transform=ax2.transAxes, fontsize=13.5, color=GRAY)
ax2.legend(fontsize=13, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.72))
ax2.grid(axis="y", alpha=0.22)
ax2.set_title("同一对预测，换阈值就换名次", fontsize=16.5, color=INK)
fig.suptitle("误差分布与排名：bad-1 与 bad-2 可以给出相反结论", fontsize=19, color=INK, y=0.99)
fig.savefig(FIG / "e118_metrics.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 读榜五问
fig, ax = plt.subplots(figsize=(11.2, 6.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
ax.text(0.35, 6.15, "读榜五问：先问协议，再看名次", fontsize=22, color=INK, weight="bold")
qs = [
 ("1", "数据集与划分？", "训练/测试是否真的分开，测试集有没有被反复调参"),
 ("2", "真值覆盖多少？", "稠密、稀疏还是带洞；缺失像素被算进分母了吗"),
 ("3", "指标与阈值？", "bad-1、bad-2、RMSE、MAE，谁对大错更敏感"),
 ("4", "遮挡与边界怎么算？", "同一张预测，换掩码就能换冠军"),
 ("5", "跨域还能用吗？", "同分布榜单第一，不等于真实场景第一"),
]
ys = [5.15, 4.05, 2.95, 1.85, 0.75]
for (num, head, body), y in zip(qs, ys):
    ax.add_patch(Rectangle((0.55, y), 9.9, 0.82, facecolor="#fffaf0", edgecolor="#b8a98a", lw=1.3))
    ax.add_patch(Circle((1.02, y + 0.41), 0.24, color=BLUE))
    ax.text(1.02, y + 0.41, num, ha="center", va="center", color="white", fontsize=16, weight="bold")
    ax.text(1.45, y + 0.52, head, fontsize=17, color=INK, weight="bold", va="center")
    ax.text(1.45, y + 0.21, body, fontsize=14.2, color=GRAY, va="center")
    if y > 0.8:
        ax.add_patch(FancyArrowPatch((10.05, y), (10.05, y - 0.28), arrowstyle="->", color=GRAY, lw=1.4, mutation_scale=12))
ax.text(0.55, 0.25, "顺序不是装饰：任何一问没答，名次都不能直接横比。", fontsize=15, color=ACCENT)
fig.tight_layout(); fig.savefig(FIG / "e118_reader.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = {
 "coverage_dense": coverage_dense, "coverage_sparse": coverage_sparse,
 "bad1_A": bad1_A, "bad1_B": bad1_B, "bad2_A": bad2_A, "bad2_B": bad2_B,
 "rmse_A": rmse_A, "rmse_B": rmse_B,
 "border_bad1_A": border_bad1_A, "border_bad1_B": border_bad1_B,
 "occl_bad1_A": occl_bad1_A, "occl_bad1_B": occl_bad1_B,
 "sparse_valid_bad1": sparse_valid_bad1, "sparse_missing_full": sparse_missing_full,
 "test_A_bad": test_A_bad, "test_B_bad": test_B_bad,
 "hold_A_bad": hold_A_bad, "hold_B_bad": hold_B_bad,
 "domain_before": domain_before, "domain_after": domain_after, "domain_factor": domain_factor,
 "n_pass": n_pass, "n_total": len(checks),
}
(OUT / "e118_bench_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] 图1 规模：Middlebury 33｜KITTI 200+200｜DTU 124×49–64｜ETH3D 13+12｜SceneFlow 39k")
print(f"[lab] 指标：bad1 A {pct(bad1_A)} / B {pct(bad1_B)}；bad2 A {pct(bad2_A)} / B {pct(bad2_B)}；RMSE A {rmse_A:.3f} / B {rmse_B:.3f}")
print(f"[lab] 掩码：去边界 A {pct(border_bad1_A)} / B {pct(border_bad1_B)}；去遮挡 A {pct(occl_bad1_A)} / B {pct(occl_bad1_B)}")
print(f"[lab] 稀疏 C：有效区 bad1 {pct(sparse_valid_bad1)}｜覆盖 {pct(coverage_sparse)}｜全图缺失 {pct(sparse_missing_full)}")
print(f"[lab] 换域：测试 A {pct(test_A_bad)} vs B {pct(test_B_bad)}｜同阈值 holdout B {pct(hold_B_bad)} vs A {pct(hold_A_bad)}｜放大 {domain_factor:.1f}×")
for name in ("e118_benchmarks.png", "e118_metrics.png", "e118_reader.png"):
    print(f"[lab] PNG sha256({name})={hashlib.sha256((FIG/name).read_bytes()).hexdigest()}")
print(f"[lab] readings → e118_bench_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e118_bench_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
