#!/usr/bin/env python3
"""E084 §9.1.1–§9.1.3：金字塔、相位相关、增量细化（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift, map_coordinates

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261104)

N = 256
BASE = 0.05 + 0.90 * (lambda t: (t - t.min()) / (t.max() - t.min()))(gaussian_filter(RNG.normal(0, 1, (N, N)), 1.3))
TRUE_SUB = (23.35, -17.45)  # (x, y)
TARGET = ndshift(BASE, shift=(TRUE_SUB[1], TRUE_SUB[0]), order=3, mode="nearest")
TRUE_INT = (int(round(TRUE_SUB[0])), int(round(TRUE_SUB[1])))


def ssd_at(a, b, ux, uy, n=N):
    yy, xx = np.mgrid[0:n, 0:n]
    coords = [yy + uy, xx + ux]
    bb = map_coordinates(b, coords, order=1, mode="nearest")
    m = (coords[0] >= 8) & (coords[0] < n - 8) & (coords[1] >= 8) & (coords[1] < n - 8)
    d = bb[m] - a[m]
    return float(np.mean(d * d))


# —— 全搜索（±24，仅用于计成本与对照）——
full_evals = 0; best_full = (1e9, 0, 0)
for uy in range(-24, 25):
    for ux in range(-24, 25):
        full_evals += 1
        v = ssd_at(BASE, TARGET, ux, uy)
        if v < best_full[0]: best_full = (v, ux, uy)

# —— 金字塔由粗到细 ——
pyr_a = [BASE]; pyr_b = [TARGET]
for _ in range(3):
    pyr_a.append(gaussian_filter(pyr_a[-1], 1.0)[::2, ::2])
    pyr_b.append(gaussian_filter(pyr_b[-1], 1.0)[::2, ::2])
levels = len(pyr_a)
hier_evals = 0
u = (0, 0)
for lv in range(levels - 1, -1, -1):
    scale = 2 ** lv
    n = pyr_a[lv].shape[0]
    r = 4 if lv == levels - 1 else 1
    best = (1e9, 0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ux = int(round(u[0] / scale)) + dx
            uy = int(round(u[1] / scale)) + dy
            hier_evals += 1
            yy, xx = np.mgrid[0:n, 0:n]
            bb = map_coordinates(pyr_b[lv], [yy + uy, xx + ux], order=1, mode="nearest")
            m = (yy + uy >= 4) & (yy + uy < n - 4) & (xx + ux >= 4) & (xx + ux < n - 4)
            e = float(np.mean((bb[m] - pyr_a[lv][m]) ** 2))
            if e < best[0]: best = (e, ux * scale, uy * scale)
    u = (best[1], best[2])
hier_shift = u
hier_err = float(np.hypot(hier_shift[0] - TRUE_SUB[0], hier_shift[1] - TRUE_SUB[1]))

# —— 相位相关 ——
P = 2 * N
win = np.outer(np.hanning(N), np.hanning(N))
Fa = np.fft.fft2((BASE - BASE.mean()) * win, s=(P, P))
Fb = np.fft.fft2((TARGET - TARGET.mean()) * win, s=(P, P))
R = Fa * np.conj(Fb)
R = R / (np.abs(R) + 1e-9)
corr = np.fft.ifft2(R).real
py, px = np.unravel_index(np.argmax(corr), corr.shape)
pdy = py if py < P // 2 else py - P
pdx = px if px < P // 2 else px - P
phase_shift = (-int(pdx), -int(pdy))
phase_peak = float(corr[py, px])

# —— 增量细化（Gauss–Newton on SSD）——
def refine(a, b, u0, iters=5):
    ux, uy = float(u0[0]), float(u0[1])
    hist = []
    gy, gx = np.gradient(gaussian_filter(b, 1.0))
    yy, xx = np.mgrid[0:N, 0:N]
    for it in range(iters):
        coords = [yy + uy, xx + ux]
        bb = map_coordinates(b, coords, order=1, mode="nearest")
        e = bb - a
        m = (coords[0] >= 10) & (coords[0] < N - 10) & (coords[1] >= 10) & (coords[1] < N - 10)
        gyb = map_coordinates(gy, coords, order=1, mode="nearest")
        gxb = map_coordinates(gx, coords, order=1, mode="nearest")
        J = np.stack([gxb[m], gyb[m]], axis=1)
        ee = e[m]
        A = J.T @ J; rhs = -(J.T @ ee)
        delta = np.linalg.solve(A + 1e-6 * np.eye(2), rhs)
        ux += float(delta[0]); uy += float(delta[1])
        hist.append(float(np.hypot(ux - TRUE_SUB[0], uy - TRUE_SUB[1])))
    return (ux, uy), hist

ref_shift, ref_hist = refine(BASE, TARGET, hier_shift, iters=5)
ref_err = float(np.hypot(ref_shift[0] - TRUE_SUB[0], ref_shift[1] - TRUE_SUB[1]))

# —— 图 1：金字塔 ——
fig, axes = plt.subplots(1, 4, figsize=(9.8, 3.2), dpi=150)
fig.patch.set_facecolor(PAPER)
for j, (img, lab) in enumerate(((pyr_a[0], "原图 256²"), (pyr_a[1], "1/2：128²"), (pyr_a[2], "1/4：64²"), (pyr_a[3], "1/8：32²"))):
    ax = axes[j]; ax.set_facecolor(PAPER)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_title(lab, fontsize=15, color=INK); ax.axis("off")
fig.suptitle("金字塔：粗层图小、搜索快；细层只在小范围微调", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, f"程序生成｜粗到细评估 {hier_evals} 次 vs 全搜索 {full_evals} 次", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(FIG / "e084_pyramid.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 2：相位相关（中心放大） ——
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
half = 12
crop = corr[py - half:py + half + 1, px - half:px + half + 1]
ax.imshow(crop, cmap="magma", vmin=0, vmax=np.percentile(crop, 99.5))
ax.plot(half, half, "o", ms=14, mfc="none", mec=ACCENT, mew=2.4)
ax.set_title("相位相关：峰值邻域放大，尖峰压过背景", fontsize=16, color=INK)
ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.02, f"程序生成｜峰 ({phase_shift[0]}, {phase_shift[1]})｜峰值 {phase_peak:.3f}", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "e084_phase.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 3：增量细化 ——
fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.semilogy(range(1, len(ref_hist) + 1), ref_hist, "o-", color=BLUE, lw=2, ms=7)
ax.set_xlabel("迭代次数", fontsize=15); ax.set_ylabel("位移误差（px，对数）", fontsize=15)
ax.set_title("增量细化：五步把误差压到约十分之一像素", fontsize=16, color=INK)
fig.text(0.5, 0.02, f"程序生成｜初始误差 {ref_hist[0]:.3f}px → 最终 {ref_hist[-1]:.4f}px", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "e084_refine.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

tests = [
    ("金字塔整数位移正确", hier_shift == TRUE_INT, f"best={hier_shift}/真值 {TRUE_INT}"),
    ("金字塔评估数远少于全搜索", hier_evals < full_evals / 5, f"{hier_evals} vs {full_evals}"),
    ("相位相关峰在真值附近", abs(phase_shift[0] - TRUE_INT[0]) <= 1 and abs(phase_shift[1] - TRUE_INT[1]) <= 1, f"peak={phase_shift}"),
    ("相位相关峰足够尖", phase_peak > 0.05, f"peak={phase_peak:.3f}"),
    ("增量细化到亚像素", ref_err < 0.12, f"err={ref_err:.4f}px"),
    ("细化误差单调下降", all(ref_hist[i + 1] <= ref_hist[i] + 1e-9 for i in range(len(ref_hist) - 1)), f"{['%.3f' % x for x in ref_hist]}"),
    ("细化不超过五步", len(ref_hist) <= 5, f"iters={len(ref_hist)}"),
    ("全搜索命中真值", best_full[1:] == TRUE_INT, f"best={best_full[1:]}"),
    ("全搜索成本高", full_evals > 1000, f"evals={full_evals}"),
    ("三图在盘", all((FIG / n).exists() for n in ("e084_pyramid.png", "e084_phase.png", "e084_refine.png")), "三张全在"),
]
lines = []; n_pass = 0
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass += bool(ok)
readings = dict(true_sub=list(TRUE_SUB), true_int=list(TRUE_INT), hier_shift=list(hier_shift), hier_evals=hier_evals,
                full_evals=full_evals, phase_shift=list(phase_shift), phase_peak=phase_peak, phase_peak_abs=float(np.max(corr)),
                ref_shift=list(ref_shift), ref_err=ref_err, ref_hist=ref_hist, ref_iters=len(ref_hist), n_pass=n_pass, n_total=len(tests))
(OUT / "e084_motion_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e084_motion_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E084 三条加速路 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e084_motion_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests) - n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests): raise SystemExit(1)
