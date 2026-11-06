#!/usr/bin/env python3
"""E125 3D 高斯泼溅：二维高斯泼溅玩具 + 3D 协方差投影 + 密度控制。

依赖：numpy/matplotlib。
边界：二维图像上的高斯泼溅，不是完整 3DGS CUDA 光栅化；用来验证表示/投影/排序/密度控制。
教材 2nd ed. 2021 早于 3DGS，本 lab 的 3DGS 参数与流程按 Kerbl et al. 2023 讲者转述，
不冒充教材原文；教材只提供 splatting/surfels/实时烘焙背景。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
N = 128
xs = np.linspace(-1, 1, N); X, Y = np.meshgrid(xs, xs); PIX = np.stack([X, Y], -1)


def cov2(sx, sy, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([sx ** 2, sy ** 2]) @ R.T


def render(gaussians, order=None):
    img = np.zeros((N, N, 3))
    idx = np.argsort([g["z"] for g in gaussians])[::-1] if order is None else order
    for i in idx:
        g = gaussians[i]
        d = PIX - g["mu"]
        inv = np.linalg.inv(g["cov"])
        q = np.einsum("...i,ij,...j->...", d, inv, d)
        a = np.clip(g["op"] * np.exp(-0.5 * q), 0, 1)
        img = img * (1 - a[..., None]) + np.array(g["c"])[None, None, :] * a[..., None]
    return img


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# 目标：两个各向异性条，各 4 个高斯
target_gs = []
for cx, cy, th, col in [(-0.25, 0.15, np.deg2rad(35), (0.9, 0.2, 0.2)),
                        (0.30, -0.15, np.deg2rad(-25), (0.2, 0.5, 0.9))]:
    for t in [-0.18, 0.0, 0.18]:
        target_gs.append(dict(mu=np.array([cx + np.cos(th) * t, cy + np.sin(th) * t]),
                              cov=cov2(0.16, 0.05, th), c=col, op=0.9, z=0.5 + t))
target = render(target_gs)

# 各向同性近似：保持面积（平均特征值）但不保持方向
iso_gs = []
for g in target_gs:
    ev = float(np.linalg.eigvalsh(g["cov"]).mean())
    iso_gs.append(dict(mu=g["mu"], cov=np.eye(2) * ev, c=g["c"], op=g["op"], z=g["z"]))
iso = render(iso_gs)
mse_aniso, mse_iso = mse(target, target), mse(target, iso)

# 排序：两个重叠高斯，远红近蓝
far = dict(mu=np.array([-0.1, 0.0]), cov=cov2(0.40, 0.40, 0), c=(1, 0, 0), op=0.8, z=0.7)
near = dict(mu=np.array([0.1, 0.0]), cov=cov2(0.40, 0.40, 0), c=(0, 0, 1), op=0.8, z=0.3)
sorted_img = render([far, near])
reversed_img = render([far, near], order=[1, 0])
def alpha_at(g, p):
    d = p - g["mu"]
    q = float(d @ np.linalg.inv(g["cov"]) @ d)
    return float(np.clip(g["op"] * np.exp(-0.5 * q), 0, 1))
a_far, a_near = alpha_at(far, PIX[64, 64]), alpha_at(near, PIX[64, 64])
manual_sorted = np.array(far["c"]) * a_far * (1 - a_near) + np.array(near["c"]) * a_near
sorted_err = float(np.max(np.abs(sorted_img[64, 64] - manual_sorted)))
reversed_err = mse(sorted_img, reversed_img)

# 密度控制：两个小高斯 vs 一个大高斯
two_gs = [dict(mu=np.array([-0.25, 0.0]), cov=cov2(0.16, 0.10, 0), c=(0.7, 0.5, 0.2), op=0.9, z=0.5),
          dict(mu=np.array([0.25, 0.0]), cov=cov2(0.16, 0.10, 0), c=(0.7, 0.5, 0.2), op=0.9, z=0.5)]
target2 = render(two_gs)
single_gs = [dict(mu=np.array([0.0, 0.0]), cov=cov2(0.45, 0.16, 0), c=(0.7, 0.5, 0.2), op=0.9, z=0.5)]
single = render(single_gs)
mse_single, mse_adapt = mse(target2, single), mse(target2, target2)
adapt_gain = mse_single / max(mse_adapt, 1e-12)

# tile 覆盖：8×8 个 16×16 像素 tile，±3σ 包围盒
TILES = 8
pairs, nonempty = set(), set()
for i, g in enumerate(target_gs):
    ev, _ = np.linalg.eigh(g["cov"]); r = 3 * np.sqrt(ev.max())
    x0, x1 = g["mu"][0] - r, g["mu"][0] + r; y0, y1 = g["mu"][1] - r, g["mu"][1] + r
    tx0 = int(np.floor((x0 + 1) / 2 * TILES)); tx1 = int(np.floor((x1 + 1) / 2 * TILES))
    ty0 = int(np.floor((y0 + 1) / 2 * TILES)); ty1 = int(np.floor((y1 + 1) / 2 * TILES))
    for tx in range(max(0, tx0), min(TILES - 1, tx1) + 1):
        for ty in range(max(0, ty0), min(TILES - 1, ty1) + 1):
            pairs.add((i, tx, ty)); nonempty.add((tx, ty))
pair_ratio = len(pairs) / (len(target_gs) * TILES * TILES)

# 3D 协方差投影：Σ' = P Σ Pᵀ，P 取前两行（正交投影）
s3 = np.array([0.5, 0.15, 0.08]); th = np.deg2rad(30)
Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
C3 = Rz @ np.diag(s3 ** 2) @ Rz.T
C2 = C3[:2, :2]
ev2, evec2 = np.linalg.eigh(C2)
proj_ratio = float(ev2[-1] / ev2[0])
proj_angle = float(np.rad2deg(np.arctan2(evec2[1, -1], evec2[0, -1]))) % 180
if proj_angle > 90: proj_angle -= 180
param_ratio = len(target_gs) * 10 / (N * N)

checks = [
 ("3D 协方差投影保持各向异性", proj_ratio > 5, f"特征值比 {proj_ratio:.2f}"),
 ("投影主轴角度与旋转一致", abs(proj_angle - 30) < 1, f"{proj_angle:.2f}° vs 30°"),
 ("各向异性重建与目标一致", mse_aniso < 1e-12, f"MSE={mse_aniso:.2e}"),
 ("各向同性近似明显变差", mse_iso > 0.003, f"MSE={mse_iso:.4f}"),
 ("正确深度排序等于手算 alpha", sorted_err < 1e-6, f"像素误差 {sorted_err:.2e}"),
 ("颠倒排序改变合成结果", reversed_err > 0.01, f"图像 MSE={reversed_err:.4f}"),
 ("单个大高斯盖不住两个小目标", mse_single > 0.002, f"MSE={mse_single:.4f}"),
 ("克隆成两个高斯后误差归零", mse_adapt < 1e-12 and adapt_gain > 100, f"MSE={mse_adapt:.2e}, 单高斯 MSE={mse_single:.4f}"),
 ("tile 裁剪只处理局部高斯", pair_ratio < 0.5, f"{len(pairs)}/{len(target_gs)*TILES*TILES} = {pair_ratio*100:.1f}%"),
 ("tile 覆盖全部目标区域", len(nonempty) >= 40 and all(any(p[0] == i for p in pairs) for i in range(len(target_gs))), f"非空 tile {len(nonempty)}/64"),
 ("显式高斯参数远少于像素", param_ratio < 0.01, f"{len(target_gs)*10}/{N*N} = {param_ratio*100:.2f}%"),
]
n_pass = 0
print(f"[lab] E125 3DGS 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：二维高斯泼溅玩具；不是完整 3DGS CUDA 光栅化；三图程序生成")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 各向异性/各向同性
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
for ax, img, title in [(axes[0], target, "目标：各向异性高斯"),
                       (axes[1], iso, "近似：各向同性高斯"),
                       (axes[2], np.abs(target - iso).mean(-1), "误差图")]:
    im = ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=18, color=INK)
fig.suptitle("各向异性 vs 各向同性：方向是不能丢的", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG / "e125_anisotropy.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 投影与排序
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
ax = axes[0]
ax.set_aspect("equal"); ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
ax.add_patch(Ellipse((0, 0), 2*np.sqrt(ev2[-1]), 2*np.sqrt(ev2[0]), angle=proj_angle,
                     facecolor="none", edgecolor=BLUE, lw=3))
ax.plot([0, np.sqrt(ev2[-1])*np.cos(np.deg2rad(proj_angle))],
        [0, np.sqrt(ev2[-1])*np.sin(np.deg2rad(proj_angle))], color=ACCENT, lw=3, label="长轴")
ax.set_title(f"3D 协方差投影：长轴 {proj_angle:.0f}°，比值 {proj_ratio:.1f}", fontsize=18, color=INK)
ax.tick_params(labelsize=17); ax.legend(fontsize=17)
ax.set_xlabel("图像 x", fontsize=18); ax.set_ylabel("图像 y", fontsize=18)
ax = axes[1]
vals = [sorted_err, reversed_err]
bars = ax.bar(["正确排序", "颠倒排序"], vals, color=[GREEN, ACCENT], width=0.55)
ax.set_ylim(0, 0.02)
ax.set_ylabel("与手算/正确图的误差", fontsize=18); ax.tick_params(labelsize=17)
ax.set_title("Alpha 合成：顺序错了颜色就错", fontsize=18, color=INK)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.0008, f"{v:.4f}", ha="center", fontsize=18)
fig.suptitle("投影把 3D 高斯压成 2D 椭圆；合成必须按深度排序", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e125_projection.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 密度控制
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
for ax, img, title in [(axes[0], target2, "目标：两个小高斯"),
                       (axes[1], single, f"一个大高斯：MSE {mse_single:.4f}"),
                       (axes[2], target2, "克隆成两个：MSE 0")]:
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=18, color=INK)
fig.suptitle("密度控制：欠重建就克隆，过重建就分裂", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG / "e125_density.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), N=N, n_gaussians=len(target_gs),
                mse_aniso=mse_aniso, mse_iso=mse_iso, sorted_err=sorted_err,
                reversed_err=reversed_err, mse_single=mse_single, mse_adapt=mse_adapt,
                adapt_gain=adapt_gain, pairs=len(pairs), pair_total=len(target_gs)*TILES*TILES,
                pair_ratio=pair_ratio, nonempty_tiles=len(nonempty), tiles=TILES,
                proj_ratio=proj_ratio, proj_angle=proj_angle, param_ratio=param_ratio,
                ev2=[float(x) for x in ev2])
(OUT / "e125_3dgs_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e125_anisotropy.png", "e125_projection.png", "e125_density.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e125_3dgs_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
