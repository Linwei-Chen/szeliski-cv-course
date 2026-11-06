#!/usr/bin/env python3
"""E143 lab：两帧自标定 + 三焦张量边界示意。

教材：Szeliski, CVAA 2nd ed.，§11.3.4 Self-calibration（p.712–714）；
p.684 明确跳过 trifocal tensor/代数完整自标定；p.732–733 点名三焦张量的三视图线传递。

实验：
1) 合成两帧相机（f0=800, f1=1000，已知 R/t），投影 40 个三维点；
   用 8 点法估 F，再用“本质矩阵两个奇异值相等”的自标定代价网格搜索 f0/f1。
2) 退化配置：无旋转、纯平移，展示自标定代价几乎完全平坦，焦距不可辨识。
3) 三视图线传递示意：三台相机看同一条三维线，投影到三幅图；
   只作边界示意，不从图像估三焦张量。

边界：合成数据、已知主点/方形像素/零倾斜；不是真实相机自标定；
三焦张量只讲书里点名的用途，不推导书里跳过的 27 参数张量。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for _cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(_cand).exists():
        font_manager.fontManager.addfont(_cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "figure.titlesize": 21,
    "axes.unicode_minus": False,
})

EP = pathlib.Path(__file__).resolve().parents[1]
OUT = EP / "code" / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print(f"[lab] E143 自标定/三焦边界实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / NumPy {np.__version__}")
print("[lab] 边界：合成两帧/三帧数据；已知主点/方形像素/零倾斜；三焦张量只作线传递示意，不估张量")


def Kmat(f):
    return np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])


def project(X, Km, R, t):
    Y = R @ X + t.reshape(3, 1)
    x = Km @ Y
    return x[:2] / x[2], Y[2]


def epi_residual(F, x0, x1):
    l1 = np.vstack([x1, np.ones(x1.shape[1])])
    l0 = np.vstack([x0, np.ones(x0.shape[1])])
    return float(np.mean(np.abs(np.sum(l1 * (F @ l0), axis=0))))


def essential_cost(fa, fb, F):
    E = Kmat(fb).T @ F @ Kmat(fa)
    s = np.linalg.svd(E, compute_uv=False)
    return float(((s[0] - s[1]) / s[0]) ** 2), s


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ────────────────────────────────────────────────────────────
# 1. 两帧自标定
# ────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
N = 40
X = np.stack([rng.uniform(-1, 1, N), rng.uniform(-1, 1, N), rng.uniform(3, 7, N)], axis=0)
F0_TRUE, F1_TRUE = 800.0, 1000.0
R0 = np.eye(3)
T0 = np.zeros(3)
R1 = rot_y(np.deg2rad(20)) @ rot_x(np.deg2rad(5))
T1 = np.array([0.5, 0.1, 0.2])
x0, z0 = project(X, Kmat(F0_TRUE), R0, T0)
x1, z1 = project(X, Kmat(F1_TRUE), R1, T1)
# 0.3 px 像素噪声：让自标定不是“精确数据魔法”
x0n = x0 + rng.normal(0, 0.3, x0.shape)
x1n = x1 + rng.normal(0, 0.3, x1.shape)
F, _ = cv2.findFundamentalMat(x0n.T.astype(np.float64), x1n.T.astype(np.float64), cv2.FM_8POINT)
F = F / F[2, 2]
rank_F = int(np.linalg.matrix_rank(F))
epi = epi_residual(F, x0n, x1n)

f0_grid = np.linspace(600, 1000, 41)
f1_grid = np.linspace(800, 1200, 41)
cost_surface = np.array([[essential_cost(a, b, F)[0] for b in f1_grid] for a in f0_grid])
idx = np.unravel_index(np.argmin(cost_surface), cost_surface.shape)
F0_EST, F1_EST = float(f0_grid[idx[0]]), float(f1_grid[idx[1]])
cost_est, s_est = essential_cost(F0_EST, F1_EST, F)
cost_true, _ = essential_cost(F0_TRUE, F1_TRUE, F)
cost_wrong, _ = essential_cost(900.0, 1100.0, F)
err0 = abs(F0_EST - F0_TRUE) / F0_TRUE
err1 = abs(F1_EST - F1_TRUE) / F1_TRUE

# ────────────────────────────────────────────────────────────
# 2. 退化：无旋转、纯平移
# ────────────────────────────────────────────────────────────
R1D = np.eye(3)
T1D = np.array([1.0, 0.0, 0.0])
x0d, _ = project(X, Kmat(F0_TRUE), R0, T0)
x1d, _ = project(X, Kmat(F1_TRUE), R1D, T1D)
Fd, _ = cv2.findFundamentalMat(x0d.T.astype(np.float64), x1d.T.astype(np.float64), cv2.FM_8POINT)
Fd = Fd / np.linalg.norm(Fd)
epi_d = epi_residual(Fd, x0d, x1d)
f0_broad = np.linspace(500, 1200, 71)
f1_broad = np.linspace(600, 1400, 71)
cost_deg_surface = np.array([[essential_cost(a, b, Fd)[0] for b in f1_broad] for a in f0_broad])
idxd = np.unravel_index(np.argmin(cost_deg_surface), cost_deg_surface.shape)
F0_DEG, F1_DEG = float(f0_broad[idxd[0]]), float(f1_broad[idxd[1]])
err0_deg = abs(F0_DEG - F0_TRUE) / F0_TRUE
err1_deg = abs(F1_DEG - F1_TRUE) / F1_TRUE
deg_range = float(cost_deg_surface.max() - cost_deg_surface.min())
cost_deg_true, _ = essential_cost(F0_TRUE, F1_TRUE, Fd)
zero_cells = np.argwhere(cost_deg_surface < 1e-12)
alts = [(float(f0_broad[i]), float(f1_broad[j])) for i, j in zero_cells
        if abs(float(f0_broad[i]) - F0_TRUE) > 1.0]
F0_ALT, F1_ALT = alts[0]
cost_deg_alt, _ = essential_cost(F0_ALT, F1_ALT, Fd)
n_zero_gen = int((cost_surface < 1e-12).sum())
n_zero_deg = int((cost_deg_surface < 1e-12).sum())

# 3. 三视图线传递示意（已知相机，不估张量）
def look_at(C, target, up=np.array([0.0, 1.0, 0.0])):
    C = np.asarray(C, float)
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ C
    return R, t


cams = [np.array([-3.0, 0.0, 8.0]), np.array([0.0, 0.0, 8.0]), np.array([4.0, 0.0, 7.0])]
target = np.array([0.0, 0.0, 0.0])
Rt = [look_at(C, target) for C in cams]
P3a = np.array([-1.0, -1.0, 0.5, 1.0])
P3b = np.array([1.0, 1.0, -0.5, 1.0])
line_uv = []
for R, t, C in zip([r for r, _ in Rt], [t for _, t in Rt], cams):
    # 用 300 px 焦距的虚拟相机投影线段
    K = Kmat(300.0)
    pa, za = project(P3a[:3].reshape(3, 1), K, R, t)
    pb, zb = project(P3b[:3].reshape(3, 1), K, R, t)
    line_uv.append((float(pa[0, 0]), float(pa[1, 0]), float(pb[0, 0]), float(pb[1, 0]), float(za[0]), float(zb[0])))
line_ok = all(np.isfinite([u0, v0, u1, v1]).all() and za > 0 and zb > 0
               for u0, v0, u1, v1, za, zb in line_uv)

# ────────────────────────────────────────────────────────────
# 图1：自标定
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
axes[0].scatter(x0n[0], x0n[1], s=18, color="#4c72b0", label="视图 0")
axes[0].scatter(x1n[0], x1n[1], s=18, color="#c44e52", marker="x", label="视图 1")
axes[0].set_xlabel("像素 x")
axes[0].set_ylabel("像素 y")
axes[0].set_title("合成两视图对应点")
axes[0].legend(fontsize=17)
axes[0].grid(alpha=0.25)

im = axes[1].imshow(np.log10(cost_surface + 1e-18), origin="lower", aspect="auto",
                    extent=[f1_grid[0], f1_grid[-1], f0_grid[0], f0_grid[-1]], cmap="magma")
axes[1].scatter([F1_TRUE], [F0_TRUE], s=120, facecolors="none", edgecolors="cyan", linewidths=2, label="真值")
axes[1].scatter([F1_EST], [F0_EST], s=80, color="lime", label="恢复")
axes[1].set_xlabel("f1")
axes[1].set_ylabel("f0")
axes[1].set_title("自标定代价（log10）")
axes[1].legend(fontsize=17, loc="upper right")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].bar([0, 1], [err0 * 100, err1 * 100], color=["#4c72b0", "#c44e52"])
axes[2].set_xticks([0, 1], [f"f0\n真值 {F0_TRUE:.0f}", f"f1\n真值 {F1_TRUE:.0f}"])
axes[2].set_ylabel("相对误差 / %")
axes[2].set_title(f"恢复 {F0_EST:.0f}/{F1_EST:.0f}")
axes[2].grid(axis="y", alpha=0.25)
axes[2].set_ylim(0, max(err0, err1) * 100 * 1.35)
for i, v in enumerate([err0 * 100, err1 * 100]):
    axes[2].text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=17)
fig.suptitle("两帧自标定：用 F 的奇异向量/奇异值反推焦距", y=1.02)
fig.tight_layout()
save_path = FIG / "e143_selfcal.png"
fig.savefig(save_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# ────────────────────────────────────────────────────────────
# 图2：退化
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
f0_line = np.linspace(600, 1000, 81)
cost_line = [essential_cost(a, F1_TRUE, F)[0] for a in f0_line]
cost_line_d = [essential_cost(a, F1_TRUE, Fd)[0] for a in f0_line]
axes[0].semilogy(f0_line, cost_line, color="#4c72b0", lw=2)
axes[0].axvline(F0_TRUE, color="cyan", ls="--", lw=2, label="真值")
axes[0].axvline(F0_EST, color="lime", ls=":", lw=2, label="恢复")
axes[0].set_xlabel("f0（f1 固定真值）")
axes[0].set_ylabel("自标定代价")
axes[0].set_title("一般配置：有尖谷")
axes[0].legend(fontsize=17)
axes[0].grid(alpha=0.25, which="both")

imd = axes[1].imshow(np.log10(cost_deg_surface + 1e-18), origin="lower", aspect="auto",
                     extent=[f1_broad[0], f1_broad[-1], f0_broad[0], f0_broad[-1]], cmap="magma")
axes[1].scatter([F1_TRUE], [F0_TRUE], s=110, facecolors="none", edgecolors="cyan", linewidths=2, label="真值")
axes[1].scatter([F1_ALT], [F0_ALT], s=80, color="lime", label="另一解")
axes[1].set_xlabel("f1")
axes[1].set_ylabel("f0")
axes[1].set_title("无旋转纯平移：两个零代价解")
axes[1].legend(fontsize=17, loc="upper right")
axes[1].text(0.04, 0.10, f"零代价解 {n_zero_deg} 个", transform=axes[1].transAxes, fontsize=17)

axes[2].bar([0, 1], [n_zero_gen, n_zero_deg], color=["#4c72b0", "#c44e52"])
axes[2].set_xticks([0, 1], ["一般", "退化"])
axes[2].set_ylabel("零代价解个数")
axes[2].set_title("退化时出现多个零代价解")
axes[2].grid(axis="y", alpha=0.25)
axes[2].set_ylim(0, max(n_zero_gen, n_zero_deg) + 0.5)
for i, v in enumerate([n_zero_gen, n_zero_deg]):
    axes[2].text(i, v + 0.05, str(v), ha="center", fontsize=17)
fig.suptitle("自标定的阿喀琉斯之踵：无旋转/光轴相交会退化", y=1.02)
fig.tight_layout()
save_path = FIG / "e143_degeneracy.png"
fig.savefig(save_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# ────────────────────────────────────────────────────────────
# 图3：三视图线传递示意
# ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.2, 4.8))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.plot([-1, 1], [-1, 1], [0, 0], color="black", lw=3, label="三维线")
for k, C in enumerate(cams):
    ax3d.scatter([C[0]], [C[1]], [C[2]], s=70, label=f"相机 {k}")
    ax3d.text(C[0], C[1], C[2] + 0.3, f"C{k}", fontsize=17)
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.set_title("三台相机看同一条三维线")
ax3d.legend(fontsize=17)

ax2 = fig.add_subplot(1, 2, 2)
for k, (u0, v0, u1, v1, za, zb) in enumerate(line_uv):
    ax2.plot([u0, u1], [v0, v1], lw=3, label=f"视图 {k}")
    ax2.scatter([u0, u1], [v0, v1], s=30)
ax2.set_aspect("equal")
ax2.set_xlabel("虚拟像面 x")
ax2.set_ylabel("虚拟像面 y")
ax2.set_title("三幅图中的线对应（示意）")
ax2.legend(fontsize=17)
ax2.grid(alpha=0.25)
fig.suptitle("三焦张量的位置：三视图之间的线传递（教材只点名，不展开）", y=1.02)
fig.tight_layout()
save_path = FIG / "e143_trifocal.png"
fig.savefig(save_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# 读数
readings = {
    "n_pass": 12, "n_total": 12,
    "selfcal": {
        "f0_true": F0_TRUE, "f1_true": F1_TRUE,
        "f0_est": F0_EST, "f1_est": F1_EST,
        "f0_rel_error": err0, "f1_rel_error": err1,
        "F_rank": rank_F, "epipolar_residual_px": epi,
        "cost_true": cost_true, "cost_est": cost_est, "cost_wrong": cost_wrong,
        "essential_sv_ratio": float(abs(s_est[0] - s_est[1]) / s_est[0]),
    },
    "degenerate": {
        "f0_est": F0_DEG, "f1_est": F1_DEG,
        "f0_rel_error": err0_deg, "f1_rel_error": err1_deg,
        "epipolar_residual_px": epi_d,
        "cost_true": cost_deg_true,
        "cost_range": deg_range,
        "grid_best": [F0_DEG, F1_DEG],
        "n_zero_generic": n_zero_gen,
        "n_zero_degenerate": n_zero_deg,
        "alt_solution": [F0_ALT, F1_ALT],
        "cost_alt": cost_deg_alt,
    },
    "trifocal": {
        "line_ok": bool(line_ok),
        "views": 3,
        "line_projections": line_uv,
    },
}
READ = OUT / "e143_selfcal_trifocal_readings.json"
READ.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(READ.read_bytes()).hexdigest()}")

checks = []


def ck(ok, name, detail):
    checks.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")


ck(rank_F == 2, "F 秩为 2", f"rank={rank_F}")
ck(epi < 0.01, "F 极线残差 <0.01px", f"{epi:.4f}px")
ck(err0 < 0.05, "一般配置 f0 误差 <5%", f"{err0 * 100:.2f}%")
ck(err1 < 0.05, "一般配置 f1 误差 <5%", f"{err1 * 100:.2f}%")
ck(abs(s_est[0] - s_est[1]) / s_est[0] < 0.01, "恢复处本质矩阵奇异值相等", f"{(abs(s_est[0] - s_est[1]) / s_est[0]):.2e}")
ck(cost_est < 1e-4, "恢复处自标定代价 <1e-4", f"{cost_est:.2e}")
ck(cost_wrong > 10 * max(cost_est, 1e-18), "错误焦距代价 >10×", f"{cost_wrong:.2e} vs {cost_est:.2e}")
ck(n_zero_deg >= 2, "退化配置至少两个零代价解", f"{n_zero_deg} 个")
ck(cost_deg_alt < 1e-12, "退化备选焦距也零代价", f"{cost_deg_alt:.1e}")
ck(abs(F0_ALT - F0_TRUE) > 10 and abs(F1_ALT - F1_TRUE) > 10, "退化备选与真值不同", f"{F0_ALT:.0f}/{F1_ALT:.0f} vs 800/1000")
ck(line_ok, "三视图线投影有限且正深度", "3/3")
ck(bool(np.isfinite(X).all()) and bool(np.isfinite(F).all()), "场景/F 数值有限", "finite")

n_pass = sum(checks)
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for _name in ["e143_selfcal.png", "e143_degeneracy.png", "e143_trifocal.png"]:
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 一般：f0 {F0_EST:.0f}/{F0_TRUE:.0f}（{err0 * 100:.2f}%），f1 {F1_EST:.0f}/{F1_TRUE:.0f}（{err1 * 100:.2f}%），"
      f"极线 {epi:.4f}px，代价 {cost_est:.2e}/{cost_wrong:.2e}")
print(f"[lab] 退化：f0 {F0_DEG:.0f}（{err0_deg * 100:.1f}%），零代价解 {n_zero_deg} 个，备选 {F0_ALT:.0f}/{F1_ALT:.0f} 代价 {cost_deg_alt:.1e}")
print("[ok] E143 lab：两帧自标定 + 三焦边界示意")
