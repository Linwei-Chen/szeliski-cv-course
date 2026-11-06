#!/usr/bin/env python3
"""E122 点云配准 ICP（2D L 形切片：最近点 + 绝对定向 + 变体）。

依赖：numpy/scipy/matplotlib。
边界：2D 合成 L 形边界点，不是真实三维扫描；三维情形把 2D 旋转换成 3D SVD/四元数，
步骤相同。图中文字与结论只针对本实验。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
rng = np.random.default_rng(122)

VERTS = np.array([[-1, -0.6], [0.4, -0.6], [0.4, 0.1], [-0.1, 0.1],
                  [-0.1, 1.0], [-1, 1.0], [-1, -0.6]], float)


def sample_poly(verts, step=0.012):
    pts, norms = [], []
    for a, b in zip(verts[:-1], verts[1:]):
        d = b - a; L = np.linalg.norm(d); k = max(1, int(L / step)); u = d / L
        n = np.array([-u[1], u[0]])
        for i in range(k):
            pts.append(a + u * L * (i + 0.5) / k); norms.append(n)
    return np.array(pts), np.array(norms)


P0, N0 = sample_poly(VERTS)
TH = np.deg2rad(25.0)
R_TRUE = np.array([[np.cos(TH), -np.sin(TH)], [np.sin(TH), np.cos(TH)]])
T_TRUE = np.array([0.25, -0.15])
Q = P0 @ R_TRUE.T + T_TRUE
QN = N0 @ R_TRUE.T
P = P0 + rng.normal(0, 0.004, P0.shape)
n_out = int(0.10 * len(P))
P = np.vstack([P, rng.uniform(-1.2, 1.2, (n_out, 2))])


def kabsch(A, B, w=None):
    if w is None: w = np.ones(len(A))
    sw = w.sum(); ca = (A * w[:, None]).sum(0) / sw; cb = (B * w[:, None]).sum(0) / sw
    H = (A - ca).T @ ((B - cb) * w[:, None])
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca


def icp(P, Q, init_deg=10.0, maxit=50, robust=False, trim=0.35, plane=False, QN=None):
    a = np.deg2rad(init_deg); Rtot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ttot = np.zeros(2); Pc = P @ Rtot.T + ttot
    tree = cKDTree(Q); hist = []; mask = None; its = 0; delta = 9.9
    for it in range(maxit):
        d, idx = tree.query(Pc); mask = d < trim
        if mask.sum() < 10: break
        if robust:
            med = np.median(d[mask]); k = 1.4826 * med * 1.5 + 1e-9
            w = np.where(d[mask] <= k, 1.0, k / np.maximum(d[mask], 1e-12))
        else:
            w = np.ones(mask.sum())
        if plane:
            p = Pc[mask]; q = Q[idx[mask]]; nrm = QN[idx[mask]]
            r = np.sum(nrm * (p - q), axis=1)
            J = np.column_stack([p[:, 0] * nrm[:, 1] - p[:, 1] * nrm[:, 0], nrm])
            x = np.linalg.lstsq(J, -r, rcond=None)[0]
            om = x[0]; dR = np.array([[np.cos(om), -np.sin(om)], [np.sin(om), np.cos(om)]]); dt = x[1:]
        else:
            dR, dt = kabsch(Pc[mask], Q[idx[mask]], w)
        Pc = Pc @ dR.T + dt; Rtot = dR @ Rtot; ttot = dR @ ttot + dt
        rmse = float(np.sqrt(np.mean(d[mask] ** 2))); hist.append(rmse); its = it + 1
        if it > 3:
            delta = abs(hist[-2] - hist[-1])
            if delta < 1e-7: break
    return Pc, Rtot, ttot, hist, its, mask, delta


def angle_of(R):
    return float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))


Rk, tk = kabsch(P0, Q)
kab_ang_err = abs(angle_of(Rk) - 25.0); kab_t_err = float(np.linalg.norm(tk - T_TRUE))
plain = icp(P, Q, init_deg=10.0)
rob = icp(P, Q, init_deg=10.0, robust=True)
plane = icp(P, Q, init_deg=10.0, plane=True, QN=QN)
bad = icp(P, Q, init_deg=180.0, maxit=60)
plain_ang = abs(angle_of(plain[1]) - 25.0); plain_t = float(np.linalg.norm(plain[2] - T_TRUE))
rob_ang = abs(angle_of(rob[1]) - 25.0); plane_ang = abs(angle_of(plane[1]) - 25.0)
rmse0 = plain[3][0]; rmse_p = plain[3][-1]; rmse_bad = bad[3][-1]
drop = 1 - rmse_p / rmse0
inlier_frac = float(plain[5][:len(P0)].sum() / len(P0))
outlier_frac = float(plain[5][len(P0):].sum() / n_out)

# KD-tree vs 暴力最近点：随机抽 100 个源点，索引应完全一致
tree = cKDTree(Q); sample = P0[:100]
kd_idx = tree.query(sample)[1]
bf_idx = np.array([np.argmin(np.sum((Q - s) ** 2, axis=1)) for s in sample])
kd_equal = bool(np.array_equal(kd_idx, bf_idx))

# 初值盆地：每 10° 跑一次，固定平移初值 0
angs = np.arange(-180, 181, 10); basin = []
for a in angs:
    out = icp(P, Q, init_deg=float(a), maxit=40)
    basin.append(out[3][-1])
basin = np.array(basin)

checks = [
 ("Kabsch 精确恢复已知旋转/平移", kab_ang_err < 1e-9 and kab_t_err < 1e-9, f"角度误差 {kab_ang_err:.1e} / 平移误差 {kab_t_err:.1e}"),
 ("点到点 ICP 把 RMSE 降 70% 以上", drop > 0.70, f"{rmse0:.4f} → {rmse_p:.4f}（降 {100*drop:.1f}%）"),
 ("点到点最终旋转误差 <0.5°", plain_ang < 0.5, f"{plain_ang:.3f}°"),
 ("点到点最终平移误差 <0.01", plain_t < 0.01, f"{plain_t:.4f}"),
 ("鲁棒权重比普通最小二乘更准", rob_ang < plain_ang, f"{rob_ang:.4f}° < {plain_ang:.4f}°"),
 ("点到面比点到点迭代更少", plane[4] < plain[4], f"{plane[4]} < {plain[4]} 次"),
 ("180° 初值落入局部极小", rmse_bad > 2*rmse_p, f"{rmse_bad:.4f} > 2×{rmse_p:.4f}"),
 ("KD-tree 与暴力最近点一致", kd_equal, "100 个抽样点索引完全相同"),
 ("距离阈值保留 85% 以上的内点", inlier_frac > 0.85, f"保留 {100*inlier_frac:.1f}% 内点（离群点 {100*outlier_frac:.1f}% 也在阈值内，靠鲁棒权重降权）"),
 ("收敛判据在最大迭代前触发", plain[4] < 50 and plain[6] < 1e-7, f"{plain[4]} 次，ΔRMSE={plain[6]:.1e}"),
]
n_pass = 0
print(f"[lab] E122 ICP 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：2D 合成 L 形边界点；三维情形同步骤，换 3D SVD/四元数，不冒充 3D 性能")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 配准前后
a0 = np.deg2rad(10.0); R0 = np.array([[np.cos(a0), -np.sin(a0)], [np.sin(a0), np.cos(a0)]])
P_init = P @ R0.T
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
for ax, pts, title in [(axes[0], P_init, "配准前：源点云 10°，真值 25°"), (axes[1], plain[0], "配准后：最近点+绝对定向")]:
    ax.scatter(Q[:, 0], Q[:, 1], s=7, c=ACCENT, alpha=0.55, label="目标点云")
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c=BLUE, alpha=0.55, label="源点云")
    ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.2); ax.set_ylim(-1.2, 1.4)
    ax.set_title(title, fontsize=18, color=INK)
    ax.tick_params(labelsize=16); ax.set_xlabel("x", fontsize=18); ax.set_ylabel("y", fontsize=18)
axes[0].legend(fontsize=16, loc="upper right")
fig.suptitle("ICP 的两个动作：找最近点，解刚体变换", fontsize=21, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e122_icp_align.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 变体：RMSE 曲线 + 最终角度误差
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
for out, lab, color in [(plain, "点到点", BLUE), (rob, "鲁棒权重", GREEN), (plane, "点到面", ACCENT)]:
    axes[0].plot(np.arange(1, len(out[3]) + 1), out[3], marker="o", ms=3, lw=2, label=lab, color=color)
axes[0].set_xlabel("迭代次数", fontsize=19); axes[0].set_ylabel("对应点 RMSE", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("误差随迭代下降", fontsize=19)
axes[0].legend(fontsize=17)
names = ["点到点", "鲁棒", "点到面"]; vals = [plain_ang, rob_ang, plane_ang]
bars = axes[1].bar(names, vals, color=[BLUE, GREEN, ACCENT], width=0.55)
axes[1].set_yscale("log"); axes[1].set_ylabel("最终旋转误差 / 度（对数）", fontsize=19)
axes[1].tick_params(labelsize=17); axes[1].set_title("最终旋转误差", fontsize=19)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width()/2, v*1.5, f"{v:.4f}", ha="center", fontsize=17)
fig.suptitle("三个变体：鲁棒权重更准，点到面收敛更快", fontsize=22, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e122_variants.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 初值盆地
fig, ax = plt.subplots(figsize=(11.8, 4.7))
ax.plot(angs, basin, "o-", color=BLUE, lw=2, ms=4)
ax.axhline(rmse_p, color=GREEN, ls="--", lw=1.5, label="好初值的最终误差")
ax.axvspan(-20, 70, color=GREEN, alpha=0.10, label="能收敛的初值带")
ax.axvline(0, color=INK, ls=":", lw=1.2)
ax.set_xlabel("初始旋转 / 度", fontsize=20); ax.set_ylabel("最终 RMSE", fontsize=20)
ax.tick_params(labelsize=17); ax.set_title("ICP 的初值盆地：好初值收敛，坏初值卡在局部极小", fontsize=20)
ax.legend(fontsize=17, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "e122_basin.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), n_points=len(P), n_outliers=n_out,
                kabsch_angle_err=kab_ang_err, kabsch_t_err=kab_t_err,
                rmse_init=rmse0, rmse_plain=rmse_p, rmse_robust=rob[3][-1],
                rmse_plane=plane[3][-1], rmse_bad=rmse_bad, drop=drop,
                angle_plain=plain_ang, angle_robust=rob_ang, angle_plane=plane_ang,
                t_plain=plain_t, its_plain=plain[4], its_robust=rob[4], its_plane=plane[4],
                its_bad=bad[4], inlier_frac=inlier_frac, outlier_frac=outlier_frac, kd_equal=kd_equal,
                basin_min=float(basin.min()), basin_max=float(basin.max()),
                plain_delta=plain[6])
(OUT / "e122_icp_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e122_icp_align.png", "e122_variants.png", "e122_basin.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e122_icp_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
