#!/usr/bin/env python3
"""E120 从点云到表面：隐式场、Marching Cubes 与 Poisson 积分（合成球）。

依赖：numpy/scipy/matplotlib/scikit-image（marching_cubes/find_contours）。
边界：合成球，不冒充真实扫描；Poisson 用 2D 切片演示同一方程，完整 3D 重建是工程实现。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes, find_contours
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse.linalg import spsolve
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
R = 1.0

def sphere_field(N, R=1.0, extent=1.2):
    xs = np.linspace(-extent, extent, N); h = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.sqrt(X**2 + Y**2 + Z**2) - R, xs, h

def mc_stats(N, level=0.0):
    f, xs, h = sphere_field(N)
    v, faces, _, _ = marching_cubes(f, level=level, spacing=(h, h, h))
    v = v - (N - 1) / 2 * h
    r = np.linalg.norm(v, axis=1)
    vol = ConvexHull(v).volume
    return dict(n=N, n_vert=len(v), n_face=len(faces), radius=float(r.mean()),
                radius_err=float(abs(r.mean() - R)), volume=float(vol),
                volume_err=float(vol / (4/3*np.pi*R**3) - 1))

mc48, mc64, mc96 = mc_stats(48), mc_stats(64), mc_stats(96)
mc_level = mc_stats(96, level=0.1)
f_center = sphere_field(48)[0][24, 24, 24]
f_far = sphere_field(48)[0][0, 0, 0]

# 2D Poisson 切片：从带噪梯度积分出隐式场
N2 = 81; xs = np.linspace(-1.2, 1.2, N2); h = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing="ij"); rr = np.sqrt(X**2 + Y**2) + 1e-12
ft = rr - R; gx, gy = X / rr, Y / rr
rng = np.random.default_rng(120)
gnx = gx + 0.15 * rng.standard_normal(gx.shape)
gny = gy + 0.15 * rng.standard_normal(gy.shape)
n = N2 - 2; idx = np.arange(n*n).reshape(n, n)
A = sparse.lil_matrix((n*n, n*n))
for i in range(n):
    for j in range(n):
        k = idx[i, j]; A[k, k] = -4 / h**2
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n:
                A[k, idx[ii, jj]] = 1 / h**2
A = A.tocsr()

def solve_poisson(gx, gy, bc):
    div = (gx[2:,1:-1] - gx[:-2,1:-1])/(2*h) + (gy[1:-1,2:] - gy[1:-1,:-2])/(2*h)
    b = np.zeros(n*n)
    for i in range(n):
        for j in range(n):
            k = idx[i, j]; b[k] = div[i, j]
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < n and 0 <= jj < n):
                    b[k] -= bc[i+1+di, j+1+dj] / h**2
    f = bc.copy(); f[1:-1,1:-1] = spsolve(A, b).reshape(n, n)
    return f

def contour_radius(field):
    cs = sorted(find_contours(field, 0.0), key=len, reverse=True)
    if not cs: return None, 0
    c = cs[0]; x = xs[0] + c[:,1]*h; y = xs[0] + c[:,0]*h
    return float(np.sqrt(x*x + y*y).mean()), len(cs)

f_poisson = solve_poisson(gnx, gny, ft)
f_noisy = ft + 0.15 * rng.standard_normal(ft.shape)
f_zero = solve_poisson(gnx, gny, np.zeros_like(ft))
ghx, ghy = gnx.copy(), gny.copy(); sel = rng.random(ghx.shape) < 0.5
ghx[sel] *= -1; ghy[sel] *= -1
f_half = solve_poisson(ghx, ghy, ft)
poisson_r, poisson_n = contour_radius(f_poisson)
noisy_r, noisy_n = contour_radius(f_noisy)
zero_r, zero_n = contour_radius(f_zero)
half_r, half_n = contour_radius(f_half)
poisson_err = abs(poisson_r - R); noisy_err = abs(noisy_r - R)

checks = [
 ("SDF 符号：球心为负、远处为正", f_center < 0 < f_far, f"{f_center:.3f} / {f_far:.3f}"),
 ("N=64 球面半径误差 <0.001", mc64["radius_err"] < 1e-3, f"{mc64['radius_err']:.6f}"),
 ("N=96 半径误差小于 N=48", mc96["radius_err"] < mc48["radius_err"], f"{mc96['radius_err']:.6f} < {mc48['radius_err']:.6f}"),
 ("N=96 三角面数 >4×N=48", mc96["n_face"] > 4*mc48["n_face"], f"{mc96['n_face']} vs {mc48['n_face']}"),
 ("N=96 网格体积误差 <0.1%", abs(mc96["volume_err"]) < 1e-3, f"{100*mc96['volume_err']:.3f}%"),
 ("等值面 +0.1：半径外移约 0.1", abs((mc_level["radius"] - R) - 0.1) < 0.01, f"{mc_level['radius']:.4f}"),
 ("Poisson 恢复半径误差 <0.005", poisson_err < 0.005, f"{poisson_err:.6f}"),
 ("Poisson 误差小于带噪场", poisson_err < noisy_err, f"{poisson_err:.6f} < {noisy_err:.6f}"),
 ("错误边界（全零）抹掉零等值面", zero_n == 0, f"轮廓 {zero_n} 条"),
 ("一半法向翻转：零等值面消失", half_n == 0, f"轮廓 {half_n} 条"),
]
n_pass = 0
print(f"[lab] E120 表面重建开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成球；Poisson 为 2D 切片演示，完整 3D 重建是工程实现")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 三种表示
f24, xs24, h24 = sphere_field(24); v24, fc24, _, _ = marching_cubes(f24, level=0, spacing=(h24,h24,h24)); v24 = v24 - 11.5*h24
fig = plt.figure(figsize=(11.4, 4.8))
ax = fig.add_subplot(1,3,1,projection="3d"); ax.plot_trisurf(v24[:,0], v24[:,1], v24[:,2], triangles=fc24, color="#9ecae1", edgecolor="#3d6a99", linewidth=0.2, alpha=0.9)
ax.set_title("显式网格：三角面", fontsize=15, color=INK); ax.set_axis_off()
ax = fig.add_subplot(1,3,2,projection="3d"); pts = v24[::15]; ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=6, c=BLUE, alpha=0.8)
ax.set_title("点云：表面采样点", fontsize=15, color=INK); ax.set_axis_off()
ax = fig.add_subplot(1,3,3); Z = np.sqrt(np.maximum(0, R**2 - X**2 - Y**2)); ax.contourf(X, Y, Z, levels=18, cmap="viridis"); ax.contour(X, Y, Z, levels=[0.0], colors=ACCENT, linewidths=2)
ax.set_title("隐式场：等值面切片", fontsize=15, color=INK); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("从点云到表面：三种表示，同一形状", fontsize=19, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e120_representations.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 Marching Cubes：切片+分辨率曲线
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios":[1,1.25]})
ax = axes[0]; Z = np.sqrt(np.maximum(0, R**2 - X**2 - Y**2)); ax.contour(X, Y, Z, levels=[0.0], colors=ACCENT, linewidths=2.2); ax.contourf(X, Y, Z, levels=16, cmap="viridis", alpha=0.75)
ax.set_title("零等值线：MC 的二维切片", fontsize=14.5, color=INK); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax = axes[1]; ns = [mc48["n"], mc64["n"], mc96["n"]]; faces = [mc48["n_face"], mc64["n_face"], mc96["n_face"]]; errs = [mc48["radius_err"], mc64["radius_err"], mc96["radius_err"]]
ax.plot(ns, faces, "o-", color=BLUE, lw=2.2, ms=8, label="三角面数")
ax.set_xlabel("体素分辨率 N", fontsize=14); ax.set_ylabel("三角面数", fontsize=14); ax.tick_params(labelsize=12)
ax2 = ax.twinx(); ax2.plot(ns, errs, "s--", color=ACCENT, lw=2.0, ms=8, label="半径误差")
ax2.set_ylabel("半径误差 / 单位", fontsize=13, color=ACCENT); ax2.tick_params(axis="y", labelsize=12.5, colors=ACCENT)
ax.set_title("分辨率越高：面更多，误差更小", fontsize=14.5, color=INK)
ax.grid(alpha=0.22)
fig.tight_layout(); fig.savefig(FIG / "e120_mc.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 Poisson：真值/带噪/恢复
fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.5))
for ax, field, title, color in [(axes[0], ft, "真值场：零等值线", GREEN), (axes[1], f_noisy, "带噪场：轮廓碎", GRAY), (axes[2], f_poisson, "Poisson 恢复：一条轮廓", BLUE)]:
    ax.contourf(X, Y, field, levels=16, cmap="RdBu_r", alpha=0.8)
    cs = sorted(find_contours(field, 0.0), key=len, reverse=True)[:3]
    for c in cs:
        ax.plot(xs[0] + c[:,1]*h, xs[0] + c[:,0]*h, color=color, lw=2.0)
    ax.set_title(title, fontsize=14, color=INK); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Poisson 积分（2D 切片）：真值误差 0.0000｜带噪 {noisy_err:.4f}｜恢复 {poisson_err:.4f}", fontsize=17, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e120_poisson.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = {
 "mc48_faces": mc48["n_face"], "mc64_faces": mc64["n_face"], "mc96_faces": mc96["n_face"],
 "mc48_radius_err": mc48["radius_err"], "mc64_radius_err": mc64["radius_err"], "mc96_radius_err": mc96["radius_err"],
 "mc96_volume_err": mc96["volume_err"], "mc_level_radius": mc_level["radius"],
 "poisson_radius": poisson_r, "poisson_err": poisson_err, "noisy_radius": noisy_r, "noisy_err": noisy_err,
 "zero_contours": zero_n, "half_contours": half_n, "f_center": float(f_center), "f_far": float(f_far),
 "n_pass": n_pass, "n_total": len(checks),
}
(OUT / "e120_surface_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] MC：48 {mc48['n_face']} 面/{mc48['radius_err']:.6f}；64 {mc64['n_face']} 面/{mc64['radius_err']:.6f}；96 {mc96['n_face']} 面/{mc96['radius_err']:.6f}")
print(f"[lab] 体积误差 {100*mc96['volume_err']:.3f}%｜等值面 +0.1 半径 {mc_level['radius']:.4f}")
print(f"[lab] Poisson：真值半径 {R:.4f}｜恢复 {poisson_r:.4f}｜带噪 {noisy_r:.4f}｜零边界 {zero_n} 轮廓｜半翻 {half_n} 轮廓")
for name in ("e120_representations.png", "e120_mc.png", "e120_poisson.png"):
    print(f"[lab] PNG sha256({name})={hashlib.sha256((FIG/name).read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e120_surface_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
