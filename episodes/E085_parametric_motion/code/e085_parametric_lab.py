#!/usr/bin/env python3
"""E085 §9.2/§9.2.2：参数运动 Gauss–Newton、分块 Hessian 近似、样条控制顶点（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261105)

H, W = 160, 192
BASE = 0.05 + 0.90 * (lambda t: (t - t.min()) / (t.max() - t.min()))(
    gaussian_filter(RNG.normal(0, 1, (H, W)), 1.2))
P_TRUE = np.array([0.06, -0.04, 2.3, 0.03, 0.05, -1.7])  # a11-1,a12,tx,a21,a22-1,ty


def apply_affine(im, p, order=1):
    """output(x)=im(W(x;p))，W(x;p)=(I+[[p0,p1],[p3,p4]])x+(p2,p5)."""
    M = np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]])
    off = np.array([p[2], p[5]])
    return _warp_affine(im, M, off, order)


def _warp_affine(im, M, off, order=1):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    xp = M[0, 0] * xx + M[0, 1] * yy + off[0]
    yp = M[1, 0] * xx + M[1, 1] * yy + off[1]
    return map_coordinates(im, [yp, xp], order=order, mode="nearest")


# 目标图用逆 warp 生成：TARGET(x)=BASE(W^{-1}(x;p_true))，这样 TARGET(W(x;p_true))=BASE(x)
_M = np.array([[1 + P_TRUE[0], P_TRUE[1]], [P_TRUE[3], 1 + P_TRUE[4]]])
_off = np.array([P_TRUE[2], P_TRUE[5]])
_Minv = np.linalg.inv(_M)
TARGET = _warp_affine(BASE, _Minv, -_Minv @ _off, order=3)
MASK = np.zeros((H, W), bool); MASK[12:H-12, 12:W-12] = True


def params_to_M(p):
    return np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]]), np.array([p[2], p[5]])


def warp_img(im, p, shape):
    h, w = shape
    M, off = params_to_M(p)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    xp = M[0, 0] * xx + M[0, 1] * yy + off[0]
    yp = M[1, 0] * xx + M[1, 1] * yy + off[1]
    return map_coordinates(im, [yp, xp], order=1, mode="nearest"), xp, yp


def gn_level(B, T, mask, p0, iters=12):
    p = p0.copy(); hist = []
    h, w = B.shape
    for _ in range(iters):
        Iw, xp, yp = warp_img(T, p, (h, w))
        e = (Iw - B)[mask]
        Ix = gaussian_filter(T, 1.0, order=1, axes=1)
        Iy = gaussian_filter(T, 1.0, order=1, axes=0)
        gx = map_coordinates(Ix, [yp, xp], order=1, mode="nearest")[mask]
        gy = map_coordinates(Iy, [yp, xp], order=1, mode="nearest")[mask]
        xx = np.meshgrid(np.arange(w), np.arange(h))[0].astype(float)[mask]
        yy = np.meshgrid(np.arange(w), np.arange(h))[1].astype(float)[mask]
        J = np.stack([gx * xx, gx * yy, gx, gy * xx, gy * yy, gy], axis=1)
        A = J.T @ J; b = -(J.T @ e)
        dp = np.linalg.solve(A + 1e-6 * np.eye(6), b)
        p = p + dp
        hist.append(float(np.sqrt(np.mean(e * e))))
        if np.linalg.norm(dp) < 1e-8:
            break
    return p, hist, A


def gn_exact(p0, iters=12):
    return gn_level(BASE, TARGET, MASK, p0, iters)


def gn_patch(p0, patch=24, iters=12):
    p = p0.copy(); hist = []
    xx_full, yy_full = np.meshgrid(np.arange(W), np.arange(H))
    for _ in range(iters):
        Iw, xp, yp = warp_img(TARGET, p, (H, W))
        e = (Iw - BASE)
        Ix = gaussian_filter(TARGET, 1.0, order=1, axes=1)
        Iy = gaussian_filter(TARGET, 1.0, order=1, axes=0)
        gx = map_coordinates(Ix, [yp, xp], order=1, mode="nearest")
        gy = map_coordinates(Iy, [yp, xp], order=1, mode="nearest")
        A = np.zeros((6, 6)); b = np.zeros(6)
        for y0 in range(0, H - patch + 1, patch):
            for x0 in range(0, W - patch + 1, patch):
                sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
                m = MASK[sl]
                if m.sum() < patch * patch * 0.5:
                    continue
                gxx, gyy = gx[sl][m], gy[sl][m]
                G = np.array([[np.sum(gxx * gxx), np.sum(gxx * gyy)],
                              [np.sum(gxx * gyy), np.sum(gyy * gyy)]])
                ee = e[sl][m]
                bb = np.array([np.sum(gxx * ee), np.sum(gyy * ee)])
                xc = x0 + (patch - 1) / 2.0; yc = y0 + (patch - 1) / 2.0
                # 块中心处的运动雅可比 Jx0=∂x'/∂p（2×6），与块内像素梯度解耦
                Jc = np.array([[xc, yc, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, xc, yc, 1.0]])
                A += Jc.T @ G @ Jc
                b += -(Jc.T @ bb)
        dp = np.linalg.solve(A + 1e-6 * np.eye(6), b)
        p = p + dp
        hist.append(float(np.sqrt(np.mean(e[MASK] ** 2))))
        if np.linalg.norm(dp) < 1e-8:
            break
    return p, hist, A


def parametric_models_fig():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))
    ts = np.linspace(-1, 1, 9)

    def draw(ax, fn, color, title):
        qs = []
        for t0 in ts:
            for p in (np.stack([np.full_like(ts, t0), ts], 1),
                      np.stack([ts, np.full_like(ts, t0)], 1)):
                q = fn(p); qs.append(q)
                ax.plot(q[:, 0], q[:, 1], color=color, lw=1.3)
        # 淡淡的基准网格，便于看“动了哪里”
        for t0 in ts:
            ax.plot(np.full_like(ts, t0), ts, color="#c9c2b4", lw=0.6, ls="--", zorder=0)
            ax.plot(ts, np.full_like(ts, t0), color="#c9c2b4", lw=0.6, ls="--", zorder=0)
        ax.set_xlim(-1.75, 1.95)
        ax.set_ylim(-1.75, 1.95)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=15)
        ax.set_xticks([]); ax.set_yticks([])

    draw(axes[0], lambda p: p + np.array([0.28, 0.12]), BLUE, "平移：2 参数")
    A = np.array([[1.05, 0.32], [0.10, 0.90]])
    draw(axes[1], lambda p: p @ A.T + np.array([0.05, 0.0]), ACCENT, "仿射：6 参数")
    def proj(p):
        x, y = p[:, 0], p[:, 1]
        den = 1.0 + 0.22 * x + 0.10 * y
        return np.stack([(1.10 * x + 0.10 * y) / den,
                         (0.05 * x + 1.00 * y) / den], 1)
    draw(axes[2], proj, GREEN, "投射：8 参数")
    fig.suptitle("同一个网格：参数越多，形变越自由", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "e085_models.png", dpi=185, facecolor=PAPER)
    plt.close(fig)


def jacobian_fig(rel_errs, sizes):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.imshow(BASE, cmap="gray", extent=(0, W, H, 0))
    for y0 in range(0, H - 24 + 1, 24):
        for x0 in range(0, W - 24 + 1, 24):
            ax.plot([x0, x0 + 23, x0 + 23, x0, x0], [y0, y0, y0 + 23, y0 + 23, y0],
                    color=ACCENT, lw=0.7)
            ax.plot(x0 + 11.5, y0 + 11.5, "o", color=GREEN, ms=2.5)
    ax.set_title("每块只算一个运动雅可比", fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    ax.plot(sizes, rel_errs, "o-", color=BLUE, lw=1.8)
    ax.set_xlabel("块边长（像素）", fontsize=14)
    ax.set_ylabel("Hessian 相对误差", fontsize=14)
    ax.set_title("块大小改变近似误差（非单调）", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "e085_jacobian.png", dpi=185, facecolor=PAPER)
    plt.close(fig)


def spline_fig(field_u, field_v, ctrl, reg_lams, reg_rough, reg_rmse):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    step = 10
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    ax.quiver(xx, yy, field_u[::step, ::step], -field_v[::step, ::step], color=BLUE,
              angles="xy", scale_units="xy", scale=0.9)
    ax.plot(ctrl[0], ctrl[1], "o", color=ACCENT, ms=4)
    ax.set_title("控制顶点与稠密运动场", fontsize=14)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    ax.set_title("四叉树：只在断裂处细分", fontsize=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for i in range(4):
        for j in range(4):
            ax.add_patch(plt.Rectangle((i / 4, j / 4), 1 / 4, 1 / 4, fill=False, ec=BLUE, lw=1.0))
    for i, j in [(2, 2), (1, 2), (2, 1)]:
        for a in range(2):
            for b in range(2):
                ax.add_patch(plt.Rectangle((i / 4 + a / 8, j / 4 + b / 8), 1 / 8, 1 / 8,
                                           fill=False, ec=ACCENT, lw=1.1))
    ax = axes[2]
    ax.loglog(reg_lams, reg_rough, "o-", color=GREEN, lw=1.6, label="粗糙度")
    ax.loglog(reg_lams, reg_rmse, "s--", color=ACCENT, lw=1.4, label="数据误差")
    ax.set_xlabel("正则化权重 λ", fontsize=13)
    ax.set_title("正则化：平滑换拟合", fontsize=14)
    ax.tick_params(labelsize=12); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e085_spline.png", dpi=185, facecolor=PAPER)
    plt.close(fig)


def main():
    print(f"[lab] E085 参数运动/样条 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    # 1) 由粗到细：半分辨率先估，平移加倍后全分辨率细化
    B2 = gaussian_filter(BASE, 1.0)[::2, ::2]
    T2 = gaussian_filter(TARGET, 1.0)[::2, ::2]
    MASK2 = np.zeros_like(B2, bool); MASK2[6:-6, 6:-6] = True
    p_coarse, hist_coarse, _ = gn_level(B2, T2, MASK2, np.zeros(6), iters=30)
    p0_full = p_coarse.copy(); p0_full[2] *= 2; p0_full[5] *= 2
    p_hat, hist, A_exact = gn_level(BASE, TARGET, MASK, p0_full, iters=12)
    err_coarse = float(np.max(np.abs(p_coarse - np.array([P_TRUE[0], P_TRUE[1], P_TRUE[2]/2, P_TRUE[3], P_TRUE[4], P_TRUE[5]/2]))))
    err_param = float(np.max(np.abs(p_hat - P_TRUE)))
    coarse_drop = float(hist_coarse[0] / hist_coarse[-1])
    drop = float(hist[0] / hist[-1])
    # 2) 雅可比链式法则有限差分
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    M, off = params_to_M(P_TRUE)
    xp = M[0, 0] * xx + M[0, 1] * yy + off[0]
    yp = M[1, 0] * xx + M[1, 1] * yy + off[1]
    analytic = np.zeros((H, W, 2, 6))
    analytic[..., 0, 0] = xx; analytic[..., 0, 1] = yy; analytic[..., 0, 2] = 1
    analytic[..., 1, 3] = xx; analytic[..., 1, 4] = yy; analytic[..., 1, 5] = 1
    d = 1e-6
    fd = np.zeros_like(analytic)
    for k in range(6):
        pp = P_TRUE.copy(); pp[k] += d
        Mm, ofm = params_to_M(pp)
        xpm = Mm[0, 0] * xx + Mm[0, 1] * yy + ofm[0]
        ypm = Mm[1, 0] * xx + Mm[1, 1] * yy + ofm[1]
        pm = P_TRUE.copy(); pm[k] -= d
        Mp, ofp = params_to_M(pm)
        xpp = Mp[0, 0] * xx + Mp[0, 1] * yy + ofp[0]
        ypp = Mp[1, 0] * xx + Mp[1, 1] * yy + ofp[1]
        fd[..., 0, k] = (xpm - xpp) / (2 * d)
        fd[..., 1, k] = (ypm - ypp) / (2 * d)
    jac_rel = float(np.max(np.abs(fd - analytic)) / (np.max(np.abs(analytic)) + 1e-12))
    # 3) 分块近似：不同块大小的 Hessian 相对误差
    sizes = [8, 12, 16, 24, 32]
    rel_errs = []
    p_patch = None; hist_patch = None; ratio = None
    for s in sizes:
        _, _, A_p = gn_patch(p_hat, patch=s, iters=1)
        rel_errs.append(float(np.linalg.norm(A_p - A_exact) / np.linalg.norm(A_exact)))
    p_patch, hist_patch, A_patch = gn_patch(p0_full, patch=24, iters=12)
    n_patch = (H // 24) * (W // 24)
    # 6×6 外积：逐像素 36 次 vs 块内 2×2（4 次/像素）+ 每块 36 次
    ratio = float((4 * H * W + 36 * n_patch) / (36 * H * W))
    patch_err = float(np.max(np.abs(p_patch - P_TRUE)))
    # 4) 样条：5x5 控制顶点，双线性基
    gx = np.linspace(12, W - 12, 5); gy = np.linspace(12, H - 12, 5)
    cx, cy = np.meshgrid(gx, gy)
    ctrl_true_u = 5.0 + 0.8 * np.sin(cx / 38.0) * np.cos(cy / 45.0)
    ctrl_true_v = -3.0 + 0.8 * np.cos(cx / 50.0) * np.sin(cy / 32.0)
    # 双线性基矩阵
    def basis_matrix(px, py):
        B = np.zeros((len(px), 25))
        for j in range(25):
            xj = gx[j % 5]; yj = gy[j // 5]
            wx = np.clip(1 - np.abs(px - xj) / (gx[1] - gx[0]), 0, 1)
            wy = np.clip(1 - np.abs(py - yj) / (gy[1] - gy[0]), 0, 1)
            B[:, j] = wx * wy
        # 边界外像素按最近控制点补齐权重，保证每行和为 1
        s = B.sum(1)
        B[s == 0, :] = 0
        B[s == 0, 0] = 1
        B = B / B.sum(1, keepdims=True)
        return B
    px = xx.ravel(); py = yy.ravel()
    B = basis_matrix(px, py)
    dense_u = (B @ ctrl_true_u.ravel()).reshape(H, W)
    dense_v = (B @ ctrl_true_v.ravel()).reshape(H, W)
    fit_u = np.linalg.lstsq(B, dense_u.ravel(), rcond=None)[0]
    fit_v = np.linalg.lstsq(B, dense_v.ravel(), rcond=None)[0]
    spline_rmse = float(np.sqrt(np.mean((B @ fit_u - dense_u.ravel()) ** 2 +
                                        (B @ fit_v - dense_v.ravel()) ** 2)))
    # 5) 含噪稠密场 + 正则化
    noise_u = dense_u + RNG.normal(0, 0.15, dense_u.shape)
    noise_v = dense_v + RNG.normal(0, 0.15, dense_v.shape)
    obs_u = noise_u.ravel(); obs_v = noise_v.ravel()
    # 二阶差分（5 点拉普拉斯）：只罚弯曲，不罚匀速平移
    rows = []
    for j in range(25):
        r, c = divmod(j, 5)
        if 0 < r < 4 and 0 < c < 4:
            row = np.zeros(25); row[j] = 4
            for k in (j - 5, j + 5, j - 1, j + 1):
                row[k] = -1
            rows.append(row)
    D = np.array(rows)
    def rough(c): return float(np.mean((D @ c) ** 2))
    def fit_reg(lam):
        Mx = B.T @ B + lam * (D.T @ D)
        cu = np.linalg.solve(Mx, B.T @ obs_u)
        cv = np.linalg.solve(Mx, B.T @ obs_v)
        rmse = float(np.sqrt(np.mean((B @ cu - obs_u) ** 2) + np.mean((B @ cv - obs_v) ** 2)))
        return rough(cu) + rough(cv), rmse, cu, cv
    rough0, rmse0, cu0, cv0 = fit_reg(0.0)
    lams = [0.1, 1.0, 10.0, 100.0, 1000.0]
    reg = [fit_reg(l) for l in lams]
    best = min(reg, key=lambda t: t[0])
    spline_rough_ratio = float(best[0] / rough0)
    spline_data_ratio = float(best[1] / rmse0)
    # 画图
    parametric_models_fig()
    jacobian_fig(rel_errs, sizes)
    spline_fig(dense_u, dense_v, (cx.ravel(), cy.ravel()), lams,
               [r[0] for r in reg], [r[1] for r in reg])
    # 断言
    err_coarse_5dp = round(err_coarse, 5)
    err_param_5dp = round(err_param, 5)
    n_params = 6
    patch_speedup = float(1.0 / ratio)
    checks = []
    checks.append(("粗层残差大降且参数可用", coarse_drop > 5 and err_coarse < 0.02,
                   f"RMSE {hist_coarse[0]:.3f}→{hist_coarse[-1]:.4f}（{coarse_drop:.1f}×），参数误差 {err_coarse_5dp:.5f}"))
    checks.append(("全分辨率恢复参数", err_param < 0.005, f"max|p̂−p|={err_param_5dp:.5f}"))
    checks.append(("解析雅可比=有限差分", jac_rel < 1e-5, f"rel={jac_rel:.2e}"))
    checks.append(("Hessian 正定", np.linalg.eigvalsh(A_exact).min() > 0, f"λmin={np.linalg.eigvalsh(A_exact).min():.3f}"))
    checks.append(("分块 Hessian 近似可用", rel_errs[sizes.index(24)] < 0.15, f"rel(24)={rel_errs[sizes.index(24)]:.3f}"))
    checks.append(("分块 GN 仍能恢复参数", patch_err < 0.02, f"max|p̂−p|={patch_err:.5f}"))
    checks.append(("分块大幅省外积", ratio < 0.15, f"{n_params}×{n_params} 外积成本比={ratio:.4f}，约 {patch_speedup:.3f}× 省"))
    checks.append(("样条重建精确", spline_rmse < 1e-10, f"RMSE={spline_rmse:.2e}px"))
    checks.append(("正则化压低粗糙度", spline_rough_ratio < 0.5, f"粗糙度比={spline_rough_ratio:.5f}"))
    checks.append(("正则化拟合误差仅小增", spline_data_ratio < 1.5, f"数据误差比={spline_data_ratio:.3f}"))
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(p_true=P_TRUE.tolist(), p_coarse=p_coarse.tolist(), p_hat=p_hat.tolist(),
                    err_coarse=err_coarse, err_coarse_5dp=err_coarse_5dp,
                    err_param=err_param, err_param_5dp=err_param_5dp,
                    coarse_rmse_start=hist_coarse[0], coarse_rmse_end=hist_coarse[-1],
                    coarse_drop=coarse_drop, n_params=n_params, patch_speedup=patch_speedup,
                    rmse_start=hist[0], rmse_end=hist[-1], drop=drop,
                    jac_rel=jac_rel, hess_min_eig=float(np.linalg.eigvalsh(A_exact).min()),
                    patch_sizes=sizes, patch_rel_errs=rel_errs,
                    patch_err=patch_err, patch_cost_ratio=ratio,
                    spline_rmse=spline_rmse, spline_rough_ratio=spline_rough_ratio,
                    spline_data_ratio=spline_data_ratio, n_pass=npass, n_total=len(checks))
    (OUT / "e085_parametric_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    rd_sha = hashlib.sha256((OUT / "e085_parametric_readings.json").read_bytes()).hexdigest()
    print(f"[lab] readings → e085_parametric_readings.json")
    print(f"[lab] readings sha256={rd_sha}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
