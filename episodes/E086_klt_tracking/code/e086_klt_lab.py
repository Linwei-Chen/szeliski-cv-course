#!/usr/bin/env python3
"""E086 §7.1.5–7.1.6：KLT 特征跟踪/漂移/仿射/大位移（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift as ndshift

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261106)

H = W = 180
BASE = 0.05 + 0.90 * (lambda t: (t - t.min()) / (t.max() - t.min()))(
    gaussian_filter(RNG.normal(0, 1, (H, W)), 1.2))
CX, CY = W // 2, H // 2
R = 15  # 31x31 patch
TEMPLATE = BASE[CY - R:CY + R + 1, CX - R:CX + R + 1].copy()
UU, VV = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))


def patch_at(im, cx, cy):
    return im[cy - R:cy + R + 1, cx - R:cx + R + 1]


def ssd(a, b):
    return float(np.mean((a - b) ** 2))


def parabolic(sm, s0, sp):
    den = sm - 2 * s0 + sp
    if abs(den) < 1e-12:
        return 0.0
    return float(0.5 * (sm - sp) / den)


def track_translation(frames, update=False, noise=0.0, rng=None):
    """整数搜索±2+抛物线亚像素；update=True 每帧重采样模板。"""
    rng = rng or np.random.default_rng(0)
    tmpl = TEMPLATE.copy()
    est = np.zeros(2)
    traj = []
    for f in frames:
        cx0, cy0 = int(round(CX + est[0])), int(round(CY + est[1]))
        best = (1e9, 0, 0)
        ssds = {}
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                p = patch_at(f, cx0 + dx, cy0 + dy)
                v = ssd(p, tmpl); ssds[(dx, dy)] = v
                if v < best[0]:
                    best = (v, dx, dy)
        _, bx, by = best
        ox = parabolic(ssds.get((bx - 1, by), best[0]), best[0], ssds.get((bx + 1, by), best[0]))
        oy = parabolic(ssds.get((bx, by - 1), best[0]), best[0], ssds.get((bx, by + 1), best[0]))
        est = np.array([cx0 + bx + ox - CX, cy0 + by + oy - CY])
        traj.append(est.copy())
        if update:
            p = patch_at(f, int(round(CX + est[0])), int(round(CY + est[1])))
            tmpl = p + rng.normal(0, noise, p.shape)
    return np.array(traj)


def feature_stats():
    n = 21
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    corner = ((xx >= 0) & (yy >= 0)).astype(float) + 0.03 * RNG.normal(size=(n, n))
    edge = (xx >= 0).astype(float) + 0.03 * RNG.normal(size=(n, n))
    flat = 0.5 + 0.03 * RNG.normal(size=(n, n))
    out = []
    for im in (corner, edge, flat):
        gx = gaussian_filter(im, 0.8, order=1, axes=1)
        gy = gaussian_filter(im, 0.8, order=1, axes=0)
        A = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                      [np.sum(gx * gy), np.sum(gy * gy)]])
        out.append(np.linalg.eigvalsh(A))
    return np.array(out)


def affine_warp(im, p, order=1):
    M = np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]])
    t = np.array([p[2], p[5]])
    x = CX + UU * M[0, 0] + VV * M[0, 1] + t[0]
    y = CY + UU * M[1, 0] + VV * M[1, 1] + t[1]
    return map_coordinates(im, [y, x], order=order, mode="nearest")


def affine_gn(frame, p0, iters=20):
    p = p0.copy(); hist = []
    for _ in range(iters):
        Iw = affine_warp(frame, p)
        e = Iw - TEMPLATE
        gx = gaussian_filter(frame, 0.8, order=1, axes=1)
        gy = gaussian_filter(frame, 0.8, order=1, axes=0)
        # 在模板坐标采目标梯度（与 E085 的链式法则同形）
        M = np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]])
        x = CX + UU * M[0, 0] + VV * M[0, 1] + p[2]
        y = CY + UU * M[1, 0] + VV * M[1, 1] + p[5]
        gxx = map_coordinates(gx, [y, x], order=1, mode="nearest")
        gyy = map_coordinates(gy, [y, x], order=1, mode="nearest")
        J = np.stack([gxx * UU, gxx * VV, gxx, gyy * UU, gyy * VV, gyy], axis=-1).reshape(-1, 6)
        ee = e.ravel()
        A = J.T @ J; b = -(J.T @ ee)
        dp = np.linalg.solve(A + 1e-6 * np.eye(6), b)
        p = p + dp
        hist.append(float(np.sqrt(np.mean(e * e))))
        if np.linalg.norm(dp) < 1e-8:
            break
    return p, hist


def true_affine(t, n):
    ang = np.deg2rad(0.35 * t); s = 1.0 + 0.0025 * t
    A = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    tx = 0.15 * t; ty = -0.10 * t
    p = np.array([A[0, 0] - 1, A[0, 1], tx, A[1, 0], A[1, 1] - 1, ty])
    return p, A, np.array([tx, ty])


def make_affine_frame(t):
    p, A, tr = true_affine(t, 0)
    M = np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]])
    # 逆映射生成：frame(x)=BASE(M^{-1}(x-cc-tr)+cc)，内容按 M,tr 正向移动
    cc = np.array([CX, CY]); Mi = np.linalg.inv(M); off = cc - Mi @ (cc + tr)
    return map_coordinates(BASE, [Mi[1, 0] * np.arange(W)[None, :] + Mi[1, 1] * np.arange(H)[:, None] + off[1],
                                  Mi[0, 0] * np.arange(W)[None, :] + Mi[0, 1] * np.arange(H)[:, None] + off[0]],
                           order=3, mode="nearest")


def search_shift(B, T, rng=4):
    best = (1e9, 0, 0)
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            s = ssd(patch_at(B, CX + dx, CY + dy), T)
            if s < best[0]:
                best = (s, dx, dy)
    return best


def pyramid_shift(B, levels=3):
    pb = [B]; pt = [BASE]
    for _ in range(levels - 1):
        pb.append(gaussian_filter(pb[-1], 1.0)[::2, ::2])
        pt.append(gaussian_filter(pt[-1], 1.0)[::2, ::2])
    u = np.zeros(2)
    for lv in range(levels - 1, -1, -1):
        scale = 2 ** lv
        Bs, Ts = pb[lv], pt[lv]
        cx, cy = Bs.shape[1] // 2, Bs.shape[0] // 2
        r = max(2, R // scale)
        tp = Ts[cy - r:cy + r + 1, cx - r:cx + r + 1]
        rr = 4 if lv == levels - 1 else 2
        best = (1e9, 0, 0)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                px = int(round(cx + u[0] / scale)) + dx
                py = int(round(cy + u[1] / scale)) + dy
                cp = Bs[py - r:py + r + 1, px - r:px + r + 1]
                if cp.shape != tp.shape:
                    continue
                s = ssd(cp, tp)
                if s < best[0]:
                    best = (s, dx, dy)
        u = np.array([u[0] + best[1] * scale, u[1] + best[2] * scale])
    return u


def brightness_track(frame, gain, bias):
    def raw(p):
        return ssd(patch_at(frame, CX + int(round(p[0])), CY + int(round(p[1]))), TEMPLATE)
    def comp(p):
        q = patch_at(frame, CX + int(round(p[0])), CY + int(round(p[1])))
        A = np.stack([q.ravel(), np.ones(q.size)], 1)
        a, b = np.linalg.lstsq(A, TEMPLATE.ravel(), rcond=None)[0]
        return float(np.mean((a * q + b - TEMPLATE) ** 2))
    def best(fn):
        vals = {}
        bb = (1e9, 0, 0)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                v = fn((dx, dy)); vals[(dx, dy)] = v
                if v < bb[0]:
                    bb = (v, dx, dy)
        _, bx, by = bb
        ox = parabolic(vals.get((bx - 1, by), bb[0]), bb[0], vals.get((bx + 1, by), bb[0]))
        oy = parabolic(vals.get((bx, by - 1), bb[0]), bb[0], vals.get((bx, by + 1), bb[0]))
        return np.array([bx + ox, by + oy], float)
    return best(raw), best(comp)


def main():
    print(f"[lab] E086 KLT/特征跟踪 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    # 1) 特征质量
    ev = feature_stats()
    corner_min, edge_min, flat_max = ev[0, 0], ev[1, 0], ev[2, 1]
    # 2) 平移序列：80 帧，0.15/-0.10 像素每帧
    n = 80
    frames = [ndshift(BASE, (-0.10 * t, 0.15 * t), order=3, mode="nearest") for t in range(n)]
    truth = np.array([[0.15 * t, -0.10 * t] for t in range(n)])
    traj_fixed = track_translation(frames, update=False)
    traj_resamp = track_translation(frames, update=True, noise=0.04)
    drift_fixed = float(np.linalg.norm(traj_fixed[-1] - truth[-1]))
    drift_resamp = float(np.linalg.norm(traj_resamp[-1] - truth[-1]))
    # 3) 亮度变化：增益 1.25/偏置 0.08
    fb = np.clip(1.8 * ndshift(BASE, (-2.7, 1.6), order=3, mode="nearest") + 0.15, 0, 1)
    raw_est, comp_est = brightness_track(fb, 1.25, 0.08)
    bright_raw_err = float(np.linalg.norm(raw_est - np.array([1.6, -2.7])))
    bright_comp_err = float(np.linalg.norm(comp_est - np.array([1.6, -2.7])))
    # 4) 仿射序列
    t_aff = 30
    aff_frame = make_affine_frame(t_aff)
    p_true, A_true, tr_true = true_affine(t_aff, 0)
    # 平移初始化：固定模板在当前帧整像素搜索 ±4
    _, ix, iy = search_shift(aff_frame, TEMPLATE, rng=4)
    p0 = np.array([0, 0, ix, 0, 0, iy], float)
    p_hat, hist = affine_gn(aff_frame, p0)
    # 角点误差：模板四角映射到目标
    corners = np.array([[-R, -R], [R, -R], [R, R], [-R, R]], float)
    def map_corners(p):
        M = np.array([[1 + p[0], p[1]], [p[3], 1 + p[4]]])
        return corners @ M.T + np.array([p[2], p[5]])
    err_trans = float(np.mean(np.linalg.norm(map_corners(p0) - map_corners(p_true), axis=1)))
    err_aff = float(np.mean(np.linalg.norm(map_corners(p_hat) - map_corners(p_true), axis=1)))
    # 5) 大位移：10 像素
    big = ndshift(BASE, (-6.0, 10.0), order=3, mode="nearest")
    local = search_shift(big, TEMPLATE, rng=3)
    local_err = float(np.linalg.norm(np.array([local[1], local[2]]) - np.array([10.0, -6.0])))
    pyr = pyramid_shift(big, levels=3)
    pyr_err = float(np.linalg.norm(pyr - np.array([10.0, -6.0])))
    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    axes[0].imshow(BASE, cmap="gray", extent=(0, W, H, 0))
    axes[0].add_patch(plt.Rectangle((CX - R, CY - R), 2 * R, 2 * R, fill=False, ec=ACCENT, lw=2))
    axes[0].set_title("跟踪窗口：双方向纹理", fontsize=14); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].plot(np.linalg.norm(traj_fixed - truth, axis=1), color=BLUE, lw=1.8, label="固定模板")
    axes[1].plot(np.linalg.norm(traj_resamp - truth, axis=1), color=ACCENT, lw=1.8, label="每帧重采样")
    axes[1].set_xlabel("帧", fontsize=13); axes[1].set_ylabel("位置误差（像素）", fontsize=13)
    axes[1].set_title("漂移：重采样会累积", fontsize=14); axes[1].tick_params(labelsize=11)
    axes[1].legend(fontsize=11, frameon=False); axes[1].grid(alpha=0.3)
    axes[2].plot(hist, color=GREEN, lw=1.8)
    axes[2].set_xlabel("迭代", fontsize=13); axes[2].set_ylabel("模板残差 RMSE", fontsize=13)
    axes[2].set_title("仿射细化：残差下降", fontsize=14); axes[2].tick_params(labelsize=11)
    axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "e086_klt.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 第二图：特征值 + 大位移
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    xs = np.arange(3); w = 0.36
    axes[0].bar(xs - w / 2, ev[:, 0], width=w, color=BLUE, label="最小特征值")
    axes[0].bar(xs + w / 2, ev[:, 1], width=w, color=GREEN, label="最大特征值")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(["角点", "边缘", "平坦"])
    axes[0].set_yscale("log"); axes[0].set_ylabel("特征值（对数）", fontsize=13)
    axes[0].set_title("好特征：两个方向都有梯度", fontsize=14)
    axes[0].tick_params(labelsize=12); axes[0].legend(fontsize=10, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axes[1].plot([0, 1], [local_err, pyr_err], "o-", color=ACCENT, lw=1.8)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["局部±3", "三层金字塔"], fontsize=12)
    axes[1].set_ylabel("位移误差（像素）", fontsize=13)
    axes[1].set_title("大位移：金字塔救场", fontsize=14); axes[1].tick_params(labelsize=11)
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "e086_search.png", dpi=185, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)
    # 第三图：仿射前后角点
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].imshow(aff_frame, cmap="gray", extent=(0, W, H, 0))
    for p, col, lab in [(p0, BLUE, "平移"), (p_hat, GREEN, "仿射"), (p_true, ACCENT, "真值")]:
        q = map_corners(p) + np.array([CX, CY])
        axes[0].plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=col, lw=2, label=lab)
    axes[0].set_title("角点对齐：平移 vs 仿射", fontsize=14); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(fontsize=11, frameon=False)
    bars = axes[1].bar(["平移", "仿射"], [err_trans, err_aff], color=[BLUE, GREEN])
    for b, v in zip(bars, [err_trans, err_aff]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.3f}",
                     ha="center", fontsize=12)
    axes[1].set_ylim(0, max(err_trans, err_aff) * 1.25)
    axes[1].set_ylabel("角点平均误差（像素）", fontsize=13)
    axes[1].set_title("仿射把形变吃进去", fontsize=14); axes[1].tick_params(labelsize=12)
    fig.tight_layout(); fig.savefig(FIG / "e086_affine.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 断言
    err_aff_3dp = round(err_aff, 3)
    checks = [
        ("角点最小特征值远高于边缘", corner_min > 10 * max(edge_min, 1e-12), f"corner={corner_min:.2f}, edge={edge_min:.2e}"),
        ("边缘最大特征值远高于平坦", ev[1, 1] > 10 * max(flat_max, 1e-12), f"edge_max={ev[1,1]:.2f}, flat={flat_max:.2e}"),
        ("固定模板漂移小", drift_fixed < 0.10, f"drift={drift_fixed:.4f}px"),
        ("重采样漂移更大", drift_resamp > 5 * max(drift_fixed, 1e-6), f"{drift_resamp:.3f} vs {drift_fixed:.4f}px"),
        ("亮度变化：原始 SSD 有偏差", bright_raw_err > 0.1, f"raw_err={bright_raw_err:.3f}px"),
        ("亮度变化：增益/偏置补偿更准", bright_comp_err < 0.1 and bright_comp_err < 0.5 * bright_raw_err, f"comp_err={bright_comp_err:.3f}px（{bright_raw_err/bright_comp_err:.1f}× 改善）"),
        ("仿射序列：平移模型误差大", err_trans > 1.0, f"corner_err={err_trans:.3f}px"),
        ("仿射细化：角点误差小", err_aff < 0.4, f"corner_err={err_aff_3dp:.3f}px"),
        ("仿射细化：残差下降", hist[-1] < 0.5 * hist[0], f"RMSE {hist[0]:.4f}→{hist[-1]:.4f}"),
        ("大位移：金字塔优于局部", pyr_err < 0.5 and local_err > 2.0, f"pyr={pyr_err:.3f}, local={local_err:.3f}px"),
    ]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(corner_min=float(corner_min), edge_min=float(edge_min), edge_max=float(ev[1, 1]),
                    step_per_frame=0.15, bright_gain=1.8, bright_bias=0.15, drift_fixed_3dp=round(drift_fixed, 3),
                    bright_speedup=float(bright_raw_err / bright_comp_err),
                    flat_max=float(flat_max), drift_fixed=drift_fixed, drift_resamp=drift_resamp,
                    bright_raw_err=bright_raw_err, bright_comp_err=bright_comp_err,
                    err_trans=err_trans, err_aff=err_aff, err_aff_3dp=err_aff_3dp, aff_hist=[float(x) for x in hist],
                    local_err=local_err, pyr_err=pyr_err, p_true=p_true.tolist(), p_hat=p_hat.tolist(),
                    n_pass=npass, n_total=len(checks))
    (OUT / "e086_klt_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e086_klt_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e086_klt_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
