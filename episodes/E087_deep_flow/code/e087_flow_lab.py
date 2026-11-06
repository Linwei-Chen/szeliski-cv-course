#!/usr/bin/env python3
"""E087 §9.3.1：相关体/迭代细化/稀疏化（RAFT 机制的最小实验，程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
from scipy.signal import correlate2d

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261107)
N = 64; R = 12; TRUE = np.array([5.4, -3.2])  # x,y


def texture(seed=0, repeat=False):
    rng = np.random.default_rng(seed)
    im = gaussian_filter(rng.normal(0, 1, (N, N)), 1.1)
    im = (im - im.min()) / (im.max() - im.min())
    if repeat:
        im[10:35, 10:35] = im[34:59, 34:59]
    return im


def ncc_map(target, tmpl):
    t = tmpl - tmpl.mean(); t = t / (np.linalg.norm(t) + 1e-12)
    ones = np.ones_like(tmpl)
    area = tmpl.size
    mean = correlate2d(target, ones, mode="valid") / area
    sq = correlate2d(target * target, ones, mode="valid") - area * mean * mean
    num = correlate2d(target, t, mode="valid")
    return num / (np.sqrt(np.maximum(sq, 1e-12)))


def subpix_peak(C):
    iy, ix = np.unravel_index(np.argmax(C), C.shape)
    def off(im, i0, ip):
        den = im - 2 * C[iy, ix] + ip
        return 0.0 if abs(den) < 1e-12 else 0.5 * (im - ip) / den
    ox = off(C[iy, ix - 1], 0, C[iy, ix + 1])
    oy = off(C[iy - 1, ix], 0, C[iy + 1, ix])
    return np.array([ix + ox, iy + oy]), float(C[iy, ix])


def iterative(C, iters=8):
    flow = np.array([20.0, 20.0]); hist = []
    for _ in range(iters):
        cx, cy = int(round(flow[0])), int(round(flow[1]))
        x0, x1 = max(0, cx - 3), min(C.shape[1], cx + 4)
        y0, y1 = max(0, cy - 3), min(C.shape[0], cy + 4)
        sub = C[y0:y1, x0:x1]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        def off(im, ip):
            den = im - 2 * sub[iy, ix] + ip
            return 0.0 if abs(den) < 1e-12 else 0.5 * (im - ip) / den
        if 0 < ix < sub.shape[1] - 1:
            ox = off(sub[iy, ix - 1], sub[iy, ix + 1])
        else:
            ox = 0.0
        if 0 < iy < sub.shape[0] - 1:
            oy = off(sub[iy - 1, ix], sub[iy + 1, ix])
        else:
            oy = 0.0
        flow = np.array([x0 + ix + ox, y0 + iy + oy])
        hist.append(flow.copy())
    return flow, hist


def main():
    print(f"[lab] E087 相关体/迭代细化 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    base = texture(1)
    tmpl = base[20:20 + 2 * R + 1, 20:20 + 2 * R + 1]
    target = ndshift(base, (TRUE[1], TRUE[0]), order=3, mode="nearest")
    C = ncc_map(target, tmpl)
    peak, pv = subpix_peak(C)
    true_top = np.array([20 + TRUE[0], 20 + TRUE[1]])
    # 相关体峰值对应模板左上角在目标中的位置（减去模板左上角 20）
    shift_est = peak - 20
    err_sub = float(np.linalg.norm(shift_est - TRUE))
    # 迭代细化：从零流开始，局部 soft-argmax
    flow, hist = iterative(C, iters=8)
    flow_err = [float(np.linalg.norm(TRUE))] + [float(np.linalg.norm(f - (np.array([20, 20]) + TRUE))) for f in hist]
    # 重复纹理的歧义
    base_rep = texture(2, repeat=True)
    tmpl_rep = base_rep[10:10 + 2 * R + 1, 10:10 + 2 * R + 1]
    target_rep = ndshift(base_rep, (TRUE[1], TRUE[0]), order=3, mode="nearest")
    C_rep = ncc_map(target_rep, tmpl_rep)
    flat = np.sort(C_rep.ravel())[::-1]
    # 取两个非极大值附近的峰（简单：排除第一个峰邻域）
    p1 = np.unravel_index(np.argmax(C_rep), C_rep.shape)
    mask = np.ones_like(C_rep, bool)
    yy, xx = np.mgrid[0:C_rep.shape[0], 0:C_rep.shape[1]]
    mask[(yy - p1[0]) ** 2 + (xx - p1[1]) ** 2 < 25] = False
    p2v = C_rep[mask].max()
    rep_ratio = float(C_rep[p1] / p2v)
    # 唯一纹理的峰/次峰
    C_u = C.copy(); mask2 = np.ones_like(C_u, bool)
    p1u = np.unravel_index(np.argmax(C_u), C_u.shape)
    mask2[(yy - p1u[0]) ** 2 + (xx - p1u[1]) ** 2 < 25] = False
    unique_ratio = float(C_u[p1u] / C_u[mask2].max())
    # 亮度变化下的 NCC
    target_b = np.clip(1.6 * target + 0.1, 0, 1)
    Cb = ncc_map(target_b, tmpl)
    peak_b, _ = subpix_peak(Cb)
    err_bright = float(np.linalg.norm((peak_b - 20) - TRUE))
    # 内存账
    full_elems = (N * N) ** 2
    sparse_elems = N * N * 5
    full_bytes = full_elems * 4
    sparse_bytes = sparse_elems * 4
    reduction = full_bytes / sparse_bytes
    # 图 1：模型时间线
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    models = [("FlowNetS", 2015, "端到端"), ("SPyNet", 2017, "金字塔"), ("PWC-Net", 2018, "特征金字塔+代价体"),
              ("IRR", 2019, "权重共享"), ("RAFT", 2020, "全相关体+迭代")]
    xs = np.arange(len(models))
    ax.plot(xs, [0] * len(models), color=INK, lw=2, zorder=1)
    for i, (name, yr, tag) in enumerate(models):
        ax.plot(i, 0, "o", color=BLUE if name != "RAFT" else ACCENT, ms=10, zorder=2)
        ax.text(i, 0.22, name, ha="center", fontsize=14, weight="bold")
        ax.text(i, -0.24, str(yr), ha="center", fontsize=12)
        ax.text(i, -0.45, tag, ha="center", fontsize=11)
    ax.set_ylim(-0.7, 0.55); ax.set_xlim(-0.6, len(models) - 0.4)
    ax.axis("off"); ax.set_title("光流网络：从端到端到全相关体", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "e087_timeline.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 图 2：相关体 + 稀疏内存
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    im = axes[0].imshow(C, cmap="magma")
    axes[0].plot(peak[0], peak[1], "o", mfc="none", mec="cyan", ms=12)
    axes[0].set_title("相关体切片：候选位置的匹配分数", fontsize=13)
    axes[0].set_xlabel("候选 x", fontsize=12); axes[0].set_ylabel("候选 y", fontsize=12)
    axes[0].tick_params(labelsize=11)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].bar(["全相关体", "稀疏 top-5"], [full_bytes / 1e6, sparse_bytes / 1e6],
                color=[BLUE, GREEN])
    axes[1].set_yscale("log"); axes[1].set_ylabel("内存（MB，对数）", fontsize=13)
    axes[1].set_title(f"稀疏化省 {reduction:.0f}×", fontsize=13)
    axes[1].tick_params(labelsize=11)
    fig.tight_layout(); fig.savefig(FIG / "e087_volume.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 图 3：迭代误差与流
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(range(len(flow_err)), flow_err, "o-", color=ACCENT, lw=1.8)
    axes[0].set_xlabel("迭代", fontsize=13); axes[0].set_ylabel("流误差（像素）", fontsize=13)
    axes[0].set_title("从零流出发：逐步逼近", fontsize=14); axes[0].tick_params(labelsize=11); axes[0].grid(alpha=0.3)
    axes[1].plot([0, TRUE[0]], [0, TRUE[1]], color=ACCENT, lw=5, alpha=0.45, label="真值")
    axes[1].plot([0, flow[0] - 20], [0, flow[1] - 20], color=GREEN, lw=2, ls="--", label="估计")
    axes[1].set_xlim(-1, 7); axes[1].set_ylim(-4.5, 1); axes[1].set_aspect("equal")
    axes[1].set_title("真值（红）与估计（绿）", fontsize=14); axes[1].tick_params(labelsize=11)
    axes[1].legend(fontsize=11, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "e087_iter.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 断言
    err_sub_3dp = round(err_sub, 3)
    err_bright_3dp = round(err_bright, 3)
    flow_err_final_3dp = round(flow_err[-1], 3)
    checks = [
        ("全相关体尺寸", full_elems == (N * N) ** 2, f"{full_elems} 元素"),
        ("全相关体内存", abs(full_bytes - 67108864) < 1, f"{full_bytes/1e6:.1f} MB"),
        ("稀疏化省百倍以上", reduction > 100, f"{reduction:.0f}×"),
        ("整数峰在真值附近", np.linalg.norm((peak - 20) - np.round(TRUE)) < 1.0, f"peak=({peak[0]-20:.2f},{peak[1]-20:.2f})"),
        ("亚像素峰误差小", err_sub < 0.2, f"err={err_sub_3dp:.3f}px"),
        ("亮度变化下 NCC 稳", err_bright < 0.2, f"err={err_bright_3dp:.3f}px"),
        ("重复纹理峰值比低", rep_ratio < 1.3, f"ratio={rep_ratio:.3f}"),
        ("唯一纹理峰值比高", unique_ratio > 2.0, f"ratio={unique_ratio:.2f}"),
        ("迭代误差单调下降", all(flow_err[i + 1] <= flow_err[i] + 1e-6 for i in range(len(flow_err) - 1)), f"{flow_err[0]:.2f}→{flow_err_final_3dp:.3f}"),
        ("迭代最终误差小", flow_err[-1] < 0.2 and flow_err[-1] < 0.1 * flow_err[0], f"err={flow_err_final_3dp:.3f}px"),
    ]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(true=list(TRUE), peak=[float(peak[0]), float(peak[1])], err_sub=err_sub, err_sub_3dp=err_sub_3dp,
                    err_bright=err_bright, err_bright_3dp=err_bright_3dp, flow_err_final_3dp=flow_err_final_3dp, rep_ratio=rep_ratio, unique_ratio=unique_ratio,
                    flow_err=flow_err, full_elems=full_elems, sparse_elems=sparse_elems,
                    full_bytes=full_bytes, sparse_bytes=sparse_bytes,
                    full_mb=full_bytes / 1e6, sparse_mb=sparse_bytes / 1e6,
                    bright_gain=1.6, bright_bias=0.1, reduction=reduction,
                    n_pass=npass, n_total=len(checks))
    (OUT / "e087_flow_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e087_flow_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e087_flow_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
