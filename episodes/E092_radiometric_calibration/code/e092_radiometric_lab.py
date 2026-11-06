#!/usr/bin/env python3
"""E092 辐射响应标定：多曝光拟合 gamma / 反演辐亮度。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 128
GAMMAS = (2.2, 2.1, 2.3)
EVS = np.arange(-7, 1)


def main():
    print(f"[lab] E092 辐射标定 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    rng = np.random.default_rng(92)
    base = gaussian_filter(rng.normal(size=(N, N)), 2.5)
    base = (base - base.min()) / (base.max() - base.min())
    rad = 10 ** (0.2 + 2.2 * base)
    dr = float(rad.max() / rad.min())
    obs = []
    for g in GAMMAS:
        ims = []
        for ev in EVS:
            im = np.clip((2.0 ** ev * rad) ** (1 / g), 0, 1) * 255
            ims.append(np.round(im))
        obs.append(np.array(ims))
    obs = np.array(obs)  # [channel, EV, y, x]
    est_g = []
    for ci, g in enumerate(GAMMAS):
        vals = []
        for i in range(N):
            for j in range(N):
                ims = obs[ci, :, i, j]
                m = (ims > 2) & (ims < 250)
                if m.sum() >= 4:
                    xx = EVS[m].astype(float); yy = np.log2(ims[m] / 255)
                    slope = np.polyfit(xx, yy, 1)[0]
                    if slope > 0: vals.append(slope)
        est_g.append(1.0 / np.median(vals))
    est_g = np.array(est_g)
    g_err = np.abs(est_g - np.array(GAMMAS))

    # 曝光饱和/覆盖
    sat = [(obs[0, k] >= 254).mean() for k in range(len(EVS))]
    short_sat = float(sat[0]); long_sat = float(sat[2])
    cover = float((obs[0] < 250).any(axis=0).mean())

    # 反演（短曝光 EV=-7）
    ev = -7; e = 2.0 ** ev
    im = obs[0, 0] / 255.0
    m = (im > 0.05) & (im < 0.95)
    rad_est = im ** est_g[0] / e
    rel = float(np.median(np.abs(rad_est[m] - rad[m]) / rad[m]))
    lin = im / e
    rel_lin = float(np.median(np.abs(lin[m] - rad[m]) / rad[m]))

    # EV 加倍输出比（选一个未饱和中灰像素）
    pix = np.unravel_index(np.argmin(np.abs(rad - 20)), rad.shape)
    ratio = float(((2.0 ** -6 * rad[pix]) ** (1 / 2.2)) / ((2.0 ** -7 * rad[pix]) ** (1 / 2.2)))
    ratio_ideal = float(2 ** (1 / 2.2))

    # 曲线单调
    x = np.linspace(0, 1, 256); curve = x ** (1 / 2.2)
    mono = bool(np.all(np.diff(curve) > 0))

    checks = [
        ("场景动态范围>100", dr > 100, f"{dr:.0f} 倍"),
        ("R 通道 gamma 误差<0.01", g_err[0] < 0.01, f"{est_g[0]:.4f} vs 2.2"),
        ("G 通道 gamma 误差<0.01", g_err[1] < 0.01, f"{est_g[1]:.4f} vs 2.1"),
        ("B 通道 gamma 误差<0.01", g_err[2] < 0.01, f"{est_g[2]:.4f} vs 2.3"),
        ("长曝光饱和>50%", long_sat > 0.5, f"{long_sat*100:.1f}%"),
        ("短曝光饱和<5%", short_sat < 0.05, f"{short_sat*100:.1f}%"),
        ("gamma 反演误差<2%", rel < 0.02, f"{rel*100:.2f}%"),
        ("线性假设误差>10%", rel_lin > 0.10, f"{rel_lin*100:.1f}%"),
        ("EV 加倍比值≈2^(1/γ)", abs(ratio - ratio_ideal) < 0.02, f"{ratio:.3f} vs {ratio_ideal:.3f}"),
        ("响应曲线单调", mono, "全段递增"),
    ]

    # 图 1：响应曲线
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    E = np.logspace(-2, 0, 200)
    axes[0].plot(E, E ** (1 / 2.2), color=ACCENT, lw=2, label="γ=2.2")
    axes[0].plot(E, E, "--", color=BLUE, lw=1.5, label="线性")
    axes[0].set_xscale("log"); axes[0].set_xlabel("相对曝光 E", fontsize=11); axes[0].set_ylabel("像素值/255", fontsize=11)
    axes[0].set_title("响应曲线：暗部被抬起", fontsize=12); axes[0].legend(fontsize=9, frameon=False); axes[0].grid(alpha=0.3)
    axes[0].tick_params(labelsize=9)
    axes[1].plot(np.log2(E), np.log2(E ** (1 / 2.2)), color=ACCENT, lw=2)
    axes[1].set_xlabel("log2 曝光", fontsize=11); axes[1].set_ylabel("log2 输出", fontsize=11)
    axes[1].set_title("双对数斜率=1/γ", fontsize=12); axes[1].grid(alpha=0.3); axes[1].tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(FIG / "e092_curve.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：三张曝光
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for k, ax in zip((0, 1, 2), axes):
        ax.imshow(obs[0, k], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"EV={EVS[k]:+d} 饱和 {sat[k]*100:.0f}%", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("同一场景三张曝光：短曝光保高光，长曝光保暗部", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(FIG / "e092_exposure.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：gamma 恢复 + 反演误差
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].bar(["R", "G", "B"], est_g, color=[ACCENT, GREEN, BLUE], width=0.55, label="估计")
    axes[0].plot([0, 1, 2], GAMMAS, "k^", ms=8, label="真值")
    axes[0].set_ylim(1.9, 2.4); axes[0].set_title("三通道 gamma 恢复", fontsize=12)
    axes[0].set_ylabel("γ", fontsize=11); axes[0].tick_params(labelsize=10); axes[0].legend(fontsize=9, frameon=False)
    axes[1].bar(["γ 校正", "线性假设"], [rel * 100, rel_lin * 100], color=[GREEN, ACCENT])
    axes[1].text(0, rel * 100 + 2, f"{rel*100:.2f}%", ha="center", fontsize=10)
    axes[1].text(1, rel_lin * 100 + 2, f"{rel_lin*100:.0f}%", ha="center", fontsize=10)
    axes[1].set_ylim(0, 105); axes[1].set_title("反演相对误差（%）", fontsize=12); axes[1].tick_params(labelsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e092_recover.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(dr=dr, gamma_true=list(GAMMAS), gamma_est=est_g.tolist(),
                    gamma_err=g_err.tolist(), short_sat=short_sat, long_sat=long_sat,
                    cover=cover, rel=rel, rel_lin=rel_lin, ratio=ratio, ratio_ideal=ratio_ideal,
                    short_sat_pct=round(short_sat * 100, 1), long_sat_pct=round(long_sat * 100, 1),
                    rel_pct=round(rel * 100, 2), rel_lin_pct=round(rel_lin * 100, 1),
                    inv_gamma=float(1/2.2), gamma_r=round(float(est_g[0]), 4), gamma_g=round(float(est_g[1]), 4),
                    gamma_b=round(float(est_g[2]), 4), dr_int=round(dr),
                    n_pass=npass, n_total=len(checks))
    (OUT / "e092_radiometric_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print("[lab] readings → e092_radiometric_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e092_radiometric_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
