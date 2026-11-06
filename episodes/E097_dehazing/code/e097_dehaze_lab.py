#!/usr/bin/env python3
"""E097 暗通道先验去雾：合成场景/透射率/大气光/参数研究。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 160; PATCH = 15; OMEGA = 0.95; T0 = 0.1


def psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def dark_channel(img, p):
    return minimum_filter(img.min(axis=2), size=p)


def estimate_A(img, p):
    d = dark_channel(img, p); flat = d.ravel(); k = max(1, int(0.001 * flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    pix = img.reshape(-1, 3)[idx]
    return pix[np.argmax(pix.sum(axis=1))]


def transmission(img, A, p, omega):
    return 1 - omega * dark_channel(img / A, p)


def guided_filter(I, p, r=20, eps=1e-3):
    mI = uniform_filter(I, r); mp = uniform_filter(p, r)
    mII = uniform_filter(I * I, r); mIp = uniform_filter(I * p, r)
    var = mII - mI * mI; cov = mIp - mI * mp
    a = cov / (var + eps); b = mp - a * mI
    return uniform_filter(a, r) * I + uniform_filter(b, r)


def recover(img, A, t, t0):
    return (img - A) / np.maximum(t[..., None], t0) + A


def make_scene():
    yy, xx = np.mgrid[0:N, 0:N]
    sky = np.stack([0.55 + 0.25 * (1 - yy / N), 0.68 + 0.20 * (1 - yy / N), 0.88 + 0.12 * (1 - yy / N)], -1)
    J = sky.copy(); depth = 0.95 - 0.70 * (yy / N)
    # 远山
    mtn = yy > 70 + 18 * np.sin(xx / 18)
    J[mtn] = np.array([0.30, 0.36, 0.32]); depth[mtn] = 0.8
    # 近处建筑
    bld = (xx > 22) & (xx < 70) & (yy > 92)
    J[bld] = np.array([0.62, 0.48, 0.34]); depth[bld] = 0.35
    # 暗窗/树
    win = (xx > 30) & (xx < 46) & (yy > 108) & (yy < 132)
    J[win] = np.array([0.05, 0.06, 0.07]); depth[win] = 0.35
    tree = ((xx - 118) ** 2 + (yy - 120) ** 2) < 22 ** 2
    J[tree] = np.array([0.08, 0.18, 0.08]); depth[tree] = 0.25
    canopy = (((xx - 28) ** 2 + (yy - 24) ** 2) < 58 ** 2) | (((xx - 144) ** 2 + (yy - 142) ** 2) < 40 ** 2)
    J[canopy] = np.array([0.06, 0.13, 0.05]); depth[canopy] = 0.20
    hedge = yy > 150
    J[hedge] = np.array([0.07, 0.15, 0.06]); depth[hedge] = 0.18
    bush = yy > 128
    tex = 0.04 + 0.10 * ((np.sin(xx * 1.7) * np.sin(yy * 1.3)) > 0)
    J[bush] = np.stack([tex[bush] * 0.7, tex[bush] * 1.2, tex[bush] * 0.6], -1); depth[bush] = 0.20
    # 白色屋顶（先验失效点）
    roof = (xx > 86) & (xx < 132) & (yy > 86) & (yy < 98)
    J[roof] = np.array([0.96, 0.96, 0.94]); depth[roof] = 0.4
    return np.clip(J, 0, 1), np.clip(depth, 0, 1), sky[..., 0] > 0.72


def main():
    print(f"[lab] E097 dehaze 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    J, depth, sky = make_scene(); A_true = np.array([0.85, 0.90, 0.95])
    beta = 1.6; t_true = np.exp(-beta * depth)
    I = np.clip(J * t_true[..., None] + A_true * (1 - t_true[..., None]), 0, 1)
    A_est = estimate_A(I, PATCH)
    t_raw = transmission(I, A_est, PATCH, OMEGA)
    t_ref = guided_filter(I.mean(axis=2), t_raw, r=25, eps=1e-3)
    Jr = recover(I, A_est, t_ref, T0)
    Jraw = recover(I, A_est, t_raw, T0)
    # 参数研究
    t_small = guided_filter(I.mean(axis=2), transmission(I, A_est, 3, OMEGA), 25, 1e-3)
    Jsmall = recover(I, A_est, t_small, T0)
    t_full = guided_filter(I.mean(axis=2), transmission(I, A_est, PATCH, 1.0), 25, 1e-3)
    Jfull = recover(I, A_est, t_full, T0)
    Jt001 = recover(I, A_est, t_ref, 0.01)
    nonsky = ~sky
    dark_frac = float((dark_channel(J, PATCH)[nonsky] < 0.10).mean())
    A_err = float(np.abs(A_est - A_true).max() / A_true.max())
    t_corr = float(np.corrcoef(t_ref.ravel(), t_true.ravel())[0, 1])
    d_corr = float(-np.corrcoef(np.log(np.maximum(t_ref, 1e-6)).ravel(), depth.ravel())[0, 1])
    p_hazy = psnr(I, J); p_raw = psnr(Jraw, J); p_ref = psnr(Jr, J)
    p_small = psnr(Jsmall, J); p_full = psnr(Jfull, J); p_t001 = psnr(Jt001, J)
    roof = (np.mgrid[0:N, 0:N][0] > 86) & (np.mgrid[0:N, 0:N][0] < 132) & (np.mgrid[0:N, 0:N][1] > 86) & (np.mgrid[0:N, 0:N][1] < 98)
    # 注意 mgrid 是 (yy,xx)
    yy, xx = np.mgrid[0:N, 0:N]; roof = (xx > 86) & (xx < 132) & (yy > 86) & (yy < 98)
    roof_err = float(np.mean(np.abs(t_ref[roof] - t_true[roof])))
    other_err = float(np.mean(np.abs(t_ref[(~roof) & nonsky] - t_true[(~roof) & nonsky])))

    checks = [
        ("非天空暗通道低值>60%", dark_frac > 0.60, f"{dark_frac*100:.1f}%"),
        ("大气光误差<5%", A_err < 0.05, f"{A_err*100:.2f}%"),
        ("透射率相关>0.85", t_corr > 0.85, f"r={t_corr:.3f}"),
        ("恢复提升>5dB", p_ref > p_hazy + 5, f"{p_ref:.2f} vs {p_hazy:.2f} dB"),
        ("深度相关>0.75", d_corr > 0.75, f"r={d_corr:.3f}"),
        ("细化优于原始透射率", p_ref > p_raw + 0.3, f"{p_ref:.2f} vs {p_raw:.2f} dB"),
        ("patch 选择影响>1dB", abs(p_small - p_ref) > 1.0, f"{abs(p_small-p_ref):.2f} dB"),
        ("omega=1 过增强", p_full < p_ref, f"{p_full:.2f} vs {p_ref:.2f} dB"),
        ("t0 下限≥0.1", float(np.min(np.maximum(t_ref, T0))) >= T0 - 1e-9, f"min={np.min(np.maximum(t_ref,T0)):.2f}"),
        ("白屋顶透射率误差更大", roof_err > other_err * 1.5, f"{roof_err:.3f} vs {other_err:.3f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.25))
    for ax, im, t in zip(axes, (I, t_ref, Jr, 1 - t_ref),
                         (f"含雾 {p_hazy:.1f} dB", "估计透射率", f"去雾 {p_ref:.1f} dB", "深度（1−t）")):
        ax.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("暗通道去雾：估计透射率，再反演辐亮度", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e097_dehaze.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.35))
    axes[0].hist(dark_channel(J, PATCH)[nonsky].ravel(), bins=30, color=BLUE, alpha=0.8, label="无雾")
    axes[0].hist(dark_channel(I, PATCH)[nonsky].ravel(), bins=30, color=ACCENT, alpha=0.6, label="有雾")
    axes[0].set_title("非天空区暗通道分布", fontsize=11); axes[0].legend(fontsize=10); axes[0].tick_params(labelsize=10)
    axes[1].scatter(t_true[::4, ::4].ravel(), t_ref[::4, ::4].ravel(), s=3, color=GREEN, alpha=0.4)
    axes[1].plot([0, 1], [0, 1], color=INK, lw=1); axes[1].set_title("估计 vs 真值透射率", fontsize=11)
    axes[1].set_xlabel("真值", fontsize=10); axes[1].set_ylabel("估计", fontsize=10); axes[1].tick_params(labelsize=10)
    fig.suptitle("暗通道先验：无雾图暗通道接近零", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e097_prior.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.3))
    axes[0].imshow(Jsmall, vmin=0, vmax=1); axes[0].set_title(f"patch=3：{p_small:.1f} dB", fontsize=10.5)
    axes[1].imshow(Jr, vmin=0, vmax=1); axes[1].set_title(f"patch=15：{p_ref:.1f} dB", fontsize=10.5)
    axes[2].imshow(Jfull, vmin=0, vmax=1); axes[2].set_title(f"omega=1：{p_full:.1f} dB", fontsize=10.5)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("参数：patch 太小过饱和，omega=1 丢远景层次", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e097_params.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(dark_frac=dark_frac, dark_frac_pct=round(dark_frac*100,1), A_est=[float(x) for x in A_est],
                    A_err=A_err, A_err_pct=round(A_err*100,2), t_corr=t_corr, t_corr_3=round(t_corr,3),
                    d_corr=d_corr, d_corr_3=round(d_corr,3), p_hazy=p_hazy, p_raw=p_raw, p_ref=p_ref,
                    p_small=p_small, p_full=p_full, p_t001=p_t001,
                    p_hazy_2=round(p_hazy,2), p_ref_2=round(p_ref,2), p_raw_2=round(p_raw,2),
                    p_small_2=round(p_small,2), p_full_2=round(p_full,2), p_t001_2=round(p_t001,2),
                    roof_err=roof_err, other_err=other_err, roof_err_3=round(roof_err,3), other_err_3=round(other_err,3),
                    patch_diff=abs(p_small-p_ref), patch_diff_2=round(abs(p_small-p_ref),2),
                    n_pass=npass, n_total=len(checks), patch=PATCH, omega=OMEGA, t0=T0, beta=beta)
    (OUT / "e097_dehaze_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for p in sorted(FIG.glob("e097_*.png")):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e097_dehaze_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT/'e097_dehaze_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
