#!/usr/bin/env python3
"""E094 色调映射：全局 gamma / 局部 base-detail / 梯度域与 WLS 实验。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 180


def sat(img):
    mx = img.max(-1); mn = img.min(-1)
    return np.where(mx > 1e-8, (mx - mn) / np.maximum(mx, 1e-8), 0.0)


def bilateral(img, sigma_s=8.0, sigma_r=0.25, radius=12):
    acc = np.zeros_like(img); wsum = np.zeros_like(img)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            w = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2)
                       - (shifted - img) ** 2 / (2 * sigma_r ** 2))
            acc += w * shifted; wsum += w
    return acc / wsum


def poisson(gx, gy):
    b = -(np.diff(gx, axis=1, prepend=0) + np.diff(gy, axis=0, prepend=0))
    u = np.arange(N); lam = (4 - 2 * np.cos(np.pi * u[:, None] / N)
                             - 2 * np.cos(np.pi * u[None, :] / N))
    lam[0, 0] = 1.0
    out = idctn(dctn(b, type=2, norm="ortho") / lam, type=2, norm="ortho")
    return out - out.mean()


def wls_1d(g, lam=8.0, alpha=1.0, eps=1e-4):
    d = np.diff(g); ws = 1.0 / (np.abs(d) ** alpha + eps)
    n = len(g)
    main = np.ones(n) + lam * (np.r_[ws, 0] + np.r_[0, ws])
    lower = -lam * ws; upper = -lam * ws
    return spsolve(diags([lower, main, upper], [-1, 0, 1], format="csc"), g)


def main():
    print(f"[lab] E094 tone mapping 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    rng = np.random.default_rng(94)
    yy, xx = np.mgrid[0:N, 0:N]
    wall = 0.9 + 0.30 * (1 - yy / N) + 0.08 * np.sin(2 * np.pi * xx / 38) * np.sin(2 * np.pi * yy / 45)
    rgb = np.stack([wall * 1.0, wall * 0.85, wall * 0.68], -1)
    win = (xx > 95) & (xx < 168) & (yy > 18) & (yy < 122)
    sky = 420 * (1.10 - 0.60 * (yy - 18) / 104)
    sky_rgb = np.stack([sky * 0.72, sky * 0.90, sky * 1.18], -1)
    ground = 55 * (0.8 + 0.4 * np.sin(2 * np.pi * xx / 30))
    ground_rgb = np.stack([ground * 0.8, ground * 1.1, ground * 0.6], -1)
    rgb[win] = sky_rgb[win]
    rgb[win & (yy > 92)] = ground_rgb[win & (yy > 92)]
    frame = ((xx > 91) & (xx < 99) | (xx > 164) & (xx < 172) |
             (yy > 14) & (yy < 22) | (yy > 118) & (yy < 126)) & (xx > 88) & (xx < 175)
    rgb[frame] *= 0.15
    rgb[(xx < 32) & (yy > 122)] *= 0.06
    dr = float(rgb.max() / rgb.min())
    scale = float(np.percentile(rgb, 60))
    x = rgb / scale
    linear = np.clip(x, 0, 1)
    clip_frac = float((linear.max(-1) > 0.995).mean())
    chan = np.clip(x, 0, 1) ** (1 / 2.2)
    y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    ymap = np.clip((y / (scale * 1.0)) ** (1 / 2.2), 0, 1)
    lum = np.clip(rgb * (ymap / np.maximum(y, 1e-12))[..., None], 0, 1)
    mask = (y > np.percentile(y, 5)) & (ymap < 0.98)
    sat_chan = float(sat(chan)[mask].mean()); sat_lum = float(sat(lum)[mask].mean())
    chroma_true = rgb / np.maximum(rgb.sum(-1, keepdims=True), 1e-12)
    chroma_chan = chan / np.maximum(chan.sum(-1, keepdims=True), 1e-12)
    chroma_lum = lum / np.maximum(lum.sum(-1, keepdims=True), 1e-12)
    chroma_err_chan = float(np.abs(chroma_chan - chroma_true)[mask].mean())
    chroma_err_lum = float(np.abs(chroma_lum - chroma_true)[mask].mean())

    # 局部 base/detail：天空亮、房子暗，边缘是主要难点
    H = 1.75 + 0.25 * (1 - yy / N) + 0.06 * np.sin(2 * np.pi * xx / 23) * np.sin(2 * np.pi * yy / 19)
    house = (xx > 42) & (xx < 132) & (yy > 98)
    roof = (np.abs(xx - 87) < 52) & (yy > 62) & (yy <= 98) & (np.abs(xx - 87) < (yy - 58) * 1.45)
    obj = house | roof
    H[obj] = (0.08 + 0.04 * np.sin(2 * np.pi * xx / 17) * np.sin(2 * np.pi * yy / 15))[obj]
    Bg = gaussian_filter(H, 8.0)
    Bb = bilateral(H, sigma_s=8.0, sigma_r=0.22, radius=12)
    Dg = H - Bg; Db = H - Bb
    step_in = float(H[~obj].mean() - H[obj].mean())
    halo_g = float(np.abs(Dg).max() / step_in)
    halo_b = float(np.abs(Db).max() / step_in)
    step_g = float(Bg[~obj].mean() - Bg[obj].mean())
    step_b = float(Bb[~obj].mean() - Bb[obj].mean())
    edge_ret_g = step_g / step_in; edge_ret_b = step_b / step_in

    # 梯度域
    gx = np.diff(H, axis=1, append=H[:, -1:])
    gy = np.diff(H, axis=0, append=H[-1:, :])
    phi = 1.0 / (1.0 + (np.hypot(gx, gy) / 0.18) ** 2)
    Gx, Gy = gx * phi, gy * phi
    energy_red = 1 - (np.sum(Gx ** 2 + Gy ** 2) / np.sum(gx ** 2 + gy ** 2))
    rec_full = poisson(gx, gy)
    rec_err = float(np.linalg.norm(rec_full - (H - H.mean())) / np.linalg.norm(H - H.mean()))
    rec_att = poisson(Gx, Gy)
    range_full = float(np.exp(H.max() - H.min()))
    range_att = float(np.exp(rec_att.max() - rec_att.min()))
    range_red = 1 - (np.log(range_att) / np.log(range_full))

    # WLS 1D：保边平滑
    n = 512; t = np.linspace(0, 1, n)
    clean = 0.5 + 1.5 * (t > 0.5) + 0.10 * np.sin(2 * np.pi * 7 * t)
    noisy = clean + 0.08 * rng.normal(size=n)
    smooth = wls_1d(noisy, lam=10.0, alpha=1.0, eps=1e-4)
    edge_ret_wls = float((smooth[t > 0.55].mean() - smooth[t < 0.45].mean()) / 1.5)
    flat = (t < 0.35) | ((t > 0.65) & (t < 0.9))
    noise_red = 1 - float(np.std((smooth - clean)[flat]) / np.std((noisy - clean)[flat]))

    checks = [
        ("HDR 动态范围>100", dr > 100, f"{dr:.0f} 倍"),
        ("线性裁切丢失>20%像素", clip_frac > 0.20, f"{clip_frac*100:.1f}%"),
        ("逐通道 gamma 更灰", sat_chan < sat_lum * 0.9, f"{sat_chan:.2f} vs {sat_lum:.2f}"),
        ("逐通道色度误差更大", chroma_err_chan > chroma_err_lum * 1.5,
         f"{chroma_err_chan:.4f} vs {chroma_err_lum:.4f}"),
        ("全局映射压进显示域", float(lum.max()) <= 1.0 and float(lum.min()) >= 0.0,
         f"[{lum.min():.2f},{lum.max():.2f}]"),
        ("高斯 base 有晕轮", halo_g > 0.10, f"{halo_g*100:.1f}%"),
        ("双边晕轮减半", halo_b < halo_g * 0.5, f"{halo_b*100:.1f}% vs {halo_g*100:.1f}%"),
        ("双边保住阶跃>80%", edge_ret_b > 0.80, f"{edge_ret_b*100:.1f}%"),
        ("梯度域能量降>50%", energy_red > 0.50, f"{energy_red*100:.1f}%"),
        ("泊松重建误差<1%", rec_err < 0.01, f"{rec_err*100:.2f}%"),
    ]

    # 图 1：全局映射三态
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.35))
    for ax, im, title in zip(axes, (linear, chan, lum),
                             ("线性裁切", "逐通道 gamma", "亮度通道 gamma")):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
    axes[1].set_xlabel(f"平均饱和 {sat_chan:.2f}", fontsize=10)
    axes[2].set_xlabel(f"平均饱和 {sat_lum:.2f}", fontsize=10)
    fig.suptitle("全局色调映射：先压亮度，再带回颜色", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e094_global.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：base/detail 与晕轮
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.3))
    for ax, im, title, cmap in zip(axes, (H, Bg, Dg, Bb),
                                   ("log 亮度 H", "高斯 base", "高斯 detail（晕轮）", "双边 base"),
                                   ("gray", "gray", "coolwarm", "gray")):
        ax.imshow(im, cmap=cmap); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    fig.suptitle("局部色调映射：线性 base 在边缘留下晕轮，双边保住边缘", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e094_local.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：梯度域 + WLS
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.3))
    axes[0].imshow(H, cmap="gray"); axes[0].set_title("log 亮度", fontsize=10.5)
    axes[1].imshow(phi, cmap="inferno"); axes[1].set_title("梯度衰减 Φ", fontsize=10.5)
    axes[2].imshow(rec_att, cmap="gray"); axes[2].set_title("重建（压缩后）", fontsize=10.5)
    axes[3].plot(t, noisy, color="#b9b0a0", lw=0.9, label="含噪")
    axes[3].plot(t, clean, color=INK, lw=1.2, label="真值")
    axes[3].plot(t, smooth, color=ACCENT, lw=1.8, label="WLS")
    axes[3].set_title("WLS：保边平滑", fontsize=10.5); axes[3].legend(fontsize=10, loc="upper left")
    axes[3].set_xlabel("位置", fontsize=10); axes[3].set_yticks([])
    for ax in axes[:3]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"梯度域衰减 {energy_red*100:.0f}% + 局部 WLS", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e094_operators.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(dr=dr, dr_int=round(dr), clip_frac=clip_frac, clip_pct=round(clip_frac * 100, 1),
                    sat_chan=sat_chan, sat_lum=sat_lum, sat_chan_2dp=round(sat_chan, 2), sat_lum_2dp=round(sat_lum, 2),
                    chroma_err_chan=chroma_err_chan, chroma_err_lum=chroma_err_lum,
                    y_min=float(lum.min()), y_max=float(lum.max()),
                    halo_g=halo_g, halo_b=halo_b, halo_g_pct=round(halo_g * 100, 1),
                    halo_b_pct=round(halo_b * 100, 1),
                    edge_ret_b=edge_ret_b, edge_ret_b_pct=round(edge_ret_b * 100, 1),
                    energy_red=energy_red, energy_red_pct=round(energy_red * 100, 1),
                    rec_err=rec_err, rec_err_pct=round(rec_err * 100, 2),
                    range_red=range_red, range_full=range_full, range_att=range_att,
                    edge_ret_wls=edge_ret_wls, noise_red=noise_red,
                    n_pass=npass, n_total=len(checks), gamma=2.2, sigma_s=8.0, sigma_r=0.22)
    (OUT / "e094_tone_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for p in sorted(FIG.glob("e094_*.png")):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e094_tone_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT/'e094_tone_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
