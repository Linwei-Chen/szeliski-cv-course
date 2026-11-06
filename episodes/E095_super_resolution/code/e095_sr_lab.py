#!/usr/bin/env python3
"""E095 超分辨率：观测模型 / 多帧 IBP / 混叠与先验。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift, zoom, laplace

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 128; SCALE = 2; SIGMA = 0.8


def make_hr():
    yy, xx = np.mgrid[0:N, 0:N]
    hr = 0.48 + 0.06 * np.sin(2 * np.pi * xx / 37) * np.sin(2 * np.pi * yy / 29)
    hr[18:52, 16:82] = 0.92
    hr[70:112, 24:58] = 0.10
    circ = (xx - 92) ** 2 + (yy - 88) ** 2 < 24 ** 2
    hr[circ] = 0.80
    stripe = ((xx + yy) // 7) % 2 == 0
    hr[stripe & (xx > 62) & (yy < 40)] = 0.18
    checker = ((xx // 6) + (yy // 6)) % 2 == 0
    hr[checker & (xx > 64) & (yy > 62)] = 0.95
    for k in range(5):
        hr[100:106, 70 + 9 * k:75 + 9 * k] = 0.05
    return np.clip(hr, 0, 1)


def down(hr, sigma=SIGMA):
    return gaussian_filter(hr, sigma)[::SCALE, ::SCALE]


def observe(hr, sh, sigma=SIGMA, noise=0.0, rng=None):
    moved = ndshift(hr, (sh[0] * SCALE, sh[1] * SCALE), order=3, mode="reflect")
    lr = down(moved, sigma)
    if noise and rng is not None:
        lr = lr + rng.normal(0, noise, lr.shape)
    return lr


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def ibp(lrs, shifts, init, iters=100, gain=0.08, reg=0.0, sigma=SIGMA):
    hr = init.copy(); hist = []
    for _ in range(iters):
        acc = np.zeros_like(hr)
        for lr, sh in zip(lrs, shifts):
            sim = down(ndshift(hr, (sh[0] * SCALE, sh[1] * SCALE), order=3, mode="reflect"), sigma)
            err = lr - sim
            up = zoom(err, SCALE, order=1)
            back = ndshift(up, (-sh[0] * SCALE, -sh[1] * SCALE), order=1, mode="reflect")
            acc += back
        hr = np.clip(hr + gain * acc / len(lrs), 0, 1)
        if reg:
            hr = np.clip(hr - reg * laplace(hr), 0, 1)
        hist.append(psnr(hr, HR))
    return hr, hist


def hf_energy(img):
    return float(np.var(laplace(img)))


def main():
    print(f"[lab] E095 SR 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    global HR
    HR = make_hr()
    rng = np.random.default_rng(95)
    sub4 = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]
    int4 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    lrs_sub4 = [observe(HR, s, noise=0.004, rng=rng) for s in sub4]
    lrs_int4 = [observe(HR, s, noise=0.004, rng=rng) for s in int4]
    clean_sub4 = [observe(HR, s, noise=0.0) for s in sub4]
    lr0 = lrs_sub4[0]
    bic = np.clip(zoom(lr0, SCALE, order=3), 0, 1)
    bic = bic[:N, :N]
    sr_sub4, hist_sub4 = ibp(lrs_sub4, sub4, bic)
    sr_int4, hist_int4 = ibp(lrs_int4, int4, bic)
    sr_single, _ = ibp(lrs_sub4[:1], sub4[:1], bic)
    sr_wrong, _ = ibp(lrs_sub4, sub4, bic, sigma=1.5)
    # 观测噪声幅度（中位绝对残差）
    model_res = float(np.median(np.abs(lrs_sub4[0] - clean_sub4[0])))
    # 多帧平均降噪
    noisy = [observe(HR, s, noise=0.035, rng=rng) for s in sub4]
    noisy_clean = clean_sub4
    flat = (slice(0, 15), slice(0, 15))
    std1 = float(np.std((noisy[0] - noisy_clean[0])[flat]))
    std4 = float(np.std((np.mean(noisy, axis=0) - np.mean(noisy_clean, axis=0))[flat]))
    noise_drop = 1 - std4 / std1
    # 混叠：有/无前置低通
    lr_no_lp = HR[::SCALE, ::SCALE]
    lr_lp = down(HR, SIGMA)
    alias_no = hf_energy(lr_no_lp); alias_lp = hf_energy(lr_lp)
    p_bic = psnr(bic, HR); p_sub4 = psnr(sr_sub4, HR); p_int4 = psnr(sr_int4, HR)
    p_single = psnr(sr_single, HR); p_wrong = psnr(sr_wrong, HR)
    edge_bic = float(np.mean(np.abs(laplace(bic)))); edge_sr = float(np.mean(np.abs(laplace(sr_sub4))))
    mse_first = 10 ** (-hist_sub4[0] / 10); mse_last = 10 ** (-hist_sub4[-1] / 10)
    err_drop = 1 - mse_last / mse_first

    checks = [
        ("观测噪声中位<0.01", model_res < 0.01, f"{model_res:.4f}"),
        ("四帧亚像素>双三次+2dB", p_sub4 > p_bic + 2.0, f"{p_sub4:.2f} vs {p_bic:.2f} dB"),
        ("整数位移增益<2.8dB", p_int4 < p_bic + 2.8, f"{p_int4:.2f} vs {p_bic:.2f} dB"),
        ("亚像素优于整数>0.5dB", p_sub4 > p_int4 + 0.5, f"{p_sub4:.2f} vs {p_int4:.2f} dB"),
        ("四帧优于单帧>0.3dB", p_sub4 > p_single + 0.3, f"{p_sub4:.2f} vs {p_single:.2f} dB"),
        ("无前置低通混叠>2倍", alias_no > 2 * alias_lp, f"{alias_no:.4f} vs {alias_lp:.4f}"),
        ("IBP 均方误差降>50%", err_drop > 0.50, f"{err_drop*100:.1f}%"),
        ("四帧平均降噪>40%", noise_drop > 0.40, f"{noise_drop*100:.1f}%"),
        ("模糊核偏大掉>1dB", p_wrong < p_sub4 - 1.0, f"{p_wrong:.2f} vs {p_sub4:.2f} dB"),
        ("边缘能量提升>20%", edge_sr > edge_bic * 1.2, f"{edge_sr/edge_bic:.2f}×"),
    ]

    # 图1 观测模型
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.3))
    axes[0].imshow(HR, cmap="gray", vmin=0, vmax=1); axes[0].set_title("高分辨真值", fontsize=10.5)
    for ax, lr, t in zip(axes[1:], (lrs_sub4[0], lrs_sub4[1], lrs_sub4[3]),
                         ("低分辨帧 1", "低分辨帧 2", "低分辨帧 4")):
        ax.imshow(lr, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=10.5)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("观测模型：平移 → 模糊 → 下采样 → 噪声", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e095_model.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图2 重建对比
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.3))
    for ax, im, t in zip(axes, (HR, bic, sr_sub4, np.abs(sr_sub4 - HR)),
                         ("真值", f"双三次 {p_bic:.1f} dB", f"四帧 IBP {p_sub4:.1f} dB", "误差 |SR−真值|")):
        ax.imshow(im, cmap="gray" if t != "误差 |SR−真值|" else "inferno", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("多帧超分：亚像素位移补回高频", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e095_sr.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图3 位移/混叠/PSNR
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.35))
    axes[0].scatter([0, 1, 0, 1], [0, 0, 1, 1], s=16, color=BLUE, label="整数位移")
    axes[0].scatter([0, .5, 0, .5], [0, 0, .5, .5], s=16, color=ACCENT, label="亚像素")
    axes[0].set_xlim(-.3, 1.6); axes[0].set_ylim(-.3, 1.6); axes[0].set_aspect("equal")
    axes[0].set_title("采样相位", fontsize=10.5); axes[0].legend(fontsize=10, loc="upper left")
    axes[1].bar(["无低通", "有低通"], [alias_no, alias_lp], color=[ACCENT, GREEN])
    axes[1].set_title("低分辨高频能量", fontsize=10.5); axes[1].tick_params(labelsize=10)
    labels = ["双三次", "单帧 IBP", "4 整数", "4 亚像素"]
    vals = [p_bic, p_single, p_int4, p_sub4]
    axes[2].bar(labels, vals, color=[PURPLE, BLUE, GREEN, ACCENT])
    axes[2].set_title("PSNR（dB）", fontsize=10.5); axes[2].tick_params(labelsize=10, rotation=15)
    for i, v in enumerate(vals): axes[2].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=10)
    fig.suptitle("相位越多、混叠越足，能补的高频越多", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e095_shifts.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(model_res=model_res, p_bic=p_bic, p_sub4=p_sub4, p_int4=p_int4,
                    p_single=p_single, p_wrong=p_wrong,
                    alias_no=alias_no, alias_lp=alias_lp, alias_ratio=alias_no / alias_lp,
                    err_drop=err_drop, err_drop_pct=round(err_drop * 100, 1),
                    noise_drop=noise_drop, noise_drop_pct=round(noise_drop * 100, 1),
                    edge_ratio=edge_sr / edge_bic, edge_ratio_pct=round(edge_sr / edge_bic * 100, 1),
                    p_bic_1=round(p_bic, 1), p_sub4_1=round(p_sub4, 1), p_int4_1=round(p_int4, 1),
                    p_single_1=round(p_single, 1), p_wrong_1=round(p_wrong, 1),
                    p_bic_2=round(p_bic, 2), p_sub4_2=round(p_sub4, 2), p_int4_2=round(p_int4, 2),
                    p_single_2=round(p_single, 2), p_wrong_2=round(p_wrong, 2),
                    model_res_4dp=round(model_res, 4), alias_no_4dp=round(alias_no, 4),
                    alias_lp_4dp=round(alias_lp, 4), edge_ratio_2=round(edge_sr / edge_bic, 2),
                    n_pass=npass, n_total=len(checks), scale=SCALE, sigma=SIGMA)
    (OUT / "e095_sr_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for p in sorted(FIG.glob("e095_*.png")):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e095_sr_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT/'e095_sr_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
