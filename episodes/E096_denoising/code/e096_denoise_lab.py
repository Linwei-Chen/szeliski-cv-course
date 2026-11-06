#!/usr/bin/env python3
"""E096 去噪/去模糊：高斯/中值/双边/NLM/连拍/Wiener。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, laplace
from scipy.signal import convolve2d

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 80


def psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def bilateral(img, sigma_s=2.0, sigma_r=0.08, radius=3):
    acc = np.zeros_like(img); wsum = np.zeros_like(img)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = np.roll(np.roll(img, dy, 0), dx, 1)
            w = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2) - (sh - img) ** 2 / (2 * sigma_r ** 2))
            acc += w * sh; wsum += w
    return acc / wsum


def nlm(img, h=0.08, patch=2, search=3):
    pad = np.pad(img, patch + search, mode="reflect")
    out = np.zeros_like(img)
    for i in range(N):
        for j in range(N):
            p = pad[i + search:i + search + 2 * patch + 1, j + search:j + search + 2 * patch + 1]
            vals = []; ws = []
            for di in range(-search, search + 1):
                for dj in range(-search, search + 1):
                    q = pad[i + search + di:i + search + di + 2 * patch + 1,
                            j + search + dj:j + search + dj + 2 * patch + 1]
                    d2 = np.mean((p - q) ** 2)
                    w = np.exp(-max(d2 - 2 * 0.01 ** 2, 0) / (h ** 2))
                    vals.append(q[patch, patch]); ws.append(w)
            out[i, j] = np.dot(vals, ws) / max(np.sum(ws), 1e-9)
    return out


def edge_corr(a, b):
    ga = laplace(a); gb = laplace(b)
    return float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])


def make_clean():
    yy, xx = np.mgrid[0:N, 0:N]
    img = 0.42 + 0.10 * np.sin(2 * np.pi * xx / 23) * np.sin(2 * np.pi * yy / 19)
    img[8:30, 8:44] = 0.88
    img[48:70, 14:34] = 0.08
    img[((xx - 58) ** 2 + (yy - 24) ** 2) < 12 ** 2] = 0.72
    checker = ((xx // 4) + (yy // 4)) % 2 == 0
    img[checker & (xx > 40) & (yy > 40)] = 0.80
    for k in range(4): img[66:70, 40 + 8 * k:45 + 8 * k] = 0.05
    return np.clip(img, 0, 1)


def motion_kernel(length=9, angle=15):
    k = np.zeros((length, length)); c = length // 2
    th = np.deg2rad(angle)
    for t in np.linspace(-c, c, 60):
        x = int(round(c + t * np.cos(th))); y = int(round(c + t * np.sin(th)))
        if 0 <= x < length and 0 <= y < length: k[y, x] = 1
    return k / k.sum()


def wiener(img, k, K=0.02):
    h, w = img.shape; kh, kw = k.shape
    kp = np.zeros((h, w)); kp[:kh, :kw] = k
    kp = np.roll(kp, (-(kh // 2), -(kw // 2)), (0, 1))
    H = np.fft.fft2(kp); G = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.real(np.fft.ifft2(G * np.fft.fft2(img)))


def main():
    print(f"[lab] E096 denoise 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    rng = np.random.default_rng(96)
    clean = make_clean()
    noisy = np.clip(clean + rng.normal(0, 25 / 255, clean.shape), 0, 1)
    sp = clean.copy(); m = rng.random(clean.shape) < 0.05
    sp[m] = rng.choice([0.0, 1.0], size=m.sum())
    gauss = gaussian_filter(noisy, 0.8)
    med = median_filter(noisy, size=3)
    bil = bilateral(noisy)
    nlm_img = nlm(noisy)
    med_sp = median_filter(sp, size=3); gauss_sp = gaussian_filter(sp, 0.8)
    # 连拍
    frames = [np.clip(clean + rng.normal(0, 25 / 255, clean.shape), 0, 1) for _ in range(8)]
    burst = np.mean(frames, axis=0)
    mis = np.mean([np.roll(f, (i % 3 - 1, i % 5 - 2), (0, 1)) for i, f in enumerate(frames)], axis=0)
    flat = (slice(0, 12), slice(52, 72))
    std1 = float(np.std((frames[0] - clean)[flat])); std8 = float(np.std((burst - clean)[flat]))
    noise_drop = 1 - std8 / std1
    # 去模糊
    k = motion_kernel(9, 15); blurred = convolve2d(clean, k, mode="same", boundary="symm")
    wiener_ok = wiener(blurred, k)
    k_bad = motion_kernel(15, 60); wiener_bad = wiener(blurred, k_bad)
    noisy_blur = np.clip(blurred + rng.normal(0, 0.01, clean.shape), 0, 1)
    wiener_noisy = wiener(noisy_blur, k)
    std_before = float(np.std((noisy_blur - blurred)[flat])); std_after = float(np.std((wiener_noisy - wiener_ok)[flat]))

    p_noisy = psnr(noisy, clean); p_gauss = psnr(gauss, clean); p_med = psnr(med, clean)
    p_bil = psnr(bil, clean); p_nlm = psnr(nlm_img, clean)
    p_gsp = psnr(gauss_sp, clean); p_msp = psnr(med_sp, clean)
    p_burst = psnr(burst, clean); p_mis = psnr(mis, clean)
    p_blur = psnr(blurred, clean); p_ok = psnr(wiener_ok, clean); p_bad = psnr(wiener_bad, clean)
    e_gauss = edge_corr(gauss, clean); e_bil = edge_corr(bil, clean); e_nlm = edge_corr(nlm_img, clean)

    checks = [
        ("高斯降噪>2dB", p_gauss > p_noisy + 2, f"{p_gauss:.2f} vs {p_noisy:.2f} dB"),
        ("椒盐中值>高斯+3dB", p_msp > p_gsp + 3, f"{p_msp:.2f} vs {p_gsp:.2f} dB"),
        ("双边保边>高斯+5%", e_bil > e_gauss + 0.05, f"{e_bil:.3f} vs {e_gauss:.3f}"),
        ("NLM>高斯+0.4dB", p_nlm > p_gauss + 0.4, f"{p_nlm:.2f} vs {p_gauss:.2f} dB"),
        ("八帧平均降噪>50%", noise_drop > 0.50, f"{noise_drop*100:.1f}%"),
        ("未对齐连拍更差>1dB", p_mis < p_burst - 1, f"{p_mis:.2f} vs {p_burst:.2f} dB"),
        ("Wiener 正确核>2dB", p_ok > p_blur + 2, f"{p_ok:.2f} vs {p_blur:.2f} dB"),
        ("错误核掉>1dB", p_bad < p_ok - 1, f"{p_bad:.2f} vs {p_ok:.2f} dB"),
        ("去模糊放大噪声>1.2×", std_after > std_before * 1.2, f"{std_after/std_before:.2f}×"),
        ("NLM 保边>高斯+8%", e_nlm > e_gauss + 0.08, f"{e_nlm:.3f} vs {e_gauss:.3f}"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(9.0, 2.1))
    for ax, im, t in zip(axes, (clean, noisy, gauss, med, bil, nlm_img),
                         ("真值", f"含噪 {p_noisy:.1f}dB", f"高斯 {p_gauss:.1f}", f"中值 {p_med:.1f}", f"双边 {p_bil:.1f}", f"NLM {p_nlm:.1f}")):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("去噪：平滑、保边、非局部，各有取舍", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e096_denoise.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.35))
    labels = ["高斯", "中值", "双边", "NLM"]; vals = [p_gauss, p_med, p_bil, p_nlm]
    axes[0].bar(labels, vals, color=[PURPLE, BLUE, GREEN, ACCENT]); axes[0].set_title("高斯噪声 PSNR（dB）", fontsize=11)
    axes[0].tick_params(labelsize=10)
    for i, v in enumerate(vals): axes[0].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=10)
    axes[1].bar(["高斯", "中值"], [p_gsp, p_msp], color=[PURPLE, ACCENT]); axes[1].set_title("椒盐噪声 PSNR（dB）", fontsize=11)
    axes[1].tick_params(labelsize=10)
    for i, v in enumerate([p_gsp, p_msp]): axes[1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=10)
    fig.suptitle("没有万能去噪器：噪声类型决定算子", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e096_tradeoff.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.2))
    for ax, im, t in zip(axes, (blurred, wiener_ok, wiener_bad, burst),
                         (f"模糊 {p_blur:.1f}dB", f"正确核 {p_ok:.1f}", f"错误核 {p_bad:.1f}", f"八帧平均 {p_burst:.1f}")):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("去模糊靠核；连拍靠对齐与平均", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e096_deblur.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(p_noisy=p_noisy, p_gauss=p_gauss, p_med=p_med, p_bil=p_bil, p_nlm=p_nlm,
                    p_gsp=p_gsp, p_msp=p_msp, p_burst=p_burst, p_mis=p_mis,
                    p_blur=p_blur, p_ok=p_ok, p_bad=p_bad, noise_drop=noise_drop,
                    noise_drop_pct=round(noise_drop * 100, 1), e_gauss=e_gauss, e_bil=e_bil, e_nlm=e_nlm,
                    std_ratio=std_after / std_before, std_ratio_2=round(std_after / std_before, 2),
                    p_noisy_1=round(p_noisy, 1), p_gauss_1=round(p_gauss, 1), p_med_1=round(p_med, 1),
                    p_bil_1=round(p_bil, 1), p_nlm_1=round(p_nlm, 1), p_gsp_1=round(p_gsp, 1),
                    p_msp_1=round(p_msp, 1), p_burst_1=round(p_burst, 1), p_mis_1=round(p_mis, 1),
                    p_blur_1=round(p_blur, 1), p_ok_1=round(p_ok, 1), p_bad_1=round(p_bad, 1),
                    p_gauss_2=round(p_gauss, 2), p_noisy_2=round(p_noisy, 2), p_msp_2=round(p_msp, 2),
                    p_gsp_2=round(p_gsp, 2), p_nlm_2=round(p_nlm, 2), p_ok_2=round(p_ok, 2),
                    p_blur_2=round(p_blur, 2), p_bad_2=round(p_bad, 2), p_burst_2=round(p_burst, 2),
                    p_mis_2=round(p_mis, 2), e_gauss_3=round(e_gauss, 3), e_bil_3=round(e_bil, 3),
                    e_nlm_3=round(e_nlm, 3), n_pass=npass, n_total=len(checks))
    (OUT / "e096_denoise_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for p in sorted(FIG.glob("e096_*.png")):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e096_denoise_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT/'e096_denoise_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
