#!/usr/bin/env python3
"""E091 计算摄影导览：多曝光 HDR / 色调映射 / 多帧去噪 / 管线图。"""
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
N = 256


def psnr(a, b):
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def main():
    print(f"[lab] E091 计算摄影 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    rng = np.random.default_rng(91)
    yy, xx = np.mgrid[0:N, 0:N]
    base = gaussian_filter(rng.normal(size=(N, N)), 3.0)
    base = (base - base.min()) / (base.max() - base.min())
    lamp = np.exp(-((xx - 190) ** 2 + (yy - 70) ** 2) / (2 * 22 ** 2))
    dark = 1 - 0.7 * np.exp(-((xx - 50) ** 2 + (yy - 200) ** 2) / (2 * 35 ** 2))
    rad = np.clip((10 ** (0.0 + 3.0 * base)) * dark * (1 + 8 * lamp), 1e-2, None)
    dr_in = float(rad.max() / rad.min())
    exposures = [1 / 2000, 1 / 200, 1 / 20]
    ims = [np.clip(e * rad + rng.normal(0, 0.002, (N, N)), 0, 1) for e in exposures]
    sat = [float((im >= 0.999).mean()) for im in ims]

    # 权重合并（高信噪比且未饱和）
    num = np.zeros((N, N)); den = np.zeros((N, N))
    for e, im in zip(exposures, ims):
        w = im ** 2 * (1 - im) ** 2
        num += w * (im / e); den += w
    est = num / np.maximum(den, 1e-9)
    valid = den > 1e-9
    corr = float(np.corrcoef(np.log(est[valid]), np.log(rad[valid]))[0, 1])
    rel = float(np.median(np.abs(est[valid] - rad[valid]) / rad[valid]))
    mid = ims[1] / exposures[1]
    rel_mid = float(np.median(np.abs(mid[valid] - rad[valid]) / rad[valid]))

    # 色调映射
    L = est / est.mean(); tm = L / (1 + L)
    dr_out = float(tm.max() / max(tm.min(), 1e-9))

    # gamma 编解码
    x = np.linspace(0, 1, 2000); gamma_err = float(np.abs((x ** (1 / 2.2)) ** 2.2 - x).max())

    # 多帧去噪：同一曝光 8 张加噪，平均
    clean = np.clip(exposures[1] * rad, 0, 1)
    noisy = [np.clip(clean + rng.normal(0, 0.06, clean.shape), 0, 1) for _ in range(8)]
    avg = np.mean(noisy, axis=0)
    ps_noisy = psnr(noisy[0], clean); ps_avg = psnr(avg, clean)

    # 暗部信噪比（短曝光）
    dm = rad < (rad.min() * 3)
    dark_snr = float(ims[0][dm].mean() / 0.002)

    checks = [
        ("场景动态范围>1000", dr_in > 1000, f"{dr_in:.0f}:1"),
        ("短曝光暗部信噪比低", dark_snr < 2, f"SNR={dark_snr:.2f}"),
        ("长曝光高光饱和多", sat[2] > 0.5, f"饱和 {sat[2]*100:.1f}%"),
        ("三曝光覆盖全部像素", float(valid.mean()) > 0.99, f"覆盖 {valid.mean()*100:.1f}%"),
        ("合并辐亮度相关>0.99", corr > 0.99, f"r={corr:.3f}"),
        ("合并相对误差<2%", rel < 0.02, f"{rel*100:.2f}%"),
        ("合并优于单张中曝光", rel < rel_mid, f"{rel*100:.2f}% vs {rel_mid*100:.2f}%"),
        ("色调映射压缩动态范围", dr_out < 0.2 * dr_in, f"{dr_in:.0f}→{dr_out:.0f}"),
        ("gamma 编解码往返", gamma_err < 1e-6, f"{gamma_err:.1e}"),
        ("多帧去噪提升>5dB", ps_avg - ps_noisy > 5, f"{ps_noisy:.1f}→{ps_avg:.1f}dB"),
    ]

    # 图 1：曝光序列/合并/色调映射
    fig, axes = plt.subplots(1, 5, figsize=(8.0, 2.0))
    panels = [(np.log10(rad), "场景辐亮度 log", "inferno"),
              (ims[0], f"短曝光 饱和{sat[0]*100:.0f}%", "gray"),
              (ims[1], f"中曝光 饱和{sat[1]*100:.0f}%", "gray"),
              (ims[2], f"长曝光 饱和{sat[2]*100:.0f}%", "gray"),
              (tm, "合并+色调映射", "gray")]
    for ax, (im, title, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap); ax.set_title(title, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("单张曝光顾此失彼；多曝光合并 + 色调映射回到屏幕", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "e091_exposure.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：成像管线
    fig, ax = plt.subplots(figsize=(7.6, 1.8)); ax.set_xlim(0, 6); ax.set_ylim(0, 1); ax.axis("off")
    stages = [("场景辐亮度", "物理"), ("镜头/PSF", "镜头模糊"), ("传感器/Bayer", "马赛克采样"), ("ISP/去噪/色调", "处理管线"), ("显示", "屏幕/打印")]
    for j, (name, ref) in enumerate(stages):
        x = 0.5 + j * 1.1
        ax.add_patch(plt.Rectangle((x - 0.42, 0.36), 0.84, 0.42, fc="#ffffff", ec=BLUE, lw=1.5))
        ax.text(x, 0.57, name, ha="center", va="center", fontsize=9.5)
        ax.text(x, 0.16, ref, ha="center", va="center", fontsize=9, color="#6b5626")
        if j < 4:
            ax.annotate("", xy=(x + 0.68, 0.57), xytext=(x + 0.44, 0.57),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_title("计算摄影的边界：从光子到像素，再到显示", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "e091_pipeline.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：章节地图
    fig, ax = plt.subplots(figsize=(7.6, 2.5)); ax.set_xlim(0, 3); ax.set_ylim(0, 2.2); ax.axis("off")
    topics = [("光度标定", "标定", 0.5, 1.55), ("HDR 合成", "合成", 1.5, 1.55),
              ("色调映射", "映射", 2.5, 1.55), ("超分/去噪", "超分", 0.5, 0.55),
              ("去雾/修补", "去雾", 1.5, 0.55), ("夜景综合", "夜景", 2.5, 0.55)]
    for name, ref, x, y in topics:
        ax.text(x, y, f"{name}\n{ref}", ha="center", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=GREEN, lw=1.6))
    ax.set_title("第 10 章：六条支线，最后汇到手机夜景", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "e091_topics.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(dr_in=dr_in, sat_short=sat[0], sat_mid=sat[1], sat_long=sat[2],
                    coverage=float(valid.mean()), corr=corr, rel=rel, rel_mid=rel_mid,
                    dr_out=dr_out, gamma_err=gamma_err, ps_noisy=ps_noisy, ps_avg=ps_avg,
                    dark_snr=dark_snr, exposures=exposures,
                    sat_long_pct=round(sat[2] * 100, 1), coverage_pct=round(float(valid.mean()) * 100, 1), rel_pct=round(rel * 100, 2),
                    rel_mid_pct=round(rel_mid * 100, 2), ps_gain=round(ps_avg - ps_noisy, 1),
                    dr_in_int=round(dr_in), dr_out_int=round(dr_out),
                    n_pass=npass, n_total=len(checks))
    (OUT / "e091_photography_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print("[lab] readings → e091_photography_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e091_photography_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
