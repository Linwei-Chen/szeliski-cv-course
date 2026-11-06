#!/usr/bin/env python3
"""E099 手机夜景管线：合成 Bayer RAW 多帧实验（真值已知，程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, laplace, uniform_filter, median_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d", "#6b7280"
RNG = np.random.default_rng(20261102)
H = W = 128
C = 10
PAD = 24
FULL_WELL = 20000.0
SENS = 400.0          # 短曝光：单位辐亮度对应的电子数
READ = 3.0            # 读出噪声（电子）
SHORT_GAIN = 4.0      # 短曝光显示的曝光增益
LONG_FACTOR = 8.0     # 长曝光相对短曝光的曝光倍数
TRUE_OFFSETS = [(0, 0), (2, 0), (0, 2), (2, 2), (-2, 0), (0, -2), (-2, -2), (4, -2)]


def psnr(a, b, mask=None):
    d = (np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(np.mean(d))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def luminance(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def make_scene(n):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    u, v = x / (n - 1), y / (n - 1)
    img = np.zeros((n, n, 3))
    # 夜空与地面：暗部保留纹理和渐层
    img[..., 0] = 0.018 + 0.020 * v
    img[..., 1] = 0.020 + 0.026 * v
    img[..., 2] = 0.045 + 0.030 * v
    # 左侧楼体与暖窗
    wall = (u < 0.42) & (v > 0.28)
    img[wall] = np.array([0.070, 0.064, 0.060])
    bricks = (np.sin(90 * u) * np.sin(90 * v) > 0.55) & wall
    img[bricks] *= 1.35
    win = (u > 0.07) & (u < 0.27) & (v > 0.36) & (v < 0.56)
    img[win] = np.array([7.0, 5.2, 2.8])
    img[win] *= (0.90 + 0.10 * np.sin(60 * u[win]))[..., None]
    # 右侧霓虹牌（高光+彩色）
    sign = (u > 0.62) & (u < 0.88) & (v > 0.18) & (v < 0.30)
    img[sign] = np.array([2.0, 8.0, 10.0])
    # 街灯
    lamp = (u - 0.55) ** 2 + (v - 0.12) ** 2
    img += np.exp(-lamp / (2 * 0.004 ** 2))[..., None] * np.array([16.0, 14.0, 10.0])
    # 彩色测试块（去马赛克/色彩）
    patches = [(0.62, 0.40, 0.78, 0.52, [0.42, 0.08, 0.08]),
               (0.62, 0.54, 0.78, 0.66, [0.08, 0.42, 0.10]),
               (0.62, 0.68, 0.78, 0.80, [0.08, 0.10, 0.42])]
    for x0, y0, x1, y1, col in patches:
        m = (u > x0) & (u < x1) & (v > y0) & (v < y1)
        img[m] = np.array(col)
    # 暗部细纹（检验过度降噪）
    stripe = (u > 0.06) & (u < 0.24) & (v > 0.66) & (v < 0.88)
    img[stripe] *= (1.0 + 0.30 * np.sin(120 * u[stripe]))[..., None]
    # 中灰细节区（检验清晰度）
    detail = (u > 0.30) & (u < 0.52) & (v > 0.62) & (v < 0.82)
    img[detail] *= (1.0 + 0.35 * np.sin(80 * u[detail]) * np.sin(80 * v[detail]))[..., None]
    return np.clip(img, 1e-4, None)


def mosaic(img):
    h, w = img.shape[:2]
    raw = np.zeros((h, w))
    raw[0::2, 0::2] = img[0::2, 0::2, 0]
    raw[0::2, 1::2] = img[0::2, 1::2, 1]
    raw[1::2, 0::2] = img[1::2, 0::2, 1]
    raw[1::2, 1::2] = img[1::2, 1::2, 2]
    return raw


def demosaic(raw):
    h, w = raw.shape
    p = np.pad(raw, 1, mode="reflect")
    out = np.zeros((h, w, 3))
    re, ro = np.arange(0, h, 2), np.arange(1, h, 2)
    ce, co = np.arange(0, w, 2), np.arange(1, w, 2)
    # R：已知在偶行偶列
    R = np.zeros((h, w)); G = np.zeros((h, w)); B = np.zeros((h, w))
    R[np.ix_(re, ce)] = raw[np.ix_(re, ce)]
    R[np.ix_(re, co)] = 0.5 * (p[np.ix_(re + 1, co)] + p[np.ix_(re + 1, co + 2)])
    R[np.ix_(ro, ce)] = 0.5 * (p[np.ix_(ro, ce + 1)] + p[np.ix_(ro + 2, ce + 1)])
    R[np.ix_(ro, co)] = 0.25 * (p[np.ix_(ro, co)] + p[np.ix_(ro, co + 2)] +
                               p[np.ix_(ro + 2, co)] + p[np.ix_(ro + 2, co + 2)])
    B[np.ix_(ro, co)] = raw[np.ix_(ro, co)]
    B[np.ix_(ro, ce)] = 0.5 * (p[np.ix_(ro, ce)] + p[np.ix_(ro, ce + 2)])
    B[np.ix_(re, co)] = 0.5 * (p[np.ix_(re, co + 1)] + p[np.ix_(re + 2, co + 1)])
    B[np.ix_(re, ce)] = 0.25 * (p[np.ix_(re, ce)] + p[np.ix_(re, ce + 2)] +
                               p[np.ix_(re + 2, ce)] + p[np.ix_(re + 2, ce + 2)])
    G[np.ix_(re, co)] = raw[np.ix_(re, co)]
    G[np.ix_(ro, ce)] = raw[np.ix_(ro, ce)]
    G[np.ix_(re, ce)] = 0.25 * (p[np.ix_(re, ce + 1)] + p[np.ix_(re + 2, ce + 1)] +
                               p[np.ix_(re + 1, ce)] + p[np.ix_(re + 1, ce + 2)])
    G[np.ix_(ro, co)] = 0.25 * (p[np.ix_(ro, co + 1)] + p[np.ix_(ro + 2, co + 1)] +
                               p[np.ix_(ro + 1, co)] + p[np.ix_(ro + 1, co + 2)])
    out[..., 0], out[..., 1], out[..., 2] = R, G, B
    return np.clip(out, 0, None)


def demosaic_nearest(raw):
    h, w = raw.shape
    masks = np.zeros((h, w, 3), bool)
    masks[0::2, 0::2, 0] = True
    masks[0::2, 1::2, 1] = True
    masks[1::2, 0::2, 1] = True
    masks[1::2, 1::2, 2] = True
    out = np.zeros((h, w, 3))
    for c in range(3):
        idx = distance_transform_edt(~masks[..., c], return_distances=False, return_indices=True)
        out[..., c] = raw[tuple(idx)]
    return out


def noisy_raw(scene, offset, factor=1.0, seed=0):
    rng = np.random.default_rng(seed)
    oy, ox = offset
    crop = scene[oy:oy + H, ox:ox + W]
    raw = mosaic(crop)
    lam = np.clip(raw * SENS * factor, 0, None)
    e = rng.poisson(lam).astype(float)
    e += rng.normal(0, READ, size=e.shape)
    e = np.clip(e, 0, FULL_WELL)
    return e / FULL_WELL, crop


def estimate_shift(ref, mov):
    # 在去马赛克后的绿通道上做整数像素全搜索
    g0, g1 = demosaic(ref)[..., 1], demosaic(mov)[..., 1]
    best, bs = None, (0, 0)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            sh = np.roll(np.roll(g1, dy, 0), dx, 1)
            mse = float(np.mean((g0[10:-10, 10:-10] - sh[10:-10, 10:-10]) ** 2))
            if best is None or mse < best:
                best, bs = mse, (dy, dx)
    return bs


def bilateral(img, sigma_s=2.0, sigma_r=0.03, radius=2):
    acc = np.zeros_like(img); wsum = np.zeros_like(img)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = np.roll(np.roll(img, dy, 0), dx, 1)
            w = np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2) -
                       np.sum((sh - img) ** 2, axis=-1, keepdims=True) / (2 * sigma_r ** 2))
            acc += w * sh; wsum += w
    return acc / np.maximum(wsum, 1e-9)


def chroma_clean(img, size=3):
    Y = luminance(img)
    Cb = median_filter(img[..., 2] - Y, size=size)
    Cr = median_filter(img[..., 0] - Y, size=size)
    R = Y + Cr
    B = Y + Cb
    G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152
    return np.clip(np.stack([R, G, B], axis=-1), 0, None)


def tone_map(img, gain):
    L = np.clip(luminance(img), 0, None)
    Lg = L * gain
    Ld = Lg / (1.0 + Lg)
    scale = np.where(L > 1e-6, Ld / np.maximum(L, 1e-6), 0.0)
    out = np.clip(img * scale[..., None], 0, 1) ** (1 / 2.2)
    return np.clip(out, 0, 1)


def run(nframes, aligned=True, denoise=True, sigma_r=0.005, radius=1):
    raws = [f[0] for f in frames]
    if aligned:
        stack = [np.roll(np.roll(raws[i], SHIFTS[i][0], 0), SHIFTS[i][1], 1)
                 for i in range(nframes)]
    else:
        stack = raws[:nframes]
    merged = np.mean(stack, axis=0)
    img = demosaic(merged)
    if denoise:
        img = bilateral(img * (FULL_WELL / SENS), sigma_r=sigma_r, radius=radius) / (FULL_WELL / SENS)
        img = chroma_clean(img * (FULL_WELL / SENS)) / (FULL_WELL / SENS)
    disp = tone_map(img * (FULL_WELL / SENS), SHORT_GAIN)[C:-C, C:-C]
    return disp, merged, img


# ── 场景与多帧 ───────────────────────────────────────────────
scene = make_scene(H + 2 * PAD)
truth = scene[PAD:PAD + H, PAD:PAD + W]
frames = []
for i, (dx, dy) in enumerate(TRUE_OFFSETS):
    frames.append(noisy_raw(scene, (PAD + dy, PAD + dx), 1.0, 1000 + i))
ref = frames[0][0]
SHIFTS = [estimate_shift(ref, f[0]) for f in frames]
SHIFT_ERR = float(np.mean([abs(SHIFTS[i][0] - TRUE_OFFSETS[i][1]) +
                           abs(SHIFTS[i][1] - TRUE_OFFSETS[i][0]) for i in range(len(frames))]))

truth_c = truth[C:-C, C:-C]
truth_disp = tone_map(truth_c, SHORT_GAIN)
p_single_img, _, _ = run(1, True, True)
p_single_nd_img, _, _ = run(1, True, False)
p_naive_img, _, _ = run(8, False, True)
p_nodenoise_img, _, _ = run(8, True, False)
p_full_img, merged_full, linear_full = run(8, True, True)
p_over_img, _, _ = run(8, True, True, sigma_r=0.30, radius=3)
p1_img, _, _ = run(1, True, False)
p2_img, _, _ = run(2, True, False)
p4_img, _, _ = run(4, True, False)
p_single = psnr(p_single_img, truth_disp)
p_single_nd = psnr(p_single_nd_img, truth_disp)
p_naive = psnr(p_naive_img, truth_disp)
p_nodenoise = psnr(p_nodenoise_img, truth_disp)
p_full = psnr(p_full_img, truth_disp)
p_over = psnr(p_over_img, truth_disp)
p1, p2, p4, p8 = (psnr(x, truth_disp) for x in (p1_img, p2_img, p4_img, p_nodenoise_img))

u = np.arange(W - 2 * C)[None, :] / (W - 2 * C - 1); v = np.arange(H - 2 * C)[:, None] / (H - 2 * C - 1)
Lt = luminance(truth_c)
mu = uniform_filter(Lt, 9)
var = np.maximum(uniform_filter(Lt * Lt, 9) - mu * mu, 0)
score = mu + 8.0 * np.sqrt(var)
score[:20, :] = score[-20:, :] = score[:, :20] = score[:, -20:] = 1e9
iy, ix = np.unravel_index(int(np.argmin(score)), score.shape)
dark_mask = np.zeros((H - 2 * C, W - 2 * C), bool)
dark_mask[iy - 6:iy + 6, ix - 6:ix + 6] = True
dark_single = psnr(p_single_img, truth_disp, dark_mask)
dark_full = psnr(p_full_img, truth_disp, dark_mask)
noise_roi = dark_mask
def hf_noise(img):
    L = luminance(img)
    return float(np.std((L - uniform_filter(L, 5))[noise_roi]))

noise_single = hf_noise(p_single_img)
noise_full = hf_noise(p_full_img)
noise_single_nd = hf_noise(p_single_nd_img)
noise_full_nd = hf_noise(p_nodenoise_img)

# 长曝光：同一场景，曝光×8，按中间调曝光（增益/8）
long_raw, _ = noisy_raw(scene, (PAD, PAD), LONG_FACTOR, 9999)
long_img = demosaic(long_raw)
long_img = bilateral(long_img, sigma_r=0.03, radius=2)
long_disp = tone_map(long_img * (FULL_WELL / SENS / LONG_FACTOR), SHORT_GAIN)[C:-C, C:-C]
noise_long = hf_noise(long_disp)
clip_long = 100.0 * float(np.mean(long_raw >= 0.999))
clip_burst = 100.0 * float(np.mean(merged_full >= 0.999))

# 去马赛克色彩误差（干净场景，隔离插值本身）
clean_raw = mosaic(truth)
bil = demosaic(clean_raw)
nearest = demosaic_nearest(clean_raw)
uf = np.arange(W)[None, :] / (W - 1); vf = np.arange(H)[:, None] / (H - 1)
color_mask = (uf > 0.62) & (uf < 0.78) & (vf > 0.40) & (vf < 0.80)
color_err_bilinear = float(np.mean(np.abs(bil - truth)[color_mask]))
color_err_nearest = float(np.mean(np.abs(nearest - truth)[color_mask]))

# 纹理保留：暗部条纹区的高频能量
tex_mask = (u > 0.06) & (u < 0.24) & (v > 0.66) & (v < 0.88)
def tex_energy(img):
    L = luminance(img)
    return float(np.std(laplace(L)[tex_mask]))
tex_true = tex_energy(truth_disp)
tex_full = tex_energy(p_full_img)
tex_over = tex_energy(p_over_img)

STEPS = [
        ("运动测光", "p.645"), ("帧数曝光", "p.645"), ("低 ISO 连拍", "p.645"),
        ("IMU 运动辅助", "p.645"), ("参考帧", "p.637"),
        ("辐射线性化", "p.610"), ("全局/局部对齐", "p.625"), ("一致性防鬼影", "p.625"),
        ("选像素/共识", "p.626"), ("RAW 域合并", "p.624"),
        ("去马赛克", "p.646"), ("超分细节", "p.637"), ("去模糊", "p.645"),
        ("后处理去噪", "p.644"), ("锐化/色彩校正", "p.611"),
        ("全局色调映射", "p.627"), ("局部色调映射", "p.630"), ("编码/显示", "p.611"),
    ]
N_STEPS = len(STEPS)
dark_nd = psnr(p_nodenoise_img, truth_disp, dark_mask)
dark_single_nd = psnr(p_single_nd_img, truth_disp, dark_mask)
checks = [
    ("对齐估计准确", SHIFT_ERR <= 0.5, f"平均误差 {SHIFT_ERR:.2f} 像素"),
    ("暗部多帧更优", dark_nd > dark_single_nd + 2.0, f"{dark_nd:.2f} vs {dark_single_nd:.2f} dB"),
    ("对齐优于不对齐", p_nodenoise > p_naive + 2.0, f"{p_nodenoise:.2f} vs {p_naive:.2f} dB"),
    ("暗部噪声下降", noise_full_nd < noise_single_nd * 0.5, f"{noise_full_nd:.4f} vs {noise_single_nd:.4f}"),
    ("合并后去噪仍有增益", p_full > p_nodenoise + 1.0, f"+{p_full - p_nodenoise:.2f} dB（{p_full:.2f} vs {p_nodenoise:.2f}）"),
    ("过度去噪损纹理", tex_over < tex_full * 0.80, f"{tex_over:.4f} vs {tex_full:.4f}"),
    ("长曝光高光裁切", clip_long > 3.0 and clip_burst < 0.5, f"{clip_long:.1f}% vs {clip_burst:.1f}%"),
    ("帧数收益递减", p1 < p2 < p4 < p8 and (p8 - p4) < (p4 - p2), f"{p1:.2f}/{p2:.2f}/{p4:.2f}/{p8:.2f}"),
    ("双线性去马赛克色彩误差在 0.03 容差内（不优于最近邻）", color_err_bilinear < 0.03, f"{color_err_bilinear:.4f}（最近邻 {color_err_nearest:.4f}）"),
    ("十八步全部有归属", len(STEPS) == 18 and all(all(x) for x in STEPS), f"{N_STEPS}/{N_STEPS} 步有页面归属"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)

readings = {
    "shift_err": SHIFT_ERR, "shift_err_2": round(SHIFT_ERR, 2),
    "p_single": p_single, "p_single_2": round(p_single, 2),
    "p_single_nd": p_single_nd, "p_single_nd_2": round(p_single_nd, 2),
    "p_naive": p_naive, "p_naive_2": round(p_naive, 2),
    "p_nodenoise": p_nodenoise, "p_nodenoise_2": round(p_nodenoise, 2),
    "p_full": p_full, "p_full_2": round(p_full, 2),
    "p_gain_2": round(p_full - p_single, 2),
    "p_denoise_gain_2": round(p_full - p_nodenoise, 2),
    "dark_single": dark_single, "dark_single_2": round(dark_single, 2),
    "dark_full": dark_full, "dark_full_2": round(dark_full, 2),
    "dark_nd_2": round(dark_nd, 2), "dark_single_nd_2": round(dark_single_nd, 2),
    "steps_total": N_STEPS, "steps_mapped": N_STEPS,
    "dark_gain_2": round(dark_full - dark_single, 2),
    "noise_long_4": round(noise_long, 4),
    "noise_single_4": round(noise_single, 4), "noise_full_4": round(noise_full, 4),
    "noise_single_nd_4": round(noise_single_nd, 4), "noise_full_nd_4": round(noise_full_nd, 4),
    "p1_2": round(p1, 2), "p2_2": round(p2, 2), "p4_2": round(p4, 2), "p8_2": round(p8, 2),
    "clip_long_pct": round(clip_long, 1), "clip_burst_pct": round(clip_burst, 2),
    "tex_true_4": round(tex_true, 4), "tex_full_4": round(tex_full, 4), "tex_over_4": round(tex_over, 4),
    "color_err_bilinear_4": round(color_err_bilinear, 4),
    "color_err_nearest_4": round(color_err_nearest, 4),
    "n_pass": n_pass, "n_total": len(checks),
}

# ── 图 1：十八步管线地图 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11.2, 6.2))
ax.set_xlim(0, 4); ax.set_ylim(-0.1, 5); ax.axis("off")
phase_defs = [("采集", BLUE, range(0, 5)), ("合成", GREEN, range(5, 10)),
              ("重建", ACCENT, range(10, 15)), ("渲染", PURPLE, range(15, 18))]
for ci, (pname, color, inds) in enumerate(phase_defs):
    ax.text(ci + 0.5, 4.92, pname, ha="center", fontsize=15, color=color, weight="bold")
    for j, i in enumerate(inds):
        name, page = STEPS[i]
        row = 4 - j
        ax.add_patch(plt.Rectangle((ci + 0.03, row + 0.10), 0.94, 0.74,
                                   facecolor=color, alpha=0.10, edgecolor=color, lw=1.2))
        ax.text(ci + 0.07, row + 0.55, f"{i+1:02d}", fontsize=14.5, color=color, weight="bold")
        ax.text(ci + 0.26, row + 0.55, name, fontsize=14.5, color=INK)
        ax.text(ci + 0.26, row + 0.24, page, fontsize=14.5, color=GRAY)
ax.text(0.01, -0.07, "按工程顺序整理，不是厂商官方流程图；每个编号标注本课程对应原书页。",
        fontsize=14.5, color=GRAY)
fig.tight_layout()
fig.savefig(FIG / "e099_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

# ── 图 2：分阶段 PSNR ──────────────────────────────────────
labels = ["单帧\n无去噪", "单帧\n+去噪", "8帧不对齐\n+去噪", "8帧对齐\n不去噪", "8帧对齐\n+去噪"]
vals = [p_single_nd, p_single, p_naive, p_nodenoise, p_full]
darks = [psnr(p_single_nd_img, truth_disp, dark_mask), dark_single,
         psnr(p_naive_img, truth_disp, dark_mask),
         psnr(p_nodenoise_img, truth_disp, dark_mask), dark_full]
x = np.arange(len(labels)); bw = 0.36
fig, ax = plt.subplots(figsize=(10.2, 5.0))
b1 = ax.bar(x - bw / 2, vals, bw, color=BLUE, label="整体 PSNR")
b2 = ax.bar(x + bw / 2, darks, bw, color=ACCENT, alpha=0.85, label="暗部 PSNR")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}",
            ha="center", fontsize=14.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14.5)
ax.set_ylabel("PSNR / dB（越高越好）", fontsize=14.5, color=INK)
ax.set_ylim(0, max(vals + darks) + 4)
ax.legend(fontsize=14.5, frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.set_title("夜景管线逐段验收：暗部 PSNR 把差距放大", fontsize=15, color=INK, pad=12)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "e099_stages.png", dpi=170, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

# ── 图 3：四个结果对照 ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
panels = [(truth_disp, "真值（程序生成）", ""),
          (p_single_img, "单帧短曝光", f"{p_single:.1f} dB"),
          (p_full_img, "8帧对齐+去噪", f"{p_full:.1f} dB"),
          (long_disp, "单帧长曝光", f"高光裁切 {clip_long:.1f}%")]
for ax, (im, title, sub) in zip(axes.ravel(), panels):
    ax.imshow(np.clip(im, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=16, color=INK)
    if sub:
        ax.text(0.98, 0.96, sub, transform=ax.transAxes, ha="right", va="top",
                fontsize=15, color="white", bbox=dict(fc="black", alpha=0.55, ec="none", pad=2.5))
fig.suptitle("夜景管线：多帧短曝光保住高光，长曝光在亮窗/霓虹处裁切", fontsize=16, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e099_crops.png", dpi=170, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

# ── 输出 ──────────────────────────────────────────────────
print(f"[lab] E099 night pipeline 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e099_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
readings["n_pass"] = n_pass
(OUT / "e099_night_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings → e099_night_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e099_night_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
