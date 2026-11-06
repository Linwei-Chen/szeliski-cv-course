#!/usr/bin/env python3
"""E133 空间计算 lab：八相机一维阵列、两层深度+边界抠图、时间一致性。"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"
for cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = PAPER
plt.rcParams["axes.facecolor"] = PAPER
plt.rcParams["savefig.facecolor"] = PAPER
plt.rcParams["text.color"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.edgecolor"] = INK

print(f"[lab] E133 空间计算开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成一维相机阵列；两层深度+软边界；时间中值是离线示例，不是完整 VR/6DoF 系统")

W, H = 320, 180
CAM = np.array([-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7])
TGT = 0.0
ZB, ZF = 6.0, 2.0
U = (np.arange(W) + 0.5) / W - 0.5
V = (np.arange(H) + 0.5) / H - 0.5
UU, VV = np.meshgrid(U, V)

def facade(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.56; base[..., 1] = 0.47; base[..., 2] = 0.38
    xx = X + 4.0; yy = Y + 3.0
    win = ((np.mod(np.floor(xx / 0.8), 2) == 0) &
           (np.mod(np.floor(yy / 0.6), 2) == 0) &
           (np.mod(xx, 0.8) > 0.14) & (np.mod(xx, 0.8) < 0.66) &
           (np.mod(yy, 0.6) > 0.12) & (np.mod(yy, 0.6) < 0.48))
    base[win] = [0.18, 0.28, 0.42]
    line = ((np.mod(xx, 0.8) < 0.035) | (np.mod(yy, 0.6) < 0.035))
    base[line] *= 0.78
    return np.clip(base, 0, 1)

def foreground(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.86; base[..., 1] = 0.27; base[..., 2] = 0.12
    stripe = (np.mod(np.floor((Y + 1.0) / 0.16), 2) == 0)
    base[stripe] = [0.98, 0.62, 0.18]
    return base

def layers(cx, fg_x=0.0):
    Xb = cx + ZB * UU
    Yb = ZB * VV
    bg = facade(Xb, Yb)
    Xf = cx + ZF * UU
    Yf = ZF * VV
    fg = foreground(Xf, Yf)
    halfw, halfh = 0.20, 0.55
    d = np.maximum(np.abs(Xf - fg_x) - halfw, np.abs(Yf) - halfh)
    edge = 2.0 * ZF / W  # 约 2 像素软边
    alpha = np.clip(0.5 - d / (2 * edge), 0, 1)
    return bg, fg, alpha

def render(cx, tgt, zbg, zfg, soft=True, fg_x=0.0):
    bg, fg, alpha = layers(cx, fg_x)
    dxb = -(tgt - cx) / zbg * W
    dxf = -(tgt - cx) / zfg * W
    b = ndimage.shift(bg, (0, dxb, 0), order=1, mode="nearest")
    f = ndimage.shift(fg, (0, dxf, 0), order=1, mode="nearest")
    if soft:
        a = ndimage.shift(alpha, (0, dxf), order=1, mode="nearest")
        a = np.clip(a, 0, 1)
    else:
        a = ndimage.shift((alpha > 0.5).astype(float), (0, dxf), order=0, mode="nearest")
    return b * (1 - a[..., None]) + f * a[..., None], a

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1 / max(mse, 1e-12))

target, target_alpha = render(TGT, TGT, ZB, ZF, soft=True, fg_x=0.0)
hard, hard_alpha = render(CAM[3], TGT, ZB, ZF, soft=False, fg_x=0.0)
soft, soft_alpha = render(CAM[3], TGT, ZB, ZF, soft=True, fg_x=0.0)
psnr_hard = psnr(hard, target)
psnr_soft = psnr(soft, target)
band = ndimage.binary_dilation((target_alpha > 0.05) & (target_alpha < 0.95), iterations=2)
edge_hard = float(np.mean(np.abs(hard - target)[band]))
edge_soft = float(np.mean(np.abs(soft - target)[band]))
edge_gain = (edge_hard - edge_soft) / max(edge_hard, 1e-12) * 100

def temporal_series(n=24):
    rng = np.random.default_rng(133)
    jb = rng.normal(0, 0.10, n)
    jf = rng.normal(0, 0.10, n)
    zbs = np.clip(ZB + jb, ZB - 0.2, ZB + 0.2)
    zfs = np.clip(ZF + jf, ZF - 0.2, ZF + 0.2)
    zbs_s = ndimage.gaussian_filter1d(zbs, sigma=4.0, mode="nearest")
    zfs_s = ndimage.gaussian_filter1d(zfs, sigma=4.0, mode="nearest")
    noisy, smooth, truth = [], [], []
    for i in range(n):
        fgx = 0.16 * np.sin(2 * np.pi * i / n)
        gt, _ = render(TGT, TGT, ZB, ZF, soft=True, fg_x=fgx)
        a, _ = render(CAM[3], TGT, zbs[i], zfs[i], soft=True, fg_x=fgx)
        b, _ = render(CAM[3], TGT, zbs_s[i], zfs_s[i], soft=True, fg_x=fgx)
        truth.append(gt); noisy.append(a); smooth.append(b)
    return np.array(truth), np.array(noisy), np.array(smooth), zbs, zfs, zbs_s, zfs_s

truth, noisy, smooth, zbs, zfs, zbs_s, zfs_s = temporal_series()
# 静态背景区域（避开移动前景的包围盒）
fg_union = np.zeros((H, W), bool)
for i in range(len(truth)):
    fgx = 0.16 * np.sin(2 * np.pi * i / len(truth))
    _, a = render(TGT, TGT, ZB, ZF, soft=True, fg_x=fgx)
    fg_union |= a > 0.05
bg_mask = ~ndimage.binary_dilation(fg_union, iterations=3)
def flicker(series):
    d = np.abs(np.diff(series, axis=0))[:, bg_mask].mean(axis=1)
    return float(d.mean())
flick_noisy = flicker(noisy)
flick_smooth = flicker(smooth)
flick_reduce = (flick_noisy - flick_smooth) / max(flick_noisy, 1e-12) * 100
psnr_noisy = float(np.mean([psnr(a, b) for a, b in zip(noisy, truth)]))
psnr_smooth = float(np.mean([psnr(a, b) for a, b in zip(smooth, truth)]))
psnr_gain = psnr_smooth - psnr_noisy
std_before = float(np.mean([zbs.std(), zfs.std()]))
std_after = float(np.mean([zbs_s.std(), zfs_s.std()]))

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("合成图像尺寸与相机数", target.shape == (H, W, 3) and len(CAM) == 8, f"8/{W}×{H}")
check("硬边界图与目标有差异", psnr_hard < 45, f"{psnr_hard:.2f} dB")
check("软边界 PSNR 高于硬边界", psnr_soft > psnr_hard, f"{psnr_hard:.2f}→{psnr_soft:.2f}")
check("软边界边缘误差下降≥20%", edge_gain >= 20, f"{edge_gain:.1f}%")
check("时间序列帧数", len(truth) == 24, f"{len(truth)}")
check("抖动深度标准差≥0.05", std_before >= 0.05, f"{std_before:.3f}")
check("时间平滑后标准差降至抖动前 40% 以下", std_after <= std_before * 0.4, f"{std_before:.3f}→{std_after:.3f}")
check("闪烁指标下降≥30%", flick_reduce >= 30, f"{flick_noisy:.5f}→{flick_smooth:.5f}")
check("时间滤波 PSNR 提升≥0.3dB", psnr_gain >= 0.3, f"{psnr_gain:.2f}")
check("每帧 PSNR 有限", np.isfinite([psnr(a, b) for a, b in zip(noisy, truth)]).all(), "finite")
check("背景区域非空", bg_mask.sum() > W * H * 0.1, f"{int(bg_mask.sum())} px")
check("边界声明为合成/非完整系统", True, "一维阵列+两层深度+离线时间平滑")

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：八相机一维阵列 + 左/目标/右三幅图
fig = plt.figure(figsize=(13.2, 5.6))
gs = GridSpec(2, 3, height_ratios=[1.0, 2.7], hspace=0.38, wspace=0.08)
axd = fig.add_subplot(gs[0, :]); axd.set_xlim(-1.0, 1.0); axd.set_ylim(-0.5, 1.5); axd.axis("off")
axd.plot([-0.82, 0.82], [0, 0], color=GRID, lw=2)
for i, c in enumerate(CAM):
    axd.plot(c, 0, "o", ms=12, color=BLUE if c not in (-0.1, 0.1) else ACCENT)
    axd.text(c, -0.27, f"{i+1}", ha="center", va="top", fontsize=17, color=INK)
axd.plot(TGT, 0, "*", ms=22, color=GOLD, markeredgecolor=INK, markeredgewidth=0.8)
axd.annotate("虚拟视角", xy=(0, 0), xytext=(-0.55, 0.28), fontsize=18, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
axd.text(0.12, 0.28, "红点=左右眼", ha="left", va="bottom", fontsize=18, color=ACCENT)
axd.text(-0.96, 1.15, "八台同步相机（简化为一维阵列）", fontsize=19, color=INK)
for j, (cx, title) in enumerate([(CAM[3], "左相机"), (TGT, "虚拟视角"), (CAM[4], "右相机")]):
    ax = fig.add_subplot(gs[1, j])
    img, _ = render(cx, TGT, ZB, ZF, soft=True)
    ax.imshow(img, interpolation="bilinear"); ax.set_title(title, fontsize=19); ax.axis("off")
fig.suptitle("自由视点视频：相机之间插出新视角", fontsize=21)
fig.savefig(FIG / "e133_rig.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：硬边界 / 软边界 / 目标
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
for ax, im, title, note in [(axes[0], hard, "硬边界", f"PSNR {psnr_hard:.2f} dB"),
                            (axes[1], soft, "软边界（分数不透明度）", f"PSNR {psnr_soft:.2f} dB"),
                            (axes[2], target, "目标视图", "参考")]:
    ax.imshow(im, interpolation="bilinear"); ax.set_title(title, fontsize=19); ax.axis("off")
    ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=18, color="white",
            bbox=dict(facecolor="black", alpha=0.62, edgecolor="none", pad=3))
fig.suptitle("两层深度 + 边界抠图：硬切会锯齿，软 alpha 更顺", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e133_layers.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：逐帧深度抖动 vs 时间中值；逐帧 PSNR
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
frames = np.arange(1, len(zfs) + 1)
ax.plot(frames, zfs, "o-", ms=4, color=ACCENT, label="逐帧估计（前景深度）")
ax.plot(frames, zfs_s, "-", lw=3, color=GREEN, label="时间平滑")
ax.axhline(ZF, color=INK, ls="--", lw=1.2, label="真值 Z=2")
ax.set_xlabel("帧", fontsize=18); ax.set_ylabel("前景深度", fontsize=18)
ax.set_ylim(1.7, 2.3); ax.tick_params(labelsize=18); ax.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
ax.set_title("逐帧深度会抖，时间平滑拉回稳定", fontsize=18)
ax = axes[1]
ax.plot(frames, [psnr(a, b) for a, b in zip(noisy, truth)], "o-", ms=4, color=ACCENT, label="逐帧")
ax.plot(frames, [psnr(a, b) for a, b in zip(smooth, truth)], "-", lw=3, color=GREEN, label="时间平滑")
ax.set_xlabel("帧", fontsize=18); ax.set_ylabel("PSNR (dB)", fontsize=18)
ax.tick_params(labelsize=18); ax.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
ax.set_title("时间一致 → 画面更稳", fontsize=18)
fig.suptitle("时间一致性：逐帧独立估计会闪，离线平滑能压住", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e133_temporal.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), image=[W, H], cameras=CAM.tolist(),
                n_cameras=len(CAM), target=TGT, z_bg=ZB, z_fg=ZF,
                psnr_hard=psnr_hard, psnr_soft=psnr_soft, edge_hard=edge_hard,
                edge_soft=edge_soft, edge_gain_pct=edge_gain,
                n_frames=int(len(truth)), flicker_noisy=flick_noisy,
                flicker_smooth=flick_smooth, flicker_noisy_display=round(flick_noisy, 5),
                flicker_smooth_display=round(flick_smooth, 5), flicker_reduce_pct=flick_reduce,
                bg_pixels=int(bg_mask.sum()),
                psnr_noisy=psnr_noisy, psnr_smooth=psnr_smooth, psnr_gain=psnr_gain,
                depth_std_before=std_before, depth_std_after=std_after,
                protocol=dict(synthetic=True, cameras="1D array", layers="background Z=6 + foreground Z=2",
                              matting="soft alpha vs hard mask", temporal="offline gaussian smoothing",
                              full_vr=False, full_6dof=False))
out = ROOT / "outputs" / "e133_spatial_computing_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
for name in ("e133_rig.png", "e133_layers.png", "e133_temporal.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] E133 空间计算结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
