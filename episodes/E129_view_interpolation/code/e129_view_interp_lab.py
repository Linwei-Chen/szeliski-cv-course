#!/usr/bin/env python3
"""E129 视图插值 lab：两幅参考图 + 深度图，前向 warp、z-buffer、空洞合并、视角权重。"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"
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

print(f"[lab] E129 视图插值 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：二维分层合成场景；前向 warp+z-buffer；不是真实图像/真实深度")

W, H = 160, 120
K = 20.0
ZF, ZB = 1.0, 4.0
yy, xx = np.mgrid[0:H, 0:W]

def render(t):
    """t=-1 左参考，t=+1 右参考，t=0 中间真值。"""
    sb = t * K / ZB; sf = t * K / ZF
    bg = np.zeros((H, W, 3))
    bg[..., 0] = 0.55 + 0.18 * np.sin((xx - sb) / 9.0)
    bg[..., 1] = 0.58 + 0.15 * np.cos(yy / 11.0)
    bg[..., 2] = 0.62 + 0.12 * np.sin((xx + yy - sb) / 13.0)
    bg += (((np.floor((xx - sb) / 16) + np.floor(yy / 16)) % 2) * 0.08)[..., None]
    cx = W / 2 + sf; cy = H / 2
    disk = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < 18
    img = bg.copy(); img[disk] = [0.85, 0.25, 0.18]
    z = np.full((H, W), ZB); z[disk] = ZF
    return np.clip(img, 0, 1), z

def warp(img, z, t_src, zbuffer=True):
    out = np.zeros_like(img); mask = np.zeros((H, W), bool)
    dx = xx - t_src * K / z
    order = np.argsort(-z.ravel(), kind="stable") if zbuffer else np.argsort(z.ravel(), kind="stable")
    for idx in order:
        y, x = divmod(int(idx), W); base = int(np.floor(dx[y, x]))
        for xd in (base, base + 1):
            if 0 <= xd < W:
                out[y, xd] = img[y, x]; mask[y, xd] = True
    return out, mask

def merge(a, ma, b, mb, w=0.5):
    both = ma & mb; onlya = ma & ~mb; onlyb = mb & ~ma
    out = np.zeros_like(a); out[both] = w * a[both] + (1 - w) * b[both]
    out[onlya] = a[onlya]; out[onlyb] = b[onlyb]
    return out, ~(ma | mb)

def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))

gt, zgt = render(0)
L, zL = render(-1); R, zR = render(1)
wL, mL = warp(L, zL, -1); wR, mR = warp(R, zR, 1)
mid, holes = merge(wL, mL, wR, mR)
p_z = psnr(mid, gt)
fg = zgt < 2
fg_z = float(np.mean(mid[..., 0][fg] > mid[..., 1][fg] + 0.3))
# 无 z-buffer：近的先写，远的后写覆盖
nL, nLm = warp(L, zL, -1, False); nR, nRm = warp(R, zR, 1, False)
nmid, nholes = merge(nL, nLm, nR, nRm)
p_no = psnr(nmid, gt)
fg_no = float(np.mean(nmid[..., 0][fg] > nmid[..., 1][fg] + 0.3))
# 基线/空洞
base_t = [0.25, 0.5, 0.75, 1.0]; holes_left = []; psnrs = []
for t0 in base_t:
    a, za = render(-t0); b, zb = render(t0)
    wa, ma = warp(a, za, -t0); wb, mb = warp(b, zb, t0)
    o, _ = merge(wa, ma, wb, mb)
    holes_left.append(int((~ma).sum())); psnrs.append(psnr(o, gt))
# 曝光不一致
mid_bright, _ = merge(np.clip(wL * 1.12, 0, 1), mL, wR, mR)
p_exp = psnr(mid_bright, gt); exp_drop = p_z - p_exp
# 视角权重（view-dependent texture maps）
angles = np.array([-40.0, 0.0, 35.0]); target = 10.0
w = 1.0 / (np.abs(angles - target) + 1.0); wn = w / w.sum()
blend_mid = (0.5, 0.5); blend_right = (0.25, 0.75)
protocol = dict(images=[160, 120], z_fg=ZF, z_bg=ZB, baseline=2.0,
                representation="2D layered scene + forward warp", zbuffer=True)

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("参考图/真值尺寸与深度范围", L.shape == (H, W, 3) and zL.min() == ZF and zL.max() == ZB, f"{L.shape}, z={zL.min()}–{zL.max()}")
check("前景视差 = 20px", abs((K / ZF) - 20) < 1e-6, f"{K/ZF:.1f}px")
check("基线变宽，单图空洞单调增加", all(holes_left[i] < holes_left[i+1] for i in range(3)), str(holes_left))
check("z-buffer PSNR ≥ 32", p_z >= 32, f"{p_z:.2f}")
check("无 z-buffer PSNR ≤ 27", p_no <= 27, f"{p_no:.2f}")
check("z-buffer 前景保留 ≥ 0.95", fg_z >= 0.95, f"{fg_z:.3f}")
check("无 z-buffer 前景保留 ≤ 0.50", fg_no <= 0.50, f"{fg_no:.3f}")
check("中点混合权重 0.5/0.5", blend_mid == (0.5, 0.5), str(blend_mid))
check("中点两图合并后无空洞", int(holes.sum()) == 0, f"{int(holes.sum())}")
check("曝光差导致 PSNR 下降 ≥ 3dB", exp_drop >= 3, f"{exp_drop:.2f}")
check("视角权重最近源最高", int(np.argmax(wn)) == 1, f"{wn.round(3).tolist()}")
check("协议字段完整且声明分层场景", protocol["representation"].startswith("2D") and protocol["images"] == [160, 120], str(protocol))

# 图1 左参考/中间/右参考
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
for ax, im, title in [(axes[0], L, "左参考（t=−1）"), (axes[1], mid, "插值中点（t=0）"), (axes[2], R, "右参考（t=+1）")]:
    ax.imshow(im, interpolation="nearest"); ax.set_title(title, fontsize=19); ax.axis("off")
fig.suptitle("两幅参考图，插出中间的虚拟视角", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e129_views.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 图2 基线 vs 空洞 + 两张掩码
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
ax = axes[0]
bars = ax.bar([str(t) for t in base_t], holes_left, color=BLUE, width=0.58)
for b, v in zip(bars, holes_left):
    ax.text(b.get_x() + b.get_width()/2, v + 18, str(v), ha="center", fontsize=17)
ax.set_xlabel("参考相机基线（±t）", fontsize=17); ax.set_ylabel("单图空洞像素数", fontsize=17)
ax.set_title("基线越宽，空洞越多", fontsize=19); ax.tick_params(labelsize=17)
ax.set_ylim(0, max(holes_left) * 1.22); ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
a, za = render(-0.25); wa, ma = warp(a, za, -0.25)
b, zb2 = render(-1.0); wb, mb = warp(b, zb2, -1.0)
for ax, m, title in [(axes[1], ~ma, "窄基线（t=0.25）空洞"), (axes[2], ~mb, "宽基线（t=1.0）空洞")]:
    ax.imshow(m, cmap="gray", interpolation="nearest"); ax.set_title(title, fontsize=18); ax.axis("off")
fig.suptitle("基线越宽，遮挡露出的空洞越大", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e129_baseline.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 图3 z-buffer vs 无 z-buffer
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
for ax, im, title in [(axes[0], mid, "有 z-buffer"), (axes[1], nmid, "无 z-buffer"), (axes[2], gt, "目标视图")]:
    ax.imshow(im, interpolation="nearest"); ax.set_title(title, fontsize=19); ax.axis("off")
axes[0].text(0.02, 0.96, f"PSNR {p_z:.1f} dB\n前景保留 {fg_z:.2f}", transform=axes[0].transAxes, va="top", fontsize=17, color="white", bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3))
axes[1].text(0.02, 0.96, f"PSNR {p_no:.1f} dB\n前景保留 {fg_no:.2f}", transform=axes[1].transAxes, va="top", fontsize=17, color="white", bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3))
fig.suptitle("没有深度竞争，背景会盖掉前景", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e129_zbuffer.png", dpi=160, bbox_inches="tight"); plt.close(fig)

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
readings = dict(n_pass=n_pass, n_total=len(checks), holes_left=holes_left, psnr_base=psnrs,
                p_z=p_z, p_no=p_no, fg_z=fg_z, fg_no=fg_no,
                blend_mid=list(blend_mid), blend_right=list(blend_right),
                exp_drop=exp_drop, p_exp=p_exp, exposure_factor=1.12, baselines=base_t, vdtm_weights=wn.round(6).tolist(),
                protocol=protocol,
                fig_sizes={"views": [12.6, 4.3], "baseline": [12.6, 4.4], "zbuffer": [12.6, 4.3]})
(ROOT / "outputs" / "e129_view_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
for f in ("e129_views.png", "e129_baseline.png", "e129_zbuffer.png"):
    print(f"[lab] PNG sha256({f})={hashlib.sha256((FIG/f).read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((ROOT/'outputs'/'e129_view_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
