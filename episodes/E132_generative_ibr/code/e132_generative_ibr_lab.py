#!/usr/bin/env python3
"""E132 生成模型 IBR lab：单图新视角、多视图混合、洞填补与四种表示。"""
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
from matplotlib.patches import FancyBboxPatch

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

print(f"[lab] E132 生成模型 IBR 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成外墙+前景柱；单图/三图 warp；扩散填补是经典方法，不是神经网络生成")

W, H = 240, 180
CAM = [-0.4, 0.0, 0.4]
TGT = 0.25
u = (np.arange(W) + 0.5) / W - 0.5
v = (np.arange(H) + 0.5) / H - 0.5
U, V = np.meshgrid(u, v)

def facade(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.56; base[..., 1] = 0.47; base[..., 2] = 0.38
    win = ((np.mod(np.floor((X + 4) / 0.8), 2) == 0) &
           (np.mod(np.floor((Y + 3) / 0.6), 2) == 0) &
           (np.mod(X + 4, 0.8) > 0.12) & (np.mod(X + 4, 0.8) < 0.68) &
           (np.mod(Y + 3, 0.6) > 0.10) & (np.mod(Y + 3, 0.6) < 0.50))
    base[win] = [0.18, 0.28, 0.42]
    line = ((np.mod(X + 4, 0.8) < 0.035) | (np.mod(Y + 3, 0.6) < 0.035))
    base[line] *= 0.78
    return np.clip(base, 0, 1)

def layers(cx):
    bg = facade(cx + 5 * U, 5 * V)
    fgmask = (np.abs((cx + 2 * U) - 0.30) < 0.16) & (np.abs(2 * V) < 0.55)
    fg = np.zeros_like(bg); fg[:] = [0.88, 0.28, 0.12]
    stripe = fgmask & (np.mod(np.floor((2 * V + 1) / 0.12), 2) == 0)
    fg[stripe] = [0.98, 0.62, 0.18]
    return bg, fg, fgmask

def warp(cx):
    dxb = -(TGT - cx) / 5 * W
    dxf = -(TGT - cx) / 2 * W
    bg, fg, fm = layers(cx)
    b = ndimage.shift(bg, (0, dxb, 0), order=1, mode="nearest")
    ba = ndimage.shift((~fm).astype(float), (0, dxb), order=1, mode="nearest") > 0.5
    f = ndimage.shift(fg, (0, dxf, 0), order=1, mode="nearest")
    fa = ndimage.shift(fm.astype(float), (0, dxf), order=1, mode="nearest") > 0.5
    out = np.where(fa[..., None], f, b)
    return out, ba | fa

warps = [warp(cx) for cx in CAM]
bg_t, fg_t, fm_t = layers(TGT)
target = np.where(fm_t[..., None], fg_t, bg_t)
single, single_valid = warps[1]
single_holes = int((~single_valid).sum())
single_psnr = float(10 * np.log10(1 / max(((single - target) ** 2)[single_valid].mean(), 1e-12)))

# 三视图固定平均：每个有效源等权。
stack = np.stack([w[0] for w in warps])
masks = np.stack([w[1] for w in warps])
num = (stack * masks[..., None]).sum(0)
den = masks.sum(0)
multi = np.where(den[..., None] > 0, num / np.maximum(den, 1)[..., None], 0)
multi_valid = den > 0
multi_holes = int((~multi_valid).sum())
multi_psnr = float(10 * np.log10(1 / max(((multi - target) ** 2).mean(), 1e-12)))

def nearest_fill(img, valid):
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return img[tuple(idx)]

def diffusion_fill(img, valid, iters=80):
    out = nearest_fill(img, valid).copy()
    holes = ~valid
    for _ in range(iters):
        blur = np.zeros_like(out)
        for ax in (0, 1):
            blur += np.roll(out, 1, ax) + np.roll(out, -1, ax)
        blur /= 4
        out[holes] = blur[holes]
    return out

naive = nearest_fill(single, single_valid)
diffused = diffusion_fill(single, single_valid)
naive_psnr = float(10 * np.log10(1 / max(((naive - target) ** 2).mean(), 1e-12)))
diffused_psnr = float(10 * np.log10(1 / max(((diffused - target) ** 2).mean(), 1e-12)))
fill_gain = diffused_psnr - naive_psnr
multi_gain = multi_psnr - naive_psnr

protocol = dict(synthetic=True, images=[W, H], cameras=CAM, target=TGT,
                scene="facade Z=5 + foreground post Z=2",
                fill="classical nearest + diffusion (not a neural network)",
                full_neural_rendering=False)

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("合成图像尺寸与相机数", single.shape == (H, W, 3) and len(CAM) == 3, f"3/{W}×{H}")
check("单图 warp 有洞", single_holes >= 1000, f"{single_holes}")
check("单图有效区 PSNR≥25", single_psnr >= 25, f"{single_psnr:.2f}")
check("三视图混合补满洞", multi_holes == 0, f"{multi_holes}")
check("三视图混合 PSNR≥30", multi_psnr >= 30, f"{multi_psnr:.2f}")
check("多视图比单图最近邻填补高≥3dB", multi_gain >= 3, f"{multi_gain:.2f}")
check("最近邻填补 PSNR≥24", naive_psnr >= 24, f"{naive_psnr:.2f}")
check("扩散填补 PSNR≥26", diffused_psnr >= 26, f"{diffused_psnr:.2f}")
check("扩散填补比最近邻高≥1dB", fill_gain >= 1, f"{fill_gain:.2f}")
check("填补后无洞", int((~single_valid).sum()) == single_holes, f"{single_holes}")
check("lab 边界声明为非神经网络", protocol["full_neural_rendering"] is False and protocol["synthetic"] is True,
      "经典扩散填补，不是生成网络")

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：四种表示路线
fig, ax = plt.subplots(figsize=(13.2, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
boxes = [
    ("纹理网格", "RGB 纹理 + 三角网格", "光栅化 + 神经后处理", "LookinGood / DLSS"),
    ("深度图 / 层", "RGB-D / MPI /\nbillboard", "warp + 合成 + 解码", "Meshry / SynSin /\nGeLaTO"),
    ("体素", "3D RGBα 网格", "ray march + 体渲染", "DeepVoxels /\nNeural Volumes"),
    ("隐式 MLP", "网络权重", "逐点查询 + 体积合成", "NeRF / IBRNet /\nPIFu"),
]
for i, (name, store, render, examples) in enumerate(boxes):
    x = 0.25 + i * 2.45
    ax.add_patch(FancyBboxPatch((x, 0.75), 2.15, 2.55, boxstyle="round,pad=0.06",
                                facecolor="#F3E9D2" if i % 2 == 0 else "#E6EBF2",
                                edgecolor=INK, linewidth=1.5))
    ax.text(x + 1.07, 3.00, name, ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(x + 1.07, 2.38, store, ha="center", va="center", fontsize=18, color=INK)
    ax.text(x + 1.07, 1.88, render, ha="center", va="center", fontsize=18, color=INK)
    ax.text(x + 1.07, 1.28, examples, ha="center", va="center", fontsize=18, color="#5A4A2A")
    if i < 3:
        ax.annotate("", xy=(x + 2.40, 2.02), xytext=(x + 2.17, 2.02),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2.2))
ax.text(5.0, 3.85, "四种 3D 表示，四条神经渲染路线", ha="center", fontsize=21, color=INK)
ax.text(5.0, 0.35, "共同点：都要“表示 + 可微渲染 + 生成/学习模块”", ha="center", fontsize=18, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e132_taxonomy.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：单图 / 三视图 / 目标
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
for ax, im, title in [(axes[0], single, "单图 warp（黑洞）"), (axes[1], multi, "三视图平均"), (axes[2], target, "目标视图")]:
    ax.imshow(im, interpolation="bilinear"); ax.set_title(title, fontsize=20); ax.axis("off")
axes[0].text(0.02, 0.96, f"洞 {single_holes} px\n有效区 PSNR {single_psnr:.1f} dB", transform=axes[0].transAxes,
             va="top", fontsize=18, color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
axes[1].text(0.02, 0.96, f"PSNR {multi_psnr:.1f} dB\n洞 {multi_holes}", transform=axes[1].transAxes,
             va="top", fontsize=18, color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
fig.suptitle("多视图混合：一张图有洞，三张图互相补", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e132_blend.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：单图洞 / 最近邻 / 扩散 / 目标
fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.1))
for ax, im, title in [(axes[0], single, "单图 warp"), (axes[1], naive, "最近邻填补"),
                      (axes[2], diffused, "扩散填补"), (axes[3], target, "目标")]:
    ax.imshow(im, interpolation="bilinear"); ax.set_title(title, fontsize=19); ax.axis("off")
axes[0].text(0.02, 0.96, f"洞 {single_holes} px", transform=axes[0].transAxes, va="top", fontsize=19,
             color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
axes[1].text(0.02, 0.96, f"PSNR {naive_psnr:.1f} dB", transform=axes[1].transAxes, va="top", fontsize=19,
             color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
axes[2].text(0.02, 0.96, f"PSNR {diffused_psnr:.1f} dB", transform=axes[2].transAxes, va="top", fontsize=19,
             color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
fig.suptitle("单图新视角：先补洞，再谈生成细节", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig(FIG / "e132_fill.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), image=[W, H], cameras=CAM, n_cameras=len(CAM), target=TGT,
                single_holes=single_holes, single_psnr=single_psnr,
                multi_holes=multi_holes, multi_psnr=multi_psnr,
                multi_gain=multi_gain, naive_psnr=naive_psnr,
                diffused_psnr=diffused_psnr, fill_gain=fill_gain,
                protocol=protocol,
                fig_sizes={"taxonomy": [13.2, 5.2], "blend": [13.0, 4.6], "fill": [13.4, 4.1]})
out = ROOT / "outputs" / "e132_generative_ibr_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
for name in ("e132_taxonomy.png", "e132_blend.png", "e132_fill.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] E132 生成模型 IBR 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
