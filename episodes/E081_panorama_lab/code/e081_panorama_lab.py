#!/usr/bin/env python3
"""E081 实验课：两图到全景——柱面投影、平移对齐、羽化/金字塔混合（程序生成场景）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter, map_coordinates

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"

WV, HS, HC = 900, 700, 500
F = 600.0
WC = int(round(2.0 * F * np.arctan((WV / 2.0) / F)))
YAW = np.deg2rad(18.0)
CX, CY_S = (WV - 1) / 2.0, (HS - 1) / 2.0
CXC, CY_C = (WC - 1) / 2.0, (HC - 1) / 2.0
GAIN_TRUE = 1.18
RNG = np.random.default_rng(20261102)

# —— 解析柱面纹理：随机傅里叶分量 + 两条 tanh 带，保证左右视图同源 ——
_comp = []
for _ in range(26):
    _comp.append((RNG.uniform(0.025, 0.065), RNG.integers(2, 23), RNG.uniform(-6, 6), RNG.uniform(0, 2 * np.pi)))

def texture(theta, h):
    v = 0.46
    for amp, m, n, ph in _comp:
        v = v + amp * np.cos(m * theta + n * h + ph)
    v = v + 0.10 * np.tanh(3.0 * np.sin(9.0 * theta + 0.7)) * np.exp(-(h / 0.30) ** 2)
    v = v + 0.07 * np.tanh(4.0 * np.sin(17.0 * theta - 0.4))
    return np.clip(v, 0.03, 0.78)

def view(psi):
    """透视视图：像素射线 (u,v,F) 旋转 yaw 后采样柱面纹理。"""
    u = (np.arange(WV) - CX)[None, :]
    v = (np.arange(HS) - CY_S)[:, None]
    theta = np.arctan2(u, F) + psi
    w = np.sqrt(u * u + F * F)
    h = v / w
    return texture(theta, h)

def to_cylinder(img):
    """投回各自相机的柱面坐标：x=Fθ，y=Fh（θ 相对光轴，不含 yaw）。"""
    xc = (np.arange(WC) - CXC)[None, :]
    yc = (np.arange(HC) - CY_C)[:, None]
    th = xc / F
    h = yc / F
    us = F * np.tan(th) + (img.shape[1] - 1) / 2.0
    cy_src = (img.shape[0] - 1) / 2.0
    vs = F * h / np.cos(th) + cy_src
    return map_coordinates(img, [np.broadcast_to(vs, (HC, WC)), np.broadcast_to(us, (HC, WC))],
                           order=3, mode="constant", cval=0.0)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else -1.0

def align_1d(a, b, min_overlap=220):
    pa = a.mean(axis=0); pb = b.mean(axis=0)
    best = (-2.0, None)
    for d in range(0, WC - min_overlap + 1):
        score = ncc(pa[d:], pb[:WC - d])
        if score > best[0]:
            best = (score, d)
    return best[1], best[0]

def align_coarse(a, b, step=4, min_overlap=180):
    a4 = gaussian_filter(a, 2.0)[::step, ::step]
    b4 = gaussian_filter(b, 2.0)[::step, ::step]
    mo = max(20, min_overlap // step)
    best = (-2.0, None, None)
    for dy in range(-3, 4):
        for dx in range(0, a4.shape[1] - mo + 1):
            if dy >= 0:
                aa = a4[dy:, dx:]; bb = b4[:aa.shape[0], :aa.shape[1]]
            else:
                aa = a4[:a4.shape[0] + dy, dx:]; bb = b4[-dy:, :aa.shape[1]]
            if aa.shape[0] < a4.shape[0] - 8 or aa.shape[1] < mo: continue
            sc = ncc(aa, bb)
            if sc > best[0]: best = (sc, dx * step, dy * step)
    return best[1], best[2], best[0]

def align_2d(a, b, dx0, rad=6):
    """柱面模型只允许水平位移：固定 dy=0，只细化 dx。"""
    best = (-2.0, None)
    for dx in range(dx0 - rad, dx0 + rad + 1):
        if dx < 0 or dx >= WC: continue
        aa = a[:, dx:]; bb = b[:, :aa.shape[1]]
        if aa.shape[1] < 220: continue
        score = ncc(aa, bb)
        if score > best[0]:
            best = (score, dx)
    return best[1], best[0]

def downsample(x):
    return gaussian_filter(x, 1.0)[::2, ::2]

def upsample(x, shape):
    y = np.zeros(shape, dtype=float)
    y[::2, ::2] = x[: (shape[0] + 1) // 2, : (shape[1] + 1) // 2]
    return gaussian_filter(y, 1.0) * 4.0

def pyr_blend(a, b, m, levels=5):
    ga, gb, gm = [a], [b], [m]
    for _ in range(levels - 1):
        ga.append(downsample(ga[-1])); gb.append(downsample(gb[-1])); gm.append(downsample(gm[-1]))
    la, lb = [], []
    for i in range(levels - 1):
        la.append(ga[i] - upsample(ga[i + 1], ga[i].shape))
        lb.append(gb[i] - upsample(gb[i + 1], gb[i].shape))
    la.append(ga[-1]); lb.append(gb[-1])
    out = la[-1] * (1 - gm[-1]) + lb[-1] * gm[-1]
    for i in range(levels - 2, -1, -1):
        out = upsample(out, la[i].shape) + la[i] * (1 - gm[i]) + lb[i] * gm[i]
    return out

# —— 生成两幅视图（右图曝光 ×1.18，各加独立噪声） ——
left = view(-YAW) + RNG.normal(0, 0.0015, (HS, WV))
u_norm = (np.arange(WV) - CX) / CX
v_norm = (np.arange(HS) - CY_S) / CY_S
vignette = 1.0 + 0.10 * u_norm[None, :]
shadow = 1.0 - 0.15 * np.exp(-((u_norm[None, :] + 0.55) / 0.18) ** 2 - (v_norm[:, None] / 0.35) ** 2)
right = np.clip(view(YAW) * GAIN_TRUE * vignette * shadow, 0, 0.999) + RNG.normal(0, 0.0015, (HS, WV))
cyl_l = to_cylinder(left)
cyl_r = to_cylinder(right)
dx_true = F * (2 * YAW)

# —— 对齐：1D 列剖面粗估 + 2D NCC 细化 ——
dx_profile, score1 = align_1d(cyl_l, cyl_r)
dx_coarse, dy_coarse, score_c = align_coarse(cyl_l, cyl_r)
dx_est, score2 = align_2d(cyl_l, cyl_r, dx_coarse)
dy_est = 0
# —— 重叠区最小二乘估增益：B ≈ g A ——
ov_a = cyl_l[:, dx_est:]
ov_b = cyl_r[:, :WC - dx_est]
gain_est = float((ov_a * ov_b).sum() / (ov_a * ov_a).sum())
# 全局增益的真值：重叠区实际用到的右图列（柱面坐标→透视 u），含渐晕与局部阴影
xr = np.arange(0, WC - dx_est)
ur = F * np.tan((xr - CXC) / F) + CX
ui = np.clip(np.round(ur).astype(int), 0, WV - 1)
gain_expected = float(GAIN_TRUE * np.mean(vignette[0, ui] * shadow[:, ui].mean(axis=0)))

# —— 画布与三种混合 ——
CW = WC + dx_est
A = np.zeros((HC, CW)); B = np.zeros((HC, CW))
A[:, :WC] = cyl_l
B[:, dx_est:dx_est + WC] = cyl_r / gain_est
cut = int((dx_est + WC) // 2)
valid = np.ones((HC, CW), bool)
valid[:, :max(1, dx_est - 2)] = True
valid[:, min(CW - 2, WC + 2):] = True

# 硬拼
hard = np.where(np.arange(CW)[None, :] < cut, A, B)
# 羽化：重叠区线性权重
alpha = np.zeros((HC, CW))
for x in range(CW):
    if x < dx_est: alpha[:, x] = 0.0
    elif x >= WC: alpha[:, x] = 1.0
    else: alpha[:, x] = (x - dx_est + 0.5) / (WC - dx_est)
feather = A * (1 - alpha) + B * alpha
# 金字塔：硬阶跃掩膜，粗层自然变宽
m0 = (np.arange(CW)[None, :] >= cut).astype(float) * np.ones((HC, 1))
pyramid = pyr_blend(A, B, m0, levels=5)

# —— 真值：同一柱面纹理在画布坐标上的解析采样 ——
xc = np.arange(CW)[None, :]; yc = (np.arange(HC) - CY_C)[:, None]
truth = texture((xc - CXC) / F - YAW, yc / F)

# —— 指标 ——
def rmse(x, y): return float(np.sqrt(np.mean((x - y) ** 2)))
def lowpass(x): return gaussian_filter(x, 8.0)
def highpass(x): return x - gaussian_filter(x, 1.2)
def seam_jump(x):
    d = np.abs(np.diff(x, axis=1))
    return float(d[:, cut - 1].mean())
def base_jump(x):
    d = np.abs(np.diff(x, axis=1))
    cols = list(range(max(0, cut - 60), max(1, cut - 30))) + list(range(min(CW - 2, cut + 30), min(CW - 1, cut + 60)))
    return float(np.median(d[:, cols]))
def hp_energy(x):
    cols = np.r_[dx_est + 20:max(dx_est + 21, cut - 30), min(CW - 21, cut + 30):WC - 20]
    h = highpass(x)[:, cols]
    return float(np.sqrt(np.mean(h * h)))

metrics = {}
for name, img in (("hard", hard), ("feather", feather), ("pyramid", pyramid)):
    metrics[name] = dict(rmse=rmse(img, truth), low_jump=seam_jump(lowpass(img)),
                         low_base=base_jump(lowpass(img)), full_jump=seam_jump(img),
                         full_base=base_jump(img), hp_energy=hp_energy(img))
metrics["feather"]["hp_vs_hard"] = metrics["feather"]["hp_energy"] / metrics["hard"]["hp_energy"]
metrics["pyramid"]["hp_vs_hard"] = metrics["pyramid"]["hp_energy"] / metrics["hard"]["hp_energy"]
for _m in metrics.values():
    _m["seam_ratio"] = _m["full_jump"] / _m["full_base"]
# 投影往返：柱面→透视→柱面
def cyl_to_view(cyl):
    """柱面 → 透视视图（与 to_cylinder 互逆）：u=F tanθ，v=F h/cosθ。"""
    u = (np.arange(WV) - CX)[None, :]
    v = (np.arange(HS) - CY_S)[:, None]
    th = np.arctan2(u, F)
    w = np.sqrt(u * u + F * F)
    h = v / w
    xc = F * th + CXC
    yc = F * h + CY_C
    return map_coordinates(cyl, [np.broadcast_to(yc, (HS, WV)), np.broadcast_to(xc, (HS, WV))],
                           order=3, mode="constant", cval=0.0)
roundtrip = rmse(to_cylinder(cyl_to_view(cyl_l)), cyl_l)

# —— 图 1：透视视图 → 柱面 ——
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), dpi=150)
fig.patch.set_facecolor(PAPER)
for ax in axes.ravel(): ax.set_facecolor(PAPER); ax.axis("off")
axes[0, 0].imshow(left, cmap="copper", vmin=0, vmax=1); axes[0, 0].set_title("左视图（透视）", fontsize=14, color=INK)
axes[0, 1].imshow(right, cmap="copper", vmin=0, vmax=1); axes[0, 1].set_title("右视图（透视，曝光 ×1.18）", fontsize=14, color=INK)
axes[1, 0].imshow(cyl_l, cmap="copper", vmin=0, vmax=1); axes[1, 0].set_title("投回柱面：竖线变直", fontsize=14, color=INK)
axes[1, 1].imshow(cyl_r, cmap="copper", vmin=0, vmax=1); axes[1, 1].set_title("投回柱面：内容左移", fontsize=14, color=INK)
fig.suptitle("柱面投影：两幅透视视图变成同一张柱面图", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, f"程序生成｜往返 RMSE {roundtrip:.3f}（灰阶 0–1）｜已知位移 {dx_true:.1f}px", ha="center", fontsize=14, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "e081_cylindrical.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 2：对齐画布 ——
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
show = feather.copy()
ax.imshow(show, cmap="copper", vmin=0, vmax=1, extent=[0, CW, HC, 0])
ax.axvspan(dx_est, WC, color=BLUE, alpha=0.10)
ax.axvline(cut, color=ACCENT, lw=2)
ax.annotate("", xy=(dx_est, 64), xytext=(0, 64), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8))
ax.text(dx_est / 2, 88, f"估计位移 {dx_est}px（真值 {dx_true:.1f}）", ha="center", fontsize=14, color=ACCENT,
        bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.85, pad=2))
ax.text((dx_est + WC) / 2, HC - 20, "重叠区：用来估增益、做过渡", ha="center", fontsize=14, color=BLUE,
        bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.85, pad=2))
ax.set_title("对齐后的画布：重叠区与接缝位置", fontsize=16, color=INK)
ax.axis("off")
fig.text(0.5, 0.02, f"列剖面粗估 {dx_coarse}px｜2D 细化 dx={dx_est} dy={dy_est}｜NCC={score2:.4f}", ha="center", fontsize=14, color=ACCENT)
fig.savefig(FIG / "e081_align.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 3：三种混合对照 ——
fig = plt.figure(figsize=(9.6, 5.3), dpi=150)
fig.patch.set_facecolor(PAPER)
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
for j, (name, label, img) in enumerate((("hard", "硬拼", hard), ("feather", "羽化", feather), ("pyramid", "5 层金字塔", pyramid))):
    ax = fig.add_subplot(gs[0, j]); ax.set_facecolor(PAPER)
    crop = img[:, max(0, cut - 65):min(CW, cut + 65)]
    ax.imshow(crop, cmap="copper", vmin=0, vmax=1, aspect="auto")
    ax.axvline(65, color=ACCENT, lw=1.5, ls="--")
    ax.set_title(label, fontsize=14, color=INK); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK)
ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(PAPER)
cols = np.arange(max(0, cut - 25), min(CW, cut + 25))
ax.plot(cols - cut, hard[HC // 2, cols], color=BLUE, lw=1.5, label="硬拼")
ax.plot(cols - cut, feather[HC // 2, cols], color=GREEN, lw=1.5, label="羽化")
ax.plot(cols - cut, pyramid[HC // 2, cols], color=PURPLE, lw=1.5, label="金字塔")
ax.axvline(0, color=ACCENT, lw=1.2, ls="--")
ax.set_xlabel("距接缝的列数", fontsize=14); ax.set_ylabel("单行亮度（灰度 0–1）", fontsize=14)
ax.legend(fontsize=14, ncol=3, frameon=False, loc="upper right")
ax.set_title("接缝附近单行剖面：硬拼有明显跳变，羽化/金字塔平滑", fontsize=14, color=INK)
fig.suptitle("三种混合：接缝跳变与高频细节的取舍", fontsize=17, color=INK, y=0.98)
fig.text(0.5, 0.01,
         f"程序生成｜全频跳变 硬 {metrics['hard']['full_jump']:.4f}/羽 {metrics['feather']['full_jump']:.4f}/金 {metrics['pyramid']['full_jump']:.4f}｜"
         f"高频保留（相对硬拼） 羽 {metrics['feather']['hp_vs_hard']:.2f}/金 {metrics['pyramid']['hp_vs_hard']:.2f}",
         ha="center", fontsize=14, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(FIG / "e081_blend.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

tests = [
    ("柱面往返误差可控", roundtrip < 0.03, f"rmse={roundtrip:.2e}"),
    ("1D 列剖面粗对齐", abs(dx_profile - dx_true) <= 2, f"dx={dx_profile}/真值 {dx_true:.1f}"),
    ("2D NCC 细化位移", abs(dx_est - dx_true) <= 1 and dy_est == 0, f"dx={dx_est}/dy={dy_est}"),
    ("全局增益估到重叠区均值", abs(gain_est - gain_expected) < 0.03, f"gain={gain_est:.4f}/期望 {gain_expected:.4f}"),
    ("硬拼全频接缝跳变明显", metrics["hard"]["full_jump"] > 3 * metrics["hard"]["full_base"], f"跳变={metrics['hard']['full_jump']:.4f}/基线 {metrics['hard']['full_base']:.5f}"),
    ("羽化压低接缝跳变", metrics["feather"]["full_jump"] < 0.30 * metrics["hard"]["full_jump"], f"羽化={metrics['feather']['full_jump']:.4f}"),
    ("金字塔压低接缝跳变", metrics["pyramid"]["full_jump"] < 0.30 * metrics["hard"]["full_jump"], f"金字塔={metrics['pyramid']['full_jump']:.4f}"),
    ("羽化会糊高频", metrics["feather"]["hp_vs_hard"] < 0.90, f"相对硬拼 hp={metrics['feather']['hp_vs_hard']:.3f}"),
    ("金字塔保住更多高频", metrics["pyramid"]["hp_vs_hard"] > 0.95 and metrics["pyramid"]["hp_vs_hard"] > metrics["feather"]["hp_vs_hard"], f"金={metrics['pyramid']['hp_vs_hard']:.3f}/羽={metrics['feather']['hp_vs_hard']:.3f}"),
    ("三图在盘", all((FIG / n).exists() for n in ("e081_cylindrical.png", "e081_align.png", "e081_blend.png")), "三张全在"),
]
lines = []; n_pass = 0
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass += bool(ok)
readings = dict(WV=WV, WC=WC, F=F, yaw_deg=18.0, dx_true=dx_true, dx_profile=dx_profile, dx_coarse=dx_coarse,
                dy_coarse=dy_coarse, dx_est=dx_est, dy_est=dy_est, ncc=score2, gain_true=GAIN_TRUE, gain_expected=gain_expected, gain_est=gain_est, roundtrip=roundtrip,
                hard=metrics["hard"], feather=metrics["feather"], pyramid=metrics["pyramid"],
                n_pass=n_pass, n_total=len(tests))
(OUT / "e081_panorama_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e081_panorama_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E081 两图到全景 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e081_panorama_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests) - n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests): raise SystemExit(1)
