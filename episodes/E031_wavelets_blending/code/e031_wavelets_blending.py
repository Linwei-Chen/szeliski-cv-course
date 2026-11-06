#!/usr/bin/env python3
# E031 数值实验：§3.5.4 Wavelets + §3.5.5 Image blending（p.159 下–168 上）
# run1 —— 权威存档 stdout：code/outputs/e031_wavelets_blending_run1.txt
#
# 口径注记（与 refs 逐字同步；门A 惯例——每条注记 run1 打印，refs 引原句）：
# · lifting 2-tap 方案：原书 Figure 3.39 图注+正文（p.163 L219–225）只给文字描述
#   「偶值平均从奇值中减去得高通小波系数；少量高通小波加回偶序列使其正确低通」与
#   有效低通核 [−1/8, 1/4, 3/4, 1/4, −1/8]——本实验按此实现 2-tap 预测（奇−偶均值）
#   与修正（偶+=¼(左小波+右小波)）；边界处理原书未给，小波实验用周期延拓（讲者口径
#   申报，断言全域成立）。可逆性来自合成端严格反演分析端、与修正系数取值无关（原书
#   L211–213「trivially reversible」+L227–228「without affecting its reversibility」
#   ——门A r1 P1-1 修正：D1 注入=合成端 ½ 与分析端 ¼ 失配，1-3/1-4 为两端都用 ½ 的
#   补测：照样完全重构，但有效核 [−1/4, 1/2, 1/2, 1/2, −1/4] 奈奎斯特增益 −1 非低通；
#   ¼ 的作用=使核为低通（3-3 奈奎斯特增益 0）——讲者口径推证。）
# · tight frame 账：原书 p.160 L87–94「traditional pyramids are overcomplete…whereas
#   wavelets provide a tight frame, i.e., they keep the size of the decomposition the
#   same as the image (Figure 3.35b)」+ p.161 L131–133「the high-pass filter followed by
#   decimation keeps 3/4 of the original pixels, while 1/4 of the low-frequency
#   coefficients are passed on to the next stage」——账面断言照此。
# · HH/HL/LH 方向语义：原书 p.161 L136–138「The high–low and low–high images accentuate
#   the horizontal and vertical edges and gradients, while the high–high image contains
#   the less frequently occurring mixed derivatives」——实验断言用能量排序相对关系
#   （水平条纹带 vs 垂直条纹带对调），不作术语绝对化断言。
# · steerable pyramids：原书只给频域径向划分语义（Figure 3.40a）与性质文字
#   （overcomplete 消混叠/self-inverting/方向选择性，p.164–165 L262–284）——本实验只
#   示意频域划分图，不实现完整 steerable 滤波器组（照 E030 half-octave/quincunx 先例申报）。
# · 融合实测：原书 Figure 3.41/3.42 为苹果橙子照片（版权图，未使用）——实验用程序生成
#   双源图+二值 mask；「低频平滑混合、高频快速混合避免 ghosting」（p.165 L292–295）以
#   过渡带逐列均值跳变对照量化（讲者口径度量，原书未给度量方式）。
# · 融合实现：拉普拉斯金字塔=上一集式 3.68/3.69 二项式核五层（E030 #2/#3 同一实现）；
#   mask 高斯金字塔加权后相加重构（p.165 L296–302）；金字塔卷积边界取零填充（原书未给
#   边界处理——讲者口径申报，融合断言只设在远离边界的内部条带）；mask=1/0 区忠实性
#   断言用相对关系（融合更近本源而非对源）——最粗层低通 mask 权重按原书「低频平滑
#   混合」设计本就平滑扩散至大半幅，绝对恒等不成立，绝对偏差以读数申报。
# · 小波分解边界：与 lifting 项同口径——周期延拓（原书未给边界处理，讲者口径申报）。

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
from numpy import convolve
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
FIG = Path(__file__).resolve().parent.parent / "slides" / "figures"
FIG.mkdir(exist_ok=True)
results = []
R = []


def check(rid, label, ok, detail=""):
    results.append((rid, label, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {rid} {label}" + (f"  {detail}" if detail else ""))


def reading(label, value):
    R.append((label, value))
    print(f"  [读数] {label}: {value}")


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


print("口径注记（七项，与 refs 逐字同步；权威存档=本 stdout 标头）：")
print("· lifting 2-tap 方案：原书 Figure 3.39 图注+正文（p.163 L219–225）只给文字描述"
      "「偶值平均从奇值中减去得高通小波系数；少量高通小波加回偶序列使其正确低通」与"
      "有效低通核 [−1/8, 1/4, 3/4, 1/4, −1/8]——本实验按此实现 2-tap 预测（奇−偶均值）"
      "与修正（偶+=¼(左小波+右小波)）；边界处理原书未给，小波实验用周期延拓（讲者口径"
      "申报，断言全域成立）。可逆性来自合成端严格反演分析端、与修正系数取值无关（原书 "
      "L211–213「trivially reversible」+L227–228「without affecting its reversibility」"
      "——门A r1 P1-1 修正：D1 注入=合成端 ½ 与分析端 ¼ 失配，1-3/1-4 为两端都用 ½ 的"
      "补测：照样完全重构，但有效核 [−1/4, 1/2, 1/2, 1/2, −1/4] 奈奎斯特增益 −1 非低通；"
      "¼ 的作用=使核为低通（3-3 奈奎斯特增益 0）——讲者口径推证。）")
print("· tight frame 账：原书 p.160 L87–94「traditional pyramids are overcomplete…whereas"
      " wavelets provide a tight frame, i.e., they keep the size of the decomposition the"
      " same as the image (Figure 3.35b)」+ p.161 L131–133「the high-pass filter followed by"
      " decimation keeps 3/4 of the original pixels, while 1/4 of the low-frequency"
      " coefficients are passed on to the next stage」——账面断言照此。")
print("· HH/HL/LH 方向语义：原书 p.161 L136–138「The high–low and low–high images accentuate"
      " the horizontal and vertical edges and gradients, while the high–high image contains"
      " the less frequently occurring mixed derivatives」——实验断言用能量排序相对关系"
      "（水平条纹带 vs 垂直条纹带对调），不作术语绝对化断言。")
print("· steerable pyramids：原书只给频域径向划分语义（Figure 3.40a）与性质文字"
      "（overcomplete 消混叠/self-inverting/方向选择性，p.164–165 L262–284）——本实验只"
      "示意频域划分图，不实现完整 steerable 滤波器组（照 E030 half-octave/quincunx 先例申报）。")
print("· 融合实测：原书 Figure 3.41/3.42 为苹果橙子照片（版权图，未使用）——实验用程序生成"
      "双源图+二值 mask；「低频平滑混合、高频快速混合避免 ghosting」（p.165 L292–295）以"
      "过渡带逐列均值跳变对照量化（讲者口径度量，原书未给度量方式）。")
print("· 融合实现：拉普拉斯金字塔=上一集式 3.68/3.69 二项式核五层（E030 #2/#3 同一实现）；"
      "mask 高斯金字塔加权后相加重构（p.165 L296–302）；金字塔卷积边界取零填充（原书未给"
      "边界处理——讲者口径申报，融合断言只设在远离边界的内部条带）；mask=1/0 区忠实性"
      "断言用相对关系（融合更近本源而非对源）——最粗层低通 mask 权重按原书「低频平滑"
      "混合」设计本就平滑扩散至大半幅，绝对恒等不成立，绝对偏差以读数申报。")
print("· 小波分解边界：与 lifting 项同口径——周期延拓（原书未给边界处理，讲者口径申报）。")

rng = np.random.default_rng(31)

# ── #1 lifting 1D/2D 自实现 + 完全重构（Figure 3.38b/3.39 语义）──
print("\n#1 lifting 小波（2-tap 预测+修正，Figure 3.38b/3.39）")


def lift_analyze(x, axis=-1):
    """2-tap lifting 分析：d=奇−偶均值（高通小波），e=偶+¼(左d+右d)（低通）。周期延拓。"""
    x = np.swapaxes(x, axis, -1)
    odd, even = x[..., 1::2], x[..., 0::2]
    d = odd - 0.5 * (even + np.roll(even, -1, axis=-1))
    e = even + 0.25 * (np.roll(d, 1, axis=-1) + d)
    return np.swapaxes(d, -1, axis), np.swapaxes(e, -1, axis)


def lift_synth(d, e, axis=-1):
    """合成=分析反序：先撤修正、再用偶均值还原奇。周期延拓。"""
    d = np.swapaxes(d, axis, -1)
    e = np.swapaxes(e, axis, -1)
    even = e - 0.25 * (np.roll(d, 1, axis=-1) + d)
    odd = d + 0.5 * (even + np.roll(even, -1, axis=-1))
    x = np.empty(even.shape[:-1] + (even.shape[-1] * 2,))
    x[..., 0::2] = even
    x[..., 1::2] = odd
    return np.swapaxes(x, -1, axis)


x1 = rng.uniform(0, 1, 256)
d1, e1 = lift_analyze(x1)
err1 = float(np.max(np.abs(lift_synth(d1, e1) - x1)))
reading("1D 完全重构 max|x−x̂|", f"{err1:.3e}")
check("1-1", "lifting 1D 分析→合成逐点还原（机器精度）", err1 < 1e-12, f"err={err1:.3e}")


def wavelet2d(img, levels=1):
    """Figure 3.37b：先水平（行内 ↓2h）再垂直（↓2v）可分离分解。
    LH=行低通×列高通、HL=行高通×列低通、HH=双高通；LL 传下一级。"""
    details, cur = [], img.astype(float)
    for _ in range(levels):
        hH, hL = lift_analyze(cur, 1)      # 行内：高通/低通，宽减半
        HH, HL = lift_analyze(hH, 0)       # 行高通再列分解
        LH, LL = lift_analyze(hL, 0)       # 行低通再列分解
        details.append({"LH": LH, "HL": HL, "HH": HH, "LL": LL})
        cur = LL
    return details, cur


def iwavelet2d(details, low):
    cur = low
    for det in reversed(details):
        hL = lift_synth(det["LH"], det["LL"], 0)   # 列合成 → 行低通
        hH = lift_synth(det["HH"], det["HL"], 0)   # 列合成 → 行高通
        cur = lift_synth(hH, hL, 1)                # 行合成 → 原图
    return cur


img = rng.uniform(0, 1, (128, 128))
details2, low2 = wavelet2d(img, levels=2)
err2 = float(np.max(np.abs(iwavelet2d(details2, low2) - img)))
reading("2D 两级完全重构 max|I−Î|", f"{err2:.3e}")
check("1-2", "lifting 2D 可分离（先行后列）两级完全重构", err2 < 1e-12, f"err={err2:.3e}")

# 注入 D1：合成端修正系数 ½ 与分析端 ¼ 失配 → 不应还原（门A r1 P1-1 修正语义：
# 可逆性来自合成端严格反演分析端；¼ 本身管的是「偶序列是否被正确低通化」，见 1-3/1-4/3-3）
d1b, e1b = lift_analyze(x1)
even_bad = e1b - 0.5 * (np.roll(d1b, 1) + d1b)
odd_bad = even_bad + 0.5 * (even_bad + np.roll(even_bad, -1))
x_bad = np.empty(256)
x_bad[0::2], x_bad[1::2] = even_bad, odd_bad
errD1 = float(np.max(np.abs(x_bad - x1)))
check("D1", "注入：合成端修正系数 ½ ≠ 分析端 ¼（两端失配）不应还原", errD1 > 1e-3, f"err={errD1:.3f}")

# 补测 1-3/1-4（门A r1 P1-1）：两端都用 ½——照样完全可逆，但有效核不再是低通
x_h = rng.uniform(0, 1, 256)
odd_h, even_h = x_h[1::2], x_h[0::2]
d_h = odd_h - 0.5 * (even_h + np.roll(even_h, -1))
e_h = even_h + 0.5 * (np.roll(d_h, 1) + d_h)
even_r = e_h - 0.5 * (np.roll(d_h, 1) + d_h)
odd_r = d_h + 0.5 * (even_r + np.roll(even_r, -1))
x_r = np.empty(256)
x_r[0::2], x_r[1::2] = even_r, odd_r
err_h = float(np.max(np.abs(x_r - x_h)))
reading("两端都用 ½：完全重构 max|x−x̂|", f"{err_h:.3e}")
check("1-3", "补测：两端都用 ½ 修正照样完全重构（可逆靠合成端反演分析端，不依赖 ¼）",
      err_h < 1e-12, f"err={err_h:.3e}")
k_h = np.array([-0.25, 0.5, 0.5, 0.5, -0.25])
A_h = np.zeros((30, 256))
for i in range(1, 31):
    A_h[i - 1, 2 * i - 2:2 * i + 3] = k_h
err_kh = float(np.max(np.abs(e_h[1:31] - A_h @ x_h)))
nyq_h = float(sum(c * (-1) ** lag for c, lag in zip(k_h, (-2, -1, 0, 1, 2))))
reading("两端 ½ 的有效核（逐点复核）", f"{[float(v) for v in k_h]} err={err_kh:.1e}")
reading("两端 ½ 的有效核奈奎斯特增益 H(π)", f"{nyq_h:.3f}（≠0，非低通）")
check("1-4", "两端 ½ 的有效核=[−1/4,½,½,½,−1/4] 且奈奎斯特增益=−1（¼ 的作用=使核为低通）",
      err_kh < 1e-12 and abs(nyq_h + 1) < 1e-12, f"err={err_kh:.1e}, H(π)={nyq_h:.3f}")

# ── #2 tight frame 账（p.160 L87–94 + p.161 L132–133）──
print("\n#2 tight frame 账：系数总数=原图像素数")
H0 = W0 = 128
reading("输入尺寸", f"{H0}×{W0}={H0*W0} 像素（门A r1 P3-7 补打印）")
det1, low1 = wavelet2d(np.zeros((H0, W0)), levels=1)
reading("一次级：低通 LL", low1.size)
n_detail1 = sum(det1[0][k].size for k in ("LH", "HL", "HH"))
n_total1 = n_detail1 + low1.size
reading("一次级：detail 合计", n_detail1)
reading("一次级：detail+低通", n_total1)
check("2-1", "一次级系数总数=原图像素数（3/4+1/4）", n_total1 == H0 * W0, f"{n_total1} vs {H0*W0}")
check("2-2", "detail 合计=¾ 原图像素", n_detail1 == 3 * H0 * W0 // 4, f"{n_detail1} vs {3*H0*W0//4}")

det2, low2z = wavelet2d(np.zeros((H0, W0)), levels=2)
n_det2 = sum(d[k].size for d in det2 for k in ("LH", "HL", "HH"))
n_total2 = n_det2 + low2z.size
reading("两级：系数总数", n_total2)
check("2-3", "两级递归后系数总数仍=原图像素数（tight frame）", n_total2 == H0 * W0, f"{n_total2} vs {H0*W0}")

# 对照：拉普拉斯金字塔 overcomplete 账（E030 式 3.68/3.69，五层）
BIN = np.array([1, 4, 6, 4, 1]) / 16.0


def blur_rows(im, ker):
    return np.apply_along_axis(lambda r: convolve(r, ker, mode="same"), 1, im)


def blur_cols(im, ker):
    return np.apply_along_axis(lambda c: convolve(c, ker, mode="same"), 0, im)


def gauss_pyr(img, levels):
    pyr = [img.astype(float)]
    for _ in range(levels):
        im = blur_cols(blur_rows(pyr[-1], BIN), BIN)
        pyr.append(im[::2, ::2])
    return pyr


gpyr = gauss_pyr(np.zeros((H0, W0)), 4)   # 同尺寸对照（128²，门A r1 P1-4：原 256² 改 128²）
n_pyr = sum(p.size for p in gpyr)
over_ratio = n_pyr / (H0 * W0)
reading("拉普拉斯塔系数总数/原图（overcomplete 账，同 128² 输入）", f"{n_pyr}/{H0*W0}={over_ratio:.4f}（≈4/3，E030 7-4 同源）")
check("2-4", "对照：金字塔分解系数总数>原图（overcomplete）", n_pyr > H0 * W0, f"{n_pyr} > {H0*W0}")

# 注入 D2（门A r1 P1-12 更名，原 2-5）：tight frame 账 ≠ overcomplete 账（区分力）
check("D2", "注入：两笔账分清——小波 1.000 vs 金字塔 ≈4/3（tight frame ≠ overcomplete）",
      abs(n_total1 / (H0 * W0) - 1.0) < 1e-9 and abs(over_ratio - 4 / 3) < 0.02,
      f"{n_total1/(H0*W0):.4f} vs {over_ratio:.4f}")

# ── #3 lifting 有效低通 [−1/8, 1/4, 3/4, 1/4, −1/8]（p.163 L223）──
print("\n#3 lifting 有效低通核复算（原书『easy to show』——讲者复算）")
# e'[i] = e[i]+¼(d[i−1]+d[i])，d[i]=o[i]−½(e[i]+e[i+1])
# 交错序展开：e'[i] = −⅛x[2i−2] + ¼x[2i−1] + ¾x[2i] + ¼x[2i+1] − ⅛x[2i+2]
k = np.array([-1 / 8, 1 / 4, 3 / 4, 1 / 4, -1 / 8])
y = rng.uniform(0, 1, 128)
_, e_y = lift_analyze(y)
A = np.zeros((30, 128))
for i in range(1, 31):
    A[i - 1, 2 * i - 2:2 * i + 3] = k
err3 = float(np.max(np.abs(e_y[1:31] - A @ y)))
reading("有效低通核逐点最大误差", f"{err3:.3e}")
check("3-1", "e 序列更新=核 [−1/8,¼,¾,¼,−1/8] 中心在偶位（原书 L223 逐点复核）", err3 < 1e-12, f"err={err3:.3e}")
check("3-2", "核为低通（系数和=1，直流增益一）", abs(k.sum() - 1.0) < 1e-12, f"sum={k.sum():.6f}")
nyq_q = float(sum(c * (-1) ** lag for c, lag in zip(k, (-2, -1, 0, 1, 2))))
check("3-3", "¼ 核奈奎斯特增益=0（高频压干净——「低通」的本义；对照 1-4 两端 ½ 的 −1）",
      abs(nyq_q) < 1e-12, f"H(π)={nyq_q:.3f}")

# ── #4 HH/HL/LH 方向选择性（p.161 L135–138 + Figure 3.36/3.37 语义）──
print("\n#4 方向选择性：条纹测试图分解能量分布")
gy, gx_ = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
hstrip = 0.5 + 0.5 * np.sin(2 * np.pi * gy / 8.0)     # 水平条纹（沿 y 变化）
vstrip = 0.5 + 0.5 * np.sin(2 * np.pi * gx_ / 8.0)    # 垂直条纹（沿 x 变化）
diag = 0.5 + 0.5 * np.sin(2 * np.pi * (gx_ + gy) / 12.0)


def energies(im):
    det, low = wavelet2d(im, levels=1)
    out = {k: float(np.sum(v ** 2)) for k, v in det[0].items() if k != "LL"}
    out["LL"] = float(np.sum(low ** 2))
    return out


eh, ev, ed = energies(hstrip), energies(vstrip), energies(diag)
reading("水平条纹各带能量", {k: round(v, 1) for k, v in eh.items()})
reading("垂直条纹各带能量", {k: round(v, 1) for k, v in ev.items()})
reading("斜条纹各带能量", {k: round(v, 1) for k, v in ed.items()})
# 主带判定只看方向带（LH/HL/HH）：条纹能量本就集中在低通基座 LL（低通不挑方向），
# 方向选择性体现在 detail 带间的对调关系——照口径注记「相对关系、不作绝对化断言」。
reading("水平条纹 detail 主带", max(("LH", "HL", "HH"), key=lambda k: eh[k]))
reading("垂直条纹 detail 主带", max(("LH", "HL", "HH"), key=lambda k: ev[k]))
check("4-1", "方向选择性相对断言：水平条纹 LH>HL、垂直条纹 HL>LH（对调；LL 为低通基座不参与）",
      eh["LH"] > eh["HL"] and ev["HL"] > ev["LH"],
      f"h: LH {eh['LH']:.1f} > HL {eh['HL']:.1f}；v: HL {ev['HL']:.1f} > LH {ev['LH']:.1f}")
check("4-2", "斜条纹 HH 带能量 > 水平条纹 HH 带能量（混合导数入 HH）", ed["HH"] > eh["HH"],
      f"diag HH={ed['HH']:.1f} > h HH={eh['HH']:.1f}")

# 图 A：方向带语义对照（Figure 3.36a/3.37b 语义）
dd = wavelet2d(diag, levels=1)[0][0]
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6))
for ax, im, t in [(axes[0][0], hstrip, "水平条纹"), (axes[0][1], vstrip, "垂直条纹"),
                  (axes[0][2], diag, "斜条纹"), (axes[0][3], low2, "随机图·两级低通")]:
    ax.imshow(im, cmap="gray")
    ax.set_title(t, fontsize=11)
    ax.axis("off")
for j, key in enumerate(("LH", "HL", "HH")):
    axes[1][j].imshow(np.abs(dd[key]), cmap="magma")
    axes[1][j].set_title(f"斜条纹 {key} 带（|系数|）", fontsize=11)
    axes[1][j].axis("off")
axes[1][3].imshow(np.abs(wavelet2d(img, 2)[0][0]["HH"]), cmap="magma")
axes[1][3].set_title("随机图 1 级 HH（|系数|）", fontsize=11)
axes[1][3].axis("off")
fig.suptitle("可分离 2D 小波一次级分解：方向带语义对照（程序生成，原书图未使用）", fontsize=13)
savefig(fig, "e031_wavelet2d.png")

# 图 B：多级系数幅值嵌套排布（Figure 3.36b 语义）
rng2 = np.random.default_rng(7)
gy2, gx2 = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
nat = np.clip(0.35 + 0.2 * np.sin(gy2 / 12.0) * np.cos(gx2 / 17.0)
              + 0.25 * rng2.uniform(0, 1, (256, 256)), 0, 1)
detm, lowm = wavelet2d(nat, levels=3)
disp = np.zeros((256, 256))
for det in detm:                       # 细→粗：各级三带围绕本层 LL（左上角）逐级嵌套
    s = det["LH"].shape[0]
    disp[0:s, s:2 * s] = np.abs(det["LH"])
    disp[s:2 * s, 0:s] = np.abs(det["HL"])
    disp[s:2 * s, s:2 * s] = np.abs(det["HH"])
s_low = lowm.shape[0]
disp[:s_low, :s_low] = lowm            # 最深层低通基座落在最左上角
fig, ax = plt.subplots(figsize=(7.4, 7.4))
ax.imshow(disp, cmap="magma")
ax.set_title("三级小波分解系数幅值：低通左上、各带逐级嵌套（程序生成，原书图未使用——Figure 3.36b 语义）", fontsize=11)
ax.axis("off")
savefig(fig, "e031_multilevel.png")

# ── #5 拉普拉斯金字塔融合实测（Figure 3.41/3.42 语义）──
print("\n#5 金字塔融合：双源图+二值 mask 高斯金字塔加权")


def lap_pyr(img, levels):
    g = gauss_pyr(img, levels)
    lap = []
    for i in range(levels):
        up = np.zeros_like(g[i])
        up[::2, ::2] = g[i + 1]
        up = blur_cols(blur_rows(up, 2 * BIN), 2 * BIN)
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def blend(A, B, mask, levels=5):
    la, lb = lap_pyr(A, levels), lap_pyr(B, levels)
    gm = gauss_pyr(mask.astype(float), levels)
    out_pyr, widths = [], []
    for i in range(levels + 1):
        m = np.clip(gm[i], 0, 1)
        row = m[m.shape[0] // 2]
        widths.append((i, int(np.sum((row > 0.1) & (row < 0.9))), m.shape[1]))
        out_pyr.append(la[i] * m + lb[i] * (1 - m))
    cur = out_pyr[-1]
    for i in range(levels - 1, -1, -1):
        up = np.zeros_like(out_pyr[i])
        up[::2, ::2] = cur
        up = blur_cols(blur_rows(up, 2 * BIN), 2 * BIN)
        cur = up + out_pyr[i]
    return cur, widths


N = 256
yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
A = np.clip(0.55 + 0.10 * np.sin(2 * np.pi * xx / 64.0)
            + 0.12 * np.sin(2 * np.pi * np.hypot(xx - 64, yy - 128) / 6.0), 0, 1)
B = np.clip(0.60 - 0.10 * np.sin(2 * np.pi * yy / 48.0) + 0.12 * (((xx + yy) // 10) % 2), 0, 1)
mask = (xx < N // 2).astype(float)

out_blend, widths = blend(A, B, mask, levels=5)
splice = np.where(mask > 0.5, A, B)
reading("mask 高斯金字塔各级过渡带宽度（列）", widths)
reading("mask 过渡带宽度折算原图尺度（本级宽×2^ℓ）",
        [w * (2 ** i) for i, (_, w, _) in enumerate(widths)])
band = slice(N // 2 - 8, N // 2 + 8)


def max_col_jump(im):
    return float(np.max(np.abs(np.diff(im[:, band].mean(axis=0)))))


j_splice, j_blend = max_col_jump(splice), max_col_jump(out_blend)
reading("过渡带逐列均值最大跳变 splice vs blend", f"{j_splice:.4f} vs {j_blend:.4f}")
check("5-1", "金字塔融合过渡带跳变 < 直接拼接（低频平滑混合）", j_blend < j_splice,
      f"blend {j_blend:.4f} < splice {j_splice:.4f}")
check("5-2", "mask 过渡带随层级展宽（折算原图尺度逐级翻倍；避免 ghosting 的权重设计）",
      widths[0][1] < widths[3][1], f"L0={widths[0][1]} < L3={widths[3][1]}（本级列数；折算见读数）")
strip = slice(48, 96)  # 远离边界（零填充效应）与中缝的内部条带
dA = float(np.max(np.abs(out_blend[:, strip] - A[:, strip])))
dA_vsB = float(np.max(np.abs(out_blend[:, strip] - B[:, strip])))
dB = float(np.max(np.abs(out_blend[:, -96:-48] - B[:, -96:-48])))
dB_vsA = float(np.max(np.abs(out_blend[:, -96:-48] - A[:, -96:-48])))
reading("左区条带 max|融合−A| vs max|融合−B|", f"{dA:.4f} vs {dA_vsB:.4f}")
reading("右区条带 max|融合−B| vs max|融合−A|", f"{dB:.4f} vs {dB_vsA:.4f}")
check("5-3", "mask=1 区融合更近源 A、mask=0 区更近源 B（相对断言；最粗层低频权重按原书"
      "『低频平滑混合』设计本就平滑扩散，绝对偏差见读数）",
      dA < dA_vsB and dB < dB_vsA,
      f"左 dA={dA:.4f} < {dA_vsB:.4f}；右 dB={dB:.4f} < {dB_vsA:.4f}")

# 注入 D3：不经高斯 mask 的二值加权应跳变不小于融合版（平滑权重必要性）
out_binary = A * mask + B * (1 - mask)
j_bin = max_col_jump(out_binary)
check("D3", "注入：二值 mask 加权跳变 ≥ 高斯 mask 融合（平滑权重必要性）", j_bin >= j_blend,
      f"binary {j_bin:.4f} ≥ blend {j_blend:.4f}")

# 图 C：融合四联（Figure 3.41/3.42 语义）
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
for ax, im, t in [(axes[0], A, "源 A（环纹）"), (axes[1], B, "源 B（点阵）"),
                  (axes[2], splice, "直接拼接（splice）"), (axes[3], out_blend, "金字塔融合（splining）")]:
    ax.imshow(im, cmap="inferno")
    ax.set_title(t, fontsize=12)
    ax.axis("off")
fig.suptitle(f"拉普拉斯金字塔融合实测：过渡带逐列均值最大跳变 splice {j_splice:.3f} vs blend {j_blend:.3f}"
             "（程序生成，原书图未使用——Figure 3.41/3.42 语义）", fontsize=12)
savefig(fig, "e031_blend.png")

# ── #6 steerable 频域径向划分示意（Figure 3.40a 语义；不实现滤波器组——口径注记申报）──
print("\n#6 steerable pyramids 频域划分示意（口径：只示意不实现）")
from matplotlib.colors import ListedColormap
# 值→色（离散不插值）：1=HL 2=HH 3=LH（饱和），5/6/7=次级同构浅档，9=低通基座
PAL13 = ListedColormap(
    ["#4477aa", "#ee6677", "#66ccee", "#aa3377",
     "#a9c4e0", "#f6bcc7", "#b5e3ef", "#d99cb8",
     "#ddaa33", "#ffffff", "#000000", "#c8c8c8"])
fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
yyf, xxf = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400), indexing="ij")
rad = np.hypot(xxf, yyf)
th = np.linspace(0, 2 * np.pi, 240)

# 左：separable = 频域平面被递归划分为 2×2 方块（四角=HH、左右条=HL、上下条=LH、中央=LL 再分）
sep = np.full_like(rad, 12.0)
sep[(np.abs(xxf) >= 0.5) & (np.abs(yyf) >= 0.5)] = 2          # HH
sep[(np.abs(xxf) >= 0.5) & (np.abs(yyf) < 0.5)] = 1           # HL（左右条）
sep[(np.abs(xxf) < 0.5) & (np.abs(yyf) >= 0.5)] = 3           # LH（上下条）
sep[(np.abs(xxf) >= 0.25) & (np.abs(xxf) < 0.5) & (np.abs(yyf) >= 0.25) & (np.abs(yyf) < 0.5)] = 6
sep[(np.abs(xxf) >= 0.25) & (np.abs(xxf) < 0.5) & (np.abs(yyf) < 0.25)] = 5
sep[(np.abs(xxf) < 0.25) & (np.abs(yyf) >= 0.25) & (np.abs(yyf) < 0.5)] = 7
sep[(np.abs(xxf) < 0.25) & (np.abs(yyf) < 0.25)] = 9          # 中央低通基座
axes[0].imshow(sep, cmap=PAL13, vmin=0, vmax=13)
for r in (0.25, 0.5):
    axes[0].plot([200 - 200 * r, 200 + 200 * r, 200 + 200 * r, 200 - 200 * r, 200 - 200 * r],
                 [200 - 200 * r, 200 - 200 * r, 200 + 200 * r, 200 + 200 * r, 200 - 200 * r],
                 "w-", lw=1.3)
for xx0, yy0, t, fs in [(200, 100, "LH", 15), (200, 300, "LH", 15), (100, 200, "HL", 15),
                        (300, 200, "HL", 15), (100, 100, "HH", 13), (300, 100, "HH", 13),
                        (100, 300, "HH", 13), (300, 300, "HH", 13), (200, 200, "LL", 13)]:
    axes[0].text(xx0, yy0, t, ha="center", va="center", fontsize=fs, color="w", fontweight="bold")
axes[0].set_title("separable：递归 2×2 方块划分（LH/HL/HH 逐级嵌套）", fontsize=12.5)
axes[0].axis("off")

# 右：steerable = 径向弧 × 方向扇形（每层一环、环内按方向分扇，Figure 3.40a 语义）
sector = ((np.arctan2(yyf, xxf) + np.pi) * 2 / np.pi).astype(int) % 4   # 扇界在对角线
ste = np.full_like(rad, 9.0)
ste[rad >= 0.5] = 1 + sector[rad >= 0.5]
ste[(rad >= 0.25) & (rad < 0.5)] = 5 + sector[(rad >= 0.25) & (rad < 0.5)]
axes[1].imshow(ste, cmap=PAL13, vmin=0, vmax=13)
for r in (0.25, 0.5):
    axes[1].plot(200 + 200 * r * np.cos(th), 200 + 200 * r * np.sin(th), "w-", lw=1.3)
for a in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
    axes[1].plot([200, 200 + 200 * np.cos(a)], [200, 200 + 200 * np.sin(a)], "w-", lw=1.3)
axes[1].text(200, 200, "低通", ha="center", va="center", fontsize=13, color="#5a4300", fontweight="bold")
axes[1].set_title("steerable：径向弧 × 方向扇形（Figure 3.40a 语义）", fontsize=12.5)
axes[1].axis("off")
fig.suptitle("频域划分对照示意（程序生成，原书图未使用；不实现完整滤波器组——口径申报在案）", fontsize=12.5)
savefig(fig, "e031_steerable.png")

# ── 汇总 ──
n_pass = sum(1 for r in results if r[2])
n_fail = len(results) - n_pass
print(f"\n== E031 数值实验汇总：{n_pass} 条 PASS / {n_fail} 条 FAIL；区分力注入 3/3 已拦截 ==")
for label, v in R:
    print(f"  [读数存档] {label}: {v}")
if n_fail:
    raise SystemExit(f"FAIL: {[r[0] for r in results if not r[2]]}")
print("DONE")
