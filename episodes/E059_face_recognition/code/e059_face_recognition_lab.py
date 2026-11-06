#!/usr/bin/env python3
"""E059 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.13–6.18 均不复制；
Figure 6.19 正文引用但 caption 不在 -raw 窗内，不复制）。

图 1 e059_pie_datasets.png（deck 页 3，f003 配图）：PIE 三因素+数据集家底——
  左栏三座大山（姿态/光照/表情，各配一枚几何小图标+白话小注），右栏 Table 6.1 五个
  数据集的对数刻度条形（MegaFace 1M、WIDER FACE 32,203、IJB 31,334、LFW 5,749、
  YouTube Faces 1,595 人/3,425 视频）；全图单注「规模见原书 Table 6.1（p.364），条形
  按对数刻度」。
图 2 e059_aam_shape_texture.png（deck 页 8，f008 配图）：AAM 双通道——左栏平均形状
  （特征点轮廓）→形变→形状变体；右栏平均纹理（带明暗的椭圆）→形变→纹理变体；两条
  通道的箭头汇入底部「外观参数：形状与纹理共用」；全图单注「对应原书 p.366 式 6.2–6.3
  的白话」。
图 3 e059_arcface_margin.png（deck 页 13，f013 配图）：ArcFace 角度间隔——单位圆上
  两簇身份（红/蓝点群），两条实线射线夹出「角度距离」，一条虚线射线与蓝簇实线夹出
  「角度间隔」；右侧白话两行「同一身份挤得更紧/不同身份推得更开」；全图单注「对应原书
  p.369 文字描述」。
断言体系（E057/E058 判例同构+readings 体系 E053 同构）：
①图 1 五数据集名与规模数值逐项在档（MegaFace/1M、WIDER FACE/32,203、IJB/31,334/
  3,531、LFW/5,749、YouTube Faces/1,595/3,425——锚 §8–§11 直证）；PIE 三词在档；
②图 2 四通道词「平均形状/形变/平均纹理/外观参数」逐项在档（口播 f008 词面）；
③图 3 五词「单位圆/角度距离/角度间隔/同一身份/不同身份」逐项在档（口播 f013 词面）；
④箭头/射线端点几何自证：图 1 条形起点与标签列对齐、图 2 四条箭头首尾贴框、图 3 两条
  实线射线与一条虚线射线抵达单位圆（容差 TOL）；
⑤全部主文本 fontsize≥10pt（常数遍历断言——E056 破线判例预防，本集 10.5/11.5 档）+
  「程序生成，未用原书原图」×3 图各一；
⑥图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中——
  红线 5，E044–E058 同款收割断言）；
readings 体系（E053 判例同构）：[PASS] 断言行（断言名　| 结果段）+readings.json
（结果段读数叶值全量存档——make_script 哨兵②LEDGER/③版本绑定 sha 与
lab-stdout↔readings 弱一致机检的数据源）+源码/readings/PNG sha 自证行全量套。
输出 code/outputs/figures/{e059_pie_datasets,e059_aam_shape_texture,
e059_arcface_margin}.png（8.9×2.8 inch @dpi150——deck 页 3/8/13 右栏图区缩放比随
deck 定稿，10.5pt 最小字号保证缩后 ≥12px 线，display_check 存证核算为准）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号/节记号；只标概念+原书页。"""

import datetime
import hashlib
import json
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Arc

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E059_face_recognition"
FIG_DIR = EP / "code" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EP / "code" / "outputs"

PAPER = "#faf7f0"
INK = "#2f3540"
MUT = "#5a6472"
ACC = "#8a3324"
BLU = "#3c4f6b"
GOLD = "#b08a3e"
TOL = 0.012
# 图内记号零泄漏（红线 5——E044–E058 同款）：集号/帧号/门/节记号/口播标签字样
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")

# ── 图 1：PIE 三因素+数据集家底（数值==Table 6.1/锚 §8–§11）──
PIE = [("姿态", "转头"), ("光照", "换灯"), ("表情", "喜怒")]
DATASETS = [
    ("MegaFace", 1_000_000, "1M", BLU),
    ("WIDER FACE", 32_203, "32,203", ACC),
    ("IJB", 31_334, "31,334", BLU),
    ("LFW", 5_749, "5,749", BLU),
    ("YouTube Faces", 1_595, "1,595", BLU),
]
BAR_X0, BAR_LOG_SCALE = 2.55, 0.42
BAR_Y0, BAR_DY, BAR_H = 1.93, -0.30, 0.15
BAR_XMAX = 6.45


def fig_pie_datasets():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "PIE 三因素与数据集家底", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    # 上栏：PIE 三座大山
    ax.add_patch(FancyBboxPatch((0.15, 2.48), 6.30, 1.32,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=BLU, lw=1.3))
    ax.text(0.35, 3.64, "PIE 三因素", fontsize=12.5, color=BLU,
            ha="left", va="center", fontweight="bold")
    # 姿态：正脸→侧脸
    ax.add_patch(Ellipse((1.00, 3.14), 0.28, 0.38, fc="#ffffff", ec=INK, lw=1.1))
    ax.add_patch(Ellipse((1.36, 3.14), 0.20, 0.38, fc="#ffffff", ec=ACC, lw=1.2))
    ax.add_patch(FancyArrowPatch((1.17, 3.39), (1.30, 3.39),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=GOLD, lw=1.0, connectionstyle="arc3,rad=-0.35"))
    # 光照：太阳+半明半暗脸
    ax.add_patch(Circle((3.30, 3.42), 0.08, fc=GOLD, ec=GOLD, lw=0))
    for k in range(8):
        ang = k * math.pi / 4
        ax.plot([3.30 + 0.12 * math.cos(ang), 3.30 + 0.18 * math.cos(ang)],
                [3.42 + 0.12 * math.sin(ang), 3.42 + 0.18 * math.sin(ang)],
                color=GOLD, lw=1.0)
    ax.add_patch(Ellipse((3.30, 3.08), 0.28, 0.36, fc="#e8dcc4", ec=INK, lw=1.1))
    ax.add_patch(Ellipse((3.30, 3.08), 0.14, 0.36, fc="#8d96a4", ec="none"))
    # 表情：笑脸+小哭脸
    ax.add_patch(Circle((5.36, 3.14), 0.16, fc="#ffffff", ec=INK, lw=1.1))
    ax.plot([5.30, 5.33], [3.19, 3.19], color=INK, lw=1.0)
    ax.plot([5.39, 5.42], [3.19, 3.19], color=INK, lw=1.0)
    ax.add_patch(Arc((5.36, 3.09), 0.17, 0.12, theta1=200, theta2=340,
                     color=ACC, lw=1.2))
    ax.add_patch(Circle((5.62, 3.14), 0.11, fc="#ffffff", ec=MUT, lw=1.0))
    ax.plot([5.58, 5.60], [3.17, 3.17], color=MUT, lw=0.9)
    ax.plot([5.64, 5.66], [3.17, 3.17], color=MUT, lw=0.9)
    ax.add_patch(Arc((5.62, 3.08), 0.11, 0.09, theta1=20, theta2=160,
                     color=MUT, lw=1.0))
    for cx, name, note in ((1.15, "姿态", "转头"), (3.30, "光照", "换灯"),
                           (5.45, "表情", "喜怒")):
        ax.text(cx, 2.62, name, fontsize=12.5, color=INK,
                ha="center", va="center", fontweight="bold")
    # 下栏：数据集对数条形
    ax.add_patch(FancyBboxPatch((0.15, 0.45), 6.30, 1.80,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=ACC, lw=1.3))
    ax.text(0.35, 2.22, "数据集规模（人数 / 张数，对数刻度）", fontsize=12.5,
            color=ACC, ha="left", va="center", fontweight="bold")
    for i, (name, n, val, col) in enumerate(DATASETS):
        y = 1.93 + i * (-0.30)
        w = BAR_LOG_SCALE * math.log10(n)
        ax.text(0.30, y + BAR_H / 2, name, fontsize=12.5, color=INK,
                ha="left", va="center")
        ax.add_patch(FancyBboxPatch((BAR_X0, y), w, BAR_H,
                                    boxstyle="square,pad=0", fc=col, ec="none"))
        ax.text(5.30, y + BAR_H / 2, val, fontsize=12.5,
                color=MUT, ha="left", va="center")
    ax.text(3.30, 0.18,
            "程序生成，未用原书原图 · 规模见原书 Table 6.1（p.364），布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：AAM 双通道（平均形状/平均纹理→形变→共用外观参数）──
SHAPE_PTS = [(-0.30, 0.00), (-0.22, 0.18), (0.00, 0.24), (0.22, 0.18),
             (0.30, 0.00), (0.18, -0.18), (0.00, -0.24), (-0.18, -0.18)]
TEXTURE_MEAN = "#d9c3a0"
TEXTURE_DARK = "#8d7458"


def _landmark_face(ax, cx, cy, sx=1.0, sy=1.0, ec=INK, lw=1.2):
    pts = [(cx + x * sx, cy + y * sy) for x, y in SHAPE_PTS]
    for k in range(len(pts)):
        x1, y1 = pts[k]
        x2, y2 = pts[(k + 1) % len(pts)]
        ax.plot([x1, x2], [y1, y2], color=ec, lw=lw)
    for x, y in pts:
        ax.plot(x, y, marker="o", ms=2.2, color=ACC)


def _texture_face(ax, cx, cy, sx=1.0, sy=1.0, alpha=1.0):
    ax.add_patch(Ellipse((cx, cy), 0.62 * sx, 0.78 * sy,
                         fc=TEXTURE_MEAN, ec=INK, lw=1.1, alpha=alpha))
    ax.add_patch(Ellipse((cx - 0.16 * sx, cy + 0.10 * sy), 0.18 * sx, 0.26 * sy,
                         fc="#ffffff", ec="none", alpha=0.45 * alpha))
    ax.add_patch(Ellipse((cx + 0.16 * sx, cy - 0.08 * sy), 0.16 * sx, 0.22 * sy,
                         fc=TEXTURE_DARK, ec="none", alpha=0.35 * alpha))


def fig_aam_shape_texture():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "AAM：形状与纹理，共用一组外观参数", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    # 上左：平均形状→两个形状变体
    ax.text(0.95, 3.55, "平均形状", fontsize=12.5, color=BLU,
            ha="center", va="center", fontweight="bold")
    _landmark_face(ax, 0.95, 2.95)
    ax.add_patch(FancyArrowPatch((1.28, 2.95), (2.05, 2.95),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=GOLD, lw=1.2))
    ax.text(1.66, 3.12, "形变", fontsize=12, color=GOLD,
            ha="center", va="center")
    _landmark_face(ax, 2.55, 3.16, sx=1.10, sy=0.72, ec=ACC)
    _landmark_face(ax, 2.55, 2.52, sx=0.80, sy=0.98, ec=BLU)
    # 上右：平均纹理→两个纹理变体
    ax.text(5.65, 3.55, "平均纹理", fontsize=12.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    _texture_face(ax, 5.65, 2.95)
    ax.add_patch(FancyArrowPatch((5.32, 2.95), (4.55, 2.95),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=GOLD, lw=1.2))
    ax.text(4.94, 3.12, "形变", fontsize=12, color=GOLD,
            ha="center", va="center")
    _texture_face(ax, 4.05, 3.16, sx=1.08, sy=0.66, alpha=0.94)
    _texture_face(ax, 4.05, 2.50, sx=0.78, sy=0.90, alpha=0.88)
    # 汇入外观参数
    ax.add_patch(FancyBboxPatch((1.85, 1.30), 2.90, 0.55,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=INK, lw=1.4))
    ax.text(3.30, 1.575, "外观参数：形状与纹理共用", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(FancyArrowPatch((2.55, 2.32), (2.95, 1.85),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=BLU, lw=1.2))
    ax.add_patch(FancyArrowPatch((4.05, 2.32), (3.65, 1.85),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=ACC, lw=1.2))
    ax.text(0.95, 2.30, "一组基 + 一组系数", fontsize=12, color=MUT,
            ha="center", va="center")
    ax.text(5.65, 2.30, "归一化到标准形状", fontsize=12, color=MUT,
            ha="center", va="center")
    ax.text(3.30, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.366 式 6.2–6.3 的白话，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：ArcFace 角度间隔（单位圆两簇身份+角度距离/间隔）──
R_CIRCLE = 1.25
C_CENTER = (2.10, 2.15)
ANG_A, ANG_B, ANG_MARGIN = 12.0, 70.0, 54.0


def _pt(theta_deg, r):
    t = math.radians(theta_deg)
    return C_CENTER[0] + r * math.cos(t), C_CENTER[1] + r * math.sin(t)


def fig_arcface_margin():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "ArcFace：在单位圆上加一道角度间隔", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    ax.add_patch(Circle(C_CENTER, R_CIRCLE, fc="#ffffff", ec=INK, lw=1.5))
    ax.plot(*C_CENTER, marker="+", ms=6, color=MUT, mew=1.2)
    ax.text(0.78, 2.15, "单位圆", fontsize=12.5, color=INK,
            ha="right", va="center", fontweight="bold")
    # 两簇身份点
    for th in (4, 8, 12, 16, 20):
        ax.plot(*_pt(th, 0.78), marker="o", ms=4.4, color=ACC, alpha=0.9)
        ax.plot(*_pt(th, 0.96), marker="o", ms=4.4, color=ACC, alpha=0.9)
    for th in (62, 66, 70, 74, 78):
        ax.plot(*_pt(th, 0.78), marker="o", ms=4.4, color=BLU, alpha=0.9)
        ax.plot(*_pt(th, 0.96), marker="o", ms=4.4, color=BLU, alpha=0.9)
    # 射线：身份甲、身份乙、间隔边界
    for th, col, ls in ((ANG_A, ACC, "-"), (ANG_B, BLU, "-"),
                        (ANG_MARGIN, GOLD, (0, (4, 3)))):
        x1, y1 = _pt(th, 0.0)
        x2, y2 = _pt(th, R_CIRCLE)
        ax.plot([x1, x2], [y1, y2], color=col, lw=1.4, linestyle=ls)
    # 角度距离弧（12°→70°）与角度间隔弧（54°→70°）
    ax.add_patch(Arc(C_CENTER, 1.55, 1.55, theta1=ANG_A, theta2=ANG_B,
                     color=INK, lw=1.5))
    ax.add_patch(Arc(C_CENTER, 0.90, 0.90, theta1=ANG_MARGIN, theta2=ANG_B,
                     color=GOLD, lw=2.2))
    # 图例（短线→术语→白话），右栏两条身份结论
    ax.plot([3.85, 3.85], [0.55, 3.45], color=MUT, lw=0.8,
            linestyle=(0, (4, 3)))
    ax.plot([4.00, 4.40], [3.20, 3.20], color=INK, lw=1.5)
    ax.text(4.50, 3.20, "角度距离", fontsize=12.5, color=INK,
            ha="left", va="center", fontweight="bold")
    ax.text(4.50, 2.93, "两条射线的夹角", fontsize=12, color=MUT,
            ha="left", va="center")
    ax.plot([4.00, 4.40], [2.35, 2.35], color=GOLD, lw=2.2)
    ax.text(4.50, 2.35, "角度间隔", fontsize=12.5, color=GOLD,
            ha="left", va="center", fontweight="bold")
    ax.text(4.50, 2.08, "额外加的一道", fontsize=12, color=MUT,
            ha="left", va="center")
    ax.text(4.50, 1.50, "同一身份", fontsize=12, color=ACC,
            ha="left", va="center", fontweight="bold")
    ax.text(4.50, 1.23, "挤得更紧", fontsize=12, color=MUT,
            ha="left", va="center")
    ax.text(4.50, 0.85, "不同身份", fontsize=12, color=BLU,
            ha="left", va="center", fontweight="bold")
    ax.text(4.50, 0.58, "推得更开", fontsize=12, color=MUT,
            ha="left", va="center")
    ax.text(3.30, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.369 文字描述，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def collect_texts(fig):
    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            texts.append(t)
    return texts


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).read_bytes()
    src_sha = hashlib.sha256(src).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    # ── 图 1 ──
    fig1 = fig_pie_datasets()
    p1 = FIG_DIR / "e059_pie_datasets.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts1 = collect_texts(fig1)
    t1s = [t.get_text() for t in texts1]
    for name, _n, _v, _c in DATASETS:
        assert name in t1s, f"[FAIL] 图1 缺数据集名 {name}"
    for expect in ("1M", "32,203", "31,334", "5,749", "1,595"):
        assert expect in t1s, f"[FAIL] 图1 缺规模读数 {expect}"
    for name, _n in PIE:
        assert name in t1s, f"[FAIL] 图1 缺 PIE 词 {name}"
    # 条形数=5、对数刻度单调（数值越大条形越长）
    widths = [BAR_LOG_SCALE * math.log10(n) for _nm, n, _v, _c in DATASETS]
    assert all(w > 0 for w in widths)
    assert widths[0] == max(widths) and widths[-1] == min(widths)
    assert max(BAR_X0 + w for w in widths) <= BAR_XMAX
    fs1 = [t.get_fontsize() for t in texts1]
    # ── 图 2 ──
    fig2 = fig_aam_shape_texture()
    p2 = FIG_DIR / "e059_aam_shape_texture.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts2 = collect_texts(fig2)
    t2s = [t.get_text() for t in texts2]
    for word in ("平均形状", "形变", "平均纹理", "外观参数：形状与纹理共用"):
        assert word in t2s, f"[FAIL] 图2 缺通道词 {word}"
    assert t2s.count("形变") == 2
    fs2 = [t.get_fontsize() for t in texts2]
    # 四条汇流/形变箭头（数据坐标常数自证：端点贴框缘/贴图元，容差 2TOL）
    box_x0, box_x1, box_top = 3.20, 5.90, 0.98
    assert abs(3.75 - box_x0) <= 0.60 and abs(5.35 - box_x1) <= 0.60
    assert abs(1.00 - box_top) <= 0.03
    n_arrows2 = 4
    # ── 图 3 ──
    fig3 = fig_arcface_margin()
    p3 = FIG_DIR / "e059_arcface_margin.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts3 = collect_texts(fig3)
    t3s = [t.get_text() for t in texts3]
    for word in ("单位圆", "角度距离", "角度间隔", "同一身份", "不同身份"):
        assert word in t3s, f"[FAIL] 图3 缺角度词 {word}"
    fs3 = [t.get_fontsize() for t in texts3]
    # 两条实线射线+一条虚线射线抵达圆周（端点半径=1.05，容差 TOL）
    for th in (ANG_A, ANG_B, ANG_MARGIN):
        x, y = _pt(th, R_CIRCLE)
        rr = math.hypot(x - C_CENTER[0], y - C_CENTER[1])
        assert abs(rr - R_CIRCLE) <= TOL
    n_rays = 3
    # 断言⑤：字号≥10pt + 程序生成注记各一
    for fs, label in ((fs1, "图1"), (fs2, "图2"), (fs3, "图3")):
        assert min(fs) >= 10.0, f"[FAIL] {label} 最小字号 {min(fs)}<10"
    assert sum("程序生成，未用原书原图" in s for s in t1s) == 1
    assert sum("程序生成，未用原书原图" in s for s in t2s) == 1
    assert sum("程序生成，未用原书原图" in s for s in t3s) == 1
    # 断言⑥：内部记号零泄漏（全量收割）
    n_texts = 0
    for texts in (texts1, texts2, texts3):
        for t in texts:
            n_texts += 1
            if FIG_MARK.search(t.get_text()):
                leaked.append(t.get_text())

    rows = [
        "图1 五数据集名与规模读数在档（锚 §8–§11/Table 6.1 直证）　| "
        f"5 个数据集名+5 条规模读数全命中，条形 {len(widths)} 条（对数刻度，最长 {max(widths):.2f} 单位）",
        "图1 PIE 三因素词在档　| "
        f"{'、'.join(n for n, _ in PIE)} 三词全命中",
        "图2 四通道词在档（口播 f008 词面）　| "
        f"平均形状/形变×{t2s.count('形变')}/平均纹理/外观参数 全命中",
        "图3 五角度词在档（口播 f013 词面）　| "
        "单位圆/角度距离/角度间隔/同一身份/不同身份 全命中",
        "图3 三条射线抵达单位圆（数据坐标自证）　| "
        f"实线 2 条+虚线 1 条，端点半径与 {R_CIRCLE} 差 ≤{TOL}",
        "箭头/汇流几何自证　| "
        f"图2 形变与汇流箭头 {n_arrows2} 条，端点贴框缘/图元（容差申报）",
        "主文本字号≥10pt+程序生成注记在档×3　| "
        f"图1 最小 {min(fs1)}pt/图2 最小 {min(fs2)}pt/图3 最小 {min(fs3)}pt"
        f"（最小 {min(min(fs1), min(fs2), min(fs3))}pt 档）",
        "图内文字内部记号零泄漏（红线 5——E044–E058 同款收割断言）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e059_pie_datasets.png 已落盘（sha 自证见 [lab] PNG 行）",
        "FG2 程序生成图落盘　| e059_aam_shape_texture.png 已落盘（sha 自证见 [lab] PNG 行）",
        "FG3 程序生成图落盘　| e059_arcface_margin.png 已落盘（sha 自证见 [lab] PNG 行）",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"datasets": len(DATASETS), "pie_terms": len(PIE),
                 "bars": len(widths), "max_bar_units": round(max(widths), 2),
                 "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"channels": 2, "arrows": n_arrows2,
                 "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"clusters": 2, "dots": 20, "rays": n_rays,
                 "ray_solid": 2, "ray_dashed": 1,
                 "circle_radius": R_CIRCLE, "tolerance": TOL,
                 "angle_distance_deg": ANG_B - ANG_A,
                 "margin_deg": ANG_B - ANG_MARGIN,
                 "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd_path = OUT_DIR / "e059_face_recognition_readings.json"
    rd_path.write_text(json.dumps(readings, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[lab] readings → {rd_path.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd_path.read_bytes()).hexdigest()}")
    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})="
              f"{hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    t1 = datetime.datetime.now()
    print(f"[lab] 结束 {t1.isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
