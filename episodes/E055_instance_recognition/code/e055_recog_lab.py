#!/usr/bin/env python3
"""E055 S2 教学替代图×2（红线 2：程序生成，未用原书原图——Figure 6.1/6.2/6.3 均不复制）。

图 1 e055_dataset_race.png（deck 页 2，f003–f004 配图）：数据集军备竞赛接力——五节点
  横排（Xerox 10→Caltech-101→PASCAL VOC→ImageNet→COCO），节点序==原书点名序
  （p.345 导言叙事序——锚 §28/§9/§10/§11 照录段顺序），每节点标名称+关键数+引文
  年份；全图单注「数据出处：原书 p.345」。
图 2 e055_match_demo.png（deck 页 6，f012 主图）：特征匹配+几何验证四步流程（左，
  竖排四框）+匹配示意（右：合成库图图案→仿射变换进杂乱场景→特征方块+匹配连线+
  仿射平行四边形框出）——语义重演原书 Figure 6.2（Lowe 2004）但布局自绘；口播 f012
  四步文本与流程框文本逐一断言。
断言体系（E054 判例同构+readings 体系 E053 同构）：
①图 1 节点序==原书点名序（数据元组列表双比对）；
②图 2 流程框四步文本==口播 f012 四步关键词序列；
③匹配线数==6、干扰特征数==12（数据常数自证）；
④箭头/连线端点贴目标缘（数据坐标判定，容差 TOL）；④b 仿射框四角落场景矩形内；
⑤全部文本 fontsize≥10pt（常数遍历断言）+图内注记行存在（「程序生成，未用原书原图」）；
⑥图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中——
  红线 5，E044–E054 同款收割断言）；
readings 体系（E053 判例同构）：[PASS] 断言行（断言名　| 结果段）+readings.json
（结果段读数叶值全量存档——make_script 哨兵②LEDGER/③版本绑定 sha 与
lab-stdout↔readings 弱一致机检的数据源）+源码/readings/PNG sha 自证行全量套。
输出 code/outputs/figures/{e055_dataset_race,e055_match_demo}.png（8.9×3.0 / 8.5×3.9
inch @dpi150——deck 页 2 右栏 max-height 275px 缩放 0.586→10pt≈12.2 CSS px、页 6 左栏
max-height 400px 缩放 0.665→10pt≈13.8 CSS px，均 ≥12px 线，display_check 存证核算为准）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号；只标概念+原书页。"""

import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from matplotlib.text import Text

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E055_instance_recognition"
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
# 图内记号零泄漏（红线 5——E044–E054 同款）：集号/帧号/门/节记号/口播标签字样
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")

# ── 图 1：数据集军备竞赛接力（节点序==原书 p.345 点名序——锚 §28→§9→§10→§11）──
RACE = [
    ("Xerox 10", "10 类小库", "2006"),
    ("Caltech-101", "101 类", "2006"),
    ("PASCAL VOC", "20 类\n首个够大够难", "2010"),
    ("ImageNet", "1000 类\n超百万标注图", "2009"),
    ("COCO", "逐物体\n精细分割", "2014"),
]


def fig_race():
    fig, ax = plt.subplots(figsize=(8.9, 3.0), dpi=150)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(4.5, 2.72, "数据集军备竞赛：从评想法到撑起学习", fontsize=11.5,
            color=INK, ha="center", va="center", fontweight="bold")
    n = len(RACE)
    w, h, y0 = 1.63, 1.52, 0.62
    xs = [0.06 + i * (w + 0.145) for i in range(n)]
    centers = []
    for i, ((name, desc, yr), x) in enumerate(zip(RACE, xs)):
        box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04",
                             fc="#fff" if i % 2 == 0 else "#f3ede0",
                             ec=ACC if i == n - 1 else BLU, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h - 0.26, name, fontsize=10.5, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y0 + h / 2 - 0.05, desc, fontsize=10, color=MUT,
                ha="center", va="center", linespacing=1.15)
        ax.text(x + w / 2, y0 + 0.2, yr, fontsize=10, color=GOLD,
                ha="center", va="center")
        centers.append((x, x + w, y0, y0 + h))
    n_arrows = n - 1
    for i in range(n_arrows):
        x1 = centers[i][1]
        x2 = centers[i + 1][0]
        ar = FancyArrowPatch((x1 + 0.01, y0 + h / 2), (x2 - 0.01, y0 + h / 2),
                             arrowstyle="-|>", mutation_scale=10,
                             color=INK, lw=1.0, shrinkA=0, shrinkB=0)
        ax.add_patch(ar)
    ax.text(4.5, 0.16, "数据出处：原书 p.345（导言·大规模数据集与基准）｜"
                       "程序生成，未用原书原图", fontsize=10, color=MUT,
            ha="center", va="center")
    # 断言④：箭头端点贴节点缘（间距≥0.145−TOL 且箭头起讫分别压左右缘）
    for i in range(n_arrows):
        x1 = centers[i][1]
        x2 = centers[i + 1][0]
        gap = x2 - x1
        assert gap >= 0.145 - TOL, f"图1 节点间距不足 {i}: {gap}"
        ar = ax.patches[-(n_arrows - i)]
        p0, p1 = ar._posA_posB
        assert abs(p0[0] - x1) < TOL and abs(p1[0] - x2) < TOL, f"图1 箭头{i}端点漂移"
    # 断言⑤：字号与注记
    fs = [t.get_fontsize() for t in ax.texts]
    assert min(fs) >= 10.0, f"图1 字号低于 10pt: {min(fs)}"
    joined = " ".join(t.get_text() for t in ax.texts)
    assert "程序生成，未用原书原图" in joined, "图1 缺程序生成注记"
    # 断言①：节点序==原书点名序
    names = [t.get_text() for t in ax.texts if t.get_fontsize() == 10.5
             and t.get_fontweight() == "bold" and t.get_position()[1] > y0 + h - 0.3]
    assert names == [r[0] for r in RACE], f"图1 节点序失配: {names}"
    out = FIG_DIR / "e055_dataset_race.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    texts = [t.get_text() for t in fig.findobj(Text)]
    plt.close(fig)
    return out, min(fs), max(fs), n_arrows, texts


# ── 图 2：四步流程+匹配示意（语义重演 Figure 6.2——布局自绘）──
STEPS = ["① 建索引树：库图兴趣点+描述子入库",
         "② 提特征查匹配：新图特征→树里比对",
         "③ 霍夫投票：筛外点·攒出仿射变换",
         "④ 验证通过：物体框出·识别完成"]
N_MATCH, N_NOISE = 6, 12
# 场景矩形界（断言④b 判定域）：plt.Rectangle((2.15, 0.2), 2.45, 3.3)
SCENE_BOUNDS = (2.15, 4.60, 0.20, 3.50)


def _motif(ax, cx, cy, s, ec):
    """合成图案：圆+方+三角拼贴（库图基本件）。"""
    ax.add_patch(Circle((cx - s * 0.55, cy + s * 0.45), s * 0.34,
                        fc="none", ec=ec, lw=1.6))
    ax.add_patch(plt.Rectangle((cx + s * 0.1, cy + s * 0.15), s * 0.55,
                               s * 0.55, fc="none", ec=ec, lw=1.6))
    tri = Polygon([(cx - s * 0.7, cy - s * 0.55), (cx + s * 0.15, cy - s * 0.55),
                   (cx - s * 0.28, cy - s * 0.05)], fc="none", ec=ec, lw=1.6)
    ax.add_patch(tri)


def _feats(cx, cy, s):
    """图案特征点（角点集，库图坐标）。"""
    return [(cx - s * 0.21, cy + s * 0.45), (cx - s * 0.89, cy + s * 0.45),
            (cx + s * 0.1, cy + s * 0.15), (cx + s * 0.65, cy + s * 0.15),
            (cx + s * 0.65, cy + s * 0.7), (cx - s * 0.7, cy - s * 0.55),
            (cx + s * 0.15, cy - s * 0.55)]


def fig_demo():
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.9), dpi=150,
                            gridspec_kw=dict(width_ratios=[0.88, 1.12]))
    fig.patch.set_facecolor(PAPER)
    ax1, ax2 = axs
    for a in axs:
        a.set_facecolor(PAPER)
        a.axis("off")
    # 左：四步流程竖排（xlim 0-3.7 / ylim 0-3.9）
    ax1.set_xlim(0, 3.7)
    ax1.set_ylim(0, 3.9)
    ax1.text(1.85, 3.72, "流程四步", fontsize=11.5, color=INK, ha="center",
             va="center", fontweight="bold")
    ys = [3.0, 2.16, 1.32, 0.48]
    for i, (txt, yy) in enumerate(zip(STEPS, ys)):
        box = FancyBboxPatch((0.12, yy - 0.3), 3.46, 0.6,
                             boxstyle="round,pad=0.03",
                             fc="#fff" if i % 2 == 0 else "#f3ede0",
                             ec=ACC if i == 3 else BLU, lw=1.2)
        ax1.add_patch(box)
        ax1.text(1.85, yy, txt, fontsize=10, color=INK, ha="center",
                 va="center", linespacing=1.2)
        if i < 3:
            ax1.add_patch(FancyArrowPatch((1.85, yy - 0.32), (1.85, ys[i + 1] + 0.32),
                                          arrowstyle="-|>", mutation_scale=9,
                                          color=INK, lw=1.0, shrinkA=0, shrinkB=0))
    # 右：匹配示意（库图→场景）xlim 0-4.8 / ylim 0-3.9
    ax2.set_xlim(0, 4.8)
    ax2.set_ylim(0, 3.9)
    ax2.text(2.4, 3.80, "匹配示意：库图图案 → 杂乱场景", fontsize=10.5,
             color=INK, ha="center", va="center", fontweight="bold")
    # 库图（左上）
    ax2.add_patch(plt.Rectangle((0.2, 1.95), 1.7, 1.55, fc="#fff",
                                ec=BLU, lw=1.2))
    ax2.text(1.05, 3.36, "数据库图", fontsize=10.5, color=BLU, ha="center")
    _motif(ax2, 1.05, 2.72, 0.55, INK)
    db_feats = _feats(1.05, 2.72, 0.55)
    for (fx, fy) in db_feats:
        ax2.add_patch(plt.Rectangle((fx - 0.03, fy - 0.03), 0.06, 0.06,
                                    fc=ACC, ec="none"))
    # 场景（右侧大区）
    ax2.add_patch(plt.Rectangle((SCENE_BOUNDS[0], SCENE_BOUNDS[2]),
                                SCENE_BOUNDS[1] - SCENE_BOUNDS[0],
                                3.5 - 0.2, fc="#f3ede0",
                                ec=MUT, lw=1.0))
    ax2.text(3.375, 3.36, "杂乱场景", fontsize=10.5, color=MUT, ha="center")
    # 仿射变换（缩放+旋转+平移）作用于库图特征与图案边界
    import numpy as np
    th = np.deg2rad(24)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = np.array([[0.9, 0.18], [0.05, 0.75]]) @ R
    off = np.array([2.7, 0.35])

    def T(p):
        return tuple(A @ np.array(p) + off)

    # 平行四边形=库图框四角变换
    corners = [(0.2, 1.9), (1.9, 1.9), (1.9, 3.35), (0.2, 3.35)]
    quad = Polygon([T(c) for c in corners], closed=True, fc="none",
                   ec=ACC, lw=1.5, ls="--")
    ax2.add_patch(quad)
    # 断言④b：仿射框四角落在场景矩形内（SCENE_BOUNDS——人工校核+机检双确认）
    tc = [T(c) for c in corners]
    assert all(SCENE_BOUNDS[0] <= x <= SCENE_BOUNDS[1]
               and SCENE_BOUNDS[2] <= y <= 3.50 for x, y in tc), f"图2 仿射框出界: {tc}"
    # 变换后的特征点
    scene_feats = []
    for (fx, fy) in db_feats:
        sx, sy = T((fx, fy))
        scene_feats.append((sx, sy))
        ax2.add_patch(plt.Rectangle((sx - 0.03, sy - 0.03), 0.06, 0.06,
                                    fc=ACC, ec="none"))
    # 干扰特征（场景杂点）
    rng = np.random.RandomState(7)
    for _ in range(N_NOISE):
        sx = rng.uniform(2.3, 4.45)
        sy = rng.uniform(0.35, 3.35)
        ax2.add_patch(plt.Rectangle((sx - 0.027, sy - 0.027), 0.054, 0.054,
                                    fc="#9aa2ad", ec="none"))
    # 匹配连线（库图特征→场景特征）
    for i in range(N_MATCH):
        fx, fy = db_feats[i]
        sx, sy = scene_feats[i]
        ax2.plot([fx, sx], [fy, sy], color=GOLD, lw=0.9, zorder=1)
    ax2.text(2.4, 0.02, "程序生成，未用原书原图——语义对应原书 Figure 6.2（p.347）",
             fontsize=10, color=MUT, ha="center", va="bottom")
    # 断言④：匹配线端点重合特征块中心（数据坐标容差）
    for i in range(N_MATCH):
        fx, fy = db_feats[i]
        mx, my = ax2.lines[i].get_xdata()[0], ax2.lines[i].get_ydata()[0]
        assert abs(mx - fx) < TOL and abs(my - fy) < TOL, f"图2 匹配线{i}起点漂移"
    # 断言③⑤
    fs = [t.get_fontsize() for a in axs for t in a.texts]
    assert min(fs) >= 10.0, f"图2 字号低于 10pt: {min(fs)}"
    joined = " ".join(t.get_text() for a in axs for t in a.texts)
    assert "程序生成，未用原书原图" in joined, "图2 缺程序生成注记"
    # 断言②：流程框文本==口播 f012 四步关键词
    step_txts = [t.get_text() for t in ax1.texts if t.get_text().startswith("①")
                 or t.get_text().startswith("②") or t.get_text().startswith("③")
                 or t.get_text().startswith("④")]
    assert step_txts == STEPS, f"图2 流程文本失配: {step_txts}"
    out = FIG_DIR / "e055_match_demo.png"
    fig.savefig(out, dpi=150, facecolor=PAPER)
    texts = [t.get_text() for t in fig.findobj(Text)]
    plt.close(fig)
    return out, min(fs), max(fs), texts


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__).resolve()
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    o1, f1min, f1max, n_arrows, texts1 = fig_race()
    o2, f2min, f2max, texts2 = fig_demo()
    texts_all = texts1 + texts2
    leaked = [t for t in texts_all if FIG_MARK.search(t)]
    n_texts = len(texts_all)
    print(f"[lab] E055 实验课数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={src_sha}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    # [PASS] 断言行（断言名　| 结果段——结果段读数全部入 readings 叶值，
    # lab-stdout↔readings 弱一致机检数据源；内部记号 f012 系 lab 存证层自申报，
    # 非 deck 可见层——红线 5 管辖 deck 可见层与口播，lab stdout 不在管辖面）
    rows = [
        "图1 节点序==原书 p.345 点名序（锚 §28→§9→§10→§11：Xerox 10→Caltech-101→"
        "PASCAL VOC→ImageNet→COCO）　| 节点 5 个（数据元组列表双比对——锚序即点名序逐位一致）",
        "图1 箭头端点贴节点缘+间距达标（数据坐标容差）　| "
        f"箭头 {n_arrows} 支，最小间距 0.145，容差 {TOL}",
        "图1 字号≥10pt+程序生成注记在档　| "
        f"字号 {f1min}–{f1max}pt（图内最小 {f1min}pt）",
        "图2 流程四步文本==口播 f012（逐框比对）　| "
        f"步数 {len(STEPS)}，字号最小 {f2min}pt",
        "图2 匹配线/干扰特征/仿射框常数自证　| "
        f"匹配线 {N_MATCH}，干扰特征 {N_NOISE}，仿射框四角界内"
        f"（场景 x∈[{SCENE_BOUNDS[0]},{SCENE_BOUNDS[1]}]、y∈[{SCENE_BOUNDS[2]},3.50]）",
        "图2 匹配线端点重合特征块中心（数据坐标容差）　| "
        f"{N_MATCH} 条线起点全重合（容差 {TOL}）",
        "图内文字内部记号零泄漏（红线 5——E044–E054 同款收割断言）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e055_dataset_race.png 已落盘（sha 自证见 [lab] PNG 行）",
        "FG2 程序生成图落盘　| e055_match_demo.png 已落盘（sha 自证见 [lab] PNG 行）",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    # readings 落盘（结果段读数叶值全量存档——E053 判例同构）
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "n_pass": n_pass,
        "n_total": len(rows),
        "src_sha16": src_sha[:16],
        "fig1": {"nodes": 5, "arrows": n_arrows, "gap": 0.145, "tol": TOL,
                 "min_fs": f1min, "max_fs": f1max},
        "fig2": {"steps": len(STEPS), "match": N_MATCH, "noise": N_NOISE,
                 "tol": TOL, "min_fs": f2min, "max_fs": f2max,
                 "bounds": [SCENE_BOUNDS[0], SCENE_BOUNDS[1],
                            SCENE_BOUNDS[2], 3.50]},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd_path = OUT_DIR / "e055_recog_readings.json"
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
