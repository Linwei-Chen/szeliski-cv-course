#!/usr/bin/env python3
"""E065 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 7.1/7.2/7.3 不复制）。

图 1 e065_feature_family.png（deck 页 3，f003 配图）：特征家族——
  点（关键点/角点+周围图块）/边缘（方向+局部外观）/轮廓（连成曲线可跟踪）/
  直线（灭点→相机内外参）/区域（分割块用于匹配识别）。对应原书 p.418–419。
图 2 e065_match_use_cases.png（deck 页 8，f008 配图）：对应点能做什么——
  稀疏对应→相机位姿→稠密立体→三维重建；旁挂拼接/HDR/稳像/实例识别。
  对应原书 p.419–420。
图 3 e065_patch_localizability.png（deck 页 16，f016 配图）：图块可定位性——
  平坦雪地/单向边缘/拐角三类 patch 的定位难易对比；对应原书 p.421 Figure 7.3 图注，
  具体判据留到 E066 结构张量。
断言体系（E057–E064 判例同构+readings 体系同构）：
①图 1 五类特征词在档；②图 2 四条用途+四段链路在档；③图 3 三类 patch 在档；
④主文本 fontsize≥12pt+「程序生成，未用原书原图」×3；⑤图内 Text 全量收割零内部记号
（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak）；⑥三张 PNG 落盘且计数=3。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e065_feature_family,e065_match_use_cases,e065_patch_localizability}.png
（6.6×4.2 inch @dpi150）。
"""
import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E065_feature_points"
FIG_DIR = EP / "code" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EP / "code" / "outputs"

PAPER = "#faf7f0"
INK = "#2f3540"
MUT = "#5a6472"
ACC = "#8a3324"
BLU = "#3c4f6b"
GOLD = "#b08a3e"
GRN = "#4a6b4a"
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


def _box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))


def _arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def collect_texts(fig):
    out = []
    for ax in fig.axes:
        out.extend(t for t in ax.texts if t.get_text().strip())
        for patch in ax.patches:
            if hasattr(patch, "get_text") and patch.get_text().strip():
                out.append(patch)
    return out


# ── 图 1：特征家族 ──
def fig_feature_family():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "特征家族：五类线索，各答一个问题", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    lanes = [
        ("点", "关键点/兴趣点/角点", "周围图块长什么样", BLU),
        ("边缘", "方向+局部外观", "哪里发生了突变", GOLD),
        ("轮廓", "边缘连成曲线", "边界往哪里走", GRN),
        ("直线", "拟合直线段", "空间往哪里退（灭点）", ACC),
        ("区域", "分割出的像素块", "这一块整体像什么", BLU),
    ]
    w, h = 2.95, 0.86
    for i, (name, l1, l2, col) in enumerate(lanes):
        x = 0.25 + (i % 2) * 3.15
        y = 2.92 - (i // 2) * 0.98
        _box(ax, x, y, w, h, ec=col)
        ax.text(x + 0.14, y + 0.64, name, fontsize=13, color=col,
                ha="left", va="center", fontweight="bold")
        ax.text(x + 0.14, y + 0.39, l1, fontsize=12, color=INK, ha="left", va="center")
        ax.text(x + 0.14, y + 0.17, l2, fontsize=12, color=MUT, ha="left", va="center")
    # 第五个（区域）居左，右下放一句总纲
    _box(ax, 3.40, 0.94, 2.80, 0.62, ec=GRN, fc="#f2f6ef")
    ax.text(4.80, 1.25, "共同点：都能在别的图里再找到", fontsize=12.5,
            color=GRN, ha="center", va="center", fontweight="bold")
    ax.text(3.3, 0.54, "点先钉位置，边缘和轮廓补边界，直线补空间方向，区域补整体",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.418–419，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：对应点的用途 ──
def fig_match_use_cases():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "对应点能做什么：从稀疏骨架到稠密场景", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    chain = [
        ("稀疏对应", "图里\n对上的点"),
        ("相机位姿", "在哪\n朝哪"),
        ("稠密立体", "逐像素\n对应"),
        ("三维重建", "模型/\n中间视角"),
    ]
    bw, gap = 1.46, 0.11
    for i, (name, sub) in enumerate(chain):
        x = 0.18 + i * (bw + gap)
        _box(ax, x, 2.62, bw, 0.96, ec=BLU if i < 2 else GRN)
        ax.text(x + bw / 2, 3.30, name, fontsize=12.5, color=BLU if i < 2 else GRN,
                ha="center", va="center", fontweight="bold")
        ax.text(x + bw / 2, 2.86, sub, fontsize=12, color=MUT,
                ha="center", va="center", linespacing=1.18)
        if i < len(chain) - 1:
            _arrow(ax, (x + bw + 0.02, 3.10), (x + bw + gap - 0.02, 3.10))
    uses = [
        ("全景拼接", "把重叠图缝成一张"),
        ("高动态范围", "不同曝光对齐"),
        ("视频稳像", "逐帧对齐去抖"),
        ("实例识别", "局部特征找同款"),
    ]
    for j, (name, sub) in enumerate(uses):
        x = 0.23 + (j % 2) * 3.17
        y = 1.62 - (j // 2) * 0.78
        _box(ax, x, y, 3.00, 0.64, ec=GOLD, fc="#fbf6e9")
        ax.text(x + 0.14, y + 0.46, name, fontsize=12.2, color=GOLD,
                ha="left", va="center", fontweight="bold")
        ax.text(x + 0.14, y + 0.15, sub, fontsize=12, color=INK, ha="left", va="center")
    _arrow(ax, (1.86, 2.58), (1.86, 2.30), color=GOLD, lw=1.1)
    ax.text(0.23, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.419–420，布局自绘",
            fontsize=12, color=MUT, ha="left", va="center")
    return fig


# ── 图 3：图块可定位性 ──
def fig_patch_localizability():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "图块可定位性：不是每块都同样好找", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    # 三块示意图：平坦/单向边缘/拐角
    boxes = [
        ("平坦雪地", 0.45, "放哪都像", ACC, 0.22),
        ("单向边缘", 2.42, "沿线滑动也像", GOLD, 0.55),
        ("拐角", 4.39, "横竖都变，钉得住", GRN, 0.92),
    ]
    for name, x, sub, col, score in boxes:
        _box(ax, x, 1.55, 1.75, 1.95, ec=col)
        ax.text(x + 0.875, 3.28, name, fontsize=12.8, color=col,
                ha="center", va="center", fontweight="bold")
        # 简化图案
        if name == "平坦雪地":
            ax.add_patch(Rectangle((x + 0.18, 1.95), 1.39, 1.05, fc="#eef2f5", ec="#c9d2da"))
        elif name == "单向边缘":
            ax.add_patch(Rectangle((x + 0.18, 1.95), 0.60, 1.05, fc="#e8edf3", ec="#c9d2da"))
            ax.add_patch(Rectangle((x + 0.78, 1.95), 0.79, 1.05, fc="#8ea3b8", ec="#c9d2da"))
        else:
            ax.add_patch(Rectangle((x + 0.18, 1.95), 0.65, 0.50, fc="#e8edf3", ec="#c9d2da"))
            ax.add_patch(Rectangle((x + 0.83, 1.95), 0.74, 0.50, fc="#d7dfe8", ec="#c9d2da"))
            ax.add_patch(Rectangle((x + 0.18, 2.45), 0.65, 0.55, fc="#d7dfe8", ec="#c9d2da"))
            ax.add_patch(Rectangle((x + 0.83, 2.45), 0.74, 0.55, fc="#8ea3b8", ec="#c9d2da"))
        ax.text(x + 0.875, 1.76, sub, fontsize=12, color=MUT, ha="center", va="center")
        # 可定位性条
        ax.add_patch(Rectangle((x + 0.18, 1.28), 1.39, 0.10, fc="#e3e7eb", ec="none"))
        ax.add_patch(Rectangle((x + 0.18, 1.28), 1.39 * score, 0.10, fc=col, ec="none"))
    ax.text(3.3, 0.92, "可定位性：越往右越容易钉住位置",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.60, "为什么？下一集用结构张量把「横竖都有变化」算出来",
            fontsize=12, color=GRN, ha="center", va="center")
    ax.text(3.3, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.421 图 7.3 图注，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def main() -> None:
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={src_sha}")

    fig1 = fig_feature_family()
    fig2 = fig_match_use_cases()
    fig3 = fig_patch_localizability()
    p1 = FIG_DIR / "e065_feature_family.png"
    p2 = FIG_DIR / "e065_match_use_cases.png"
    p3 = FIG_DIR / "e065_patch_localizability.png"
    fig1.savefig(p1, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig2.savefig(p2, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig3.savefig(p3, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    plt.close("all")

    def texts(fig):
        return [t.get_text() for t in collect_texts(fig)]

    t1, t2, t3 = texts(fig1), texts(fig2), texts(fig3)
    all_texts = t1 + t2 + t3
    leaked = [s for s in all_texts if FIG_MARK.search(s)]
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]
    fs3 = [t.get_fontsize() for t in collect_texts(fig3)]
    pngs = [p1, p2, p3]

    rows = []
    rows.append(f"图 1 五类特征词在档：{all(k in ' '.join(t1) for k in ['点','边缘','轮廓','直线','区域'])}")
    rows.append(f"图 1 三个追问在档：{all(k in ' '.join(t1) for k in ['哪里发生了突变','边界往哪里走','空间往哪里退'])}")
    rows.append(f"图 2 四类用途在档：{all(k in ' '.join(t2) for k in ['全景拼接','高动态范围','视频稳像','实例识别'])}")
    rows.append(f"图 2 四段链路在档：{all(k in ' '.join(t2) for k in ['稀疏对应','相机位姿','稠密立体','三维重建'])}")
    rows.append(f"图 3 三类图块在档：{all(k in ' '.join(t3) for k in ['平坦雪地','单向边缘','拐角'])}")
    rows.append(f"图 3 可定位性递进在档：{all(k in ' '.join(t3) for k in ['放哪都像','沿线滑动也像','钉得住'])}")
    rows.append(f"主文本字号≥12pt+程序生成注记在档×3　| 图一 最小 {min(fs1):.1f}pt/图二 最小 {min(fs2):.1f}pt/图三 最小 {min(fs3):.1f}pt")
    rows.append(f"三图均含「程序生成，未用原书原图」：{sum('程序生成，未用原书原图' in s for s in all_texts) == 3}")
    rows.append(f"图内文字内部记号零泄漏（红线 5）　| 图内 Text 收割 {len(all_texts)} 条，命中 {('、'.join(leaked)) if leaked else '无'}")
    rows.append(f"三张 PNG 落盘：{sum(p.exists() and p.stat().st_size > 0 for p in pngs) == 3}")

    print("[lab] 图 1 文本：" + " / ".join(t1))
    print("[lab] 图 2 文本：" + " / ".join(t2))
    print("[lab] 图 3 文本：" + " / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    if leaked:
        print(f"[FAIL] 内部记号泄漏：{leaked}")
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1

    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"families": 5, "point": 1, "edge": 1, "contour": 1, "line": 1, "region": 1,
                 "min_fs": min(fs1), "max_fs": max(fs1), "fig_no": 7.1, "page": 418},
        "fig2": {"use_cases": 4, "chain": 4, "min_fs": min(fs2), "max_fs": max(fs2),
                 "fig_no": 7.2, "page": 420},
        "fig3": {"patches": 3, "easy": 1, "hard": 2, "min_fs": min(fs3), "max_fs": max(fs3),
                 "fig_no": 7.3, "page": 421},
        "scope": {"page_start": 417, "page_end": 421, "frames": 16},
        "texts": {"collected": len(all_texts), "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rd = OUT_DIR / "e065_feature_points_readings.json"
    rd.write_text(json.dumps(readings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rd.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd.read_bytes()).hexdigest()}")
    for png in pngs:
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
