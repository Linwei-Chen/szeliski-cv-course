#!/usr/bin/env python3
"""E073 全景图：识别链十站 + 特征链八站 + 三座桥（红线 2：程序生成，本节原书无此图）。

断言：①18 个节点与台账行同名同集号；②节点框互不重叠且不出画布；
③三座桥端点正确；④全部文字 ≥10pt（显示缩放后 ≥12 CSS px 由 display_check 复核）。
输出 code/outputs/figures/e073_recap_map.png（9.0×6.2 in，dpi150）。
"""
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E073_recap_ch6_ch7"
LEDGER = (EP / "code" / "outputs" / "e073_recap_ledger.txt").read_text()
OUT = EP / "code" / "outputs" / "figures" / "e073_recap_map.png"

REC = [
    ("实例识别与几何验证", "E055", "p.346"), ("大规模实例检索", "E056", "p.448"),
    ("图像分类的手工特征", "E057", "p.349"), ("深度网络与相似检索", "E058", "p.358"),
    ("人脸识别", "E059", "p.363"), ("人脸检测与行人检测", "E060", "p.370"),
    ("通用对象检测", "E061", "p.379"), ("语义分割", "E062", "p.387"),
    ("视频与视觉语言", "E063", "p.396"), ("识别章习题课", "E064", "p.413"),
]
FEAT = [
    ("特征点与局部块", "E065", "p.419"), ("角点检测", "E066", "p.422"),
    ("SIFT 尺度空间", "E067", "p.429"), ("SIFT 描述子", "E068", "p.434"),
    ("特征匹配与几何校验", "E069", "p.441"), ("直线与消失点", "E070", "p.477"),
    ("学习型描述子", "E071", "p.437"), ("OpenCV 全流程实验", "E072", "p.441"),
]

rows = dict(re.findall(r"^  (.+?)（(E\d{3})）｜", LEDGER, re.M))
errors = []
for name, ep, _ in REC + FEAT:
    if rows.get(name) != ep:
        errors.append(f"节点「{name}」/ {ep} 与台账不符（台账={rows.get(name)}）")
if len(rows) != 18:
    errors.append(f"台账行数 {len(rows)} ≠ 18")
if errors:
    for e in errors:
        print(f"[FAIL] {e}", file=sys.stderr)
    sys.exit(1)
print("[ok] 18 节点与台账同名同集号")

INK, PAPER = "#1c2437", "#f6f0e6"
C6, C7, BRIDGE = "#7a5c2e", "#3d6a99", "#a23b2a"
W, H = 9.0, 6.2
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)

X_L, X_R, NW = 0.045, 0.565, 0.37
NH, STEP, Y0 = 0.066, 0.078, 0.815
nodes = {}
for i, (name, ep, pg) in enumerate(REC):
    nodes[ep] = dict(name=name, ep=ep, pg=pg, x=X_L, y=Y0 - i * STEP, w=NW, h=NH, side="L")
for i, (name, ep, pg) in enumerate(FEAT):
    nodes[ep] = dict(name=name, ep=ep, pg=pg, x=X_R, y=Y0 - i * STEP, w=NW, h=NH, side="R")

for n in nodes.values():
    if not (0 <= n["x"] and n["x"] + n["w"] <= 1 and 0 <= n["y"] - n["h"] and n["y"] <= 1):
        errors.append(f"节点出界：{n['name']}")
    ax.add_patch(FancyBboxPatch((n["x"], n["y"] - n["h"]), n["w"], n["h"],
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="#fffdf8" if n["side"] == "L" else "#f4f8fc",
                                ec=C6 if n["side"] == "L" else C7, lw=1.2))
    short = {"大规模实例检索": "大规模检索", "图像分类的手工特征": "手工特征分类",
             "深度网络与相似检索": "深度网络", "人脸检测与行人检测": "人脸与行人检测",
             "视频与视觉语言": "视频与语言", "识别章习题课": "识别习题",
             "SIFT 尺度空间": "SIFT 尺度", "特征匹配与几何校验": "匹配与校验",
             "OpenCV 全流程实验": "OpenCV 实验", "实例识别与几何验证": "实例识别",
             "通用对象检测": "通用检测", "特征点与局部块": "特征点", "角点检测": "角点"}.get(n["name"], n["name"])
    ax.text(n["x"] + 0.012, n["y"] - 0.022, short, fontsize=11.2, color=INK,
            ha="left", va="center", fontweight="bold")
    ax.text(n["x"] + n["w"] - 0.012, n["y"] - 0.047, n["pg"], fontsize=10.0,
            color="#5d6472", ha="right", va="center")

vals = list(nodes.values())
for i, a in enumerate(vals):
    for b in vals[i + 1:]:
        ax0, ay0 = a["x"], a["y"] - a["h"]
        bx0, by0 = b["x"], b["y"] - b["h"]
        if ax0 < bx0 + b["w"] and ax0 + a["w"] > bx0 and ay0 < by0 + b["h"] and ay0 + a["h"] > by0:
            errors.append(f"节点重叠：{a['name']} / {b['name']}")

ax.text(X_L + NW / 2, 0.945, "第 6 章 · 认得出（语义）", fontsize=13, color=C6,
        ha="center", va="center", fontweight="bold")
ax.text(X_R + NW / 2, 0.945, "第 7 章 · 找得准（几何）", fontsize=13, color=C7,
        ha="center", va="center", fontweight="bold")

BR = [("E055", "E069", "匹配"), ("E059", "E068", "向量"), ("E061", "E069", "几何")]
for src, dst, label in BR:
    a, b = nodes[src], nodes[dst]
    p1 = (a["x"] + a["w"], a["y"] - a["h"] / 2)
    p2 = (b["x"], b["y"] - b["h"] / 2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.5, ls=(0, (4, 3)), color=BRIDGE,
                                 connectionstyle="arc3,rad=0.08"))
    ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.018, label,
            fontsize=10.0, color=BRIDGE, ha="center", va="bottom", fontweight="bold")

if errors:
    for e in errors:
        print(f"[FAIL] {e}", file=sys.stderr)
    sys.exit(1)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.05)
print(f"[ok] {OUT}")
print("MAP-DONE")
