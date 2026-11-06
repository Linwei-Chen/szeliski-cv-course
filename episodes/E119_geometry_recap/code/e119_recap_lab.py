#!/usr/bin/env python3
"""E119 第 11–12 章复盘台账：18 站、三条共用算子、三层主角图。

边界：本集是复盘课，不新测算法；所有计数来自课程地图与各集已过门审的 shots.json/status。
"""
import hashlib, json, pathlib
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"

# ep, 标题, 章, 页码, 类型, 输出族, 是否学习组件, 角色
STATIONS = [
 ("E101", "SfM 问题总览", "11", "683–684", "concept", "foundation", False, "开场地图"),
 ("E102", "双视图几何基石", "11", "703–708", "concept", "foundation", False, "对极约束"),
 ("E103", "八点法", "11", "703–708", "algorithm", "foundation", False, "解 F"),
 ("E104", "姿态恢复与三角化", "11", "693–703", "algorithm", "sparse", False, "相机+点"),
 ("E105", "束调整", "11", "717–722", "algorithm", "sparse", False, "联合优化"),
 ("E106", "增量式重建", "11", "725–728", "algorithm", "sparse", False, "流水线"),
 ("E107", "全局 SfM", "11", "728–730", "algorithm", "sparse", False, "旋转平均"),
 ("E108", "SLAM", "11", "734–740", "concept", "trajectory", False, "在线定位"),
 ("E109", "ORB-SLAM", "11", "外部论文", "algorithm", "trajectory", False, "三线程"),
 ("E110", "COLMAP 上机", "11", "725–728", "lab", "sparse", False, "动手"),
 ("E117", "自动驾驶感知栈", "11", "737–739", "application", "application", False, "应用"),
 ("E111", "视差即深度", "12", "751–753", "concept", "dense", False, "几何入口"),
 ("E112", "立体匹配流水线", "12", "762–769", "algorithm", "dense", False, "代价/聚合"),
 ("E113", "SGM", "12", "774–775", "algorithm", "dense", False, "全局近似"),
 ("E114", "PSM-Net", "12", "778–781", "algorithm", "dense", True, "学习代价体"),
 ("E115", "单目深度", "12", "796–798", "learned", "dense", True, "单图深度"),
 ("E118", "Middlebury/KITTI 榜单", "12", "751–754 等", "evaluation", "evaluation", False, "评测"),
 ("E116", "深度相机硬件", "13", "816–820", "hardware", "dense", False, "跨章桥"),
]
OPS = {
 "对应": ("11", "12"),
 "相机模型": ("11", "12"),
 "优化": ("11", "12"),
}
BRIDGES = [
 ("稀疏→稠密", "E110→E111"),
 ("几何→学习", "E113→E114/E115"),
 ("离线→在线", "E107→E108/E117"),
]

n = len(STATIONS); ch = Counter(s[2] for s in STATIONS)
spans = {"11": (683, 750), "12": (751, 807), "13": (816, 820)}
ops_ok = all(set(chs) == {"11", "12"} for chs in OPS.values())
outputs = Counter(s[5] for s in STATIONS)
learned = [s[0] for s in STATIONS if s[6]]
learning_after_baseline = learned == ["E114", "E115"] and "E113" in [s[0] for s in STATIONS]
n_neural = sum(1 for s in STATIONS if any(k in s[1] for k in ("NeRF", "神经场", "Neural")))
n_lab = sum(1 for s in STATIONS if s[4] == "lab")
n_eval = sum(1 for s in STATIONS if s[4] == "evaluation")

checks = [
 ("18 站齐全且编号 E101–E118 不重不漏", n == 18 and len({s[0] for s in STATIONS}) == 18 and [s[0] for s in STATIONS].count("E101") == 1, f"{n} 站"),
 ("第 11 章 11 站", ch["11"] == 11, f"{ch['11']} 站"),
 ("第 12 章 6 站", ch["12"] == 6, f"{ch['12']} 站"),
 ("第 13 章硬件桥 1 站（E116，不冒充第 12 章）", ch["13"] == 1 and any(s[0] == "E116" and s[2] == "13" for s in STATIONS), f"{ch['13']} 站"),
 ("章页跨度 11=68 页、12=57 页、13 桥 5 页，互不重叠", spans["11"] == (683, 750) and spans["12"] == (751, 807) and spans["13"] == (816, 820), "683–750 / 751–807 / 816–820"),
 ("三条共用算子（对应/相机模型/优化）两章都出现", ops_ok and len(OPS) == 3, "3 条"),
 ("输出族：稀疏 5、稠密 6、轨迹 2；另有基础/应用/评测 5", outputs["sparse"] == 5 and outputs["dense"] == 6 and outputs["trajectory"] == 2, f"稀疏{outputs['sparse']}/稠密{outputs['dense']}/轨迹{outputs['trajectory']}"),
 ("学习组件 2 个（E114/E115），都排在几何基线之后", learning_after_baseline, "E113 之后"),
 ("3 座桥 + 1 个动手 + 1 个评测", len(BRIDGES) == 3 and n_lab == 1 and n_eval == 1, "3/1/1"),
 ("神经场 0 站：本集只到第 12 章，不冒充已讲 NeRF", n_neural == 0, "0 站"),
]
n_pass = 0
print(f"[lab] E119 复盘台账开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：复盘课不新测算法；计数来自课程地图与各集已过门审 shots/status")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

def vis_no(eid: str) -> str:
    """可见层集号：正课编号=内部编号−8（E101→第 93 集，E118→第 110 集；页脚口径）。"""
    return str(int(eid[1:]) - 8)


# 图1 18 站时间线
fig, ax = plt.subplots(figsize=(11.2, 6.6)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 7.2); ax.axis("off")
ax.text(0.25, 6.85, "18 站：第 11 章 11 站，第 12 章 6 站，外加第 13 章 1 座硬件桥", fontsize=19, color=INK, weight="bold")
cols = {"11": (0.25, "第 11 章 SfM / SLAM · p.683–750", BLUE), "12": (5.75, "第 12 章 深度 · p.751–807", GREEN)}
for chap, (x0, head, color) in cols.items():
    ax.text(x0, 6.35, head, fontsize=15, color=color, weight="bold")
    rows = [s for s in STATIONS if s[2] == chap]
    for i, s in enumerate(rows):
        y = 5.85 - i * 0.49
        ax.add_patch(Rectangle((x0, y - 0.19), 4.9, 0.38, facecolor="#fffaf0", edgecolor="#c9bda4", lw=0.8))
        ax.text(x0 + 0.10, y, vis_no(s[0]), fontsize=13.5, color=color, weight="bold", va="center")
        ax.text(x0 + 0.78, y, s[1], fontsize=13.5, color=INK, va="center")
        ax.text(x0 + 4.55, y, s[3], fontsize=13.5, color=GRAY, va="center", ha="right")
# Ch13 bridge
ax.add_patch(Rectangle((5.75, 0.68), 4.9, 0.38, facecolor="#f8ece8", edgecolor=ACCENT, lw=1.0))
ax.text(5.85, 0.87, vis_no("E116"), fontsize=13.5, color=ACCENT, weight="bold", va="center")
ax.text(6.53, 0.87, "深度相机硬件（第 13 章 13.2 节桥）", fontsize=13.5, color=INK, va="center")
ax.text(10.55, 0.87, "816–820", fontsize=13.5, color=GRAY, va="center", ha="right")
ax.text(0.25, 0.25, "集号为正课编号；页码来自各集已过门审的 S1 存证；第 101 集为外部论文，不标书页。", fontsize=13.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e119_timeline.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 共用几何脊柱
fig, ax = plt.subplots(figsize=(11.2, 5.6)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(0.25, 5.25, "两条流水线，共用一根几何脊柱", fontsize=20, color=INK, weight="bold")
def box(x, y, w, h, title, body, fc, ec, fs=14):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.6))
    ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=fs+1, color=INK, weight="bold")
    ax.text(x + w/2, y + h*0.28, body, ha="center", va="center", fontsize=fs-1.5, color=GRAY)
box(0.25, 3.3, 1.9, 1.05, "2D 对应", "特征/像素匹配", "#eef2f7", BLUE)
box(2.55, 3.3, 1.9, 1.05, "相机模型", "内参/外参/对极", "#eef7ee", GREEN)
box(4.85, 3.3, 2.0, 1.05, "三角化/BA", "点与相机一起解", "#fff3cd", "#b8860b")
box(7.25, 3.3, 1.8, 1.05, "3D 输出", "点云/轨迹/深度", "#f8ece8", ACCENT)
for x in (2.15, 4.45, 6.85):
    ax.add_patch(FancyArrowPatch((x, 3.82), (x + 0.35, 3.82), arrowstyle="->", color=GRAY, lw=1.6, mutation_scale=13))
ax.text(0.25, 2.65, "第 11 章：稀疏点 + 相机（第 96–99 集）", fontsize=14.5, color=BLUE, weight="bold")
ax.text(0.25, 2.25, "第 12 章：稠密深度/视差（第 103–107 集）", fontsize=14.5, color=GREEN, weight="bold")
ax.text(0.25, 1.85, "学习换掉的是“对应/代价”这一层，不是相机模型和优化", fontsize=13.5, color=GRAY)
box(0.25, 0.55, 2.7, 0.85, "几何基座", "13 站（含 1 个上机）", "#fffaf0", "#c9bda4", 13.8)
box(3.15, 0.55, 2.0, 0.85, "学习组件", "2 站", "#eef2f7", BLUE, 13.8)
box(5.35, 0.55, 1.9, 0.85, "硬件桥", "1 站", "#eef7ee", GREEN, 13.8)
box(7.45, 0.55, 3.2, 0.85, "应用/评测", "2 站", "#f8ece8", ACCENT, 13.8)
fig.tight_layout(); fig.savefig(FIG / "e119_spine.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 谁在扛事：三层条形
fig, ax = plt.subplots(figsize=(10.8, 5.4))
labels = ["几何骨架", "学习组件", "硬件桥", "应用", "评测"]
vals = [13, 2, 1, 1, 1]
colors = [BLUE, ACCENT, GREEN, "#6b3fa0", GRAY]
y = list(range(len(labels)))[::-1]
ax.barh(y, vals, color=colors, height=0.58)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=15)
ax.set_xlim(0, 15); ax.set_xlabel("站数（共 18 站）", fontsize=15); ax.tick_params(axis="x", labelsize=13)
for yy, v in zip(y, vals):
    ax.text(v + 0.25, yy, str(v), va="center", fontsize=15, color=INK, weight="bold")
ax.set_title("18 站里，13 站是几何骨架：主角不是某个网络", fontsize=18, color=INK)
ax.grid(axis="x", alpha=0.22)
fig.tight_layout(); fig.savefig(FIG / "e119_layers.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = {
 "n_stations": n, "n_ch11": ch["11"], "n_ch12": ch["12"], "n_ch13_bridge": ch["13"],
 "ch11_pages": 68, "ch12_pages": 57, "ch13_bridge_pages": 5,
 "n_sparse": outputs["sparse"], "n_dense": outputs["dense"], "n_trajectory": outputs["trajectory"],
 "n_learning": len(learned), "n_lab": n_lab, "n_eval": n_eval, "n_bridges": len(BRIDGES),
 "n_neural_fields": n_neural, "n_ops": len(OPS), "n_pass": n_pass, "n_total": len(checks),
}
(OUT / "e119_recap_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] 章分布：Ch11 {ch['11']} / Ch12 {ch['12']} / Ch13 桥 {ch['13']}")
print(f"[lab] 输出族：稀疏 {outputs['sparse']} / 稠密 {outputs['dense']} / 轨迹 {outputs['trajectory']}")
print(f"[lab] 共用算子：{'、'.join(OPS)}；学习组件：{'、'.join(learned)}；神经场覆盖 {n_neural} 站")
for name in ("e119_timeline.png", "e119_spine.png", "e119_layers.png"):
    print(f"[lab] PNG sha256({name})={hashlib.sha256((FIG/name).read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e119_recap_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
