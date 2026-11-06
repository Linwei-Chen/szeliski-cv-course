#!/usr/bin/env python3
"""E054 S2 全景图：模具链（6 站）+学习链（12 站）节点流图（红线 2：程序生成，本节原书无图表）。
断言体系（E035 判例同构）：
①节点与台账行逐一核对（概念名+集号双比对，不渲染集号）；
②水平箭头端点贴节点缘（数据坐标判定，容差 TOL）+流拓扑断言——(来源→目标) 配对序列
  必须与两链预期流逐一相等；
③桥肘线（模具链终点→学习链起点）与蛇形折返肘线走链间空带；标注纸底盒压折返水平段
  （E035 判例形态——盒不透明纸底，线自盒两侧露出）；
④fig.canvas.draw() 后用真实窗口 extent 断言标注盒与全部节点框不相交（bbox pad 计入
  实测，不由常数自比）；
⑤--negative-test 注入错误桥（末支槽位换错目标），贴缘仍成立但流拓扑配对错→必须 FAIL
  （负例记录）。
输出 code/outputs/figures/e054_recap_map.png（1350×930@dpi150，deck 页 2 显示 max-height 560px——缩放 0.602，图内最小 10pt→20.8px×0.602≈12.5 CSS px ≥12px 线，display_check 核算为准）。
红线 5：可见层无内部记号——节点只标概念+原书节·页（集号仅用于台账断言，不渲染）。"""

import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E054_recap_ch4_ch5"
FIG_DIR = EP / "code" / "outputs" / "figures"
LEDGER = (EP / "code" / "outputs" / "e054_recap_ledger.txt").read_text()
NEG = "--negative-test" in sys.argv

# 与台账 ROWS 同序同名（(概念, 集, 页锚渲染文案)）
CH4 = [
    ("最小二乘与正规方程", "E036", "附录 A.2 p.927"),
    ("鲁棒损失与 IRLS", "E037", "4.1.3 节 p.202"),
    ("RANSAC", "E038", "8.1.4 节 p.510"),
    ("RANSAC 实战三件", "E039", "8.1.4 节续 p.511"),
    ("不确定度与协方差", "E040", "附录 B.6 p.952"),
    ("非线性最小二乘与 LM", "E041", "附录 A.3 p.930"),
]
CH5 = [
    ("神经元与激活函数", "E042", "5.3.1 p.270"),
    ("损失函数与梯度下降", "E043", "5.3.4 p.280"),
    ("反向传播", "E044", "5.3.5 p.285"),
    ("优化器与初始化", "E045", "5.3.6 p.287"),
    ("卷积与权值共享", "E046", "5.4 p.291"),
    ("感受野与池化", "E047", "5.4.1 p.295"),
    ("架构演进史", "E048", "5.4.3 p.299"),
    ("可视化与对抗样本", "E049", "5.4.5 p.307"),
    ("3D CNN 与 RNN", "E050", "5.5.1 p.317"),
    ("Transformer 与 ViT", "E051", "5.5.3 p.322"),
    ("生成模型", "E052", "5.5.4 p.328"),
    ("实验课", "E053", "5.7 p.337"),
]

errors = []
ledger_rows = dict(re.findall(r"^  (.+?)（(E[\d+E]+)）｜", LEDGER, re.M))
for name, eps, _ in CH4 + CH5:
    if name not in ledger_rows:
        errors.append(f"节点「{name}」不在台账行中")
    elif ledger_rows[name] != eps:
        errors.append(f"节点「{name}」集号 {eps} ≠ 台账 {ledger_rows[name]}")
if len(CH4) + len(CH5) != 18 or len(ledger_rows) != 18:
    errors.append(f"节点数 {len(CH4)+len(CH5)} ≠ 台账行数 {len(ledger_rows)}")
if errors:
    for e in errors:
        print(f"[FAIL] {e}", file=sys.stderr)
    sys.exit(1)
print(f"[ok] 节点 {len(CH4)+len(CH5)} 个与台账 18 行逐一对应（含集号比对）")

INK, PAPER, C4, C5 = "#1c2437", "#f6f0e6", "#7a5c2e", "#3d6a99"
FIG_W, FIG_H = 9.0, 6.2

# 布局：模具链 6 站 y=0.78；学习链两行各 6 站（y=0.44 上行 / y=0.10 下行，均左→右）
NW, NH, DX, X0 = 0.148, 0.095, 0.158, 0.045
NODES = []
for i, (name, eps, pg) in enumerate(CH4):
    NODES.append(dict(name=name, eps=eps, pg=pg, chain=4, x=X0 + i * DX, y=0.78, w=NW, h=NH))
for i, (name, eps, pg) in enumerate(CH5[:6]):
    NODES.append(dict(name=name, eps=eps, pg=pg, chain=5, x=X0 + i * DX, y=0.44, w=NW, h=NH))
for i, (name, eps, pg) in enumerate(CH5[6:]):
    NODES.append(dict(name=name, eps=eps, pg=pg, chain=5, x=X0 + i * DX, y=0.10, w=NW, h=NH))
ND = {(n["eps"], n["name"]): n for n in NODES}
BRIDGE_Y = 0.615   # 桥横段空带（模具链底 0.78 与学习上行顶 0.528 之间）
TURN_Y = 0.272     # 折返横段空带（学习上行底 0.44 与下行顶 0.188 之间）

# 预期流：模具链 5 支顺流+桥（E041→E042）+学习上行 5 支+折返（E047→E048）+下行 5 支
FLOW = [(CH4[i], CH4[i + 1]) for i in range(5)] + [(CH4[5], CH5[0])]
FLOW += [(CH5[i], CH5[i + 1]) for i in range(5)] + [(CH5[5], CH5[6])]
FLOW += [(CH5[i], CH5[i + 1]) for i in range(6, 11)]
if NEG:
    FLOW[4] = (CH4[4], CH5[2])

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)


# 渲染层长名拆行（台账双比对仍用全名——name 不变）
SPLIT = {
    "最小二乘与正规方程": "最小二乘\n与正规方程",
    "鲁棒损失与 IRLS": "鲁棒损失\n与 IRLS",
    "RANSAC 实战三件": "RANSAC\n实战三件",
    "不确定度与协方差": "不确定度\n与协方差",
    "非线性最小二乘与 LM": "非线性最小二乘\n与 LM",
    "神经元与激活函数": "神经元\n与激活函数",
    "损失函数与梯度下降": "损失函数\n与梯度下降",
    "卷积与权值共享": "卷积\n与权值共享",
    "可视化与对抗样本": "可视化\n与对抗样本",
    "Transformer 与 ViT": "Transformer\n与 ViT",
}


def draw(n):
    color = C4 if n["chain"] == 4 else C5
    ax.add_patch(FancyBboxPatch((n["x"], n["y"]), n["w"], n["h"],
                                boxstyle="round,pad=0.005,rounding_size=0.012",
                                fc="white", ec=color, lw=1.6))
    disp = SPLIT.get(n["name"], n["name"])
    nl = "\n" in disp
    ax.text(n["x"] + n["w"] / 2, n["y"] + n["h"] * (0.70 if nl else 0.66), disp,
            ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.04)
    ax.text(n["x"] + n["w"] / 2, n["y"] + n["h"] * (0.16 if nl else 0.22), n["pg"],
            ha="center", va="center", fontsize=10, color="#5a6472")


for n in NODES:
    draw(n)

ax.text(0.5, 0.925, "第 4 章 一枚模具：拟合与优化（从猜到证明）",
        ha="center", fontsize=15, color=C4)
ax.text(0.5, 0.585, "第 5 章 一整套学习：深度学习（从证明到学会）",
        ha="center", fontsize=15, color=C5)


def geom_of(a, b):
    ha, hb = ND[(a[1], a[0])], ND[(b[1], b[0])]
    if ha["y"] == hb["y"]:  # 同行水平箭头
        if hb["x"] > ha["x"]:
            x0, x1 = ha["x"] + ha["w"], hb["x"]
        else:
            x0, x1 = ha["x"], hb["x"] + hb["w"]
        return dict(kind="h", x0=x0, x1=x1, y=ha["y"] + ha["h"] / 2, a=ha, b=hb)
    # 肘线三段（竖→横→竖入顶缘）：竖段走节点中线，横段走空带，末段下行入目标顶缘
    xc = ha["x"] + ha["w"] / 2
    xm = hb["x"] + hb["w"] / 2
    ymid = BRIDGE_Y if ha["chain"] == 4 else TURN_Y  # 桥走模具链下空带；折返走学习两行间空带
    return dict(kind="elbow", a=ha, b=hb, x0=xc, xm=xm, y0=ha["y"],
                ymid=ymid, y1=hb["y"] + hb["h"])


for a, b in FLOW:
    g = geom_of(a, b)
    if g["kind"] == "h":
        c = "#8a7a5c" if g["a"]["chain"] == 4 else "#6d8cab"
        ax.add_patch(FancyArrowPatch((g["x0"], g["y"]), (g["x1"], g["y"]),
                                     arrowstyle="-|>", mutation_scale=11, color=c,
                                     lw=1.3, shrinkA=0, shrinkB=0))
    else:
        c = "#a0522d" if (g["a"]["chain"] == 4 and g["b"]["chain"] == 5) else "#6d8cab"
        ax.plot([g["x0"], g["x0"]], [g["y0"], g["ymid"]], color=c, lw=1.3)
        ax.plot([g["x0"], g["xm"]], [g["ymid"], g["ymid"]], color=c, lw=1.3)
        ax.add_patch(FancyArrowPatch((g["xm"], g["ymid"]), (g["xm"], g["y1"]),
                                     arrowstyle="-|>", mutation_scale=11, color=c,
                                     lw=1.3, shrinkA=0, shrinkB=0))

# 标注纸底盒压折返横段（E035 判例形态）
ann = ax.annotate("一枚硬币的两面：同一套「假设→损失→梯度→迭代」模具——\n"
                  "第 4 章用它证明一张图，第 5 章用它学会看图",
                  xy=(0.5, TURN_Y), ha="center", va="center", fontsize=11.5, color=INK,
                  bbox=dict(boxstyle="round,pad=0.42", fc="#fdf8ee", ec="#c9b98a", lw=1.0),
                  zorder=5)

# 断言②：端点贴缘（数据坐标，容差 TOL）+流拓扑
TOL = 0.004
errors_post = []
for gi, (a, b) in enumerate(FLOW):
    g = geom_of(a, b)
    ha, hb = g["a"], g["b"]
    if g["kind"] == "h":
        if hb["x"] > ha["x"]:
            ok = abs(g["x0"] - (ha["x"] + ha["w"])) <= TOL and abs(g["x1"] - hb["x"]) <= TOL
        else:
            ok = abs(g["x0"] - ha["x"]) <= TOL and abs(g["x1"] - (hb["x"] + hb["w"])) <= TOL
        if not ok:
            errors_post.append(f"端点未贴缘: {a[0]}→{b[0]}")
    else:
        ok = (abs(g["x0"] - (ha["x"] + ha["w"] / 2)) <= TOL
              and abs(g["xm"] - (hb["x"] + hb["w"] / 2)) <= TOL
              and abs(g["y0"] - ha["y"]) <= TOL
              and abs(g["y1"] - (hb["y"] + hb["h"])) <= TOL)
        if not ok:
            errors_post.append(f"肘线端点未贴缘: {a[0]}→{b[0]}")

EXP_PAIRS = [(a[0], b[0]) for a, b in (
    [(CH4[i], CH4[i + 1]) for i in range(5)] + [(CH4[5], CH5[0])]
    + [(CH5[i], CH5[i + 1]) for i in range(5)] + [(CH5[5], CH5[6])]
    + [(CH5[i], CH5[i + 1]) for i in range(6, 11)])]
got_pairs = [(a[0], b[0]) for a, b in FLOW]
if got_pairs != EXP_PAIRS:
    for gi, (e, g2) in enumerate(zip(EXP_PAIRS, got_pairs)):
        if e != g2:
            errors_post.append(f"流拓扑错位 #{gi}: 预期 {e[0]}→{e[1]}，实际 {g2[0]}→{g2[1]}")
if NEG and not errors_post:
    errors_post.append("负例注入未被拦截")

# 断言④：标注盒与全部节点框真实窗口 extent 不相交（bbox pad 计入实测）
fig.canvas.draw()
r = fig.canvas.get_renderer()
ann_bb = ann.get_window_extent(r)
for n in NODES:
    xy0 = ax.transData.transform((n["x"] - 0.005, n["y"] - 0.005))
    xy1 = ax.transData.transform((n["x"] + n["w"] + 0.005, n["y"] + n["h"] + 0.005))
    nb = Bbox((xy0, xy1))
    if ann_bb.overlaps(nb):
        errors_post.append(f"标注盒与节点「{n['name']}」相交")
if errors_post:
    for e in errors_post:
        print(f"[FAIL] {e}", file=sys.stderr)
    if NEG:
        print("[negative-test] 预期 FAIL——负例记录完成")
    sys.exit(1)

FIG_DIR.mkdir(parents=True, exist_ok=True)
out = FIG_DIR / "e054_recap_map.png"
fig.savefig(out, dpi=150, facecolor=PAPER)
print(f"[ok] {out}")
if NEG:
    print("NEG-UNEXPECTED-PASS")
    sys.exit(1)
print("MAP-DONE")
