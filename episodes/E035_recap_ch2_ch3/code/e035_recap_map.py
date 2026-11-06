#!/usr/bin/env python3
"""E035 S2 全景图实验：成像链+处理链节点流图（红线 2：程序生成，本节原书无图表）。
节点集与 e035_recap_ledger.txt 台账行逐一核对（概念名+集号双比对断言，round2 P2-g）。
箭头断言（round2 P1-a）：每支水平箭头端点贴节点缘（容差 TOL）+流拓扑断言——识别出的
(来源→目标) 配对序列必须与两链预期流逐一相等；折返/桥箭头单列断言。
--negative-test 注入 round1 旧坐标公式（ARROWS[13]，行 2 rl 第 3 支槽位、预期流=
金字塔→插值与抽取）：旧坐标端点能贴上但配对错，端点+流拓扑断言必须 FAIL（负例记录
进 status.gatea_r2/r3）。
桥（round2 P1-b / round3 P2-3）：肘线三段（竖→横→下指箭头）走两链空带 CORRIDOR_Y
（arc3 弧解析证实 rad=±0.10 均入节点 bbox，弃用）；标注纸底盒压水平段；除数据坐标
断言外，fig.canvas.draw() 后用真实窗口 extent 断言标注盒与 17 节点框不相交且 y 净空
≥2px（bbox pad 计入实测，不由常数自比）。
输出 code/outputs/figures/e035_recap_map.png（860×560，deck 页 2 右栏显示）。
红线 5：可见层无内部记号——节点只标概念+原书节·页（E0XX 仅用于台账断言，不渲染）。"""

import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E035_recap_ch2_ch3"
FIG_DIR = EP / "code" / "outputs" / "figures"
LEDGER = (EP / "code" / "outputs" / "e035_recap_ledger.txt").read_text()

# 与台账 ROWS 同序同名的节点表（(概念, 集, 页锚)）——集号与台账双比对（round2 P2-g），不渲染
CH2 = [
    ("几何基元与变换", "E013+E014", "2.1 节 p.36"),
    ("相机投影与内参", "E015+E016", "2.1.4 节 p.51"),
    ("镜头畸变", "E017", "2.1.5 节 p.63"),
    ("光度成像", "E018", "2.2 节 p.66"),
    ("光学", "E019", "2.2.3 节 p.74"),
    ("相机与色彩", "E020", "2.3.2 节 p.87"),
]
CH3 = [
    ("点操作", "E021", "3.1 节 p.109"),
    ("直方图均衡", "E022", "3.1.4 节 p.115"),
    ("线性滤波", "E023", "3.2 节 p.119"),
    ("可分离滤波", "E024", "3.2.1 节 p.124"),
    ("非线性滤波", "E025", "3.3 节 p.131"),
    ("二值与形态", "E026", "3.3.3 节 p.138"),
    ("傅里叶变换", "E027+E028", "3.4 节 p.142"),
    ("插值与抽取", "E029", "3.5 节 p.149"),
    ("金字塔", "E030", "3.5.3 节 p.154"),
    ("小波", "E031", "3.5.4 节 p.159"),
    ("几何变换", "E032", "3.6 节 p.168"),
]

errors = []
ledger_rows = dict(re.findall(r"^  (\S+?)（(E[\d+E]+)）｜", LEDGER, re.M))
for name, eps, _ in CH2 + CH3:
    if name not in ledger_rows:
        errors.append(f"节点「{name}」不在台账行中")
    elif ledger_rows[name] != eps:
        errors.append(f"节点「{name}」集号 {eps} ≠ 台账 {ledger_rows[name]}")
if len(CH2) + len(CH3) != 17 or len(ledger_rows) != 17:
    errors.append(f"节点数 {len(CH2)+len(CH3)} ≠ 台账行数 {len(ledger_rows)}")
if errors:
    for e in errors:
        print(f"[FAIL] {e}", file=sys.stderr)
    sys.exit(1)
print(f"[ok] 节点 {len(CH2)+len(CH3)} 个与台账 17 行逐一对应（含集号比对）")

INK, PAPER, CH2C, CH3C = "#1c2437", "#f6f0e6", "#3d5a80", "#98c1d9"
BRIDGEC = "#7a5c2e"

fig = plt.figure(figsize=(8.6, 5.6), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])          # 全幅 axes：0–1 坐标即整张图（无边距缩水）
fig.patch.set_facecolor(PAPER)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# 纵向预算（y 自下而上）：脚注 0.028｜Ch3 行2 0.115–0.275｜Ch3 行1 0.36–0.52
# ｜Ch3 标题 0.583｜Ch2 节点 0.615–0.775｜Ch2 标题 0.845｜桥句 0.90｜主标题 0.965
X0, W, N6, H = 0.035, 0.135, 6, 0.16
GAP6 = (1 - 2 * X0 - W * N6) / (N6 - 1)          # 六节点行：列距（两链共用列网格）
PAD, TOL = 0.004, 0.006
ROW1, ROW2 = 0.36, 0.115                          # Ch3 两行节点下缘
CH3_TITLE_Y = 0.583                               # 上移留出桥标注带（round2 P1-b）

NODES = []                                        # (name, x, y)
ARROWS = []                                       # (kind, xa, ya, xb, yb)
PATCHES = []                                      # 节点框 artist（round3 P2-3 真实 extent 断言用）


def node(x, y, name, page, color):
    dark = color == CH2C
    tcol = "white" if dark else INK
    p = FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.004",
                       fc=color, ec=INK, lw=1.1, alpha=0.94)
    ax.add_patch(p)
    PATCHES.append(p)
    ax.text(x + W / 2, y + H * 0.72, name, ha="center", va="center",
            fontsize=11, fontweight="bold", color=tcol)
    ax.text(x + W / 2, y + H * 0.30, page, ha="center", va="center",
            fontsize=10.5, color=tcol)            # round2 P2-a：9.5→10.5pt
    NODES.append((name, x, y))


def harrow(kind, xa, xb, y):
    """水平箭头：kind= lr 左→右（画在左节点右缘与右节点左缘之间）/ rl 右→左。"""
    if kind == "lr":
        p = FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>",
                            mutation_scale=11, color=INK, lw=1.2)
    else:
        p = FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>",
                            mutation_scale=11, color=INK, lw=1.2)
    ax.add_patch(p)
    ARROWS.append((kind, xa, y, xb, y))


def check_horiz(a):
    """round2 P1-a 端点断言（单支）：起/终点必须贴节点缘（容差 TOL）、方向与跨度受检。
    返回识别出的 (来源节点, 目标节点) 概念名——lr 起于左节点右缘、rl 起于右节点左缘。"""
    kind, xa, ya, xb, yb = a
    if kind == "lr":
        src = [n for n in NODES if abs(xa - (n[1] + W)) <= TOL and abs(ya - (n[2] + H / 2)) <= 0.03]
        dst = [n for n in NODES if abs(xb - n[1]) <= TOL and abs(yb - (n[2] + H / 2)) <= 0.03]
    else:
        src = [n for n in NODES if abs(xa - n[1]) <= TOL and abs(ya - (n[2] + H / 2)) <= 0.03]
        dst = [n for n in NODES if abs(xb - (n[1] + W)) <= TOL and abs(yb - (n[2] + H / 2)) <= 0.03]
    assert len(src) == 1 and len(dst) == 1, (
        f"{kind} 箭头端点不贴节点缘: ({xa:.4f},{ya:.3f})→({xb:.4f},{yb:.3f})"
        f" src={ [n[0] for n in src] } dst={ [n[0] for n in dst] }")
    span = xb - xa
    if kind == "lr":
        assert 0.005 < span <= GAP6, f"lr 方向/跨度错: span={span:.4f}（应在一个列隙 {GAP6:.3f} 内）"
    else:
        assert -0.005 > span >= -GAP6, f"rl 方向/跨度错: span={span:.4f}（应在一个列隙 {GAP6:.3f} 内）"
    return src[0][0], dst[0][0]


# ── 上链：第 2 章（一行六站）──
ax.text(0.5, 0.845, "一枚镜头 —— 第 2 章 Image formation（印刷 p.33 起）",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
ax.text(0.5, 0.795, "世界怎么变成一张图：几何 → 光度 → 相机",
        ha="center", va="center", fontsize=10.5, color="#5a6478")
for i, (name, _, page) in enumerate(CH2):
    x = X0 + i * (W + GAP6)
    node(x, 0.615, name, page, CH2C)
    if i:
        harrow("lr", x - GAP6 + PAD, x - PAD, 0.695)

# ── 下链：第 3 章（蛇形两行：6 站左→右，5 站右→左）──
ax.text(0.5, CH3_TITLE_Y, "一张图 —— 第 3 章 Image processing（印刷 p.107 起）",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
for i, (name, _, page) in enumerate(CH3[:6]):
    x = X0 + i * (W + GAP6)
    node(x, ROW1, name, page, CH3C)
    if i:
        harrow("lr", x - GAP6 + PAD, x - PAD, ROW1 + H / 2)
fx = X0 + 5 * (W + GAP6) + W / 2                   # 列 6 中线（二值与形态/傅里叶变换共用）
ARROWS.append(("td", fx, ROW1 - 0.008, fx, ROW2 + H + 0.008))
ax.add_patch(FancyArrowPatch((fx, ROW1 - 0.008), (fx, ROW2 + H + 0.008),
                             arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.2))
for j, (name, _, page) in enumerate(CH3[6:]):      # 行2 自右向左流
    col = 5 - j                                    # 傅里叶落在第 6 列正下方
    x = X0 + col * (W + GAP6)
    node(x, ROW2, name, page, CH3C)
    if j:                                          # 起于右邻站左缘、止于本站右缘
        harrow("rl", x + W + GAP6 - PAD, x + W + PAD, ROW2 + H / 2)

# ── 桥（虚线肘线）：成像链终点「相机与色彩」→ 处理链起点「点操作」——相机交出的像素阵列=点操作的输入 ──
# round2 P1-b：arc3 弧在首末位差下上鼓穿 Ch2 节点、下探穿 row1 节点（解析+断言证实），改走
# 两链之间空带（row1 顶+pad 0.526 之上、Ch3 标题底 0.567 之下）的肘线：竖→横→箭头下指。
CORRIDOR_Y = 0.545
bx_x = X0 + 5 * (W + GAP6) + W / 2                  # 相机与色彩 中线（=列 6 中线）
ARROWS.append(("bridge_v", bx_x, 0.615 - 0.008, bx_x, CORRIDOR_Y))
ARROWS.append(("bridge_h", bx_x, CORRIDOR_Y, X0 + W / 2, CORRIDOR_Y))
ARROWS.append(("bridge_a", X0 + W / 2, CORRIDOR_Y, X0 + W / 2, ROW1 + H + 0.006))
ax.plot([bx_x, bx_x], [0.607, CORRIDOR_Y], ls="--", color=BRIDGEC, lw=1.4)
ax.plot([bx_x, X0 + W / 2], [CORRIDOR_Y, CORRIDOR_Y], ls="--", color=BRIDGEC, lw=1.4)
ax.add_patch(FancyArrowPatch((X0 + W / 2, CORRIDOR_Y), (X0 + W / 2, ROW1 + H + 0.006),
                             arrowstyle="-|>", mutation_scale=13, color=BRIDGEC, lw=1.4,
                             linestyle="--"))
# 标注居中压在水平段上，纸底小圆角盒掩罩身后虚线（保证可读）
bridge_label = ax.text(0.5, CORRIDOR_Y, "桥：像素阵列", ha="center", va="center", fontsize=10,
                       fontweight="bold", color=BRIDGEC,
                       bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec=BRIDGEC, lw=0.8))


def check_special():
    kinds = [a[0] for a in ARROWS]
    assert kinds.count("lr") == 10 and kinds.count("rl") == 4, f"水平箭头数错: {kinds}"
    # 流拓扑断言：识别出的 (来源→目标) 序列必须与两链+折返的预期流完全一致（按 ARROWS 顺序）
    expected = [(CH2[i - 1][0], CH2[i][0]) for i in range(1, 6)]
    expected += [(CH3[i - 1][0], CH3[i][0]) for i in range(1, 6)]
    expected += [(CH3[6 + j - 1][0], CH3[6 + j][0]) for j in range(1, 5)]
    got = []
    for a in ARROWS:
        if a[0] in ("lr", "rl"):
            got.append(check_horiz(a))
    assert got == expected, "流拓扑错:\n got=" + str(got) + "\n exp=" + str(expected)
    td = [a for a in ARROWS if a[0] == "td"][0]
    _, xa, ya, xb, yb = td
    assert abs(xa - xb) < 1e-12, "折返箭头不竖直"
    assert abs(xa - fx) <= TOL, "折返箭头不在列 6 中线"
    assert abs(ya - (ROW1 - 0.008)) <= TOL and abs(yb - (ROW2 + H + 0.008)) <= TOL, "折返箭头端点离节点缘过远"
    bd = {}
    for a in ARROWS:
        if a[0].startswith("bridge"):
            bd[a[0]] = a
    assert set(bd) == {"bridge_v", "bridge_h", "bridge_a"}, f"桥段缺失: {set(bd)}"
    _, xa, ya, xb, yb = bd["bridge_v"]
    src = [n for n in NODES if n[0] == "相机与色彩"][0]
    dst = [n for n in NODES if n[0] == "点操作"][0]
    assert abs(xa - xb) < 1e-12 and abs(xa - (src[1] + W / 2)) <= TOL, "桥竖段不在相机与色彩中线"
    assert abs(ya - (src[2] - 0.008)) <= 0.002, "桥竖段起点不贴节点下缘"
    assert abs(yb - CORRIDOR_Y) < 1e-12, "桥竖段终点不在空带中线"
    _, xa, ya, xb, yb = bd["bridge_h"]
    assert abs(ya - CORRIDOR_Y) < 1e-12 and abs(yb - CORRIDOR_Y) < 1e-12, "桥水平段不在空带中线"
    assert ROW1 + H + PAD + 0.002 < CORRIDOR_Y < CH3_TITLE_Y - 0.016 - 0.002, "空带不够容纳桥水平段"
    assert dst[1] <= xb <= dst[1] + W, "桥水平段终点不在点操作跨度内"
    _, xa, ya, xb, yb = bd["bridge_a"]
    assert abs(xa - xb) < 1e-12 and dst[1] <= xa <= dst[1] + W, "桥箭头不在点操作跨度内"
    assert abs(ya - CORRIDOR_Y) <= 0.002 and 0 < yb - (dst[2] + H) <= 0.01, "桥箭头不落在点操作顶部上方"
    # 标注墨迹整段落入空带（含纸底盒不碰标题/节点）
    assert CORRIDOR_Y + 0.0124 < CH3_TITLE_Y - 0.016 and CORRIDOR_Y - 0.0124 > ROW1 + H + PAD, "桥标注墨迹出空带"


check_special()

# ── 标题与注脚（红线 5：无内部记号；红线 2：程序生成声明）──
ax.text(0.5, 0.965, "一枚镜头一张图：第 2、3 章全景串联（讲者复盘台账）",
        ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
ax.text(0.5, 0.905, "桥：成像的每一步都决定图的模样——成像链的终点（2.3 节 相机与色彩）就是处理链的起点（3.1 节 点操作）",
        ha="center", va="center", fontsize=11, color=BRIDGEC)
ax.text(0.99, 0.028, "程序生成，本节原书无图表 ｜ 节点与页锚=复盘台账（17 行断言一致）",
        ha="right", va="center", fontsize=12, color="#8a94a8")

# ── round3 P2-3：真实渲染 extent 断言（含 bbox pad 实测，不由常数自比）──
# draw 后取标注盒（含 bbox patch）与 17 节点框（含描边）的窗口 extent，转回数据坐标：
# ① 与任一节点框相交即 FAIL；② x 重叠的节点（桥标注正下方的线性滤波/可分离滤波），
#    标注盒下沿到框上沿的净空折算显示像素须 ≥2px。脚注 extent 一并打印备查（P2-1）。
fig.canvas.draw()
_inv = ax.transData.inverted()
_rend = fig.canvas.get_renderer()


def _dbox(artist):
    bb = artist.get_window_extent(renderer=_rend)
    (x0, y0), (x1, y1) = _inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


_lb = _dbox(bridge_label)
_fn = _dbox(ax.texts[-1])                              # 脚注（刚绘制的最后一条 text）
print(f"[extent] 桥标注盒数据 bbox=({_lb[0]:.4f},{_lb[1]:.4f})-({_lb[2]:.4f},{_lb[3]:.4f})；"
      f"脚注 bbox=({_fn[0]:.4f},{_fn[1]:.4f})-({_fn[2]:.4f},{_fn[3]:.4f})")
for (name, _, ny), patch in zip(NODES, PATCHES):
    nb = _dbox(patch)
    _sep = (_lb[2] < nb[0] or nb[2] < _lb[0] or _lb[3] < nb[1] or nb[3] < _lb[1])
    assert _sep, f"桥标注盒与节点框 {name} 相交（真实 extent 含 bbox pad）"
    if nb[2] > _lb[0] and nb[0] < _lb[2]:              # x 方向重叠——量 y 净空（节点在下方取标注下沿−框上沿，在上方取框下沿−标注上沿）
        gap_px = max(_lb[1] - nb[3], nb[1] - _lb[3]) * 560   # 图高 560px
        assert gap_px >= 2.0, f"桥标注盒与节点 {name} 净空 {gap_px:.1f}px < 2px"
        print(f"[extent] 标注盒↔{name} y 净空 {gap_px:.1f}px ≥2")

if "--negative-test" in sys.argv:
    # round3 P2-2 负例：round1 旧坐标公式原样注入 ARROWS[len-5]=ARROWS[13]
    # （行 2 rl 第 3 支槽位、预期流=金字塔→插值与抽取）。该坐标端点能贴上节点缘
    # （gap 3|4 合法），只有流拓扑断言（got==expected 逐一比较）能拦截——正是
    # round1 漏网机理。FAIL 输出完整 got/exp 差异（失配下标+配对），不截断。
    x = X0 + 4 * (W + GAP6)
    ARROWS[len(ARROWS) - 5] = ("rl", x - PAD, ROW2 + H / 2, x - GAP6 + PAD, ROW2 + H / 2)
    try:
        check_special()
    except AssertionError as e:
        print("NEGATIVE-TEST-OK: round1 旧坐标被端点+流拓扑断言拦截，完整差异如下")
        print(str(e))
        sys.exit(0)
    print("NEGATIVE-TEST-FAIL: 旧坐标竟通过断言", file=sys.stderr)
    sys.exit(1)

FIG_DIR.mkdir(parents=True, exist_ok=True)
out = FIG_DIR / "e035_recap_map.png"
fig.savefig(out, facecolor=PAPER)
print(f"[ok] 箭头断言过（lr10+rl4+td1+桥三段，流拓扑 14 配对全对，桥肘线走空带）；{out}")
