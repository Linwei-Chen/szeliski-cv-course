#!/usr/bin/env python3
"""
E026 · 二值图像处理：形态学、距离变换与连通域（§3.3.3）

范围：§3.3.3 Binary image processing（印刷 pp.138–142 上）：节引言与阈值化
（式 3.44 θ(f,t)=1 if f≥t else 0，p.138）＋Figure 3.22（p.138，二值形态学六
格，结构元素 5×5 方形；图注「The effects of majority are a subtle rounding of
sharp corners. Opening fails to eliminate the dot, as it is not wide enough.」）
＋Morphology（p.138 底–139：原书统一框架=先用结构元素卷积再对计数结果阈值化，
式 3.45 c=f⊗s、S=结构元素像素数；五操作 dilation θ(c,1)/erosion θ(c,S)/
majority θ(c,S/2)/opening dilate(erode(f,s),s)/closing erode(dilate(f,s),s)；
「dilation grows (thickens) objects … erosion shrinks (thins) them」）
＋Distance transforms（pp.139–141：两趟光栅算法；式 3.46 d1=|k|+|l|、式
3.47 d2=√(k²+l²)、式 3.48 D(i,j)=min_{k,l:b(k,l)=0} d(i−k,j−l)=到最近背景
像素的距离；Figure 3.23 p.140 前向（北/西）+后向（南/东）光栅扫描；「just
keeping the minimum scalar distance … is not sufficient. Instead, a
vector-valued distance … must be kept and compared using the squared distance
(hypotenuse) rule」；grassfire/chamfer 别名；脊=skeleton/MAT；signed distance
transform=「compute the distance transforms for both the original binary image
and its complement and to negate one of them before combining」）
＋Connected components（pp.141–142 顶：N4/N8 邻接；面积/周长/质心/二阶矩式
3.49，「from which the major and minor axis orientation and lengths can be
computed using eigenvalue analysis」）。切点=§3.4 Fourier transforms 标题
（p.142，归 E027）。

数值实验（23 条 = 16 数据 + 4 常数核对 C1–C4 + #S 同步 1 项（5 张图）+ #D 区分力 2 项；
注入类判据由 #D 承担不在正式断言复写——E019 门A round3 新P1-2 惯例）：
  #1  形态学统一框架与五操作（p.138 底–139 式 3.45 与五操作定义；4 条）：
      统一框架（结构元素计数 c=f⊗s + 阈值化 θ）与集合定义（结构元素平移的
      并/交）逐位一致（dilation/erosion 各一）；opening 去除小于结构元素的
      亮特征、closing 填掉暗小孔（图注「Opening fails to eliminate the dot」
      的极性机理：对极性不符的特征 opening 无效——暗点 opening 后保留）；
      majority 圆化尖角（直角凸角像素被删、直边像素保留）+dilation 不减、
      erosion 不增（Figure 3.22「dilation grows / erosion shrinks」语义）
  #2  距离变换定义与两趟光栅（式 3.46–3.48 pp.139–140+Figure 3.23 语义；
      4 条）：city-block 两趟光栅（前向=1+北/西最小、后向=并入 1+南/东）与
      暴力定义逐位一致（随机 20×20 网格）；前向顺序忠实——单列边界网格下
      前向只传播北/西（构造用例）；向量版两趟欧氏（存 (Δx,Δy)、按平方距离
      「hypotenuse」规则比较）与暴力欧氏逐位一致；标量两趟欧氏存在高估
      （≥1 像素 D_scalar>D_true——原书「not sufficient」语义）
  #3  两度量对照（式 3.46/3.47；2 条）：同一点 (6,8)：d1=14、d2=10（比值
      1.4；3-4-5 勾股放大）；同阈值下 d2 圆等值线在 45° 对角方向比 d1 菱形
      更远（对角点 (a,a)：d1=2a vs d2=a√2，d2 更小→圆更「胖」）
  #4  连通域与区域统计（N4/N8+式 3.49 pp.141–142；4 条）：斜对角相连两块
      N8 连通数 1、N4 连通数 2；手工网格的面积/周长/质心逐项断言；式 3.49
      M 矩阵逐项复现（3 点手工例）；特征值分析主轴方向——斜置条带主特征
      向量角度=条带方向（±1° 容差）
  #5  signed distance transform（p.140–141「negate one of them before
      combining」；2 条）：signed D 在含边界行剖面连续正→0→负（0 恰在边界
      像素）；city-block 度量下相邻像素 signed 差 ≤1（1-Lipschitz 平滑性）
  C1–C4  常数：式 3.44 边界语义（f=t 记 1）；d1(3,4)=7；d2(3,4)=5（勾股数）；
      majority 阈值整数化（S=25：θ(c, 12.5) ⇔ c≥13）
  #S  同步：5 张程序生成图写入 code/figures/ 与 slides/figures/（md5 一致）：
      e026_morph_ops（五操作）/e026_dt_raster（两趟光栅）/
      e026_dt_iso（等值线对照）/e026_cc_stats（连通域与统计）/
      e026_signed_profile（signed 一维剖面）——均标「原书图未使用」
  #D  区分力（不计数）：D-1 用 erosion 冒充 dilation → #1 拦截；D-2 标量
      两趟欧氏冒充精确欧氏 → #2 高估断言拦截；D-3 两趟方向与邻居不配对 → #2 拦截

输出：stdout 存档 code/outputs/e026_binary_run1.txt
"""

import hashlib
import pathlib
import sys

import numpy as np

# ---------- 基础设施 ----------
FAILS = []
COUNT = {"n": 0}


def chk(label, cond, detail=""):
    COUNT["n"] += 1
    ok = bool(cond)
    if not ok:
        FAILS.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


rng = np.random.default_rng(2026042)


def b_grid_ref():
    """D-3 用的固定随机网格（独立种子，保证有前景且背景非空）。"""
    r = np.random.default_rng(2026042)
    b = (r.random((20, 20)) < 0.12).astype(np.int64)
    b[0, :] = 0
    return b

# ---------- 原书算子（忠实式 3.44/3.45 与五操作定义） ----------

def theta(f, t):
    """式 3.44：θ(f,t) = 1 if f ≥ t, else 0（f=t 边界记 1）。"""
    return (np.asarray(f) >= t).astype(np.int64)


def conv_count(f, s):
    """式 3.45：c = f ⊗ s —— 结构元素扫过时窗口内 1 的整数计数（same 边界，
    越界按 0 计，与二值图「结构元素完整落在图内」的卷积计数一致）。"""
    H, W = f.shape
    kh, kw = s.shape
    r_h, r_w = kh // 2, kw // 2
    padded = np.zeros((H + 2 * r_h, W + 2 * r_w), dtype=np.int64)
    padded[r_h:r_h + H, r_w:r_w + W] = f
    c = np.zeros((H, W), dtype=np.int64)
    for u in range(kh):
        for v in range(kw):
            c += s[u, v] * padded[u:u + H, v:v + W]
    return c


def dilate(f, s):
    return theta(conv_count(f, s), 1)


def erode(f, s):
    S = int(s.sum())
    return theta(conv_count(f, s), S)


def majority(f, s):
    S = int(s.sum())
    return theta(conv_count(f, s), S / 2)


def opening(f, s):
    return dilate(erode(f, s), s)


def closing(f, s):
    return erode(dilate(f, s), s)


def dilate_set(f, s):
    """集合定义：dilation = 结构元素平移位置的并（原点对齐）。"""
    H, W = f.shape
    kh, kw = s.shape
    out = np.zeros_like(f)
    offs = [(u - kh // 2, v - kw // 2) for u in range(kh) for v in range(kw) if s[u, v]]
    for du, dv in offs:
        shifted = np.zeros_like(f)
        src = f[max(0, -du):H - max(0, du), max(0, -dv):W - max(0, dv)]
        dst = shifted[max(0, du):H - max(0, -du), max(0, dv):W - max(0, -dv)]
        dst[:] = src
        out |= shifted
    return out


def erode_set(f, s):
    """集合定义：erosion = 结构元素平移位置的交（负原点对齐）。"""
    H, W = f.shape
    kh, kw = s.shape
    out = np.ones_like(f)
    offs = [(u - kh // 2, v - kw // 2) for u in range(kh) for v in range(kw) if s[u, v]]
    for du, dv in offs:
        shifted = np.zeros_like(f)
        src = f[max(0, du):H - max(0, -du), max(0, dv):W - max(0, -dv)]
        dst = shifted[max(0, -du):H - max(0, du), max(0, -dv):W - max(0, dv)]
        dst[:] = src
        out &= shifted
    return out


# ---------- 距离变换（式 3.46–3.48 与两趟光栅） ----------

def dt_brute(b, metric):
    """式 3.48 暴力定义：D(i,j)=min_{k,l:b(k,l)=0} d(i−k,j−l)。"""
    H, W = b.shape
    bg = np.argwhere(b == 0)
    if len(bg) == 0:
        return np.full((H, W), np.inf)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)
    out = np.empty(len(pts))
    for n, p in enumerate(pts):
        d = np.abs(bg[:, 0] - p[0]) + np.abs(bg[:, 1] - p[1]) if metric == "d1" \
            else np.hypot(bg[:, 0] - p[0], bg[:, 1] - p[1])
        out[n] = d.min()
    return out.reshape(H, W)


def dt_two_pass_city(b, return_fwd=False):
    """Figure 3.23 语义：city-block 两趟光栅——前向 1+min(北,西)，后向再并入
    1+min(南,东)，与当前值取 min（原书「the minimum is both over the current
    value D and 1 + the distance of the south and east neighbors」）。"""
    H, W = b.shape
    INF = H * W + 1
    D = np.where(b == 0, 0.0, INF)
    for i in range(H):            # 自顶向下（forward）
        for j in range(W):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i > 0:
                cand.append(D[i - 1, j] + 1)   # 北
            if j > 0:
                cand.append(D[i, j - 1] + 1)   # 西
            D[i, j] = min(cand)
    if return_fwd:
        return D.copy()
    for i in range(H - 1, -1, -1):  # 自底向上（backward）
        for j in range(W - 1, -1, -1):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i + 1 < H:
                cand.append(D[i + 1, j] + 1)   # 南
            if j + 1 < W:
                cand.append(D[i, j + 1] + 1)   # 东
            D[i, j] = min(cand)
    return D


def dt_two_pass_scalar_euclid(b):
    """标量两趟欧氏（原书「not sufficient」所指的错误省事版）。"""
    H, W = b.shape
    INF = (H + W) ** 2
    D = np.where(b == 0, 0.0, float(INF))
    for i in range(H):
        for j in range(W):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i > 0:
                cand.append(D[i - 1, j] + 1)
            if j > 0:
                cand.append(D[i, j - 1] + 1)
            D[i, j] = min(cand)
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i + 1 < H:
                cand.append(D[i + 1, j] + 1)
            if j + 1 < W:
                cand.append(D[i, j + 1] + 1)
            D[i, j] = min(cand)
    return D


def dt_two_pass_scalar_euclid8(b):
    """8 邻域标量两趟欧氏（对角步长 √2）——「只存最小标量距离」的最好情况版。
    原书 p.140「just keeping the minimum scalar distance … is not sufficient」：
    即便步长补了对角，链式标量在非轴向非对角方向仍高估。"""
    H, W = b.shape
    INF = float((H + W) * 2)
    D = np.where(b == 0, 0.0, INF)
    r2 = np.sqrt(2.0)
    for i in range(H):                      # 前向：北、西北、西、东北
        for j in range(W):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i > 0:
                cand.append(D[i - 1, j] + 1)
                if j > 0:
                    cand.append(D[i - 1, j - 1] + r2)
                if j + 1 < W:
                    cand.append(D[i - 1, j + 1] + r2)
            if j > 0:
                cand.append(D[i, j - 1] + 1)
            D[i, j] = min(cand)
    for i in range(H - 1, -1, -1):          # 后向：南、东南、东、西南
        for j in range(W - 1, -1, -1):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i + 1 < H:
                cand.append(D[i + 1, j] + 1)
                if j + 1 < W:
                    cand.append(D[i + 1, j + 1] + r2)
                if j > 0:
                    cand.append(D[i + 1, j - 1] + r2)
            if j + 1 < W:
                cand.append(D[i, j + 1] + 1)
            D[i, j] = min(cand)
    return D


def dt_two_pass_city_broken(b):
    """注入错误：两趟方向与邻居不配套（前向取北+东、后向取南+西）。
    前向扫到 (i,j) 时东邻未扫（INF）、后向扫到 (i,j) 时西邻未扫（INF），
    退化为只能沿列传播——用于 D-3 验证「方向配错，结果就不对」。"""
    H, W = b.shape
    INF = float((H + W) ** 2)
    D = np.where(b == 0, 0.0, INF)
    for i in range(H):                      # 前向，邻居取「北、东」
        for j in range(W):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i > 0:
                cand.append(D[i - 1, j] + 1)
            if j + 1 < W:                   # 东邻尚未扫描（INF）
                cand.append(D[i, j + 1] + 1)
            D[i, j] = min(cand)
    for i in range(H - 1, -1, -1):          # 后向，邻居取「南、西」
        for j in range(W - 1, -1, -1):
            if b[i, j] == 0:
                continue
            cand = [D[i, j]]
            if i + 1 < H:
                cand.append(D[i + 1, j] + 1)
            if j > 0:                       # 西邻尚未扫描（INF）
                cand.append(D[i, j - 1] + 1)
            D[i, j] = min(cand)
    return D


def dt_two_pass_vector_euclid(b):
    """向量版两趟欧氏：每个像素存到边界偏移 (Δi,Δj)，按平方距离（斜边）规则
    比较——原书「a vector-valued distance … must be kept and compared using
    the squared distance (hypotenuse) rule」。"""
    H, W = b.shape
    BIG = 10 ** 9
    yy0, xx0 = np.meshgrid(np.arange(b.shape[0]), np.arange(b.shape[1]), indexing="ij")
    vy = np.where(b == 0, yy0, BIG).astype(np.int64)   # 背景像素指针=自身（边界集合）
    vx = np.where(b == 0, xx0, BIG).astype(np.int64)

    def better(i, j, cy, cx):
        dy, dx = cy - i, cx - j
        return dy * dy + dx * dx, cy, cx

    for i in range(H):
        for j in range(W):
            if b[i, j] == 0:
                continue
            best = (BIG ** 2, vy[i, j], vx[i, j])
            for cy, cx in [(i - 1, j), (i, j - 1)]:
                if 0 <= cy < H and 0 <= cx < W:
                    d2v, ry, rx = better(i, j, vy[cy, cx], vx[cy, cx])
                    if d2v < best[0]:
                        best = (d2v, ry, rx)
            vy[i, j], vx[i, j] = best[1], best[2]
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            if b[i, j] == 0:
                continue
            best = (vy[i, j] - i) ** 2 + (vx[i, j] - j) ** 2, vy[i, j], vx[i, j]
            for cy, cx in [(i + 1, j), (i, j + 1)]:
                if 0 <= cy < H and 0 <= cx < W:
                    d2v, ry, rx = better(i, j, vy[cy, cx], vx[cy, cx])
                    if d2v < best[0]:
                        best = (d2v, ry, rx)
            vy[i, j], vx[i, j] = best[1], best[2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return np.hypot(vy - yy, vx - xx)


# ---------- 连通域（N4/N8 与区域统计） ----------

def connected_components(b, n8=False):
    """连通域标注：N4（上下左右）或 N8（加对角）。返回 (labels, n)。"""
    H, W = b.shape
    labels = np.zeros((H, W), dtype=np.int64)
    if n8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n = 0
    for i in range(H):
        for j in range(W):
            if b[i, j] and labels[i, j] == 0:
                n += 1
                stack = [(i, j)]
                labels[i, j] = n
                while stack:
                    y, x = stack.pop()
                    for dy, dx in nbrs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and b[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = n
                            stack.append((ny, nx))
    return labels, n


def region_stats(f, labels, lab):
    """p.141 区域统计：面积/周长/质心/二阶矩（式 3.49）。坐标 (x,y) 按
    原书「centroid (average x and y values)」用列 x、行 y。"""
    ys, xs = np.nonzero((labels == lab) & (f == 1))
    area = len(ys)
    xbar, ybar = xs.mean(), ys.mean()
    mask = labels == lab
    boundary = np.zeros_like(mask)
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dy, j + dx
                    if not (0 <= ni < H and 0 <= nj < W) or not mask[ni, nj]:
                        boundary[i, j] = True
                        break
    perim = int(boundary.sum())
    M = np.array([
        [((xs - xbar) ** 2).sum(), ((xs - xbar) * (ys - ybar)).sum()],
        [((xs - xbar) * (ys - ybar)).sum(), ((ys - ybar) ** 2).sum()],
    ])
    return {"area": area, "perimeter": perim, "centroid": (xbar, ybar), "M": M}


# ---------- 实验 #1：统一框架与五操作 ----------

def exp1():
    print("#1 形态学统一框架与五操作（式 3.45+五操作定义，p.138 底–139）")
    s5 = np.ones((5, 5), dtype=np.int64)
    f = np.zeros((16, 16), dtype=np.int64)
    f[2:10, 2:10] = 1          # 大方块（含直角尖角）
    f[3, 3] = 0                # 内部暗孔（closing 的目标）
    f[12, 6] = 1               # 孤立亮小点（opening 的目标）
    f[11:13, 2:4] = 1          # 亮小块（2×2，opening 去掉）
    f[11:16, 9:14] = 1         # 大亮块（5×5，opening 原样保留——原书 dot 去不掉的机理面）

    dil_u, dil_s = dilate(f, s5), dilate_set(f, s5)
    ero_u, ero_s = erode(f, s5), erode_set(f, s5)
    chk("#1-1 统一框架 dilation θ(c,1)=集合定义并集（逐位）", np.array_equal(dil_u, dil_s),
        f"差异像素 {int((dil_u != dil_s).sum())}/{dil_u.size}")
    chk("#1-2 统一框架 erosion θ(c,S)=集合定义交集（逐位）", np.array_equal(ero_u, ero_s),
        f"差异像素 {int((ero_u != ero_s).sum())}/{ero_u.size}")

    op, cl = opening(f, s5), closing(f, s5)
    dot_before = f[12, 6]
    dot_after = op[12, 6]
    hole_before = int(1 - f[3, 3])
    hole_after = int(1 - cl[3, 3])
    blk_alive = int(op[11:16, 9:14].sum())
    chk("#1-3 opening 去亮小点/小块、大亮块原样保留 / closing 填暗小孔（图注极性机理）",
        dot_before == 1 and dot_after == 0 and hole_before == 1 and hole_after == 0
        and blk_alive == 25,
        f"亮点 opening 后残留亮像素={dot_after}（应 0，即去亮）；暗孔 closing 后残留暗像素={hole_after}（应 0，即填平）；"
        f"5×5 大亮块 opening 后存活 {blk_alive}/25（dot 至少容得下结构元素就啃不动——图注 fail 句机理面）")

    # 图注「Opening fails to eliminate the dot」机理面：暗点经 opening 保留
    dark_dot = f.copy()
    dark_dot[6:8, 6:8] = 0    # 大方块内的 2×2 暗点
    op_dark = opening(dark_dot, s5)
    dark_gone = int((dark_dot[6:8, 6:8] == 0).sum() - (op_dark[6:8, 6:8] == 0).sum())
    dd_filled = int(closing(dark_dot, s5)[6:8, 6:8].sum())
    chk("#1-4 closing 填平 2×2 暗孔（f011 口播「2×2 的暗孔一闭就平」读数）",
        dd_filled == 4 and dark_gone == 0,
        f"2×2 暗孔 f[6:8,6:8]=0 → closing 后孔区全 1（填充读数 {dd_filled}/4）；"
        f"对照：同暗点 opening 后消失 {dark_gone} 像素（应 0，即保留——开闭极性对照面：closing 填暗孔、opening 不去暗点）")
    maj = majority(f, s5)
    corner = maj[2, 2]        # 直角凸角
    edge = maj[2, 5]          # 直边中点
    grow_ok = dil_u.sum() >= f.sum()
    shrink_ok = ero_u.sum() <= f.sum()
    chk("#1-5 majority 圆化尖角（凸角删/直边留）+grows/shrinks 单调",
        corner == 0 and edge == 1 and grow_ok and shrink_ok and dark_gone == 0,
        f"凸角 majority={corner}（应 0）、直边={edge}（应 1）、暗点经 opening 后消失 {dark_gone} 像素（应 0，即保留）")


# ---------- 实验 #2：距离变换两趟光栅 ----------

def exp2():
    print("#2 距离变换定义与两趟光栅（式 3.46–3.48，pp.139–140；Figure 3.23 语义）")
    b = (rng.random((20, 20)) < 0.12).astype(np.int64)
    b[0, :] = 0                # 保证背景非空
    ref1 = dt_brute(b, "d1")
    got1 = dt_two_pass_city(b)
    chk("#2-1 city-block 两趟光栅=暴力定义（逐位，随机网格）",
        np.array_equal(ref1, got1),
        f"网格 {b.shape[0]}×{b.shape[1]}，两趟 vs 暴力差异像素 "
        f"{int((ref1 != got1).sum())}/{b.size}")

    src = np.ones((7, 7), dtype=np.int64)
    src[0, 0] = 0              # 唯一背景源在 (0,0)（前向方向的起点）
    fwd_only = dt_two_pass_city(src, return_fwd=True)
    yy, xx = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    chk("#2-2 前向顺序忠实（北/西递推——唯一背景 (0,0)，前向即得精确曼哈顿 i+j）",
        np.array_equal(fwd_only, (yy + xx).astype(float)),
        f"前向 D[2,3]={fwd_only[2, 3]}（应 5）")

    ref2 = dt_brute(b, "d2")
    vec = dt_two_pass_vector_euclid(b)
    chk("#2-3 向量版两趟欧氏=暴力欧氏（随机 20×20 一例，斜边规则）",
        np.allclose(ref2, vec, atol=1e-9),
        f"网格 {b.shape[0]}×{b.shape[1]}（随机），最大差 {np.abs(ref2 - vec).max():.3e}")

    diag = np.ones((11, 11), dtype=np.int64)
    diag[0, 0] = 0             # 唯一背景 (0,0)：真欧氏 √(i²+j²)
    scal8 = dt_two_pass_scalar_euclid8(diag)
    ref_d = dt_brute(diag, "d2")
    over = int((scal8 > ref_d + 1e-9).sum())
    # (5,2)：真欧氏 √29≈5.385；8 邻域标量最优链=3 轴向+2 对角=3+2√2≈5.828（非轴非对角必高估）
    chain = 3 + 2 * np.sqrt(2)
    chk("#2-4 标量两趟欧氏高估（原书「not sufficient」：对角步长已补仍不够）",
        over >= 1 and abs(scal8[5, 2] - chain) < 1e-9 and abs(ref_d[5, 2] - np.sqrt(29)) < 1e-9
        and scal8[5, 2] > ref_d[5, 2],
        f"11×11 唯一背景 (0,0) 网格高估像素数 {over}；"
        f"D_scalar8(5,2)={scal8[5, 2]:.3f}（=3+2√2≈{chain:.3f}）"
        f" vs 真欧氏 {ref_d[5, 2]:.3f}（=√29≈{np.sqrt(29):.3f}）")

    # f020 口播「在这张图上两趟就算对」的直接支撑：与 #2-4 同一张 diag 网格
    vec_diag = dt_two_pass_vector_euclid(diag)
    chk("#2-5 向量版两趟欧氏在唯一背景 (0,0) 网格=暴力（f020「在这张图上」支撑）",
        np.allclose(ref_d, vec_diag, atol=1e-9),
        f"11×11 唯一背景 (0,0) 网格（与 #2-4 同一张），最大差 "
        f"{np.abs(ref_d - vec_diag).max():.3e}")


# ---------- 实验 #3：两度量对照 ----------

def exp3():
    print("#3 两度量对照（式 3.46/3.47）")
    d1_68 = abs(6) + abs(8)
    d2_68 = np.hypot(6, 8)
    chk("#3-1 点 (6,8)：d1=14、d2=10（勾股放大，比值 1.4）",
        d1_68 == 14 and d2_68 == 10, f"d1={d1_68} d2={d2_68}")
    a = 5
    d1_diag = a + a
    d2_diag = np.hypot(a, a)
    chk("#3-2 对角方向：d1=2a vs d2=a√2——圆等值线更「胖」",
        d1_diag == 10 and abs(d2_diag - a * np.sqrt(2)) < 1e-12 and d2_diag < d1_diag,
        f"d1={d1_diag} d2={d2_diag:.4f}")


# ---------- 实验 #4：连通域与区域统计 ----------

def exp4():
    print("#4 连通域与区域统计（N4/N8+式 3.49，pp.141–142）")
    g = np.zeros((6, 6), dtype=np.int64)
    g[1:3, 1:3] = 1            # 块 A
    g[3:5, 3:5] = 1            # 块 B（与 A 仅 (2,2)-(3,3) 对角相邻）
    _, n4 = connected_components(g, n8=False)
    _, n8 = connected_components(g, n8=True)
    chk("#4-1 斜对角相连：N8 连通数 1、N4 连通数 2", n8 == 1 and n4 == 2,
        f"N4={n4} N8={n8}")

    h = np.zeros((7, 9), dtype=np.int64)
    h[1, 1:6] = 1              # 水平条带 5 像素 (行1, 列1..5)
    h[4, 2:5] = 1              # 第二条带 3 像素 (行4, 列2..4)
    labels, n = connected_components(h, n8=False)
    st = region_stats(h, labels, 1)
    chk("#4-2 区域统计：面积/周长/质心（条带 1）",
        st["area"] == 5 and st["perimeter"] == 5
        and abs(st["centroid"][0] - 3.0) < 1e-9 and abs(st["centroid"][1] - 1.0) < 1e-9,
        f"area={st['area']} perim={st['perimeter']} "
        f"centroid=({st['centroid'][0]:.1f}, {st['centroid'][1]:.1f})")

    pts = [(0, 0), (2, 1), (1, 3)]     # (x,y) 三点手工例（非对角，测交叉项）
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    xb, yb = xs.mean(), ys.mean()
    M_hand = np.array([
        [((xs - xb) ** 2).sum(), ((xs - xb) * (ys - yb)).sum()],
        [((xs - xb) * (ys - yb)).sum(), ((ys - yb) ** 2).sum()],
    ])
    M_expect = np.array([[2.0, 1.0], [1.0, 14.0 / 3.0]])  # x̄=1,ȳ=4/3：Σ(x−x̄)²=2、Σ(x−x̄)(y−ȳ)=1、Σ(y−ȳ)²=42/9
    chk("#4-3 式 3.49 M 矩阵逐项（3 点手工例）",
        np.allclose(M_hand, M_expect), f"M=\n{M_hand}")

    band = np.zeros((20, 40), dtype=np.int64)
    for x in range(4, 36):
        y = int(round(0.5 * x)) + 2   # 斜率 0.5 条带（约 26.57°）
        band[y:y + 2, x] = 1
    lb, _ = connected_components(band, n8=True)
    stb = region_stats(band, lb, 1)
    w, v = np.linalg.eigh(stb["M"])
    main = v[:, -1]
    ang = np.degrees(np.arctan2(main[1], main[0])) % 180.0
    expect = np.degrees(np.arctan2(1, 2))        # atan(0.5)≈26.57°
    chk("#4-4 特征值分析主轴方向=条带方向（±1°）",
        abs(ang - expect) < 1.0, f"主轴 {ang:.2f}° vs 期望 {expect:.2f}°")


# ---------- 实验 #5：signed distance ----------

def exp5():
    print("#5 signed distance transform（p.140–141「negate one of them before combining」）")
    b = np.zeros((12, 12), dtype=np.int64)
    b[3:9, 3:9] = 1            # 6×6 前景方块
    d_in = dt_two_pass_city(b)          # 前景内到背景的距离
    d_out = dt_two_pass_city(1 - b)     # 补图（原书「its complement」）
    signed = d_in - d_out      # 原书最简合并（讲者选向：前景正、背景负）
    fg = b == 1
    chk("#5-1 符号场：前景全正（≥1）/背景全负（≤−1），最简合并无 0 值像素",
        (signed[fg] >= 1).all() and (signed[~fg] <= -1).all(),
        f"行5剖面 {signed[5, :].tolist()}（边界两侧 +1/−1，0 值不出现——两参考集各自含 0 层使然）")
    dy = np.abs(np.diff(signed, axis=0)); dx = np.abs(np.diff(signed, axis=1))
    same = []
    for Dm, axis in [(dy, 0), (dx, 1)]:
        s0 = fg[:-1, :] if axis == 0 else fg[:, :-1]
        s1 = fg[1:, :] if axis == 0 else fg[:, 1:]
        same.append(Dm[s0 == s1].max())
    cross = []
    for Dm, axis in [(dy, 0), (dx, 1)]:
        s0 = fg[:-1, :] if axis == 0 else fg[:, :-1]
        s1 = fg[1:, :] if axis == 0 else fg[:, 1:]
        cross.append(Dm[s0 != s1].max())
    chk("#5-2 平滑性：同侧相邻差 ≤1（各自 1-Lipschitz）、跨界恰为 2（1−(−1)）",
        max(same) <= 1.0 + 1e-9 and max(cross) == 2.0,
        f"同侧最大差 {max(same)}、跨界最大差 {max(cross)}")


# ---------- 常数核对 C1–C4 ----------

def constants():
    print("常数核对 C1–C4")
    chk("C1 式 3.44 边界语义（f=t 记 1）",
        theta(np.array([2.9, 3.0, 3.1]), 3.0).tolist() == [0, 1, 1],
        f"θ([2.9, 3.0, 3.1], t=3.0)={[int(x) for x in theta(np.array([2.9, 3.0, 3.1]), 3.0)]}"
        f"（边界值 f=t 记 1，应 [0, 1, 1]）")
    chk("C2 d1(3,4)=|3|+|4|=7", abs(3) + abs(4) == 7)
    chk("C3 d2(3,4)=√(3²+4²)=5（勾股数）", np.hypot(3, 4) == 5.0)
    s5 = np.ones((5, 5), dtype=np.int64)
    f = np.zeros((9, 9), dtype=np.int64)
    f[4, 4] = 1
    c12 = conv_count(f, s5)[4, 4]
    maj_t = theta(np.array([c12]), 25 / 2).tolist()
    chk("C4 majority 阈值整数化（S=25：θ(c,12.5) ⇔ c≥13）",
        c12 == 1 and maj_t == [0], f"c={c12}（单像素在 5×5 窗口计数 1，12.5 阈值下 majority=0）")


# ---------- 图（#S 同步） ----------

def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    figdir = pathlib.Path(__file__).resolve().parent / "figures"
    slide_dir = pathlib.Path(__file__).resolve().parent.parent / "slides" / "figures"
    figdir.mkdir(exist_ok=True)
    slide_dir.mkdir(exist_ok=True)
    made = []

    def save(fig, name):
        for d in (figdir, slide_dir):
            fig.savefig(d / name, dpi=150, bbox_inches="tight")
        made.append(name)
        plt.close(fig)

    def show_grid(ax, g, title, cmap="gray"):
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # 图1 五操作（Figure 3.22 语义，程序生成）
    s5 = np.ones((5, 5), dtype=np.int64)
    f = np.zeros((20, 20), dtype=np.int64)
    f[3:13, 3:13] = 1
    f[5, 5] = 0                # 内部暗孔
    f[16, 17] = 1              # 亮小点
    f[15:17, 2:4] = 1          # 亮小块
    f[14:19, 9:14] = 1         # 大亮块（5×5，opening 原样保留）
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 6.4))
    show_grid(ax[0, 0], f, "(a) 原二值图（大方块+暗孔+亮小点/块+大亮块）")
    show_grid(ax[0, 1], dilate(f, s5), "(b) dilation θ(c,1)：长粗")
    show_grid(ax[0, 2], erode(f, s5), "(c) erosion θ(c,S)：缩细")
    show_grid(ax[1, 0], majority(f, s5), "(d) majority θ(c,S/2)：圆化尖角")
    show_grid(ax[1, 1], opening(f, s5), "(e) opening：去小（点/2×2）留大（5×5 块）")
    show_grid(ax[1, 2], closing(f, s5), "(f) closing：填暗孔")
    fig.suptitle("二值形态学五操作（5×5 方形结构元素；Figure 3.22 语义，程序生成，原书图未使用）", fontsize=11)
    fig.tight_layout()
    save(fig, "e026_morph_ops.png")

    # 图2 两趟光栅（Figure 3.23 语义，程序生成）
    b = np.zeros((11, 11), dtype=np.int64)
    b[2:9, 2:9] = 1            # 中央 7×7 前景块：距离从边界向内 1..4 递增
    fwd = dt_two_pass_city(b)
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
    show_grid(ax[0], b, "(a) 原二值图（中央 7×7 前景块）")
    im = ax[1].imshow(np.where(b == 0, np.nan, fwd), cmap="viridis", interpolation="nearest")
    ax[1].set_title("(b) city-block 距离（两趟光栅）", fontsize=9)
    for (i, j), val in np.ndenumerate(fwd):
        if b[i, j]:
            ax[1].text(j, i, int(val), ha="center", va="center", fontsize=8, color="w")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].axis("off")
    ax[2].text(0.02, 0.86, "前向（自顶向下）：D ← min(D, 1+北, 1+西)", fontsize=10)
    ax[2].text(0.02, 0.60, "后向（自底向上）：D ← min(D, 1+南, 1+东)", fontsize=10)
    ax[2].text(0.02, 0.34, "与暴力定义逐位一致（实验 #2-1）", fontsize=10)
    ax[2].text(0.02, 0.08, "Figure 3.23 语义，程序生成，原书图未使用", fontsize=9, color="gray")
    fig.suptitle("两趟光栅距离变换（式 3.48 + Figure 3.23 语义）", fontsize=11)
    fig.tight_layout()
    save(fig, "e026_dt_raster.png")

    # 图3 等值线对照（式 3.46/3.47）
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
    yy, xx = np.meshgrid(np.linspace(-10, 10, 201), np.linspace(-10, 10, 201), indexing="ij")
    for a, met, ttl in [(ax[0], lambda y, x: np.abs(x) + np.abs(y), "d1 城市街区（式 3.46）等值线=菱形"),
                        (ax[1], lambda y, x: np.hypot(x, y), "d2 欧氏（式 3.47）等值线=圆")]:
        a.contour(xx[0], yy[:, 0], met(yy, xx), levels=[3, 5, 7, 9], colors="k", linewidths=1)
        a.contourf(xx[0], yy[:, 0], met(yy, xx), levels=20, cmap="Blues")
        a.plot(0, 0, "r+", ms=14)
        a.set_aspect("equal")
        a.set_title(ttl, fontsize=10)
    fig.suptitle("同源点两种度量：菱形 vs 圆（对角 45° 上 d2 更近——程序生成，原书图未使用）", fontsize=11)
    fig.tight_layout()
    save(fig, "e026_dt_iso.png")

    # 图4 连通域与统计（N4/N8+式 3.49）
    g = np.zeros((8, 8), dtype=np.int64)
    g[1:3, 1:3] = 1
    g[3:5, 3:5] = 1
    lab4, n4 = connected_components(g, n8=False)
    lab8, n8 = connected_components(g, n8=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for a, lb, nn, ttl in [(ax[0], lab4, n4, f"N4 邻接：{n4} 个连通域（分裂，各画质心）"),
                           (ax[1], lab8, n8, f"N8 邻接：{n8} 个连通域（对角并入，画主轴）")]:
        a.imshow(np.where(lb > 0, lb, np.nan), cmap="tab10", vmin=1, vmax=2, interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        for lab in range(1, nn + 1):
            ys, xs = np.nonzero(lb == lab)
            a.plot(xs.mean(), ys.mean(), "r*", ms=13)
            if nn == 1:   # 单一合并域才谈主轴（分裂的 2×2 方块各向同性无主轴）
                st = region_stats(g, lb, lab)
                w, v = np.linalg.eigh(st["M"])
                vec = v[:, -1] * 2.2
                a.plot(xs.mean() + vec[0] * np.array([-1.0, 1.0]), ys.mean() + vec[1] * np.array([-1.0, 1.0]), "r--", lw=1.2)
    fig.suptitle("N4 vs N8（★=质心，红虚线=主轴；程序生成，原书图未使用）", fontsize=13)
    fig.tight_layout()
    save(fig, "e026_cc_stats.png")

    # 图5 signed distance 一维剖面（p.140–141「negate one of them before combining」）
    b5 = np.zeros((12, 12), dtype=np.int64)
    b5[3:9, 3:9] = 1
    s5d = dt_two_pass_city(b5) - dt_two_pass_city(1 - b5)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs5 = np.arange(12)
    ax.step(xs5, s5d[5, :], where="mid", color="k", lw=1.8)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvspan(2.5, 8.5, color="gold", alpha=0.18)
    ax.text(5.5, -3.4, "前景（方块内部）", ha="center", fontsize=10)
    ax.text(1.0, 2.6, "背景", ha="center", fontsize=10)
    ax.set_xlabel("x（行 5 剖面）", fontsize=10)
    ax.set_ylabel("signed 距离", fontsize=10)
    ax.set_title("signed distance：前景正、背景负（原书合并做法边界两侧 ±1；程序生成，原书图未使用）", fontsize=10.5)
    fig.tight_layout()
    save(fig, "e026_signed_profile.png")

    # #S 同步断言
    ok = True
    for name in made:
        h1 = hashlib.md5((figdir / name).read_bytes()).hexdigest()
        h2 = hashlib.md5((slide_dir / name).read_bytes()).hexdigest()
        ok = ok and (h1 == h2)
    chk("#S 同步（5 张程序生成图，两处 md5 一致）", ok and len(made) == 5,
        f"生成 {len(made)} 张：{made}；md5 两处一致 "
        + " ".join(f"{n}:{hashlib.md5((figdir / n).read_bytes()).hexdigest()[:8]}" for n in made))


# ---------- 区分力注入（不计数） ----------

def 区分力():
    print("区分力自检（注入错误实现，验证断言拦截）")
    try:
        s5 = np.ones((5, 5), dtype=np.int64)
        f = np.zeros((16, 16), dtype=np.int64)
        f[2:10, 2:10] = 1
        f[12, 6] = 1
        fake_open = erode(dilate(f, s5), s5)   # 错：closing 当 opening 用
        dot_survives = fake_open[12, 6]
        real_open = opening(f, s5)
        chk("D-1 closing 冒充 opening → #1-3 拦截（真 opening 滤点、假的不滤）",
            real_open[12, 6] == 0 and dot_survives == 1,
            f"假 opening 亮点={dot_survives}（应 1 被拦截）")
        diag = np.ones((11, 11), dtype=np.int64)
        diag[0, 0] = 0
        ref2 = dt_brute(diag, "d2")
        scal = dt_two_pass_scalar_euclid8(diag)
        over = int((scal > ref2 + 1e-9).sum())
        chk("D-2 标量两趟欧氏冒充精确欧氏 → #2-4 高估断言拦截",
            over >= 1 and scal[5, 2] - ref2[5, 2] > 0.4,
            f"高估像素 {over}；D_scalar8(5,2)={scal[5, 2]:.3f} vs √29={ref2[5, 2]:.3f}，差 {scal[5, 2] - ref2[5, 2]:.2f}")
        diagb = np.ones((11, 11), dtype=np.int64)
        diagb[0, 0] = 0          # 唯一背景 (0,0)：正确两趟=曼哈顿 i+j
        broken = dt_two_pass_city_broken(diagb)
        ref1b = dt_brute(diagb, "d1")
        diff = int((broken != ref1b).sum())
        same_rc = np.argwhere(broken == ref1b)
        cols_same = sorted({int(j) for _, j in same_rc})
        comp = (set(map(tuple, same_rc)) ==
                {(i, j) for i in range(11) for j in range(11) if j in cols_same})
        chk("D-3 两趟方向与邻居不配对（前向北东/后向南西）→ 同类断言拦截（11×11 口径）",
            diff >= 1,
            f"配错方向后与精确街区距离差异像素 {diff}/121（11×11 唯一背景 (0,0) 网格口径，应 >0 被拦截）；"
            f"未受影响 {len(same_rc)} 像素=第 {cols_same} 列全部（构成整列成立={comp}）")
    except Exception as e:
        chk("D 区分力注入执行", False, f"异常: {e}")


# ---------- 主流程 ----------

def main():
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    constants()
    make_figs()
    区分力()
    print(f"\n汇总：{COUNT['n'] - len(FAILS)} 条 PASS / {len(FAILS)} FAIL")
    if FAILS:
        print("FAIL 明细：", FAILS)
        sys.exit(1)
    print("DONE")


if __name__ == "__main__":
    main()
