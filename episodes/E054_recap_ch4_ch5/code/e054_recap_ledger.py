#!/usr/bin/env python3
"""E054 S2 串联台账：读 E036–E053 十八集 shots.json（页码引用单源）+lab run 存档，
产出「概念→节锚→回链页→回响实验」台账并跑三层断言——回链/存档/章界。
stdout 落 code/outputs/e054_recap_ledger.txt（权威档），stderr 分离。
判例：E035_recap_ch2_ch3/code/e035_recap_ledger.py（A1 回链+A2 存档+A3 章界三层同构）。
E054 变体申报：E036–E053 全为新集形态（页码引用在档）——18 行全 link 无 own；
A3 章界=第 5 章组（E042–E053）硬断言 [235,342]+第 4 章主题组（E036–E041）容器分布
实测申报（其取材页天然散布 Ch4 正文/§8.1.4/附录 A/B——地图 E036–E041 行 ⚠️ 核改申报
如实记录的结构事实，各集 scope 已过门审，不拍区间硬断）。"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # code→集→episodes→课程根
EP = ROOT / "episodes" / "E054_recap_ch4_ch5"
OUT = EP / "code" / "outputs" / "e054_recap_ledger.txt"

PAGE = re.compile(r"原书第 (\d+) 页")

# 台账行：（概念, 集, 原书节锚, 回链页, 回响 lab, 回响说明）
ROWS = [
    ("最小二乘与正规方程", "E036", "附录 A.2 Linear least squares p.927", 927,
     "正规方程手推与三诊断（欠定/病态/外点）"),
    ("鲁棒损失与 IRLS", "E037", "§4.1.3 Robust data fitting p.202+附录 B.3 Robust statistics p.945", 202,
     "二次/绝对/Huber 对外点敏感性对照"),
    ("RANSAC", "E038", "§8.1.4 Robust least squares and RANSAC p.510", 510,
     "抽样投票与内点一致集"),
    ("RANSAC 实战三件", "E039", "§8.1.4（续）p.511", 511,
     "阈值/迭代次数/置信度公式实算"),
    ("不确定度与协方差", "E040", "附录 B.6 Uncertainty estimation p.952", 952,
     "后验协方差与 Cramér–Rao 下界"),
    ("非线性最小二乘与 LM", "E041", "附录 A.3 Non-linear least squares p.930", 930,
     "泰勒展开与 LM 阻尼插值"),
    ("神经元与激活函数", "E042", "§5.3.1 Weights and layers p.270", 270,
     "线性塌缩与 ReLU 死单元"),
    ("损失函数与梯度下降", "E043", "§5.3.4 Loss functions p.280+5.3.5 前半 p.284", 280,
     "损失地形与一步梯度"),
    ("反向传播", "E044", "§5.3.5 Backpropagation 后半 p.285–287", 285,
     "链式法则流水线与解析梯度对照"),
    ("优化器与初始化", "E045", "§5.3.6 Training and optimization p.287+结论 5.61 p.283", 287,
     "momentum/Adam 与 He 初始化"),
    ("卷积与权值共享", "E046", "§5.4 前半 p.291–294", 291,
     "多通道卷积与参数量口算"),
    ("感受野与池化", "E047", "§5.4.1 Pooling and unpooling p.295", 295,
     "感受野演化与 max/average 对照"),
    ("架构演进史", "E048", "§5.4.3–5.4.4 p.299–304", 299,
     "AlexNet→VGG→GoogLeNet→ResNet 接力"),
    ("可视化与对抗样本", "E049", "§5.4.5–5.4.7 p.307–317", 307,
     " saliency 与贴纸攻击"),
    ("3D CNN 与 RNN", "E050", "§5.5.1–5.5.2 p.317–322", 317,
     "时序融合四式与权重跨时共享"),
    ("Transformer 与 ViT", "E051", "§5.5.3 Transformers p.322", 322,
     "self-attention 三权重矩阵"),
    ("生成模型", "E052", "§5.5.4 Generative models p.328–336", 328,
     "VAE 码本量化与 GAN 博弈"),
    ("实验课", "E053", "§5.7 Exercises p.337–342", 337,
     "19 题三层结构与三优化器扫描"),
]

EXC = {"E041": "p.235 引用系第 5 章扉页预告（E041 锚 §4 实测章界——预登记）"}


def cited_pages(slug_dir: pathlib.Path) -> set[int]:
    s = json.loads((slug_dir / "production" / "shots.json").read_text())
    pages = set()
    for c in s["chunks"]:
        for t in (c["narration"], *c.get("refs", [])):
            pages.update(int(m.group(1)) for m in PAGE.finditer(t))
    return pages


SLUG = {}
for d in (ROOT / "episodes").glob("E0*"):
    m = re.match(r"(E\d+)_(.+)", d.name)
    if m:
        SLUG[m.group(1)] = m.group(2)


def ep_dir(num: str) -> pathlib.Path:
    return ROOT / "episodes" / f"{num}_{SLUG[num]}"


lines = []
errors = []
n_link = n_run = 0
MAIN_TOTAL = 0


def emit(s=""):
    lines.append(s)


emit("=" * 78)
emit("E054 · 视觉问题的万能模具：拟合与学习是一枚硬币的两面 —— S2 串联台账（程序生成）")
emit("=" * 78)
emit("回链口径：复盘课不新摘原书正文页（E035 判例同款）——页码支撑=回链各集已过门审")
emit("shots.json refs/narration 引用集合；本台账 18 行全 link（E036+ 全为新集形态）。")
emit("")

emit("── 一枚模具（第 4 章主题·拟合与优化——从猜到证明，E036–E041）──")
for name, eps, anchor, link_p, echo in ROWS[:6]:
    emit(f"  {name}（{eps}）｜ {anchor} ｜ 回链页={link_p} ｜ 回响：{echo}")
emit("── 一整套学习（第 5 章 Deep learning——从证明到学会，E042–E053）──")
for name, eps, anchor, link_p, echo in ROWS[6:]:
    emit(f"  {name}（{eps}）｜ {anchor} ｜ 回链页={link_p} ｜ 回响：{echo}")
emit("")

emit("── 断言 ──")
# A1 页码锚回链：回链页在该集（shots.json 引用集合）
for name, eps, anchor, link_p, echo in ROWS:
    if link_p not in cited_pages(ep_dir(eps)):
        errors.append(f"A1 回链失配: {name} p.{link_p} 不在 {eps} 引用集合")
    n_link += 1
emit(f"  [{'PASS' if not any(e.startswith('A1') for e in errors) else 'FAIL'}] "
     f"A1 页码锚：回链 {n_link} 行全命中（18 集引用集合实测）")

# A2 回响实验存档：十八集全部要有含 PASS 汇总的 run 存档
SUM_RE = re.compile(r"(汇总：[^\n]+|\d+\s*条[^\n]{0,10}(?:全部\s*)?PASS[^\n]{0,40}|\d+\s*条，\s*PASS\s*\d+)")
run_report = []
for name, eps, anchor, link_p, echo in ROWS:
    d = ep_dir(eps)
    outs = sorted((d / "code" / "outputs").glob("*.txt")) if (d / "code" / "outputs").exists() else []
    best = None
    for r in outs:
        if "regression" in r.name or "display" in r.name:
            continue
        t = r.read_text()
        if "PASS" in t:
            mm = SUM_RE.search(t)
            if mm:
                best = (r.name, mm.group(0)[:56])
                break
    if best:
        run_report.append((eps, best[0], best[1]))
        n_run += 1
        m = re.search(r"(\d+)\s*条", best[1])
        if m:
            MAIN_TOTAL += int(m.group(1))
    else:
        run_report.append((eps, "-", "无含 PASS 汇总的存档"))
        errors.append(f"A2 回响存档缺: {eps} 无可用 run 汇总")
emit(f"  [{'PASS' if not any(e.startswith('A2') for e in errors) else 'FAIL'}] "
     f"A2 回响存档：{n_run}/18 集含 PASS 汇总行；主数合计={MAIN_TOTAL}（口播引用以本行实测为准）")
for one, fn, summ in run_report:
    emit(f"      {one}: {fn} —— {summ}")

# A3 章界：第 5 章组硬断言 [235,342]；第 4 章主题组容器分布实测申报
viol = []
for num in [f"E{i:03d}" for i in range(42, 54)]:
    pages = cited_pages(ep_dir(num))
    out_rng = sorted(p for p in pages if not 235 <= p <= 342)
    if out_rng:
        viol.append((num, out_rng))
emit(f"  [{'PASS' if not viol else 'FAIL'}] A3 章界：第 5 章组 E042–E053 引用页⊆[235,342]"
     f"（Ch5 印刷 235 起=ToC 实测 e054_book_toc_p4.txt；Ch6 切点 343——E053 §23 判例兑现）")
for num, rng in viol:
    emit(f"      ⚠ {num} 出界页: {rng}")
    errors.append(f"A3 章界: {num} 出界 {rng}")
emit("  A3 申报：第 4 章主题组（E036–E041）取材页容器分布实测——")
dist = {}
for num in [f"E{i:03d}" for i in range(36, 42)]:
    for p in sorted(cited_pages(ep_dir(num))):
        if p == 235:
            tag = "Ch5 扉页预告（E041 例外预登记）"
        elif p in (193, 202, 203):
            tag = "Ch4 正文 [191,234]"
        elif 510 <= p <= 512:
            tag = "§8.1.4（鲁棒拟合—RANSAC 载体节）"
        elif 927 <= p <= 931:
            tag = "附录 A（最小二乘）"
        elif 945 <= p <= 952:
            tag = "附录 B（鲁棒/不确定度）"
        else:
            tag = "⚠ 未分类"
        dist.setdefault(tag, []).append(f"{num} p.{p}")
for tag in sorted(dist):
    emit(f"      {tag}: {', '.join(dist[tag])}")

emit("")
emit(f"台账行数：{len(ROWS)}（模具链 6+学习链 12）；断言错误 {len(errors)} 项")
if errors:
    for e in errors:
        emit(f"  [ERROR] {e}")
    emit("LEDGER-FAIL")
    OUT.write_text("\n".join(lines) + "\n")
    sys.exit(1)
emit("LEDGER-DONE")
OUT.write_text("\n".join(lines) + "\n")
print(f"[ok] {OUT}（{len(lines)} 行）")
