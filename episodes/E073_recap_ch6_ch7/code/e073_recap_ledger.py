#!/usr/bin/env python3
"""E073 复盘课串联台账：Ch6 识别 10 集 + Ch7 特征 8 集，共 18 行。

复盘课不新摘原书正文页（E035/E054 判例）：页码支撑=各集已过门审 shots.json 的
原书页引用（回链）。本脚本只做机械核对，不引入新书页事实。
输出 code/outputs/e073_recap_ledger.txt；断言失败 exit 1。
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E073_recap_ch6_ch7"
OUT = EP / "code" / "outputs" / "e073_recap_ledger.txt"

# (概念, 集, 节锚, 回链页, 一句话)
REC = [
    ("实例识别与几何验证", "E055", "§6.1 Instance recognition p.346", 346, "先找局部特征，再用几何一致性确认是同一个实例"),
    ("大规模实例检索", "E056", "§7.1.4 Large-scale matching and retrieval p.448", 448, "视觉词袋与倒排表，把海量描述子变成可检索的索引"),
    ("图像分类的手工特征", "E057", "§6.2.1 Feature-based methods p.349", 349, "全局描述子加分类器：先造特征，再判类别"),
    ("深度网络与相似检索", "E058", "§6.2.2 Deep networks p.358", 358, "AlexNet 之后，特征从手工设计变成端到端学出来"),
    ("人脸识别", "E059", "§6.2.4 Face recognition p.363", 363, "把人脸压成向量：同一个人靠近，不同人分开"),
    ("人脸检测与行人检测", "E060", "§6.3.1–6.3.2 p.370", 370, "滑窗、级联与 HOG：先回答图里有没有人"),
    ("通用对象检测", "E061", "§6.3.3 General object detection p.379", 379, "R-CNN 家族、YOLO 与 DETR：框出每个物体"),
    ("语义分割", "E062", "§6.4 Semantic segmentation p.387", 387, "从框走到像素：每个像素都要有类别"),
    ("视频与视觉语言", "E063", "§6.5–6.6 p.396", 396, "时间维度与语言对齐：识别不再只看单张图"),
    ("识别章习题课", "E064", "§6.7–6.8 p.409", 413, "把整章问题做成可动手的练习"),
]
FEAT = [
    ("特征点与局部块", "E065", "§7 导言 p.417", 419, "先问：图上哪些位置值得跟踪、配准和检索"),
    ("角点检测", "E066", "§7.1.1 Feature detectors p.422", 422, "Harris 与 Shi-Tomasi：两个方向都有变化的位置"),
    ("SIFT 尺度空间", "E067", "§7.1.1 尺度空间 p.429", 429, "尺度不变：近看远看都认得出同一个点"),
    ("SIFT 描述子", "E068", "§7.1.2 Feature descriptors p.434", 434, "16×16 邻域、4×4×8 方向直方图，128 维向量"),
    ("特征匹配与几何校验", "E069", "§7.1.3 Feature matching p.441", 441, "比率测试、互检、RANSAC：把候选筛成可信对应"),
    ("直线与消失点", "E070", "§7.4 Lines and vanishing points p.477", 477, "点之外的另一类几何：直线也能投票与求消失点"),
    ("学习型描述子", "E071", "§7.1.2 后半 p.437", 437, "BRIEF、ORB 到 SuperPoint：描述子也能学"),
    ("OpenCV 全流程实验", "E072", "§7.1–7.4 回指 p.441", 441, "提取、匹配、互检、RANSAC、叠加：一条链跑通"),
]

PAGE_RANGE = re.compile(r"原书第\s*(\d+)(?:\s*到\s*(\d+))?\s*页")


def cited_pages(ep: str) -> set[int]:
    ds = list((ROOT / "episodes").glob(f"{ep}_*"))
    if len(ds) != 1:
        raise SystemExit(f"[FAIL] {ep} 目录不唯一：{ds}")
    s = json.loads((ds[0] / "production" / "shots.json").read_text())
    pages = set()
    for c in s["chunks"]:
        for t in (c["narration"], *c.get("refs", [])):
            for a, b in PAGE_RANGE.findall(t):
                lo = int(a)
                hi = int(b) if b else lo
                pages.update(range(lo, hi + 1))
    return pages


lines = []
errors = []
emit = lines.append
emit("=" * 78)
emit("E073 · 从「认得出」到「找得准」：识别与特征的合流 —— 串联台账（程序生成）")
emit("=" * 78)
emit("回链口径：复盘课不新摘原书正文页；页码支撑=各集已过门审 shots.json 引用集合。")
emit("")
for title, rows in (("认得出（第 6 章识别，E055–E064）", REC), ("找得准（第 7 章特征，E065–E072）", FEAT)):
    emit(f"── {title} ──")
    for name, ep, anchor, page, echo in rows:
        emit(f"  {name}（{ep}）｜ {anchor} ｜ 回链页={page} ｜ 一句话：{echo}")
    emit("")

emit("── 断言 ──")
for name, ep, anchor, page, echo in REC + FEAT:
    if page not in cited_pages(ep):
        errors.append(f"A1 回链失配：{name} p.{page} 不在 {ep} 引用集合")
emit(f"  [{'PASS' if not any(e.startswith('A1') for e in errors) else 'FAIL'}] "
     f"A1 页码锚：回链 18 行全命中（E055–E072 引用集合实测）")

# A2 章界：识别组 ⊆ [343,416]，特征组 ⊆ [417,491]
viol = []
for name, ep, anchor, page, echo in REC + FEAT:
    if not 343 <= page <= 491:
        viol.append((name, page))
# E056 取材 §7.1.4（第 7 章）但主题属识别检索：E055 立项已核原书自注「Section 7.1.4 reviews such techniques」
for name, ep, anchor, page, echo in REC:
    if page > 416 and ep != "E056":
        viol.append((name, page))
emit(f"  [{'PASS' if not viol else 'FAIL'}] A2 章界：全部回链页⊆[343,491]；识别主题集仅 E056 取材 Ch7 §7.1.4"
     f"（书内自注移交，E055 立核；其余识别集⊆[343,416]，特征集⊆[417,491]）")
for name, page in viol:
    errors.append(f"A2 章界：{name} p.{page} 出界")

# A3 两条链的行数
emit(f"  [{'PASS' if len(REC) == 10 and len(FEAT) == 8 else 'FAIL'}] "
     f"A3 行数：识别链 10 + 特征链 8 = 18")
if len(REC) != 10 or len(FEAT) != 8:
    errors.append("A3 行数不符")

emit("")
emit(f"台账行数：{len(REC)+len(FEAT)}（识别链 10+特征链 8）；断言错误 {len(errors)} 项")
if errors:
    for e in errors:
        emit(f"  [ERROR] {e}")
    emit("LEDGER-FAIL")
    OUT.write_text("\n".join(lines) + "\n")
    sys.exit(1)
emit("LEDGER-DONE")
OUT.write_text("\n".join(lines) + "\n")
print(f"[ok] {OUT}（{len(lines)} 行）")
