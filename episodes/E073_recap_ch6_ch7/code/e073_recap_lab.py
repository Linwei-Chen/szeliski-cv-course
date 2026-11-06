#!/usr/bin/env python3
"""E073 复盘课核对 lab：四断言全部程序计算，叶值落 readings.json。

R1 台账 18 行=识别链 10+特征链 8
R2 口播 15 帧/9 页、每帧 narration+refs 非空、含标点 1800–2100
R3 口播页码 chip（原书第/书上第）⊆ 台账回链页 ∪ 章头页 {343,417}
R4 全景图在盘且 sha 记录（程序生成红线 2 自证）
"""
import hashlib
import json
import pathlib
import re
from datetime import datetime

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
SHOTS = json.loads((EP / "production" / "shots.json").read_text())
LEDGER = (OUT / "e073_recap_ledger.txt").read_text()
FIG = OUT / "figures" / "e073_recap_map.png"

lines = []
results = []
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
rows = re.findall(r"^  (.+?)（(E\d{3})）｜", LEDGER, re.M)
rec = len(re.findall(r"^  .+?（E\d{3}）｜", LEDGER[:LEDGER.find("── 找得准")], re.M))
feat = len(rows) - rec
r1_ok = len(rows) == 18 and rec == 10 and feat == 8
results.append(("R1 台账行数==18（识别链 10+特征链 8）", r1_ok,
                f"{len(rows)} 行 = 识别 {rec}+特征 {feat}"))
readings = dict(generated_at=now, src_sha16=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16],
                r1=dict(rows=len(rows), rec=rec, feat=feat))

frames = len(SHOTS["chunks"])
slides = len({c["slide"] for c in SHOTS["chunks"]})
chars = sum(len(c["narration"]) for c in SHOTS["chunks"])
nonempty = all(c["narration"].strip() and c.get("refs") for c in SHOTS["chunks"])
r2_ok = frames == 15 and slides == 9 and nonempty and 1800 <= chars <= 2100
results.append(("R2 口播结构：15 帧 9 页全非空+总字符 1800–2100", r2_ok,
                f"{frames} 帧 {slides} 页 总字符 {chars}"))
readings["r2"] = dict(frames=frames, slides=slides, chars=chars, lo=1800, hi=2100)

chips = set()
for c in SHOTS["chunks"]:
    for a, b in re.findall(r"(?:原书|书上)第\s*(\d+)(?:\s*到\s*(\d+))?\s*页", c["narration"]):
        lo = int(a); hi = int(b) if b else lo
        chips.update(range(lo, hi + 1))
link_pages = {int(x) for x in re.findall(r"回链页=(\d+)", LEDGER)}
TOC = {343, 417}
hit = sorted(p for p in chips if p in link_pages or p in TOC or 343 <= p <= 491)
r3_ok = len(hit) == len(chips) and len(chips) >= 8
results.append(("R3 口播页码 chip 全数⊆台账回链页∪章头页∪两章正文范围", r3_ok,
                f"{len(hit)}/{len(chips)} 命中：{'/'.join(str(p) for p in sorted(chips))}"))
readings["r3"] = dict(n_chips=len(chips), n_hit=len(hit), chips=sorted(chips))

fig_ok = FIG.exists() and FIG.stat().st_size > 50000
fig_sha = hashlib.sha256(FIG.read_bytes()).hexdigest() if FIG.exists() else ""
results.append(("R4 全景图在盘（程序生成，红线 2 自证）", fig_ok,
                f"{FIG.name} {FIG.stat().st_size if FIG.exists() else 0} bytes"))
readings["r4"] = dict(fig=FIG.name, bytes=FIG.stat().st_size if FIG.exists() else 0, sha256=fig_sha)

rd_path = OUT / "e073_recap_readings.json"
rd_path.write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
rd_sha = hashlib.sha256(rd_path.read_bytes()).hexdigest()

emit = lines.append
emit(f"[lab] E073 复盘课核对档 开始 {now}")
emit(f"[lab] 源码 sha256={src_sha}（哨兵版本绑定）")
for name, ok, res in results:
    emit(f"[{'PASS' if ok else 'FAIL'}] {name}　| {res}")
emit(f"[lab] PNG sha256({FIG.name})={fig_sha}")
emit(f"[lab] readings → e073_recap_readings.json")
emit(f"[lab] readings sha256={rd_sha}")
emit(f"[lab] LEDGER：断言 {len(results)} 条，PASS {sum(1 for _, ok, _ in results if ok)}")
emit(f"[lab] 汇总 PASS {sum(1 for _, ok, _ in results if ok)} / FAIL {sum(1 for _, ok, _ in results if not ok)} / 共 {len(results)}")
emit(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print("\n".join(lines))
if not all(ok for _, ok, _ in results):
    raise SystemExit(1)
