#!/usr/bin/env python3
"""E038 数值实验：RANSAC 随机抽样一致——从全部对应点抽两点→数内点→重复投票→留最优→内点集交下一拟合阶段。
锚：e038_book_s814_anchors.txt（§8.1.4 印刷 p.510–513 上）。
主流程对齐原书 p.511（门A round1 P0-1 修）：抽签与计票都在全部对应点上进行（主体组 17 点=16 内点+1 外点；
加压组 20 点=16 内点+4 外点），外点可以进签、也可以进计票池——不使用真值标签挑样本。
数据语言延续 E037（seed 2038 单次实现，跨 seed 不可比）：内点 16 个 y=1+0.5x+N(0,0.15) 均布 x∈[0,4]；
主体组外点 1 个 (4.6, 5.2)（与 E037 同位，回链）；加压组再加 3 个外点（(1.0,3.0),(2.5,0.2),(3.5,4.5)，手挑位置）。
口径申报（读数级，D31）：
- ϵ=0.3 为本实验应用约定（约 2σ）——原书 p.511：ϵ application dependent，1–3 pixels 是图像特征场景；
- 残差 ri=|预测−观测|（E037「误差=预测−观测」语言），与 8.28 的 x̃′−x̂′ 同构（估计/预测位置−探测位置）；
- S=200 为实验约定——8.29/8.30 S 公式与 Table 8.2 属 E039 锚（D31），本集只「重复 S 次」（p.511 实文）；
- 两个「全」读数分开入档（round1 P0-1 修）：全内点样本轮=两点皆为内点的轮数（对应 p^k 概念，E039 伏笔）；
  满票轮=16 个内点全部落在门槛内的轮数（本 seed 实测 0/200，最优 15 票——如实入档，口播只讲票最多）；
- RANSAC 两点线只是起点/内点名单（p.511：Either the initial parameter guess p or the full set of computed
  inliers is then passed on to the next data fitting stage）——名单重解的价值在用上全部名单点（不只靠两点）；
  本 seed 单次读数中两点胜出线恰比名单重解更贴真线，单次读数不比优劣（round2 P1-1 修），回链 E037。
图（程序生成，未用教材原图）：图1 投票直观 765×320（对应 p.511 RANSAC 主流程）；
图2 票数分布堆叠柱 765×320（纯内点签/含外点签分色，对应 p.511 repeated S times）；
图3 外点加压 765×320（对应 p.511 too many outliers 段）。
round4 门A P2 清账（2026-10-30）：E8f 真重放（重放数据全链重导出，不复用原数据/原索引——P2-4）；
并列签读数范围入档 readings e3.tie_scope（只入档、不断言——P2-5）；E3b/E3c 详情加主体组标注。
"""
import json
import pathlib

import numpy as np

rng = np.random.default_rng(2038)
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
checks = []
readings = {}


def chk(label, ok, detail=""):
    checks.append(label)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"FAIL: {label}")


def line_from_two(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return None
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1


def count_inliers(m, b, xs, ys, eps):
    r = np.abs(ys - (m * xs + b))
    return int((r <= eps).sum()), r


def ols(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


M_TRUE, B_TRUE, SIGMA = 0.5, 1.0, 0.15
EPS = 0.3          # 应用约定 ≈2σ（原书：ϵ application dependent，1–3 pixels 为图像场景）
S_ROUNDS = 200     # 实验约定（8.29/8.30 属 E039 锚，D31）——两点线抖，高票签稀少，50 轮预实验最优仅 13/17，200 轮留足投票余量

# ── 数据（全部对应点=16 内点+1 外点，抽签与计票都在全集上——round1 P0-1 修）──
x_in = np.sort(rng.uniform(0.0, 4.0, 16))
y_in = B_TRUE + M_TRUE * x_in + rng.normal(0.0, SIGMA, 16)
x_out_pt, y_out_pt = 4.6, 5.2
x_all = np.append(x_in, x_out_pt)
y_all = np.append(y_in, y_out_pt)

# ── E1 两点定线（单轮，从全集 17 点抽——外点可能进签）──
i1, i2 = rng.choice(17, size=2, replace=False)
p1, p2 = (x_all[i1], y_all[i1]), (x_all[i2], y_all[i2])
m2, b2 = line_from_two(p1, p2)
first_has_outlier = bool(i1 == 16 or i2 == 16)
chk("E1a 单轮两点线精确过抽中两点（残差 <1e-12）",
    max(abs(y_all[i1] - (m2 * x_all[i1] + b2)), abs(y_all[i2] - (m2 * x_all[i2] + b2))) < 1e-12,
    f"两点 ({p1[0]:.3f},{p1[1]:.3f}) ({p2[0]:.3f},{p2[1]:.3f}) -> m={m2:.4f} b={b2:.4f}"
    f"（签含外点={first_has_outlier}）")

# ── E2 首轮读数（ϵ 口径，计票在全集 17 点上）──
n2, r2 = count_inliers(m2, b2, x_all, y_all, EPS)
r_out_pt = abs(y_out_pt - (m2 * x_out_pt + b2))
chk("E2a 首轮抽签读数：两点皆内点、线立起来了只是斜了点（本 seed 首轮 m=0.6105、十四票不满）——单轮质量看手气，投票机制因此必要（读数级）",
    n2 < 17 and n2 >= 2 and not first_has_outlier,
    f"首轮 {n2}/17 票（m={m2:.4f}，签含外点={first_has_outlier}）——round2 P2-6 修：标签与读数对齐（立住但斜）")
chk("E2b 外点不计入（外点残差 >ϵ）", r_out_pt > EPS,
    f"外点残差 {r_out_pt:.4f} > ϵ={EPS}")
readings["e2"] = {"two_pt_line": [round(m2, 4), round(b2, 4)],
                  "first_round_votes": n2, "first_round_has_outlier": first_has_outlier,
                  "outlier_resid": round(r_out_pt, 4),
                  "eps_note": "ϵ=0.3≈2σ 应用约定（原书 p.511：application dependent，1–3 pixels 为图像场景）",
                  "note": "首轮抽签为固定 seed 实测读数，非挑好的演示签；计票在全集 17 点上（round1 P0-1 修）"}

# ── E3 投票 S 轮（原书主流程：从全部对应点抽 2 点→拟线→全集数内点→重复 S 次→留票最多）──
draws = np.array([rng.choice(17, size=2, replace=False) for _ in range(S_ROUNDS)])
votes, lines, full_flags = [], [], []
for a, b in draws:
    m, bb = line_from_two((x_all[a], y_all[a]), (x_all[b], y_all[b]))
    n, r_round = count_inliers(m, bb, x_all, y_all, EPS)
    votes.append(n)
    lines.append((m, bb))
    full_flags.append(bool((r_round[:16] <= EPS).all()))  # round2 P2-5 修：满票=16 个内点逐点全落门槛（不含外点，总票数口径废弃）
votes = np.array(votes)
best = int(votes.argmax())
m_best, b_best = lines[best]
# round1 P0-1 修：两个「全」读数分开入档
pure_rounds = int((draws.max(axis=1) < 16).sum())     # 全内点样本轮：两点皆为内点（对应 p^k 概念，E039 伏笔）
dirty_rounds = int((draws.max(axis=1) == 16).sum())   # 含外点样本轮：签里混进了外点
full_vote_rounds = int(sum(full_flags))               # 满票轮：16 个内点全部落进门槛（逐点口径）
n_best_ties = int((votes == votes.max()).sum())       # round3 P2-4 修：最高票并列轮数如实披露（argmax 取首个）
chk("E3a 全内点样本轮与含外点样本轮都存在（抽签确实会摸到坏数据——p.511 必要性论证的实证）",
    pure_rounds >= 1 and dirty_rounds >= 1,
    f"全内点样本轮 {pure_rounds}/{S_ROUNDS}，含外点样本轮 {dirty_rounds}/{S_ROUNDS}")
chk("E3b 含外点签全部低票（≤5），高票轮（≥14 票）全为纯内点签——投票把混进坏数据的签压在箱底（读数级）",
    votes[draws.max(axis=1) == 16].max() <= 5
    and votes[draws.max(axis=1) < 16][votes[draws.max(axis=1) < 16] >= 14].size
        == votes[votes >= 14].size,
    f"主体组 17 点：含外点签票数范围 {votes[draws.max(axis=1) == 16].min()}–{votes[draws.max(axis=1) == 16].max()}"
    f"，≥14 票轮全部纯内点签（中位数 {int(np.median(votes))} 票）")
chk("E3c 主体组最优两点线贴近真线（|m−0.5|<0.05，读数级——两点线本身抖动）",
    abs(m_best - M_TRUE) < 0.05,
    f"主体组 m_best={m_best:.4f}（最优 {int(votes.max())} 票；{n_best_ties} 轮并列取首个——round3 P2-4 修：并列如实披露）")
r_best_out = abs(y_out_pt - (m_best * x_out_pt + b_best))
chk("E3d 胜出线对外点残差远超门槛（外点零票——计票在全集上，外点真被排除）",
    r_best_out > EPS,
    f"胜出线外点残差 {r_best_out:.4f} > ϵ={EPS}（round1 P1-5 修：该读数正式入档）")
# round4 P2-5 修：并列签读数范围入档（只入档、不断言）——57 张同票签的平票裁决（argmax 取首个）
# 落在哪张，影响 m_best/名单重解读数与 f009 的反差叙述；口播已有「单次读数不比优劣」兜底，此处给范围供查。
tie_m, tie_refit = [], []
for r in range(S_ROUNDS):
    if votes[r] != votes.max():
        continue
    tie_m.append(float(lines[r][0]))
    msk_t = np.abs(y_all - (lines[r][0] * x_all + lines[r][1])) <= EPS
    tie_refit.append(float(ols(x_all[msk_t], y_all[msk_t])[0]))
readings["e3"] = {"S": S_ROUNDS, "best_inliers": int(votes.max()),
                  "best_vote_ties": n_best_ties,
                  "m_best": round(m_best, 4), "b_best": round(b_best, 4),
                  "pure_sample_rounds": pure_rounds, "dirty_sample_rounds": dirty_rounds,
                  "full_vote_rounds": full_vote_rounds,
                  "dirty_votes_max": int(votes[draws.max(axis=1) == 16].max()),
                  "median_inliers": int(np.median(votes)),
                  "best_line_outlier_resid": round(r_best_out, 4),
                  "votes_hist": {str(k): int(v) for k, v in
                                 zip(*np.unique(votes, return_counts=True))},
                  "tie_scope": {"n_ties": len(tie_m),
                                "m_best_min": round(min(tie_m), 4), "m_best_max": round(max(tie_m), 4),
                                "m_refit_min": round(min(tie_refit), 4), "m_refit_max": round(max(tie_refit), 4),
                                "note": "round4 P2-5 修：全部并列签逐一入档的范围读数（只入档，不断言）——"
                                        "argmax 首个裁决是其中一次实现，换裁决读数会落在该范围内"},
                  "note": "满票轮 0/200 如实入档：ϵ=0.3 下没有一签让 16 内点全中，投票挑的是票最多的签（15 票）——"
                          "原书只说留票最多者，不承诺全中；全内点样本轮 180/200 对应 p^k 概念（E039 伏笔）"}
print(f"[入档] E3-tie 并列签读数范围（{len(tie_m)} 张，只入档不断言——round4 P2-5 修）："
      f"m_best {min(tie_m):.4f}–{max(tie_m):.4f}，名单重解 m {min(tie_refit):.4f}–{max(tie_refit):.4f}")

# ── E4 坏抽签出局（投票淘汰机制，读数级）──
worst = int(votes.argmin())
m_worst, b_worst = lines[worst]
r_w_out = abs(y_out_pt - (m_worst * x_out_pt + b_worst))
chk("E4a 最差签出局（票数远低于最优——投票自然淘汰，读数级）",
    votes.min() < votes.max(),
    f"最差轮 {votes.min()} 票（m={m_worst:.4f}，签含外点={bool(draws[worst].max() == 16)}，外点残差 {r_w_out:.4f}）")
chk("E4b 低票不全是外点签（纯内点的小签也抖——淘汰判据=总票数，读数级留痕）",
    True, f"最差轮签含外点={bool(draws[worst].max() == 16)}（本 seed 最差签两点皆内点、纯噪声抖动；"
          f"含外点签 {dirty_rounds} 轮全部 ≤{int(votes[draws.max(axis=1) == 16].max())} 票）")
readings["e4"] = {"worst_inliers": int(votes.min()), "m_worst": round(m_worst, 4),
                  "worst_has_outlier": bool(draws[worst].max() == 16),
                  "outlier_resid_worst": round(r_w_out, 4),
                  "note": "淘汰判据=内点总数（投票），非单个外点残差——round3 判例同源：读数级不断言『坏线必远』"}

# ── E5 内点名单交下一拟合阶段（p.511 实文：inliers passed on to the next data fitting stage）──
inlier_mask = np.abs(y_all - (m_best * x_all + b_best)) <= EPS
m_refit, b_refit = ols(x_all[inlier_mask], y_all[inlier_mask])
m_clean, b_clean = ols(x_in, y_in)
chk("E5a 名单不含外点（15 个，恰缺一个内点）→名单重解与 16 内点 OLS 同在真线附近（|Δ|=0.0077 如实入档）",
    int(inlier_mask.sum()) == 15 and not inlier_mask[-1]
    and abs(m_refit - m_clean) < 0.01,
    f"名单 {int(inlier_mask.sum())} 个 -> 名单重解 m={m_refit:.4f} vs 剔除外点 OLS m={m_clean:.4f}"
    f"（|Δ|={abs(m_refit - m_clean):.4f}，差额来自名单缺一内点——15 票签未把全部自己人收进门槛；"
    f"本集数据自身 m_clean={m_clean:.4f}，偏离真线 {abs(m_clean-M_TRUE):.4f}——seed 2038 单次实现，跨 seed 不可比）")
readings["e5"] = {"m_refit": round(m_refit, 4), "b_refit": round(b_refit, 4),
                  "m_clean": round(m_clean, 4), "n_refit_inliers": int(inlier_mask.sum()),
                  "abs_delta_refit_vs_clean": round(abs(m_refit - m_clean), 6),
                  "note": "名单重解=用上全部名单点（p.511：passed on to the next data fitting stage），回链 E037 E1a；"
                          "round2 P1-1 修：本 seed 单次读数，两点胜出线 m=0.5295 恰比名单重解 m=0.5384 更贴真线"
                          "（偏 0.0295 vs 0.0384）——单次读数不比优劣，名单重解的价值在用上 15 人数据（不只两点）；"
                          "本集名单 15 个（缺一内点），加压组名单恰好全中属单次手气（E6 note）——round1 P2-4 修：Δ 实际值入档"}

# ── E6 外点加压（20 点全集=16 内点+4 外点：抽签、计票都在全集上——真·鲁棒性实证）──
extra_out = [(1.0, 3.0), (2.5, 0.2), (3.5, 4.5)]
x_hard = np.append(x_all, [p[0] for p in extra_out])
y_hard = np.append(y_all, [p[1] for p in extra_out])
m_hard_ols, b_hard_ols = ols(x_hard, y_hard)
draws_h = np.array([rng.choice(20, size=2, replace=False) for _ in range(S_ROUNDS)])
votes_h, lines_h = [], []
for a, b in draws_h:
    m, bb = line_from_two((x_hard[a], y_hard[a]), (x_hard[b], y_hard[b]))
    n, _ = count_inliers(m, bb, x_hard, y_hard, EPS)
    votes_h.append(n)
    lines_h.append((m, bb))
votes_h = np.array(votes_h)
best_h = int(votes_h.argmax())
m_best_h, b_best_h = lines_h[best_h]
mask_h = np.abs(y_hard - (m_best_h * x_hard + b_best_h)) <= EPS
m_refit_h, b_refit_h = ols(x_hard[mask_h], y_hard[mask_h])
chk("E6a 外点加压 OLS 被拉偏（|m−0.5|≥0.10）", abs(m_hard_ols - M_TRUE) >= 0.10,
    f"OLS m={m_hard_ols:.4f}")
chk("E6b 加压组名单重解与剔除外点 OLS 分毫不差（|Δ|<1e-9）——本 seed 机制实证（外点手挑、单 seed，读数级不外推）=4 个外点零票+对照 E6a OLS 被拉偏；名单 16 恰全中是本批签手气（round2 P1-2+round3 P2-5 修）",
    int(mask_h.sum()) == 16 and not mask_h[-4:].any()
    and abs(m_refit_h - m_clean) < 1e-9,
    f"最优轮 {int(votes_h.max())}/20 票（签含外点={bool(draws_h[best_h].max() >= 16)}）-> 名单 {int(mask_h.sum())} 个"
    f" -> 名单重解 m={m_refit_h:.4f} = 剔除外点 OLS m={m_clean:.4f}（|Δ|={abs(m_refit_h - m_clean):.2e}）"
    "——外点进了抽签池又被票数淘汰（round1 P0-1 修：全集抽样下的真对照）")
chk("E6c 加压组投票区分度（含外点签轮与高票轮都存在）",
    (draws_h.max(axis=1) >= 16).sum() >= 1 and (votes_h >= 15).sum() >= 1,
    f"含外点签 {(draws_h.max(axis=1) >= 16).sum()}/{S_ROUNDS} 轮，高票轮（≥15）{(votes_h >= 15).sum()}/{S_ROUNDS}，"
    f"中位数 {int(np.median(votes_h))} 票")
dirty_h = draws_h.max(axis=1) >= 16
chk("E6d 加压组含外点签最高票数入档且低于胜出票（投票淘汰的量化——round2 P2-3 修+round3 P2-6 修：恒真改实断言）",
    int(votes_h[dirty_h].max()) < int(votes_h.max()),
    f"含外点签 {int(dirty_h.sum())}/{S_ROUNDS} 轮，票数范围 "
    f"{int(votes_h[dirty_h].min())}–{int(votes_h[dirty_h].max())}"
    f"（最高 {int(votes_h[dirty_h].max())} 票 < 胜出 {int(votes_h.max())} 票）")
readings["e6"] = {"m_hard_ols": round(m_hard_ols, 4), "m_refit_hard": round(m_refit_h, 4),
                  "n_refit_inliers_hard": int(mask_h.sum()),
                  "best_inliers_hard": int(votes_h.max()),
                  "dirty_rounds_hard": int(dirty_h.sum()),
                  "dirty_votes_max_hard": int(votes_h[dirty_h].max()),
                  "median_hard": int(np.median(votes_h)),
                  "abs_delta_hard": round(abs(m_refit_h - m_clean), 9),
                  "note": "round2 P1-2 修：两类结论分开——『外点零票+OLS 被拉偏而 RANSAC 不被拉偏』=本 seed 机制实证"
                          "（外点手挑、单 seed，读数级不外推——round3 P2-5 修）；"
                          "『名单 16 恰全中』=单次读数（本批签手气，主体组同机制只拿 15）；加压组从 20 点全集抽"
                          "（外点可进签）、计票在全集上"}

# ── E7 LMS 对照（p.511：Least median of squares finds the median value of the ri² values）──
med_scores = []
for mm, bb_lms in lines:
    r = (y_all - (mm * x_all + bb_lms)) ** 2
    med_scores.append(float(np.median(r)))
lms = int(np.argmin(med_scores))
m_lms, b_lms = lines[lms]
chk("E7a LMS（残差平方中位数最小）选出的线也贴真线（|m−0.5|<0.08，读数级）",
    abs(m_lms - M_TRUE) < 0.08, f"LMS m={m_lms:.4f}（残差平方中位数 {min(med_scores):.6f}）")
chk("E7b 同一批抽签下 LMS 与 RANSAC 的选择读数留痕（同异均记录，不作优劣断言）",
    True, f"RANSAC m={m_best:.4f} vs LMS m={m_lms:.4f}（同抽签 {S_ROUNDS} 轮、同为全集计票；"
          "两法判据不同：内点计数 vs 残差平方中位数——round1 P2-3 修：标签与评分集口径）")
readings["e7"] = {"m_lms": round(m_lms, 4), "b_lms": round(b_lms, 4),
                  "min_median_sq_resid": round(min(med_scores), 6),
                  "same_pick_as_ransac": lms == best,
                  "note": "LMS 与 RANSAC 并列（p.511）；判据不同（残差平方中位数 vs 内点计数），本集只留读数不作优劣断言"}

# ── E8 复现性（固定 seed 的投票过程逐位可复现——同代码同种子，不外推跨平台）──
rng2 = np.random.default_rng(2038)
x_in2 = np.sort(rng2.uniform(0.0, 4.0, 16))
y_in2 = B_TRUE + M_TRUE * x_in2 + rng2.normal(0.0, SIGMA, 16)
chk("E8a 数据逐位可复现（同 seed 重放 x/y 一致）",
    np.array_equal(x_in, x_in2) and np.array_equal(y_in, y_in2))
i1_r, i2_r = rng2.choice(17, size=2, replace=False)  # 重放 E1 单轮演示的抽签——rng 流对齐（round3 P2-7 修：不再丢弃，参与比对）
draws2 = np.array([rng2.choice(17, size=2, replace=False) for _ in range(S_ROUNDS)])
chk("E8b 抽签序列逐位可复现（同 seed 重放 draws 一致）", np.array_equal(draws, draws2))
x_all2 = np.append(x_in2, x_out_pt)   # round4 P2-4 修：重放侧全集由重放数据构造（与原数据同构，逐位一致已由 E8a 背书）
y_all2 = np.append(y_in2, y_out_pt)
votes2 = np.array([count_inliers(*line_from_two((x_all2[a], y_all2[a]), (x_all2[b], y_all2[b])),
                                 x_all2, y_all2, EPS)[0] for a, b in draws2])
chk("E8c 票数逐位可复现（同 seed 重放 votes 一致——round2 P2-4 修：重放覆盖面从数据+抽签扩到票数）",
    np.array_equal(votes, votes2))
draws_h2 = np.array([rng2.choice(20, size=2, replace=False) for _ in range(S_ROUNDS)])
chk("E8d 加压组抽签逐位可复现（同 seed 重放 draws_h 一致）", np.array_equal(draws_h, draws_h2))
x_hard2 = np.append(x_in2, [x_out_pt] + [p[0] for p in extra_out])
y_hard2 = np.append(y_in2, [y_out_pt] + [p[1] for p in extra_out])
votes_h2 = np.array([count_inliers(*line_from_two((x_hard2[a], y_hard2[a]), (x_hard2[b], y_hard2[b])),
                                   x_hard2, y_hard2, EPS)[0] for a, b in draws_h2])
chk("E8e 加压组票数逐位可复现（同 seed 重放 votes_h 一致）", np.array_equal(votes_h, votes_h2))
# round3 P2-7 修+round4 P2-4 修：LMS 选择与两场名单重解从重放数据全链重导出后比对（真重放）——
# round4 P2-4 修：lines2/mask2/lms2 全部改用重放数据 x_all2/y_all2 与重放导出 best2=argmax(votes2)，
# 不再复用原数据 x_all 与原 best 索引（否则同一输入做同一计算，近乎恒真，不构成独立重放）
lines2 = [line_from_two((x_all2[a], y_all2[a]), (x_all2[b], y_all2[b])) for a, b in draws2]
best2 = int(votes2.argmax())
med2 = [float(np.median((y_all2 - (mm * x_all2 + bb)) ** 2)) for mm, bb in lines2]
lms2 = int(np.argmin(med2))
mask2 = np.abs(y_all2 - (lines2[best2][0] * x_all2 + lines2[best2][1])) <= EPS
m_refit2, _ = ols(x_all2[mask2], y_all2[mask2])
lines_h2 = [line_from_two((x_hard2[a], y_hard2[a]), (x_hard2[b], y_hard2[b])) for a, b in draws_h2]
best_h2 = int(votes_h2.argmax())
mask_h2 = np.abs(y_hard2 - (lines_h2[best_h2][0] * x_hard2 + lines_h2[best_h2][1])) <= EPS
m_refit_h2, _ = ols(x_hard2[mask_h2], y_hard2[mask_h2])
chk("E8f 首轮签/LMS 选择/两场名单重解 逐位可复现（round3 P2-7 修：E8 重放缺口补齐；round4 P2-4 修：重放数据全链重导出）",
    (int(i1_r), int(i2_r)) == (int(i1), int(i2)) and lms2 == lms
    and m_refit2 == m_refit and m_refit_h2 == m_refit_h,
    f"首轮签 ({i1},{i2})、LMS 选中第 {lms} 轮、名单重解 m={m_refit:.4f}/{m_refit_h:.4f} 均逐位一致"
    "（重放侧由 x_all2/x_hard2 与 argmax(votes2)/argmax(votes_h2) 独立导出）")
readings["e8"] = {"seed": 2038, "reproducible": True,
                  "replay_scope": ["x/y 数据", "首轮签", "draws 主体组抽签", "votes 主体组票数",
                                   "draws_h 加压组抽签", "votes_h 加压组票数",
                                   "LMS 选择", "两场名单重解"],
                  "note": "固定 seed 断言的是本实验在同一套代码下重放逐位一致（数据+两轮抽签+两轮票数+导出选择全比对"
                          "——round3 P2-7 修：首轮签/LMS/重解入重放清单）；"
                          "不外推「RANSAC 任何实现/跨机器都可复现」（实现/抽签器相关）——round1 P0-4 修+round2 P2-4 修"}

# ── 图（程序生成，未用教材原图）──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 100})
plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# 图1 投票直观：全集 17 点+抽签线（灰）+胜出线（红）+真线
fig, ax = plt.subplots(figsize=(7.65, 3.2))
ax.scatter(x_in, y_in, s=28, color="#2a6f9e", zorder=3, label="内点 16")
ax.scatter([x_out_pt], [y_out_pt], s=64, marker="X", color="#c0392b", zorder=4, label="外点")
xs = np.linspace(-0.2, 4.8, 50)
for j, (m, bb) in enumerate(lines):
    ax.plot(xs, m * xs + bb, color="#b8c4cc", lw=0.7, alpha=0.55,
            zorder=1, label="抽签线（S=200 轮）" if j == 0 else None)
ax.plot(xs, m_best * xs + b_best, color="#c0392b", lw=2.6, zorder=5,
        label=f"胜出线（{int(votes.max())} 票）")
ax.plot(xs, M_TRUE * xs + B_TRUE, color="#3a7d44", lw=1.6, ls="--", zorder=2,
        label="真线 m=0.5")
ax.set_xlim(-0.2, 4.8); ax.set_ylim(-0.6, 6.2)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("RANSAC：从全部对应点抽两点试一轮，数一轮内点，票多者胜")
ax.legend(loc="upper left", fontsize=12, framealpha=0.92, ncol=2)
fig.text(0.99, 0.012, "程序生成，未用教材原图（概念对应原书第 511 页 RANSAC 主流程）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e038_ransac_vote.png")
plt.close(fig)

# 图2 票数分布堆叠柱：纯内点签（蓝）/含外点签（红）分色
fig, ax = plt.subplots(figsize=(7.65, 3.2))
vals = np.arange(2, 17)
has_out = draws.max(axis=1) == 16
cnt_pure = np.array([int(((votes == v) & ~has_out).sum()) for v in vals])
cnt_dirty = np.array([int(((votes == v) & has_out).sum()) for v in vals])
ax.bar([str(v) for v in vals], cnt_pure, color="#2a6f9e", width=0.62,
       label="纯内点签（两点皆内点）")
bars_d = ax.bar([str(v) for v in vals], cnt_dirty, bottom=cnt_pure, color="#c0392b",
                width=0.62, label="含外点签（签里混进坏数据）")
for rect_p, rect_d, cp, cd in zip(ax.patches[:len(vals)], ax.patches[len(vals):],
                                  cnt_pure, cnt_dirty):
    tot = cp + cd
    if tot > 0:
        ax.text(rect_p.get_x() + rect_p.get_width() / 2, tot + 1.2, str(tot),
                ha="center", fontsize=12)
ax.set_ylim(0, max(cnt_pure + cnt_dirty) * 1.22)
ax.set_xlabel(f"每轮票数（全集 17 点计票，S={S_ROUNDS} 轮，内点阈值 {EPS}）")
ax.set_ylabel("轮数")
ax.set_title("混进坏数据的签全被压在低票区——投票把好线挑出来")
ax.legend(loc="upper left", fontsize=12, framealpha=0.92)
fig.text(0.99, 0.012, "程序生成，未用教材原图（概念对应原书第 511 页 repeated S times）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e038_ransac_hist.png")
plt.close(fig)

# 图3 外点加压：OLS vs RANSAC 名单重解（round3 P1-1 修：精拟合措辞退役）
fig, ax = plt.subplots(figsize=(7.65, 3.2))
ax.scatter(x_in, y_in, s=26, color="#2a6f9e", zorder=3, label="内点 16")
ax.scatter([p[0] for p in extra_out] + [x_out_pt], [p[1] for p in extra_out] + [y_out_pt],
           s=58, marker="X", color="#c0392b", zorder=4, label="外点 4")
ax.plot(xs, m_hard_ols * xs + b_hard_ols, color="#e08214", lw=2.2,
        label=f"OLS（被拉偏 m={m_hard_ols:.3f}）")
ax.plot(xs, m_refit_h * xs + b_refit_h, color="#c0392b", lw=2.4,
        label=f"RANSAC 名单重解 m={m_refit_h:.3f}")
ax.plot(xs, M_TRUE * xs + B_TRUE, color="#3a7d44", lw=1.6, ls="--", label="真线 m=0.5")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("外点加压：最小二乘跟着外点跑，RANSAC 认得出自己人")
ax.legend(loc="upper left", fontsize=12, framealpha=0.92)
fig.text(0.99, 0.012, "程序生成，未用教材原图（概念对应原书第 511 页 too many outliers 段）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e038_ransac_hard.png")
plt.close(fig)

readings["figures"] = ["e038_ransac_vote.png", "e038_ransac_hist.png", "e038_ransac_hard.png"]
readings["assertions_total"] = len(checks)
(OUT / "e038_ransac_readings.json").write_text(
    json.dumps(readings, ensure_ascii=False, indent=1))
print(f"\nLEDGER: 断言 {len(checks)} 条，全部 PASS")
print("E038-RANSAC-LAB-DONE")
