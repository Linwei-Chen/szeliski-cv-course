#!/usr/bin/env python3
"""
E027 · 傅里叶变换Ⅰ：把图像听成和弦（§3.4 前半）

范围：§3.4 Fourier transforms 前半（印刷 pp.142–145 上）：§3.4 引言+正弦探针
（式 3.50 s(x)=sin(2πfx+φi)=sin(ωx+φi)、ω=2πf；式 3.51 o(x)=h(x)∗s(x)=
A sin(ωx+φo)——「convolve the sinusoidal signal s(x) with a filter whose
impulse response is h(x), we get another sinusoid of the same frequency but
different magnitude A and phase φo」；Figure 3.24 p.143「The Fourier Transform
as the response of a filter h(x) to an input sinusoid …」；「The new magnitude
A is called the gain or magnitude of the filter, while the phase difference
∆φ = φo − φi is called the shift or phase」）
＋复值正弦（式 3.52 s(x)=e^{jωx}=cos ωx+j sin ωx；式 3.53 o(x)=h∗s=A e^{j(ωx+φ)}）
＋傅里叶变换定义（式 3.54 H(ω)=F{h(x)}=A e^{jφ}，「a tabulation of the magnitude
and phase response at each frequency」；式 3.55 h(x) ↔ H(ω) 变换对记法）
＋连续域式 3.56 H(ω)=∫h(x)e^{−jωx}dx＋离散域式 3.57 H(k)=(1/N)Σ h(x)e^{−j2πkx/N}
（DFT；「it only makes sense for values in the range k ∈ [−N/2, N/2]…larger
values of k alias with lower frequencies」）
＋FFT（「O(N²)…there exists a faster algorithm called the Fast Fourier
Transform (FFT), which requires only O(N log₂N) operations」；butterfly 蝶形级）
＋性质清单（superposition/shifting/reversal/convolution/correlation/
multiplication/differentiation/domain scaling/energy preservation——二版
「I have removed all of these details」引 Szeliski 2010 §3.1 Tables 3.1–3.3）
＋Table 3.1（p.145）六个可分离小核的频响：「Fourier transforms of the
separable kernels shown in Figure 3.14, obtained by evaluating
Σ_k h(k)e^{−jkω}」——box-3 (1/3)(1+2cosω)、box-5 (1/5)(1+2cosω+2cos2ω)、
linear (1/2)(1+cosω)、binomial (1/4)(1+cosω)²、Sobel 因子 (1/2)[−1,0,1]→
幅值 sinω、corner (1/2)(1−cosω)；「the moving average filters do not
uniformly dampen higher frequencies and hence can lead to ringing artifacts」。
切点=§3.4.1 Two-dimensional Fourier transforms 标题（p.146，归 E028）。

记法（原书 p.142 明言）：本节用 x, y 表空间坐标（非 i, j），j 表虚数单位
（「we use the letter j for the imaginary number」）。

数值实验（32 条正式断言 = 27 数据 + 4 常数核对 C1–C4 + #S 同步 1 项（5 张图）；
另有 #D 区分力注入 4 项——E019 门A round3 新P1-2 惯例，注入类判据由 #D 承担，
不进正式计数、汇总单列）：
  #1  正弦探针：增益与相移（式 3.50/3.51+Figure 3.24 语义；5 条）：box-3 与
      linear 核对已知频率正弦的数值卷积输出（圆形卷积口径），拟合法测增益 A
      与输出相位 φo，与解析频响 H(ω) 的 |H|、arg H 逐点对照（A=|H|、
      ∆φ=φo−φi=arg H）；同频不同初相增益不变；Sobel 探针 A=|sin ω| 与
      Δφ=−π/2 两条专项（图 e027_probe 同参数）
  #2  Table 3.1 六核频响复算（p.145；8 条）：box-3/box-5/linear/binomial/
      Sobel 因子/corner 因子各自按 DTFT 定义 Σ_k h(k)e^{−jkω} 数值复算（17
      个频率点）与解析式逐点对照；binomial 双解析式恒等（(1/4)(1+cosω)² ≡
      (1/8)(3+4cosω+cos2ω)）；原书表 corner 行核列/变换列不一致实证 1 条
  #3  DFT 与 FFT（式 3.57；6 条）：式 3.57 直接实现（含 1/N 归一）与 numpy
      FFT（×1/N）全谱对照（随机实信号 N=32）；实信号 DFT 共轭对称
      H(−k)=conj H(k)（k∈[−N/2,N/2] 意义下）；三例读数 δ/方波/正弦（图
      e027_dft_pairs 同参数）；FFT 提速账（N=2¹³ 直接 O(N²) vs numpy FFT
      计时对照打印，结果逐位一致断言；预热+中位协议）
  #4  频域看卷积（式 3.53 蕴含的逐频率增益相移语义；6 条）：时域卷积 y 由直接
      定义求和 conv_circ_direct 计算（O(N²)、不经 FFT，防循环论证——门A r3
      N3-P1-1），其谱与逐频点乘积对照（fft 口径 fft(h∗s)=fft(s)·fft(h)+式 3.57
      归一口径换算 DFT(y)=N·H(k)·S(k)——「把卷积搬到频域去看」的数值兑现）；
      FFT 卷积实现与直接求和逐点一致；补零正向断言：补零到 N+K−1 后
      ifft(fft·fft)≡np.convolve full（「补零即一致」，门A r4 P3-5）；正弦探针
      的复数版（式 3.53）：h∗e^{jωx}=H(ω)e^{jωx} 两个频率点逐点对照
  #5  ringing 伏笔（p.144「do not uniformly dampen higher frequencies and
      hence can lead to ringing artifacts」，L100–101；2 条）：box-3 频响在
      ω=2π/3 过零、ω>2π/3 为负（负瓣存在），binomial 无负瓣（ω∈[0,π] 恒正）
  C1–C4  常数：式 3.50 ω=2πf 换算（f=1/4 ⇒ sin(πx/2)）；式 3.52 欧拉公式
      （e^{jθ}=cosθ+j sinθ 多点）；式 3.54/3.55 DC 点 H(0)=1=A·e^{j0}（box-3）；
      式 3.57 归一 H(0)=mean(h)（box-5 ⇒ 1/5）
  #S  同步：5 张程序生成图写入 code/figures/ 与 slides/figures/（md5 一致）：
      e027_probe（正弦探针输入→滤波→输出增益/相移标注，Figure 3.24 语义）/
      e027_kernels_freq（Table 3.1 六核频响曲线 2×3 重绘，标「原书表未使用，
      程序重绘」）/e027_dft_pairs（脉冲/方波/正弦三例的信号↔DFT 幅相对照）/
      e027_conv_theorem（时域卷积 vs 频域逐点乘积对照）/
      e027_fft_speed（直接 O(N²) vs FFT 计时 log-log）——均标「原书图未使用」
  #D  区分力（4 项）：D-1 用相关（核翻转）冒充卷积（非对称核 Sobel 因子，
      对称核相关=卷积测不出差异）→ #1 相移断言拦截；D-2 binomial 解析式漏
      cos2ω 项 → #2 拦截；D-3 DFT 漏 1/N 归一 → #3 拦截；D-4 不补零线性
      卷积冒充循环卷积 → #4 拦截（为口播「做普通卷积要先补零」提供实证）

输出：stdout 存档 code/outputs/e027_fourier_run7.txt
"""
import hashlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

FAILS = []
COUNT = {"n": 0}


def chk(label, cond, detail=""):
    COUNT["n"] += 1
    ok = bool(cond)
    if not ok:
        FAILS.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


rng = np.random.default_rng(20261029)

# ---------- 原书算子（忠实式 3.50–3.57 与 Table 3.1） ----------

def dtft(h, w):
    """Table 3.1 口径：Σ_k h(k)·e^{−jkω}，k 为核索引且**中心对齐原点**
    （原书 Figure 3.14 核索引 k=−1,0,1 口径，Table 3.1 解析式为零相位形式）。
    偶长核无中心点，本集六核均为奇长。"""
    h = np.asarray(h, dtype=complex)
    k = np.arange(len(h)) - len(h) // 2
    return (h[None, :] * np.exp(-1j * np.outer(w, k))).sum(axis=1)


def dft_book(x):
    """式 3.57：H(k) = (1/N) Σ_x h(x)·e^{−j2πkx/N}（含 1/N 归一，k=0..N−1）。"""
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)
    return (W @ x) / N


def conv_circ(sig, ker):
    """圆形卷积（周期延拓口径，DFT 对角化语义）：核中心对齐 index 0 后补零，
    ifft(fft(sig)·fft(ker)).real。正弦探针与频域看卷积均用此口径（整周期正弦
    在圆形卷积下无边界效应，与式 3.51「同频正弦只改幅度相位」严格成立）。"""
    N = len(sig)
    ker_full = np.zeros(N)
    ker_full[:len(ker)] = ker
    ker_full = np.roll(ker_full, -(len(ker) // 2))
    out = np.fft.ifft(np.fft.fft(sig) * np.fft.fft(ker_full))
    return out.real if np.isrealobj(sig) else out


def conv_circ_direct(sig, ker):
    """直接按定义的循环卷积：y[n]=Σ_m h[m]·x[(n−m) mod N]——核先补零再 roll 做
    中心对齐，idx=(n−m) mod N 取模索引矩阵 + 一次矩阵乘实现逐点求和（向量化，
    等价 O(N²) 双循环、不经任何 FFT）——卷积定理验证的独立时域参照，防止
    「用 fft 卷积算出的 y 再去验证 fft(y)=fft·fft」的循环论证（门A r3 N3-P1-1）。"""
    N = len(sig)
    ker_full = np.zeros(N)
    ker_full[:len(ker)] = ker
    ker_full = np.roll(ker_full, -(len(ker) // 2))
    idx = (np.arange(N)[None, :] - np.arange(N)[:, None]) % N
    return ker_full @ sig[idx]


# Table 3.1 六核（1D separable 因子；box/linear/binomial 为对称归一 1D 核，
# Sobel/corner 取 Table 3.1 Kernel 列的 1D 因子）
K_BOX3 = np.array([1, 1, 1], float) / 3
K_BOX5 = np.array([1, 1, 1, 1, 1], float) / 5
K_LINEAR = np.array([1, 2, 1], float) / 4
K_BINOM = np.array([1, 4, 6, 4, 1], float) / 16
K_SOBEL1D = np.array([-1, 0, 1], float) / 2          # (1/2)[−1,0,1]
K_CORNER1D = np.array([1, -2, 1], float) / 4         # 幅值谱 ⇒ (1/2)(1−cosω)


# ---------- #1 正弦探针：增益与相移 ----------

def probe_response(h, w, phi_i=0.7, N=512):
    """正弦探针（式 3.50→3.51）：输入 sin(ωx+φi) 过核 h（周期延拓卷积），
    拟合输出正弦的幅度 A 与相位 φo（最小二乘 on sin/cos 基）。"""
    x = np.arange(N)
    s = np.sin(w * x + phi_i)
    o = conv_circ_direct(s, h)
    G = np.stack([np.sin(w * x), np.cos(w * x)], axis=1)
    coef, *_ = np.linalg.lstsq(G, o, rcond=None)
    A = float(np.hypot(coef[0], coef[1]))
    phi_o = float(np.arctan2(coef[1], coef[0]))
    return A, phi_o


def exp1():
    print("== #1 正弦探针：增益与相移（式 3.50/3.51+Figure 3.24，pp.142–143） ==")
    cases = [(K_BOX3, 2 * np.pi / 16, "box-3", 0.7),
             (K_BOX3, 2 * np.pi * 5 / 16, "box-3", -1.3),
             (K_LINEAR, 2 * np.pi * 3 / 16, "linear", 0.7),
             (K_SOBEL1D, 2 * np.pi * 5 / 32, "Sobel", 0.7)]
    worst_A = worst_p = 0.0
    for h, w, name, phi_i in cases:
        A, phi_o = probe_response(h, w, phi_i)
        H = dtft(h, np.array([w]))[0]
        dA = abs(A - abs(H))
        dp = abs((phi_o - phi_i) - np.angle(H)) % (2 * np.pi)
        dp = min(dp, 2 * np.pi - dp)
        worst_A, worst_p = max(worst_A, dA), max(worst_p, dp)
        print(f"    {name} ω={w:.4f} φi={phi_i:+.1f}: A={A:.9f} vs |H|={abs(H):.9f}"
              f"  ∆φ={(phi_o - phi_i) % (2 * np.pi):.9f} vs argH={np.angle(H) % (2 * np.pi):.9f}")
    chk("#1 增益 A=|H(ω)|（3 组探针，max 差<1e-9）", worst_A < 1e-9, f"max|ΔA|={worst_A:.2e}")
    chk("#1 相移 ∆φ=φo−φi=arg H(ω)（3 组，max 差<1e-6）", worst_p < 1e-6, f"max|Δφ|={worst_p:.2e}")
    # 同频不同相位：增益不变（式 3.51 对 φi 的线性性）
    A1, _ = probe_response(K_BOX3, 2 * np.pi / 16, phi_i=0.7)
    A2, _ = probe_response(K_BOX3, 2 * np.pi / 16, phi_i=2.1)
    chk("#1 同频不同初相增益不变（|A(φi=0.7)−A(φi=2.1)|<1e-9）", abs(A1 - A2) < 1e-9,
        f"A1={A1:.12f} A2={A2:.12f}")
    # Sobel 因子解析核对（图 e027_probe 同参数）：|H|=|sin ω|≈0.8315、Δφ=−π/2（差分核滞后 90°）
    w_s = 2 * np.pi * 5 / 32
    A_s, phi_s = probe_response(K_SOBEL1D, w_s, 0.7)
    dphi_s = (phi_s - 0.7 + np.pi) % (2 * np.pi) - np.pi
    chk("#1 Sobel 探针 A=|sin ω|（ω=2π·5/32=图 e027_probe 参数，|A−|sinω||<1e-9）",
        abs(A_s - abs(np.sin(w_s))) < 1e-9, f"A={A_s:.6f} |sin ω|={abs(np.sin(w_s)):.6f}")
    chk("#1 Sobel 探针 Δφ=−π/2（差分核滞后 90°，|Δφ+π/2|<1e-9）",
        abs(dphi_s + np.pi / 2) < 1e-9, f"Δφ={dphi_s:.6f} rad")


# ---------- #2 Table 3.1 六核频响复算 ----------

def exp2():
    print("== #2 Table 3.1 六核频响复算（p.145，DTFT 定义 vs 解析式） ==")
    print("    口径注记：原书表 corner 行 Kernel 列 (1/2)[−1,2,−1] 与 Transform 列 (1/2)(1−cosω)")
    print("    不自洽——前者 DTFT=1−cosω，恰差 2 倍（原书排印不一致；PDF p.171 版面核验，")
    print("    存档 evidence/e027_p145_table31_render.png）。本实验取 Transform 列口径")
    print("    K=(1/4)[1,−2,1]，频响曲线与 Transform 列解析式严格一致（见下 chk）。")
    ws = np.linspace(0, np.pi, 17)
    tables = [
        (K_BOX3, lambda w: (1 + 2 * np.cos(w)) / 3, "box-3 (1/3)(1+2cosω)"),
        (K_BOX5, lambda w: (1 + 2 * np.cos(w) + 2 * np.cos(2 * w)) / 5, "box-5 (1/5)(1+2cosω+2cos2ω)"),
        (K_LINEAR, lambda w: (1 + np.cos(w)) / 2, "linear (1/2)(1+cosω)"),
        (K_BINOM, lambda w: (1 + np.cos(w)) ** 2 / 4, "binomial (1/4)(1+cosω)²"),
        (K_SOBEL1D, lambda w: np.sin(w), "Sobel 因子 (1/2)[−1,0,1] ⇒ |H|=sinω"),
        (K_CORNER1D, lambda w: (1 - np.cos(w)) / 2, "corner 因子 (1/4)[1,−2,1] ⇒ (1/2)(1−cosω)"),
    ]
    for h, ana, name in tables:
        num = dtft(h, ws)
        if "Sobel" in name:
            err = float(np.abs(np.abs(num) - ana(ws)).max())   # 纯虚谱取幅值
        elif "corner" in name:
            # corner 因子谱为负实值 [−(1/2)(1−cosω)]（二阶差分方向约定，E023 CORNER3
            # 为同核整体反向）；Table 3.1 Transform 列画幅值曲线 → 取幅值对照
            err = float(np.abs(np.abs(num) - ana(ws)).max())
        else:
            err = float(np.abs(num.real - ana(ws)).max())
        chk(f"#2 {name}（17 点 max 差<1e-12）", err < 1e-12, f"max err={err:.2e}")
    ident = float(np.abs((1 + np.cos(ws)) ** 2 / 4 - (3 + 4 * np.cos(ws) + np.cos(2 * ws)) / 8).max())
    chk("#2 binomial 双解析式恒等（(1/4)(1+cosω)² ≡ (1/8)(3+4cosω+cos2ω)，17 点）",
        ident < 1e-15, f"max diff={ident:.2e}")
    # 原书表内不一致实证：核列 (1/2)[−1,2,−1] 的 DTFT 恰为 1−cosω（=2×Transform 列）
    k_tbl = np.array([-1, 2, -1], float) / 2
    H_tbl = dtft(k_tbl, ws).real
    d_full = float(np.abs(H_tbl - (1 - np.cos(ws))).max())
    d_half = float(np.abs(H_tbl - (1 - np.cos(ws)) / 2).max())
    chk("#2 原书表核列 (1/2)[−1,2,−1] DTFT=1−cosω、≠Transform 列 (1/2)(1−cosω)（差 2 倍，表内不一致实证）",
        d_full < 1e-12 and d_half > 1e-3, f"max|DTFT−(1−cosω)|={d_full:.2e} max|DTFT−(1/2)(1−cosω)|={d_half:.2e}")


# ---------- #3 DFT 与 FFT ----------

def exp3():
    print("== #3 DFT 与 FFT（式 3.57，p.144） ==")
    x = rng.random(32) - 0.5
    H_book = dft_book(x)
    H_np = np.fft.fft(x) / 32
    err = float(np.abs(H_book - H_np).max())
    chk("#3 式 3.57 直接实现（含 1/N）与 numpy FFT/32 全谱一致（N=32，max 差<1e-12）",
        err < 1e-12, f"max err={err:.2e}")
    # 实信号共轭对称：H(−k)=conj H(k)（k∈[1,N/2]，负频率=H(N−k)）
    cs = float(np.abs(H_book[1:16] - np.conj(H_book[31:16:-1])).max())
    chk("#3 实信号 DFT 共轭对称 H(N−k)=conj H(k)（k∈[−N/2,N/2] 意义，max 差<1e-12）",
        cs < 1e-12, f"max err={cs:.2e}")
    # 三例 DFT 读数（图 e027_dft_pairs 同参数 N=32）：δ/方波/正弦
    d = dft_book(np.r_[1.0, np.zeros(31)])
    d_err = float(np.abs(np.abs(d) - 1 / 32).max())
    chk("#3 δ 谱全点等幅 |H(k)|=1/N=1/32（N=32，max 偏差<1e-12）",
        d_err < 1e-12, f"max||H|−1/32|={d_err:.2e}")
    sq = np.r_[np.ones(8), np.zeros(24)]
    S = dft_book(sq)
    chk("#3 方波（宽 8）谱：主瓣峰在 k=0、|H(0)|=8/32=0.25，第一零点 k=±4（sinc 形包络）",
        int(np.abs(S).argmax()) == 0 and abs(abs(S[0]) - 0.25) < 1e-12
        and abs(S[4]) < 1e-12 and abs(S[-4]) < 1e-12,
        f"|H(0)|={abs(S[0]):.4f} |H(±4)|={abs(S[4]):.2e}/{abs(S[28]):.2e}")
    sn = dft_book(np.sin(2 * np.pi * 3 / 16 * np.arange(32)))
    mag = np.abs(sn)
    nz = np.flatnonzero(mag > 1e-9)
    chk("#3 正弦 f=3/16 谱仅 ±k₀ 两条非零（k=6 与 26，即 ±3/16，|H|=0.5 各半）",
        set(nz.tolist()) == {6, 26} and abs(mag[6] - 0.5) < 1e-9 and abs(mag[26] - 0.5) < 1e-9,
        f"非零 bin={sorted(nz.tolist())} |H|={mag[6]:.4f}/{mag[26]:.4f}")
    # FFT 提速账（图 e027_fft_speed 同协议）：预热 1 次弃掉，再取中位（直接法 3 次、
    # FFT 5 次）压单次抖动——单次计时 run 间可差近一倍（冷启动/降频），
    # 预热+中位让读数只反映稳态性能、跨次可复现；直接法 3 次（r4 P3-7：
    # 两样本中位=均值，3 次才是真中位）
    def t_med(fn, reps, warmup=1):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]

    print("    全 N 计时（预热+中位：预热 1 次弃掉，直接法 3 次、FFT 5 次各取中位；图 e027_fft_speed 的数据另跑、不计入本表，逐次有波动，读数以本存档为准）：")
    sweep = []
    for e in range(5, 14):
        Nn = 2 ** e
        xg = rng.random(Nn) - 0.5
        td = t_med(lambda: dft_book(xg), 3)
        tf = t_med(lambda: np.fft.fft(xg), 5)
        sweep.append((Nn, td, tf))
        print(f"      N=2^{e}={Nn:5d}: 直接 {td * 1000:9.3f} ms  FFT {tf * 1000:7.4f} ms"
              f"  ≈{td / max(tf, 1e-9):8.0f}×")
    N = 2 ** 13
    x = rng.random(N) - 0.5
    H_direct = dft_book(x)
    H_fast = np.fft.fft(x) / N
    err = float(np.abs(H_direct - H_fast).max())
    t_direct, t_fast = sweep[-1][1], sweep[-1][2]
    speed = t_direct / max(t_fast, 1e-9)
    print(f"    主读数 N=8192（=扫描末行中位）：直接 O(N²) {t_direct * 1000:.1f} ms vs FFT {t_fast * 1000:.3f} ms"
          f"（提速 ≈{speed:.0f}×；理论 O(N/log₂N)≈{N / np.log2(N):.0f}×）")
    chk("#3 直接 O(N²) 与 FFT 结果逐位一致（N=2¹³，max 差<1e-9）", err < 1e-9, f"max err={err:.2e}")


# ---------- #4 频域看卷积 ----------

def exp4():
    print("== #4 频域看卷积（式 3.53 语义：卷积对每个复正弦只乘 H(k)；循环卷积口径） ==")
    print("    口径注记：fft(h∗s)=fft(s)·fft(h) 对应**循环**卷积（周期延拓）——本实验")
    print("    两侧均按循环卷积计算（信号按周期、核补零后 roll 对齐），恒等式成立的")
    print("    原因是两侧同为循环口径，而非「无环绕」；做普通线性卷积需先补零再变换。")
    print("    独立性说明：时域卷积 y 由**直接定义求和** conv_circ_direct 计算（O(N²)、")
    print("    不经 FFT），再对照其谱与逐频点乘积——避免「用 fft 卷积造 y 再验证 fft(y)")
    print("    =fft·fft」的循环论证（门A r3 N3-P1-1）；FFT 卷积实现另与直接求和逐点对照。")
    N = 64
    x = rng.random(N) - 0.5
    h = K_BOX5
    y = conv_circ_direct(x, h)
    h_full = np.roll(np.pad(h, (0, N - len(h))), -(len(h) // 2))   # 核中心对齐 index 0 补零
    H = dft_book(h_full)                # 核的谱（圆形卷积定理口径）
    S = dft_book(x)
    Y = dft_book(y)
    # 卷积定理（fft 口径）：fft(y)=fft(s)·fft(h_full)——y 为直接定义求和所得，
    # 断言非平凡；换算到式 3.57 的 1/N 归一口径即 DFT(y)=N·H(k)·S(k)
    err = float(np.abs(np.fft.fft(y) - np.fft.fft(x) * np.fft.fft(h_full)).max())
    y_fftref = conv_circ(x, h)          # FFT 卷积实现产出（对照路径，非主断言对象）
    err_fftref = float(np.abs(np.fft.fft(y_fftref) - np.fft.fft(x) * np.fft.fft(h_full)).max())
    print(f"    y 来源标记：主断言 y=conv_circ_direct(x,h)（直接定义求和，不经 FFT）")
    print(f"    双路 err 对照：y_direct err={err:.2e}（主断言对象）；y_fft err={err_fftref:.2e}"
          f"（FFT 卷积实现产出——两路仅舍入差异，主断言不使用 y_fft）")
    chk("#4 时域卷积谱 = 逐频点乘积（y=直接定义求和·循环卷积：fft(h∗s)=fft(s)·fft(h)，box-5，N=64）",
        err < 1e-9, f"max err={err:.2e}")
    # 补零正向断言（门A r4 P3-5）：补零到线性卷积长度 N+K−1 后，频域乘积的
    # ifft 应逐点等于 np.convolve(...,'full')——「补零即一致」的正向实证，
    # 与 D-4（不补零→边界失配）合起来完整支撑口播「做普通卷积，要先补零」
    K = len(h)
    y_lin = np.convolve(x, h, mode="full")
    y_freq = np.fft.ifft(np.fft.fft(x, N + K - 1) * np.fft.fft(np.pad(h, (0, N - 1)))).real
    err_zp = float(np.abs(y_freq - y_lin).max())
    chk("#4 补零后频域乘积=线性卷积（正向：ifft(fft·fft)≡convolve full，长 N+K−1，max 差<1e-12）",
        err_zp < 1e-12, f"max err={err_zp:.2e}")
    err_c = float(np.abs(conv_circ(x, h) - y).max())
    chk("#4 FFT 卷积实现与直接定义求和逐点一致（conv_circ vs conv_circ_direct，防实现漂移）",
        err_c < 1e-12, f"max err={err_c:.2e}")
    err_n = float(np.abs(Y - len(x) * H * S).max())
    chk("#4 式 3.57 归一口径换算：DFT(y)=N·H(k)·S(k)（N 因子来自 1/N 归一）",
        err_n < 1e-9, f"max err={err_n:.2e}")
    # 复正弦版（式 3.53）：h ∗ e^{jωx} = H(ω)·e^{jωx}（卷积用直接定义求和）
    xs = np.arange(N)
    for w in (2 * np.pi * 3 / 16, 2 * np.pi * 7 / 16):
        s = np.exp(1j * w * xs)
        o = conv_circ_direct(s.real, h) + 1j * conv_circ_direct(s.imag, h)
        Hw = dtft(h, np.array([w]))[0]
        err = float(np.abs(o - Hw * s).max())
        chk(f"#4 h∗e^{{jωx}}=H(ω)e^{{jωx}}（式 3.53，卷积=直接定义求和，ω={w:.3f}，max 差<1e-9）", err < 1e-9,
            f"max err={err:.2e}")


# ---------- #5 ringing 伏笔 ----------

def exp5():
    print("== #5 ringing 伏笔（p.144「do not uniformly dampen…ringing artifacts」，L100–101） ==")
    ws = np.linspace(0, np.pi, 2001)
    H_exact = dtft(K_BOX3, np.array([2 * np.pi / 3]))[0]
    H3 = dtft(K_BOX3, ws).real
    w_star = 2 * np.pi / 3
    chk("#5 box-3 频响在 ω=2π/3 精确过零（|H|<1e-15）且 ω>2π/3 出现负瓣（min<0）",
        abs(H_exact) < 1e-15 and H3[ws > w_star].min() < 0,
        f"|H(2π/3)|={abs(H_exact):.2e} 负瓣 min={H3.min():.4f} @ω={ws[H3.argmin()]:.3f}")
    Hb = dtft(K_BINOM, ws).real
    edge_zero = Hb.argmin() == 0 or Hb.argmin() == len(Hb) - 1
    chk("#5 binomial 频响无负瓣（ω∈[0,π] H≥0，零点仅在 ω=π 端点→抑制振铃）",
        Hb.min() >= 0 and edge_zero and abs(Hb[-1]) < 1e-15,
        f"min={Hb.min():.4f} argmin@ω={ws[Hb.argmin()]:.3f}")


# ---------- 常数核对 ----------

def constants():
    print("== C1–C4 常数核对 ==")
    x = np.arange(8)
    chk("C1 式 3.50 换算：f=1/4 ⇒ sin(2π·(1/4)x)=sin(πx/2) 逐点一致",
        np.abs(np.sin(2 * np.pi * 0.25 * x) - np.sin(np.pi * x / 2)).max() < 1e-15)
    th = np.linspace(0, 2 * np.pi, 9)[:-1]
    euler = np.exp(1j * th) - (np.cos(th) + 1j * np.sin(th))
    chk("C2 式 3.52 欧拉公式 e^{jθ}=cosθ+j sinθ（8 点，max 差<1e-15）",
        float(np.abs(euler).max()) < 1e-15)
    H0 = dtft(K_BOX3, np.array([0.0]))[0]
    chk("C3 式 3.54/3.55 DC 点：H(0)=A·e^{j0}=1（box-3 归一化和为 1）",
        abs(H0 - 1.0) < 1e-15, f"H(0)={H0:.12f}")
    H0b = dft_book(K_BOX5)[0]
    chk("C4 式 3.57 归一：H(0)=mean(h)（box-5 ⇒ 1/5）", abs(H0b - 0.2) < 1e-15,
        f"H(0)={H0b:.12f}")


# ---------- 图生成 ----------

def save_fig(fig, name):
    for d in ("figures", "../slides/figures"):
        p = pathlib.Path(__file__).parent / d
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / name, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    NOTE = "（程序生成，原书图未使用）"
    names = []

    # ① e027_probe：正弦探针（Figure 3.24 语义）——用非对称 Sobel 因子：
    # 对称核圆形卷积零相移（Δφ=0），Sobel 因子 Δφ=π/2 相移演示显著
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    xs = np.arange(128)
    w = 2 * np.pi * 5 / 32
    phi_i = 0.7
    s = np.sin(w * xs + phi_i)
    o = conv_circ_direct(s, K_SOBEL1D)
    A, phi_o = probe_response(K_SOBEL1D, w, phi_i, N=128)
    axes[0].plot(xs, s, color="#888", lw=1.2, label="输入 s(x)=sin(ωx+φi)")
    axes[1].plot(xs, o, color="#b56a2c", lw=1.6, label="输出 o(x)=h∗s")
    env = float(np.abs(dtft(K_SOBEL1D, np.array([w]))[0]))
    for ax in axes:
        ax.axhline(0, color="#bbb", lw=0.6)
    axes[1].plot(xs, env * np.sin(w * xs + phi_o), color="#2c5f8a", lw=0.9, ls="--",
                 label="A·sin(ωx+φo) 拟合")
    axes[0].set_ylabel("幅度"); axes[1].set_ylabel("幅度")
    axes[1].set_xlabel("x（空间坐标，原书 §3.4 记法）")
    axes[0].legend(fontsize=8, loc="upper right"); axes[1].legend(fontsize=8, loc="upper right")
    dphi = (phi_o - phi_i + np.pi) % (2 * np.pi) - np.pi      # wrap 到 (−π,π]
    axes[1].set_title(f"正弦探针（Sobel 因子 (1/2)[−1,0,1]）：增益 A=|H(ω)|={env:.3f}，"
                      f"相移 Δφ=φo−φi={dphi:.3f} rad（差分核滞后 90°）{NOTE}",
                      fontsize=10)
    fig.tight_layout(); save_fig(fig, "e027_probe.png"); names.append("e027_probe.png")

    # ② e027_kernels_freq：Table 3.1 六核频响曲线重绘
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
    ws = np.linspace(0, np.pi, 400)
    six = [("box-3", K_BOX3, lambda w: (1 + 2 * np.cos(w)) / 3),
           ("box-5", K_BOX5, lambda w: (1 + 2 * np.cos(w) + 2 * np.cos(2 * w)) / 5),
           ("linear", K_LINEAR, lambda w: (1 + np.cos(w)) / 2),
           ("binomial", K_BINOM, lambda w: (1 + np.cos(w)) ** 2 / 4),
           ("Sobel", K_SOBEL1D, lambda w: np.sin(w)),
           ("corner（幅值）", K_CORNER1D, lambda w: (1 - np.cos(w)) / 2)]
    for ax, (name, h, ana) in zip(axes.ravel(), six):
        num = dtft(h, ws)
        plot_val = np.abs(num) if name in ("Sobel", "corner") else num.real
        ax.plot(ws / (2 * np.pi), plot_val, lw=1.6,
                color="#2c5f8a", label="DTFT 复算")
        ax.plot(ws / (2 * np.pi), ana(ws), lw=0.9, ls="--", color="#b56a2c", label="解析式")
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.6, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Table 3.1 六个可分离小核的频响（对应原书第 145 页；原书表未使用，程序重绘）", fontsize=11)
    fig.supxlabel("归一化频率 f=ω/2π")
    fig.supylabel("H(ω)")
    fig.tight_layout(); save_fig(fig, "e027_kernels_freq.png"); names.append("e027_kernels_freq.png")

    # ③ e027_dft_pairs：三例信号↔DFT 幅值
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 6.4))
    N = 32
    sigs = [("单位脉冲 δ(x)", np.r_[1.0, np.zeros(N - 1)]),
            ("方波（宽 8）", np.r_[np.ones(8), np.zeros(N - 8)]),
            ("正弦 f=3/16", np.sin(2 * np.pi * 3 / 16 * np.arange(N)))]
    for (name, sg), (ax_t, ax_f) in zip(sigs, axes):
        ax_t.plot(np.arange(N), sg, lw=1.4, color="#2c5f8a")
        ax_t.set_title(f"{name}（时域）", fontsize=9)
        S = dft_book(sg)
        freqs = np.fft.fftfreq(N)     # k∈[−N/2, N/2) 口径
        order = np.argsort(freqs)
        ax_f.stem(freqs[order], np.abs(S)[order], markerfmt=".", basefmt=":")
        ax_f.set_title("幅值 |H(k)|（频域，k∈[−N/2,N/2]）", fontsize=9)
        ax_f.set_xlim(-0.55, 0.55)
    fig.suptitle("三例「信号 ↔ DFT 幅值」对照（式 3.57，1/N 归一）" + NOTE, fontsize=11)
    fig.tight_layout(); save_fig(fig, "e027_dft_pairs.png"); names.append("e027_dft_pairs.png")

    # ④ e027_conv_theorem：时域卷积 vs 频域逐点乘积
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    N = 64
    # 图数据用独立同种子生成器（r4 材料不足 2：来源可查证、逐次可复现、与主
    # 序列无关）；本图自身 err 打印进存档，图注引用本图读数
    x = np.random.default_rng(20261029).random(N) - 0.5
    y = conv_circ_direct(x, K_BOX5)
    axes[0].plot(np.arange(N), x, lw=1.0, color="#888", label="输入 s(x)")
    axes[0].plot(np.arange(N), y, lw=1.4, color="#b56a2c", label="卷积输出 h∗s")
    axes[0].set_title("时域：box-5 卷积（随机信号）", fontsize=10)
    axes[0].legend(fontsize=8)
    h_full_f = np.roll(np.pad(K_BOX5, (0, N - len(K_BOX5))), -2)
    Hk = dft_book(h_full_f); Sk = dft_book(x); Yk = dft_book(y)
    # 频域侧独立计算（fft 未归一口径；y 来自 conv_circ_direct，不经 FFT——
    # 两条曲线是否重合才是定理的真检验，门A r3 N3-P2-2 处方 a；
    # 旧版第二条曲线误画 Yk_fft 自身，已修）
    Yk_fft = np.abs(np.fft.fft(y)); HkS_fft = np.abs(np.fft.fft(x) * np.fft.fft(h_full_f))
    axes[1].plot(np.fft.fftfreq(N), Yk_fft, lw=1.6, color="#b56a2c", label="|fft(h∗s)|")
    axes[1].plot(np.fft.fftfreq(N), HkS_fft, lw=0.9, ls="--", color="#2c5f8a",
                 label="|fft(h)·fft(s)|")
    axes[1].set_xlim(-0.55, 0.55); axes[1].set_ylim(bottom=0)
    axes[1].set_title("频域：逐频点乘积（fft 口径，两曲线重合）", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle("把卷积搬到频域去看（上一集承诺兑现；式 3.53 语义）" + NOTE, fontsize=11)
    err_fig = float(np.abs(np.fft.fft(y) - np.fft.fft(x) * np.fft.fft(h_full_f)).max())
    print(f"    [图] e027_conv_theorem：N={N}、x=default_rng(20261029) 前 {N} 点−0.5"
          f"（独立同种子、可复现）；本图数据自身 err={err_fig:.2e}（图注引用此值）")
    fig.tight_layout(); save_fig(fig, "e027_conv_theorem.png"); names.append("e027_conv_theorem.png")

    # ⑤ e027_fft_speed：提速账
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ns, t_dir, t_fft = [], [], []
    for e in range(5, 14):
        N = 2 ** e
        x = rng.random(N) - 0.5
        t0 = time.perf_counter(); dft_book(x); t_dir.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); np.fft.fft(x); t_fft.append(time.perf_counter() - t0)
        ns.append(N)
    ax.loglog(ns, t_dir, "o-", lw=1.5, color="#b56a2c", label="直接按式 3.57 求和 O(N²)")
    ax.loglog(ns, t_fft, "s-", lw=1.5, color="#2c5f8a", label="FFT O(N log₂N)")
    # 提速倍数不进图：计时逐次波动，图上标数字会与文本存档对不齐——倍数结论只由 run 存档承载
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FixedFormatter(
        ["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"]))
    ax.set_xlabel("信号长度 N"); ax.set_ylabel("耗时（s）")
    ax.set_title("FFT 提速账（原书第 144 页「O(N²)…only O(N log₂N)」）" + NOTE, fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); save_fig(fig, "e027_fft_speed.png"); names.append("e027_fft_speed.png")

    # 同步检查：figures/ 与 slides/figures/ md5 一致
    ok = True
    det = []
    for nm in names:
        a = pathlib.Path(__file__).parent / "figures" / nm
        b = pathlib.Path(__file__).parent / "../slides/figures" / nm
        if not a.exists() or not b.exists() or hashlib.md5(a.read_bytes()).hexdigest() != hashlib.md5(b.read_bytes()).hexdigest():
            ok = False
        det.append(f"{nm}={hashlib.md5(a.read_bytes()).hexdigest()[:8] if a.exists() else 'MISSING'}")
    chk("#S 程序生成图 5 张写入两处且 md5 一致（probe/kernels_freq/dft_pairs/conv_theorem/fft_speed）",
        ok and len(names) == 5, " ".join(det))


# ---------- 区分力注入 ----------

def 区分力():
    print("== #D 区分力注入（4 项，不计数） ==")
    # D-1 相关（核翻转）冒充卷积 → #1 相移断言应拦截
    try:
        # 非对称核（Sobel 因子 (1/2)[−1,0,1]）：相关 = 翻转核卷积，两者相移差
        # 2·ω·center≠0（对称核如 box-3 相关=卷积，测不出差异）
        w = 2 * np.pi * 5 / 16
        x = np.arange(512)
        s = np.sin(w * x + 0.7)
        o_corr = conv_circ_direct(s, K_SOBEL1D[::-1])           # 相关 = 翻转核卷积（直接定义求和）
        G = np.stack([np.sin(w * x), np.cos(w * x)], axis=1)
        coef, *_ = np.linalg.lstsq(G, o_corr, rcond=None)
        phi_o = float(np.arctan2(coef[1], coef[0]))
        H = dtft(K_SOBEL1D, np.array([w]))[0]
        dp = abs((phi_o - 0.7) - np.angle(H)) % (2 * np.pi)
        dp = min(dp, 2 * np.pi - dp)
        chk("D-1 相关冒充卷积 → 相移断言拦截", dp > 1e-3, f"|Δφ|={dp:.4f}（应显著非零）")
    except Exception as e:
        chk("D-1 相关冒充卷积 → 相移断言拦截", False, f"异常: {e}")
    # D-2 binomial 解析式漏 cos2ω 项 → #2 拦截
    ws = np.linspace(0, np.pi, 17)
    wrong = (1 + np.cos(ws)) / 2            # 用 linear 冒充 binomial
    right = (1 + np.cos(ws)) ** 2 / 4
    chk("D-2 binomial 解析式漏 cos2ω 项 → #2 拦截", float(np.abs(wrong - right).max()) > 1e-3,
        f"max diff={float(np.abs(wrong - right).max()):.4f}")
    # D-3 DFT 漏 1/N 归一 → #3 拦截
    x = rng.random(32) - 0.5
    err = float(np.abs(dft_book(x) * 32 - np.fft.fft(x) / 32).max())
    chk("D-3 DFT 漏 1/N 归一 → #3 拦截", err > 1e-3, f"max err={err:.4f}")
    # D-4 不补零的线性卷积冒充循环卷积 → #4 拦截（门A r3 N3-P1-1 处方 2：
    # np.convolve 无环绕，边界处与循环口径的逐频点乘积不符——同时为口播
    # 「做普通卷积要先补零」提供实证）
    x = rng.random(64) - 0.5
    y_lin = np.convolve(x, K_BOX5, mode="same")     # 线性卷积（无环绕）
    h_full = np.roll(np.pad(K_BOX5, (0, 64 - len(K_BOX5))), -(len(K_BOX5) // 2))
    err = float(np.abs(np.fft.fft(y_lin) - np.fft.fft(x) * np.fft.fft(h_full)).max())
    chk("D-4 不补零线性卷积冒充循环卷积 → #4 拦截", err > 1e-6,
        f"max err={err:.4f}（边界环绕差异应显著）")


# ---------- 主流程 ----------

def main():
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    constants()
    make_figs()
    n_main, f_main = COUNT["n"], len(FAILS)
    区分力()
    n_d, f_d = COUNT["n"] - n_main, len(FAILS) - f_main
    print(f"\n汇总：{n_main - f_main} 条 PASS / {f_main} FAIL"
          f"＋区分力 {n_d - f_d}/{n_d}（注入自检，不计数）")
    if FAILS:
        print("FAIL 明细：", FAILS)
        sys.exit(1)
    print("DONE")


if __name__ == "__main__":
    main()
