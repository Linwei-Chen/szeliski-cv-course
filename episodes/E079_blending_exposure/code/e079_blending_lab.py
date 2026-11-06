#!/usr/bin/env python3
"""E079 实验：羽化/拉普拉斯金字塔/梯度域混合与曝光补偿（程序生成，一维剖面）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP/"code/outputs"; FIG=OUT/"figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER,INK,ACCENT,BLUE,GREEN,PURPLE="#f6f0e6","#1c2437","#a23b2a","#3d6a99","#2e7d32","#7b4f9d"
rng=np.random.default_rng(11)
N=256; x=np.arange(N)
base=0.50+0.15*np.sin(2*np.pi*x/32)+0.08*np.sin(2*np.pi*x/8)
L=base+rng.normal(0,0.004,N)
R=1.25*base+rng.normal(0,0.004,N)
ov=slice(96,161); left_only=np.arange(N)<=95; right_only=np.arange(N)>=161
alpha=np.ones(N); alpha[96:161]=np.linspace(1,0,65); alpha[161:]=0
hard=np.where(x<=127,L,R)
feather=alpha*L+(1-alpha)*R
# 曝光补偿：重叠区线性拟合 L≈aR+b
a,b=np.polyfit(R[ov],L[ov],1); Rc=a*R+b
# 对数域增益
c=np.mean(np.log(L[ov])-np.log(R[ov])); log_gain=float(np.exp(c))
feather_c=alpha*L+(1-alpha)*Rc
hard_c=np.where(x<=127,L,Rc)
# 拉普拉斯金字塔
def down(v): return gaussian_filter1d(v,1.2)[::2]
def up(v,n): return np.interp(np.linspace(0,len(v)-1,n),np.arange(len(v)),v)
def lap(v,levels=3):
    G=[v]
    for _ in range(levels-1): G.append(down(G[-1]))
    return [G[i]-up(G[i+1],len(G[i])) for i in range(levels-1)]+[G[-1]]
def blend_pyr(A,B,al,levels=3):
    LA=lap(A,levels); LB=lap(B,levels); alv=[al]
    for _ in range(levels-1): alv.append(down(alv[-1]))
    C=alv[-1]*LA[-1]+(1-alv[-1])*LB[-1]
    for i in range(levels-2,-1,-1):
        C=up(C,len(LA[i]))+alv[i]*LA[i]+(1-alv[i])*LB[i]
    return C
pyr=blend_pyr(L,Rc,alpha,3)
# 梯度域：拼接目标梯度后积分
d=alpha[:-1]*np.diff(L)+(1-alpha[:-1])*np.diff(Rc)
grad=np.r_[L[0],L[0]+np.cumsum(d)]
def maxstep(v): return float(np.max(np.abs(np.diff(v[100:156]))))
def seamjump(v): return float(abs(v[128]-v[127]))
def rmse(v): return float(np.sqrt(np.mean((v-base)**2)))
base_jump=seamjump(base)
sj_hard=seamjump(hard); sj_feather=seamjump(feather); sj_hard_c=seamjump(hard_c)
sj_feather_c=seamjump(feather_c); sj_pyr=seamjump(pyr); sj_grad=seamjump(grad)
ms_hard=maxstep(hard); ms_feather=maxstep(feather); ms_hard_c=maxstep(hard_c)
ms_feather_c=maxstep(feather_c); ms_pyr=maxstep(pyr); ms_grad=maxstep(grad)
rmse_hard=rmse(hard); rmse_feather=rmse(feather); rmse_feather_c=rmse(feather_c)
rmse_pyr=rmse(pyr); rmse_grad=rmse(grad)
gain_rel=float(abs(log_gain-a)/a)
double_edge=float(np.max(np.abs(np.diff(grad[100:156])))/max(np.max(np.abs(np.diff(L[100:156]))),np.max(np.abs(np.diff(Rc[100:156])))))

# ── 图 1：混合管线 ──
fig,ax=plt.subplots(figsize=(9.4,5.8),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
def box(x,y,w,h,t,b,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc="#fffdf8",ec=c,lw=1.6))
    ax.text(x+w/2,y+h-0.06,t,ha="center",va="center",fontsize=16.5,color=c,fontweight="bold")
    ax.text(x+w/2,y+h/2-0.04,b,ha="center",va="center",fontsize=15,color=INK,linespacing=1.5)
box(0.02,0.44,0.20,0.32,"两张图","曝光不同\n位置已对齐",BLUE)
box(0.27,0.44,0.20,0.32,"找缝","避开运动物\n选重叠区",ACCENT)
box(0.52,0.44,0.20,0.32,"曝光补偿","先把亮度\n拉到同水平",PURPLE)
box(0.77,0.44,0.21,0.32,"混合","羽化/金字塔\n梯度域",GREEN)
for a0,b0 in [((0.22,0.60),(0.27,0.60)),((0.47,0.60),(0.52,0.60)),((0.72,0.60),(0.77,0.60))]:
    ax.add_patch(FancyArrowPatch(a0,b0,arrowstyle="-|>",mutation_scale=16,lw=1.6,color="#7a5c2e",connectionstyle="arc3,rad=0.05"))
ax.text(0.5,0.92,"无缝不是抹平，是分频段处理",ha="center",va="center",fontsize=18,color=INK,fontweight="bold")
ax.text(0.5,0.14,"顺序：先补偿亮度，再做分频段混合",ha="center",va="center",fontsize=14.5,color=ACCENT)
fig.savefig(FIG/"e079_pipeline.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 2：方法对照 ──
fig,ax=plt.subplots(figsize=(10.2,5.6),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.axhline(0,color=INK,lw=1.2,ls=":",label="真实信号=0")
ax.plot(x,hard-base,color=ACCENT,lw=1.9,label=f"硬拼接 RMSE {rmse_hard:.3f}")
ax.plot(x,feather-base,color=BLUE,lw=1.9,label=f"羽化 RMSE {rmse_feather:.3f}")
ax.plot(x,feather_c-base,color=GREEN,lw=1.9,label=f"曝光+羽化 RMSE {rmse_feather_c:.3f}")
ax.plot(x,pyr-base,color=PURPLE,lw=1.9,label=f"金字塔 RMSE {rmse_pyr:.3f}")
ax.plot(x,grad-base,color="#c58a00",lw=1.9,label=f"梯度域 RMSE {rmse_grad:.3f}")
ax.axvspan(96,160,color="#ddd2bd",alpha=0.35)
ax.set_xlabel("像素位置",fontsize=15); ax.set_ylabel("与真实信号的亮度差",fontsize=15)
ax.set_title("同一条缝，五种处理：离真实越近越好",fontsize=17,color=INK)
ax.tick_params(labelsize=12.5); ax.legend(fontsize=12.5,ncol=2); ax.grid(alpha=0.22)
fig.tight_layout(); fig.savefig(FIG/"e079_compare.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 3：接缝跳变条形图 ──
fig,ax=plt.subplots(figsize=(8.8,5.4),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
labels=["硬拼接","羽化","曝光+羽化","金字塔","梯度域"]
vals=[rmse_hard,rmse_feather,rmse_feather_c,rmse_pyr,rmse_grad]
colors=[ACCENT,BLUE,GREEN,PURPLE,"#c58a00"]
bars=ax.bar(labels,vals,color=colors,width=0.6)
for bb,v in zip(bars,vals): ax.text(bb.get_x()+bb.get_width()/2,v+0.002,f"{v:.3f}",ha="center",fontsize=13.5,color=INK)
ax.set_ylabel("整体 RMSE",fontsize=15); ax.tick_params(labelsize=12.5); ax.grid(axis="y",alpha=0.25)
ax.set_title("曝光补偿把整体误差降一个量级",fontsize=17,color=INK)
fig.tight_layout(); fig.savefig(FIG/"e079_jump.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

tests=[
 ("硬拼接有明显台阶",ms_hard>0.10,f"最大跳变 {ms_hard:.3f}"),
 ("羽化压到真实台阶附近",abs(sj_feather-base_jump)<0.02,f"羽化 {sj_feather:.4f} 真实 {base_jump:.4f}"),
 ("曝光补偿后接近真实台阶",abs(sj_feather_c-base_jump)<0.01,f"补偿 {sj_feather_c:.4f} 真实 {base_jump:.4f}"),
 ("补偿后误差下降",rmse_feather_c<rmse_feather,f"羽化 {rmse_feather:.4f} 补偿 {rmse_feather_c:.4f}"),
 ("金字塔接近真实台阶",abs(sj_pyr-base_jump)<0.01,f"金字塔 {sj_pyr:.4f} 真实 {base_jump:.4f}"),
 ("金字塔整体误差小",rmse_pyr<0.06,f"RMSE {rmse_pyr:.4f}"),
 ("梯度域接近真实台阶",abs(sj_grad-base_jump)<0.01,f"梯度域 {sj_grad:.4f} 真实 {base_jump:.4f}"),
 ("梯度域整体误差小",rmse_grad<0.08,f"RMSE {rmse_grad:.4f}"),
 ("对数增益与线性增益一致",gain_rel<0.01,f"相对差 {gain_rel:.4f}"),
 ("梯度域不造双边缘",double_edge<1.2,f"比值 {double_edge:.3f}"),
 ("三图在盘",all((FIG/n).exists() for n in ("e079_pipeline.png","e079_compare.png","e079_jump.png")),"三张全在"),
]
lines=[]; n_pass=0
for name,ok,detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass+=bool(ok)
readings=dict(base_jump=base_jump,seam_hard=sj_hard,seam_feather=sj_feather,seam_hard_c=sj_hard_c,
              seam_feather_c=sj_feather_c,seam_pyr=sj_pyr,seam_grad=sj_grad,
              maxstep_hard=ms_hard,maxstep_feather=ms_feather,maxstep_hard_c=ms_hard_c,
              maxstep_feather_c=ms_feather_c,maxstep_pyr=ms_pyr,maxstep_grad=ms_grad,
              rmse_hard=rmse_hard,rmse_feather=rmse_feather,rmse_feather_c=rmse_feather_c,
              rmse_pyr=rmse_pyr,rmse_grad=rmse_grad,gain_linear=float(a),gain_log=log_gain,
              gain_rel=gain_rel,double_edge=double_edge,n_pass=n_pass,n_total=len(tests))
(OUT/"e079_blending_readings.json").write_text(json.dumps(readings,ensure_ascii=False,indent=1)+"\n")
rd_sha=hashlib.sha256((OUT/"e079_blending_readings.json").read_bytes()).hexdigest()
src_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E079 混合与曝光补偿 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")): print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e079_blending_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass!=len(tests): raise SystemExit(1)
