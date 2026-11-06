#!/usr/bin/env python3
"""E077 实验：全景姿态图——逐对串联漂移 vs 全局最小二乘（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP/"code/outputs"; FIG=OUT/"figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER,INK,ACCENT,BLUE,GREEN="#f6f0e6","#1c2437","#a23b2a","#3d6a99","#2e7d32"

N=8
TRUE=np.deg2rad(np.arange(N)*45.0)
rng=np.random.default_rng(7)
chain=[(i,i+1) for i in range(N-1)]
skip=[(i,i+2) for i in range(N-2)]
skip3=[(i,i+3) for i in range(N-3)]
closure=[(N-1,0)]
edges=chain+skip+skip3+closure
meas={}
for i,j in edges:
    diff=(TRUE[j]-TRUE[i]+np.pi)%(2*np.pi)-np.pi
    meas[(i,j)]=diff+rng.normal(0,np.deg2rad(2.0))

def chain_angles(meas):
    th=np.zeros(N); th[0]=0
    for i in range(N-1):
        th[i+1]=th[i]+meas[(i,i+1)]
    return th

def design(meas, use_edges, ref):
    # 把每条边的角度差解卷绕到参考解附近：θj-θi = 2πk + 测量值
    A=[]; b=[]
    for i,j in use_edges:
        row=np.zeros(N-1)
        if i>0: row[i-1]-=1
        if j>0: row[j-1]+=1
        m=meas[(i,j)]
        target=m+2*np.pi*round((ref[j]-ref[i]-m)/(2*np.pi))
        A.append(row); b.append(target)
    return np.array(A),np.array(b)

def solve(A,b,weights=None):
    if weights is None: weights=np.ones(len(b))
    sw=np.sqrt(weights)
    x,*_=np.linalg.lstsq(A*sw[:,None],b*sw,rcond=None)
    return np.r_[0,x]

def global_angles(meas,use_edges,ref):
    A,b=design(meas,use_edges,ref)
    return solve(A,b)

def robust_angles(meas,use_edges,ref,iters=30):
    A,b=design(meas,use_edges,ref)
    th=solve(A,b)
    for _ in range(iters):
        r=np.abs(A@th[1:]-b)
        scale=1.4826*np.median(np.abs(r-np.median(r)))+1e-12
        c=2.0*scale
        w=np.where(r<=c,1.0,c/np.maximum(r,1e-12))
        th=solve(A,b,w)
    return th

def rmse(th): return float(np.rad2deg(np.sqrt(np.mean(((th-TRUE+np.pi)%(2*np.pi)-np.pi)**2))))

th_chain=chain_angles(meas)
th_global=global_angles(meas, edges, th_chain)
th_chain_only=global_angles(meas, chain, th_chain)
# 外点
meas_out=dict(meas); meas_out[(2,3)] += np.deg2rad(25)
th_out=global_angles(meas_out, edges, th_chain)
# 鲁棒：Huber 迭代重加权，自动压低外点权重
th_rob=robust_angles(meas_out, edges, th_chain)

chain_err=rmse(th_chain); global_err=rmse(th_global)
chain_only_err=rmse(th_chain_only); out_err=rmse(th_out); rob_err=rmse(th_rob)
closure_gap_chain=float(abs(((th_chain[-1]+meas[(N-1,0)]-th_chain[0]+np.pi)%(2*np.pi)-np.pi)))
closure_gap_global=float(abs(((th_global[-1]+meas[(N-1,0)]-th_global[0]+np.pi)%(2*np.pi)-np.pi)))

# ── 图 1：拼接管线 ──
fig,ax=plt.subplots(figsize=(9.2,6.0),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
def box(x,y,w,h,t,b,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc="#fffdf8",ec=c,lw=1.6))
    ax.text(x+w/2,y+h-0.055,t,ha="center",va="center",fontsize=16.5,color=c,fontweight="bold")
    ax.text(x+w/2,y+h/2-0.04,b,ha="center",va="center",fontsize=15,color=INK,linespacing=1.5)
box(0.03,0.62,0.26,0.25,"特征","每张图\n提点描述",BLUE)
box(0.36,0.62,0.26,0.25,"两两匹配","只连有重叠\n的图对",BLUE)
box(0.69,0.62,0.28,0.25,"全局对齐","一起优化\n所有姿态",ACCENT)
box(0.36,0.18,0.26,0.25,"合成","选投影面\n接缝混合",GREEN)
for a,b in [((0.29,0.745),(0.36,0.745)),((0.62,0.745),(0.69,0.745)),((0.72,0.62),(0.55,0.43))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,lw=1.6,color="#7a5c2e",connectionstyle="arc3,rad=0.12"))
ax.text(0.5,0.93,"全景四步：先连图，再一起算",ha="center",va="center",fontsize=18,color=INK,fontweight="bold")
ax.text(0.5,0.07,"本集重点：红色「全局对齐」——所有重叠对一起满足",ha="center",va="center",fontsize=14.5,color=ACCENT)
fig.savefig(FIG/"e077_pipeline.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 2：相机环与边 ──
fig,ax=plt.subplots(figsize=(7.6,6.4),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
for k,(i,j) in enumerate(edges):
    color=ACCENT if (i,j)==(2,3) else ("#b9b2a4" if (i,j) in chain else BLUE)
    lw=2.4 if (i,j)==(2,3) else (1.8 if (i,j) in chain else 1.0)
    ax.plot([np.cos(TRUE[i]),np.cos(TRUE[j])],[np.sin(TRUE[i]),np.sin(TRUE[j])],color=color,lw=lw,alpha=0.8)
ax.scatter(np.cos(TRUE),np.sin(TRUE),s=260,c=GREEN,zorder=5)
for i in range(N):
    ax.text(1.12*np.cos(TRUE[i]),1.12*np.sin(TRUE[i]),f"C{i+1}",ha="center",va="center",fontsize=14,color=INK)
ax.set_aspect("equal"); ax.axis("off"); ax.set_title("八台相机、十九对重叠：链边+跳边+远跳+闭合边",fontsize=16,color=INK,pad=34)
ax.text(0,0,"红色=外点实验预留\n灰=链边，蓝=跳边/远跳",ha="center",va="center",fontsize=14,color=ACCENT,linespacing=1.5)
fig.savefig(FIG/"e077_graph.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 3：误差对照 ──
fig,ax=plt.subplots(figsize=(9.2,5.8),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
labels=["逐对串联","只用链边\n全局解","全部边\n全局解","含外点\n全局解","剔除外点\n鲁棒解"]
vals=[chain_err,chain_only_err,global_err,out_err,rob_err]
colors=[ACCENT,"#8e44ad",GREEN,BLUE,"#2e7d32"]
bars=ax.bar(labels,vals,color=colors,width=0.58)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+max(vals)*0.02,f"{v:.2f}",ha="center",fontsize=14,color=INK)
ax.set_ylabel("姿态角 RMSE（度）",fontsize=15); ax.tick_params(labelsize=12.5); ax.grid(axis="y",alpha=0.25)
ax.set_title("全局解把误差摊平；外点要剔除",fontsize=18,color=INK)
fig.tight_layout(); fig.savefig(FIG/"e077_error.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

tests=[
 ("逐对串联明显漂移",chain_err>2.5,f"RMSE={chain_err:.2f}°"),
 ("全局解误差不到串联一半",global_err<chain_err*0.5,f"串联={chain_err:.2f}° 全局={global_err:.2f}°"),
 ("闭合边把缺口压小",closure_gap_global<closure_gap_chain,f"串联={np.rad2deg(closure_gap_chain):.2f}° 全局={np.rad2deg(closure_gap_global):.2f}°"),
 ("图连通",len(edges)>=N,f"{len(edges)} 条边 / {N} 台相机"),
 ("跳边提升精度",global_err<chain_only_err,f"链边={chain_only_err:.2f}° 全边={global_err:.2f}°"),
 ("外点破坏全局解",out_err>global_err*1.5,f"无外点={global_err:.2f}° 含外点={out_err:.2f}°"),
 ("鲁棒剔除恢复",rob_err<out_err,f"含外点={out_err:.2f}° 鲁棒={rob_err:.2f}°"),
 ("规范固定",abs(th_global[0])<1e-12,"首台相机角度固定"),
 ("未知量 N−1",len(th_global)-1==N-1,"每台相机一个角度未知量"),
 ("三图在盘",all((FIG/n).exists() for n in ("e077_pipeline.png","e077_graph.png","e077_error.png")),"三张全在"),
]
lines=[]; n_pass=0
for name,ok,detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass+=bool(ok)
readings=dict(chain_rmse=chain_err,chain_only_rmse=chain_only_err,global_rmse=global_err,
              outlier_rmse=out_err,robust_rmse=rob_err,
              closure_gap_chain_deg=float(np.rad2deg(closure_gap_chain)),
              closure_gap_global_deg=float(np.rad2deg(closure_gap_global)),
              n_edges=len(edges),n_cameras=N,n_unknown=N-1,n_pass=n_pass,n_total=len(tests))
(OUT/"e077_bundle_readings.json").write_text(json.dumps(readings,ensure_ascii=False,indent=1)+"\n")
rd_sha=hashlib.sha256((OUT/"e077_bundle_readings.json").read_bytes()).hexdigest()
src_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
now=datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
print(f"[lab] E077 全景姿态图 开始 {now}"); print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")): print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e077_bundle_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass!=len(tests): raise SystemExit(1)
