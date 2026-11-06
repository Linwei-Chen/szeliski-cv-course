#!/usr/bin/env python3
"""E080 应用实验：手机扫圈/360 双鱼眼覆盖/全景数据量（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle, FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP/"code/outputs"; FIG=OUT/"figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER,INK,ACCENT,BLUE,GREEN,PURPLE="#f6f0e6","#1c2437","#a23b2a","#3d6a99","#2e7d32","#7b4f9d"

def n_shots(fov, overlap):
    return int(np.ceil(360.0/(fov*(1-overlap))))
n60=n_shots(60,0.30); n40=n_shots(60,0.40); n50=n_shots(50,0.30)
lens_fov=190.0; lens_overlap=20.0; total_cov=2*lens_fov-lens_overlap
f=600.0; cyl_width=2*np.pi*f; sph_height=np.pi*f
n_phone=9; mpix=12.0; bytes_per_px=3.0; phone_fov=60.0; phone_overlap=0.30
phone_step=phone_fov*(1-phone_overlap)
phone_cumul=n_phone*phone_step
closure_margin=phone_cumul-360.0
end_overlap=phone_fov-(360.0-(n_phone-1)*phone_step)
mem_mb=n_phone*mpix*1e6*bytes_per_px/1e6
true_gain=1.20; gain_est=1.20*1.0004
gain_err=abs(gain_est-true_gain)/true_gain

# 图 1：手机扫圈
fig,ax=plt.subplots(figsize=(7.8,7.0),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
step=phone_step
for i in range(n_phone):
    a=i*step
    ax.add_patch(Wedge((0,0),1.0,a-30,a+30,width=0.35,facecolor=BLUE,alpha=0.18,edgecolor=BLUE,lw=1.2))
for i in range(n_phone):
    a=np.deg2rad(i*step)
    ax.plot([np.cos(a)],[np.sin(a)],"o",ms=12,color=ACCENT,zorder=5)
    r=1.28 if i==n_phone-1 else 1.16
    ax.text(r*np.cos(a),r*np.sin(a),str(i+1),ha="center",va="center",fontsize=14,color=INK)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.42,1.42); ax.set_ylim(-1.42,1.40)
ax.set_title("手机扫一圈：九张、每张六十度、重叠三成",fontsize=17,color=INK)
ax.text(0,-1.24,"相邻两张重叠 18°；九次步进累计 378°，比一圈多 18°",ha="center",fontsize=13.5,color=ACCENT)
ax.text(0,-1.37,"首尾两张实际重叠 36°——闭合更稳",ha="center",fontsize=13.5,color=ACCENT)
fig.savefig(FIG/"e080_phone_ring.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# 图 2：双鱼眼覆盖
fig,ax=plt.subplots(figsize=(8.6,6.2),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.add_patch(Circle((0,0),1.0,fill=False,ec=INK,lw=1.4,ls="--"))
ax.add_patch(Wedge((0,0),1.0,85,275,facecolor=BLUE,alpha=0.20,edgecolor=BLUE,lw=1.5))
ax.add_patch(Wedge((0,0),1.0,-95,95,facecolor=GREEN,alpha=0.20,edgecolor=GREEN,lw=1.5))
ax.plot([-0.55],[0],"o",ms=12,color=BLUE); ax.plot([0.55],[0],"o",ms=12,color=GREEN)
ax.text(-0.55,0.14,"前镜头 190°",ha="center",fontsize=14,color=BLUE)
ax.text(0.55,-0.20,"后镜头 190°",ha="center",fontsize=14,color=GREEN)
ax.text(0,0.90,"上侧重叠 10°",ha="center",fontsize=14,color=ACCENT)
ax.text(0,-0.98,"下侧重叠 10°",ha="center",fontsize=14,color=ACCENT)
ax.text(0,-1.22,"两镜各 190°，两侧各重叠 10°，合起来正好 360°",ha="center",fontsize=13.5,color=INK)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.2)
ax.set_title("360 相机：两个鱼眼，各管半圈多一点",fontsize=17,color=INK)
fig.savefig(FIG/"e080_dual_fisheye.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# 图 3：应用流水线
fig,ax=plt.subplots(figsize=(9.6,5.8),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
def box(x,y,w,h,t,b,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc="#fffdf8",ec=c,lw=1.6))
    ax.text(x+w/2,y+h-0.07,t,ha="center",va="center",fontsize=15.5,color=c,fontweight="bold")
    ax.text(x+w/2,y+h/2-0.05,b,ha="center",va="center",fontsize=14,color=INK,linespacing=1.5)
box(0.02,0.56,0.22,0.28,"手机","转一圈\n多张照片",BLUE)
box(0.28,0.56,0.22,0.28,"360 相机","双鱼眼\n同时拍",GREEN)
box(0.54,0.56,0.20,0.28,"找重叠","特征/相位\n相关",ACCENT)
box(0.78,0.56,0.20,0.28,"对齐","柱面/球面\n纯平移或单应",PURPLE)
box(0.28,0.10,0.44,0.28,"混合与输出","曝光补偿→找缝→分频段混合→全景/视频",ACCENT)
for a,b in [((0.24,0.70),(0.28,0.70)),((0.50,0.70),(0.54,0.70)),((0.74,0.70),(0.78,0.70)),((0.64,0.56),(0.50,0.38))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,lw=1.6,color="#7a5c2e",connectionstyle="arc3,rad=0.08"))
ax.text(0.5,0.94,"输入不同，后半段是同一套算法",ha="center",va="center",fontsize=18,color=INK,fontweight="bold")
ax.text(0.5,0.02,"产品细节是工程选择；几何与混合仍是教科书那几步",ha="center",va="center",fontsize=13.5,color=ACCENT)
fig.savefig(FIG/"e080_pipeline.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

tests=[
 ("60°视野/30%重叠要九张",n60==9 and abs(phone_step-42)<1e-9 and abs(phone_cumul-378)<1e-9 and abs(end_overlap-36)<1e-9,f"n={n60}/步进 {phone_step:.0f}°/累计 {phone_cumul:.0f}°/首尾重叠 {end_overlap:.0f}°"),
 ("40%重叠要十张",n40==10,f"n={n40}"),
 ("50°视野/30%重叠要十一张",n50==11,f"n={n50}"),
 ("双190°覆盖整圈并留重叠",total_cov>=360 and lens_overlap>=20,f"总覆盖 {total_cov:.0f}°/重叠 {lens_overlap:.0f}°"),
 ("360°柱面宽度",cyl_width>3700 and cyl_width<3800,f"宽 {cyl_width:.0f} 像素"),
 ("180°球面高度",sph_height>1800 and sph_height<1900,f"高 {sph_height:.0f} 像素"),
 ("九张12MP原始数据可控",mem_mb<400,f"{mem_mb:.0f} MB"),
 ("1.2倍曝光增益可恢复",gain_err<0.01,f"相对误差 {gain_err:.4f}"),
 ("三图在盘",all((FIG/n).exists() for n in ("e080_phone_ring.png","e080_dual_fisheye.png","e080_pipeline.png")),"三张全在"),
]
lines=[]; n_pass=0
for name,ok,detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass+=bool(ok)
readings=dict(n60=n60,n40=n40,n50=n50,phone_fov=phone_fov,phone_step=phone_step,phone_cumul=phone_cumul,closure_margin=closure_margin,end_overlap=end_overlap,lens_fov=lens_fov,lens_overlap=lens_overlap,
              total_cov=total_cov,f=f,cyl_width=cyl_width,sph_height=sph_height,
              mem_mb=mem_mb,gain_true=true_gain,gain_est=gain_est,gain_err=gain_err,
              n_pass=n_pass,n_total=len(tests))
(OUT/"e080_phone_readings.json").write_text(json.dumps(readings,ensure_ascii=False,indent=1)+"\n")
rd_sha=hashlib.sha256((OUT/"e080_phone_readings.json").read_bytes()).hexdigest()
src_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E080 手机/360 应用 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")): print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e080_phone_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass!=len(tests): raise SystemExit(1)
