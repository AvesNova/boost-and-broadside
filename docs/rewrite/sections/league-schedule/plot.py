import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import rl, dm_wl, start
from analyse import modes
from rules import E, show

def rule(first=5.0, add=5.0, growth=0.17, end=500.0):
    points=[first]
    while True:
        nxt=points[-1]+max(add, growth*points[-1])
        if nxt>end: return points
        points.append(nxt)
from doubling import doubling
pts=doubling(5, 50)
show("every 5M to 50M, then the interval doubles with the steps", pts)
print("to 500M:", pts, len(pts))

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"frontline":"#2a78d6", "elimination":"#eb6834", "deathmatch (win/loss)":"#1baf7a"}
plt.rcParams.update({"font.size":10, "axes.edgecolor":MUTED, "axes.labelcolor":INK,
    "xtick.color":MUTED, "ytick.color":MUTED, "figure.facecolor":SURF, "axes.facecolor":SURF})
fig,(top,bottom)=plt.subplots(2,1,figsize=(11,11),gridspec_kw=dict(height_ratios=[3,2]))

grid=np.linspace(0,161,1611); mean=E(grid)
inside=[p for p in pts if p<=161]
top.plot(grid,mean,color=INK,lw=2,zorder=3,label="mean of the three modes (monotone fit)")
for p in inside:
    e=float(E(p))
    top.plot([p,p],[800,e],color=MUTED,lw=0.6,alpha=0.6,zorder=1)
    top.plot([0,p],[e,e],color=MUTED,lw=0.6,alpha=0.6,zorder=1)
    top.annotate(f"{p}M", (p,e), xytext=(4,-12), textcoords="offset points", fontsize=8, color=MUTED)
top.scatter(inside,E(np.array(inside)),s=36,color=INK,edgecolor=SURF,linewidth=1.5,zorder=4,
    label="members: every 5M to 50M, then the interval doubles when the steps double")
top.set(xlim=(0,165),ylim=(800,2450),xlabel="RL steps (millions)",ylabel="Elo (Imugi = 1000)")
top.set_title("rl-v20 (game v20): mean Elo over the three modes, with the staged league snapshot rule (5 per doubling)",
    loc="left",color=INK,fontsize=12)
top.legend(loc="lower right",frameon=False)
top.spines[["top","right"]].set_visible(False)

for name,(s,fit) in modes.items():
    raw = dm_wl if name.startswith("deathmatch") else rl[name]
    x=[0]+[r[0] for r in raw]; y=[start[name.split()[0]]]+[r[1] for r in raw]; err=[0]+[r[2] for r in raw]
    bottom.errorbar(x,y,yerr=err,fmt="o",ms=4,color=COL[name],alpha=0.6,elinewidth=1,capsize=0)
    bottom.plot(s,fit,color=COL[name],lw=2,label=name)
    bottom.annotate(name,(s[-1],fit[-1]),xytext=(6,0),textcoords="offset points",color=INK,fontsize=9,va="center")
bottom.set(xlim=(0,190),xlabel="RL steps (millions)",ylabel="Elo (Imugi = 1000)")
bottom.set_title("Per mode: rated points (± 1 SE) and monotone fits; step 0 is bc-v20's last rated checkpoint (own pool)",
    loc="left",color=INK,fontsize=10)
bottom.legend(loc="lower right",frameon=False)
bottom.spines[["top","right"]].set_visible(False)
for ax in (top,bottom): ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
fig.text(0.01,0.005,"All ratings are of the exponential-average checkpoints (averaged agent), live-Elo pools anchored at Imugi = 1000.",
    color=MUTED,fontsize=8)
fig.tight_layout(rect=(0,0.015,1,1))
fig.savefig("league_schedule.png",dpi=130)
