import re, numpy as np
from sklearn.isotonic import IsotonicRegression
LIVE="/home/vizia/avesnova/live-elo/"
def parse(path):
    out={}
    for line in open(path):
        m=re.match(r"real_scale_(\w+)\s+\S*step_(\d+)/exponential_average\s+([-\d.]+)\s+\+-\s+([\d.]+)",line)
        if m: out.setdefault(m[1],[]).append((int(m[2])/1e6,float(m[3]),float(m[4])))
    return {k:sorted(v) for k,v in out.items()}
rl=parse(LIVE+"rl-v20/latest.txt")
bc=parse(LIVE+"bc-v20/latest.txt")
dm_wl=sorted(parse(LIVE+"rl-v20-winloss/log.txt")["deathmatch"])
start={k:v[-1][1] for k,v in bc.items()}  # bc-v20's last rated checkpoint (84.9M), own pool
