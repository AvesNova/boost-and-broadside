import json,sys
from bench.action_mix import STICKS, POWERS
for f in sys.argv[1:]:
    d=json.load(open(f))
    name=d.get('condition','deathmatch-1v1')
    for k in ('Yongma','Jujak'):
        st=d['stick'][k]; pw=d['power'][k]; sh=d.get('shoot',{}).get(k,{}).get('fire',float('nan'))
        print(f"{name:16s} {k:6s} " + " ".join(f"{s}={st[s]*100:4.1f}" for s in STICKS) + " | " + " ".join(f"{p}={pw[p]*100:4.1f}" for p in POWERS) + f" | fire={sh*100:4.1f}")
