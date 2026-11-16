import json,sys
runs=[(f.split('/')[-1].replace('.json',''), json.load(open(f))) for f in sys.argv[1:]]
def row(label, fn):
    print(f"| {label} | " + " | ".join(fn(d) for _,d in runs) + " |")
print("| metric | " + " | ".join(n for n,_ in runs) + " |")
print("|---|" + "---|"*len(runs))
row("deaths/game", lambda d: f"{d['deaths_per_game']:.2f}")
row("kills/game", lambda d: f"{d['kills_per_game']:.2f}")
row("merges/game (post-merge windows)", lambda d: f"{d['windows']/d['games']:.1f}")
row("merges/kill", lambda d: f"{d['windows']/max(d['kills_per_game']*d['games'],1):.1f}")
row("Yongma kill / point share", lambda d: f"{d['kill_share_a'][0]:.3f} / {d['point_share_a'][0]:.3f}")
row("game-loop scissors / one / two circle", lambda d: "{2:.0f} / {0:.0f} / {1:.0f}".format(*d['post_merge_circles_one_two_scissors']))
row("both >=2 reversals at 15 deg/s (share of windows)", lambda d: f"{d['both_two_reversals_share']:.3f}")
for k in ('Yongma','Jujak'):
    row(f"{k} reversals/window (15 deg/s)", lambda d: f"{d['per_kind'][k]['reversals_per_window']:.2f}")
    row(f"{k} sustained turn (>15 deg/s, held 1 s)", lambda d: f"{d['per_kind'][k]['sustained_turn_share']:.3f}")
    row(f"{k} turn >15 deg/s", lambda d: f"{d['per_kind'][k]['turn_above_15_share']:.3f}")
    row(f"{k} sustained turn (old def)", lambda d: f"{d['per_kind'][k]['sustained_turn_share_old']:.3f}")
    row(f"{k} stick side flips/min", lambda d: f"{d['per_kind'][k]['stick_flips_per_minute']:.1f}")
    row(f"{k} hits/game", lambda d: f"{d['per_kind'][k]['hits_per_game']:.0f}")
