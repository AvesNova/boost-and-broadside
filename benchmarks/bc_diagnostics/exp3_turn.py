"""Exp 3: turn-head decomposition and teacher-variable stratification (offline, CPU)."""

import sys
import torch
import torch.nn.functional as F

import os as _os
S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
NAMES = ["straight", "left", "right", "sharp_L", "sharp_R", "brake", "sharp_brake"]

d = torch.load(f"{S}/dump_rollout11.pt")
exp = d["expert"][..., 3:10]           # (T,B,N,7) teacher turn probs (bf16-stored -> fp32)
logq = d["turn_logp"]                  # (T,B,N,7)
m = d["keep"]                          # (T,B,N) bool: bc_valid & actor & alive
teach = d["teacher"]

p = exp.clamp(min=1e-8)
logp = p.log()
q = logq.exp()
per_class = exp * (logp - logq)        # (T,B,N,7) KL contribution by class
kl = per_class.sum(-1)                 # (T,B,N)
M = m.float()
den = M.sum()
print(f"tokens={int(den)}  mean turn KL={float((kl*M).sum()/den):.4f}")
print(f"teacher turn entropy={float((-(p*logp).sum(-1)*M).sum()/den):.4f} "
      f"policy turn entropy={float((-(q*logq).sum(-1)*M).sum()/den):.4f}")

t_arg = exp.argmax(-1)
q_arg = q.argmax(-1)

print("\n== per turn action ==")
print(f"{'action':12s} {'t_mass':>7s} {'t_argmax%':>9s} {'q_mass':>7s} {'KLcontrib':>9s} "
      f"{'cond_KL':>8s} {'agree%':>7s} {'q@t_arg':>8s}")
for a in range(7):
    tm = float((exp[..., a] * M).sum() / den)
    qm = float((q[..., a] * M).sum() / den)
    klc = float((per_class[..., a] * M).sum() / den)
    sel = (t_arg == a) & m
    n = float(sel.sum())
    if n > 0:
        cond = float((kl * sel.float()).sum() / n)
        agree = float(((q_arg == a) & sel).sum() / n)
        qat = float((q[..., a] * sel.float()).sum() / n)
    else:
        cond = agree = qat = float("nan")
    print(f"{NAMES[a]:12s} {tm:7.4f} {100*n/den:9.2f} {qm:7.4f} {klc:9.4f} "
          f"{cond:8.4f} {100*agree:7.2f} {qat:8.4f}")

print("\n== confusion (rows teacher argmax, cols policy argmax, % of tokens) ==")
print(f"{'':12s}" + "".join(f"{n[:7]:>9s}" for n in NAMES))
for a in range(7):
    row = []
    for b in range(7):
        row.append(100 * float((((t_arg == a) & (q_arg == b)) & m).sum()) / den)
    print(f"{NAMES[a]:12s}" + "".join(f"{v:9.3f}" for v in row))

# ---- exact chain-rule decomposition: group {straight},{left:1,3},{right:2,4},{brake:5,6}
groups = [[0], [1, 3], [2, 4], [5, 6]]
gnames = ["straight", "LEFT", "RIGHT", "brake"]
pg = torch.stack([exp[..., g].sum(-1) for g in groups], -1)
qg = torch.stack([q[..., g].sum(-1) for g in groups], -1)
kl_group = (pg * ((pg.clamp(min=1e-12)).log() - qg.clamp(min=1e-12).log())).sum(-1)
# within-group conditional (sharp vs normal) for LEFT and RIGHT
kl_within = torch.zeros_like(kl_group)
for gi, g in enumerate(groups):
    if len(g) < 2:
        continue
    pc = exp[..., g] / pg[..., gi : gi + 1].clamp(min=1e-12)
    qc = q[..., g] / qg[..., gi : gi + 1].clamp(min=1e-12)
    kl_within += pg[..., gi] * (pc * (pc.clamp(min=1e-12).log() - qc.clamp(min=1e-12).log())).sum(-1)
print(f"\n== chain rule: KL_turn = KL(straight/left/right/brake) + E_g[KL(sharp|g)] ==")
print(f"  coarse direction KL : {float((kl_group*M).sum()/den):.4f}")
print(f"  sharp-vs-normal KL  : {float((kl_within*M).sum()/den):.4f}")
print(f"  sum                 : {float(((kl_group+kl_within)*M).sum()/den):.4f}  (total {float((kl*M).sum()/den):.4f})")

# direction-only error: teacher's signed side mass vs policy's
p_left, p_right = pg[..., 1], pg[..., 2]
q_left, q_right = qg[..., 1], qg[..., 2]
wrong_side = ((p_left > p_right) & (q_right > q_left)) | ((p_right > p_left) & (q_left > q_right))
print(f"  tokens where policy picks the opposite side: {100*float((wrong_side&m).sum())/den:.2f}%  "
      f"their mean turn KL={float((kl*(wrong_side&m).float()).sum()/max(float((wrong_side&m).sum()),1)):.3f}")


def strat(name, var, edges):
    v = var
    print(f"\n== turn KL by {name} ==")
    print(f"{'bin':>18s} {'tokens':>9s} {'%':>6s} {'KL':>7s} {'KL_dir':>7s} {'KL_sharp':>8s} {'H_teach':>7s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v >= lo) & (v < hi) & m
        n = float(sel.sum())
        if n < 50:
            continue
        f = sel.float()
        print(f"{f'[{lo:g},{hi:g})':>18s} {n:9.0f} {100*n/den:6.2f} "
              f"{float((kl*f).sum()/n):7.4f} {float((kl_group*f).sum()/n):7.4f} "
              f"{float((kl_within*f).sum()/n):8.4f} {float((-(p*logp).sum(-1)*f).sum()/n):7.4f}")


import math
alpha = teach["alpha"]
rel_front = teach["rel_front"]
rel_comb = teach["rel_combat"]
# effective bearing the blended teacher is mostly following
eff = torch.where(alpha > 0.5, rel_front, rel_comb)
disagree = (rel_front - rel_comb).abs()
disagree = torch.minimum(disagree, 2 * math.pi - disagree)

strat("|effective rel angle| (rad)", eff.abs(),
      [0, 0.03, 0.06, 0.09, 0.12, 0.2, 0.3, 0.42, 0.7, 1.2, 2.0, 3.2])
strat("alpha (0=combat,1=frontline)", alpha, [0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01])
strat("|front-vs-combat bearing disagreement| (rad)", disagree,
      [0, 0.05, 0.2, 0.5, 1.0, 2.0, 3.2])
strat("closest enemy distance", teach["closest_dist"].clamp(max=9e5),
      [0, 100, 200, 400, 600, 1000, 2000, 1e6])
strat("speed", teach["speed"], [0, 20, 60, 100, 140, 200])
strat("alive ships", teach["n_alive"], [0, 3, 5, 7, 8, 9])
strat("episode step_count", teach["step_count"], [0, 500, 1500, 3000, 6000, 9001])

# near-ramp indicator: |angle| inside the two ramp windows
a_ = eff.abs()
near_turn_ramp = (a_ >= 0.03) & (a_ < 0.12)
near_sharp_ramp = (a_ >= 0.24) & (a_ < 0.42)
for nm, sel0 in [("inside turn ramp [0.03,0.12)", near_turn_ramp),
                 ("inside sharp ramp [0.24,0.42)", near_sharp_ramp),
                 ("outside both ramps", ~(near_turn_ramp | near_sharp_ramp))]:
    sel = sel0 & m
    n = float(sel.sum())
    print(f"{nm:32s} tokens={n:8.0f} ({100*n/den:5.2f}%)  KL={float((kl*sel.float()).sum()/n):.4f}")

# mask-variant sanity: what other (wrong) denominators would report
alive = d["alive"]; actor = d["actor"]
for nm, mm in [("bc_valid&actor&alive (production)", m),
               ("alive only", alive),
               ("actor&alive", actor & alive),
               ("all tokens", torch.ones_like(alive))]:
    f = mm.float()
    print(f"{nm:34s} den={float(f.sum()):9.0f} turnKL={float((kl*f).sum()/f.sum()):.4f}")
