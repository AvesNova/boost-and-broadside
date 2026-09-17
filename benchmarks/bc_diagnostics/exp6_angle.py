"""Exp 6: reconstruct the teacher's turn head from the bearing and price bearing error in KL."""

import torch, torch.nn.functional as F
import os as _os
S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
cfg = StochasticAgentConfig()

d = torch.load(f"{S}/dump_rollout11.pt")
exp = d["expert"][..., 3:10]; logq = d["turn_logp"]; m = d["keep"]
teach = d["teacher"]; alpha = teach["alpha"]; rel_front = teach["rel_front"]


def ramp(x, lo, hi, plo, phi):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return plo + t * (phi - plo)


def teacher_turn(rel):
    a = rel.abs()
    pn = ramp(a, *cfg.turn_angle_ramp, *cfg.turn_angle_prob)
    ps = ramp(a, *cfg.sharp_turn_angle_ramp, *cfg.sharp_turn_angle_prob)
    right = (rel > 0).float(); left = (rel < 0).float()
    cols = [1 - pn, pn * left * (1 - ps), pn * right * (1 - ps), pn * left * ps, pn * right * ps,
            torch.zeros_like(pn), torch.zeros_like(pn)]
    p = torch.stack(cols, -1)
    return p / (p.sum(-1, keepdim=True) + 1e-8)


sel = (alpha > 0.99) & m                       # pure-frontline tokens: exact reconstruction
n = float(sel.sum())
rec = teacher_turn(rel_front)
err = (rec - exp).abs().max(-1).values
print(f"pure-frontline tokens (alpha>0.99 & valid): {n:.0f} ({100*n/float(m.sum()):.1f}% of valid)")
print(f"  reconstruction max |p_rec - p_stored|: mean={float((err*sel).sum()/n):.2e} "
      f"max={float(err[sel].max()):.2e}   (stored teacher probs are bf16)")

f = sel.float()
kl_real = float(((exp * ((exp.clamp(min=1e-8)).log() - logq)).sum(-1) * f).sum() / n)
print(f"  measured policy turn KL on these tokens: {kl_real:.4f}")
print("\n  KL(teacher(theta) || teacher(theta+eps)) — the cost of a pure bearing error:")
for eps in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):
    for sgn in (1, -1):
        p2 = teacher_turn(rel_front + sgn * eps)
        kl = (exp * ((exp.clamp(min=1e-8)).log() - p2.clamp(min=1e-8).log())).sum(-1)
        if sgn == 1:
            v1 = float((kl * f).sum() / n)
        else:
            v2 = float((kl * f).sum() / n)
    print(f"    eps={eps:5.3f} rad ({eps*57.3:5.2f} deg):  KL+={v1:7.4f}  KL-={v2:7.4f}")
# random-sign error of magnitude eps (a symmetric estimator error)
print("\n  symmetric random-sign bearing error:")
g = torch.Generator().manual_seed(0)
for eps in (0.01, 0.02, 0.05, 0.1, 0.2):
    s = (torch.randint(0, 2, rel_front.shape, generator=g).float() * 2 - 1) * eps
    p2 = teacher_turn(rel_front + s)
    kl = (exp * ((exp.clamp(min=1e-8)).log() - p2.clamp(min=1e-8).log())).sum(-1)
    print(f"    eps={eps:5.3f} rad ({eps*57.3:5.2f} deg):  KL={float((kl*f).sum()/n):7.4f}")

# what a *smoothed* teacher would cost: policy's best achievable if it knew the
# bearing distribution but not the exact value (Gaussian bearing uncertainty)
print("\n  KL(teacher || E_sigma[teacher(theta+N(0,sigma))]) — irreducible loss under"
      " Gaussian bearing uncertainty:")
for sigma in (0.02, 0.05, 0.1, 0.2):
    acc = torch.zeros_like(exp)
    K = 24
    for i in range(K):
        acc += teacher_turn(rel_front + torch.randn(rel_front.shape, generator=g) * sigma)
    q = acc / K
    kl = (exp * ((exp.clamp(min=1e-8)).log() - q.clamp(min=1e-8).log())).sum(-1)
    print(f"    sigma={sigma:5.3f} rad ({sigma*57.3:5.2f} deg): KL={float((kl*f).sum()/n):7.4f}")

# non-actor token KL (the unmasked-evaluator contribution)
alive = d["alive"]; actor = d["actor"]
kl_all = (exp * ((exp.clamp(min=1e-8)).log() - logq)).sum(-1)
for nm, mm in [("actor & alive (production)", (actor & alive)),
               ("alive & NOT actor (opponent team)", (alive & ~actor)),
               ("dead tokens", ~alive)]:
    ff = mm.float(); dd = float(ff.sum())
    print(f"{nm:36s} den={dd:9.0f} turnKL={float((kl_all*ff).sum()/max(dd,1)):.4f}")
