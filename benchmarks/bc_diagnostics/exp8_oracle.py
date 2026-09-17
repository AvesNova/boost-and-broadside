"""Exp 8: oracle-bearing control, achievable floor, and angular error -> KL.

Three turn heads are fitted on the frozen final latent of the checkpoint and
scored on independent held-out rollouts:

  latent        what the real action head has to work with;
  latent+oracle the same latent with the teacher's true sin/cos bearings appended;
  oracle only   the achievable floor -- can a tiny net map exact bearings to the
                teacher's turn distribution at all?

If "latent+oracle" collapses toward "oracle only" while "latent" does not, the
action head is exonerated and the missing quantity is the bearing itself.
"""

import json
import os as _os

import torch
import torch.nn as nn
import torch.nn.functional as F

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
DEV = "cuda"
LATENT = "b1.temporal0"


def load(tag):
    d = torch.load(f"{S}/probe_{tag}.pt")
    keep = d["keep"].reshape(-1)
    teach = {k: v.reshape(-1)[keep] for k, v in d["teacher"].items() if v.dim() == 3}
    return dict(
        latent=d["acts"][LATENT].reshape(-1, d["acts"][LATENT].shape[-1])[keep].float(),
        raw=d["acts"]["raw_input"].reshape(-1, d["acts"]["raw_input"].shape[-1])[keep].float(),
        teach=teach,
        expert=d["expert"].reshape(-1, d["expert"].shape[-1])[keep].float(),
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[keep].float(),
    )


def oracle_feats(t):
    """Everything the teacher's turn head actually consumes, in its own units."""
    cols = [
        t["rel_front"].sin(), t["rel_front"].cos(),
        t["rel_personal"].sin(), t["rel_personal"].cos(),
        t["alpha"],
        t["rel_front"].abs(), t["rel_personal"].abs(),
    ]
    return torch.stack(cols, -1)


def kl(expert, logq):
    p = expert.clamp_min(1e-8)
    return ((expert * (p.log() - logq)).sum(-1)).mean()


def fit_head(xtr, ptr, xte, hidden=512, epochs=60, bs=4096, lr=2e-3):
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    xtr_n, xte_n = ((xtr - mu) / sd).to(DEV), ((xte - mu) / sd).to(DEV)
    ptr = ptr.to(DEV)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, ptr.shape[1]),
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(xtr_n) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr_n), device=DEV)
        for i in range(0, len(xtr_n) - bs + 1, bs):
            idx = perm[i : i + bs]
            loss = -(ptr[idx] * F.log_softmax(net(xtr_n[idx]), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        out = [F.log_softmax(net(xte_n[i : i + 65536]), -1).cpu() for i in range(0, len(xte_n), 65536)]
    return torch.cat(out)


def main():
    tr, te = load("train"), load("heldout")
    otr, ote = oracle_feats(tr["teach"]), oracle_feats(te["teach"])
    rows = []

    base = float(kl(te["expert"], te["logq"]))
    rows.append(dict(head="frozen policy action head (checkpoint)", heldout_turn_kl=round(base, 4)))
    print(json.dumps(rows[-1]), flush=True)

    cells = {
        "latent only": (tr["latent"], te["latent"]),
        "latent + oracle bearing": (torch.cat([tr["latent"], otr], 1), torch.cat([te["latent"], ote], 1)),
        "oracle bearing only": (otr, ote),
        "raw ego features only": (tr["raw"], te["raw"]),
    }
    preds = {}
    for name, (a, b) in cells.items():
        lq = fit_head(a, tr["expert"], b)
        preds[name] = lq
        v = float(kl(te["expert"], lq))
        rows.append(dict(head=name, heldout_turn_kl=round(v, 4)))
        print(json.dumps(rows[-1]), flush=True)

    # --- angular error -> KL, measured rather than simulated -------------------
    # Reuse the fitted frontline probe from exp7 by refitting it here on the same
    # split, so the error attached to each token is an honest held-out error.
    import sys
    sys.path.insert(0, S)
    from exp7_probe import ang_err_deg, fit_probe

    ytr = torch.stack([tr["teach"]["rel_front"].sin(), tr["teach"]["rel_front"].cos()], -1)
    _, _, pred = fit_probe(tr["latent"], ytr, te["latent"], epochs=40)
    err = ang_err_deg(pred, te["teach"]["rel_front"])

    a = te["teach"]["alpha"]
    sel = a > 0.99  # the tokens where the frontline bearing is the whole teacher
    pol_kl = (te["expert"].clamp_min(1e-8) * (te["expert"].clamp_min(1e-8).log() - te["logq"])).sum(-1)
    targ = te["expert"].argmax(-1)
    polarg = te["logq"].argmax(-1)
    LEFT, RIGHT = {1, 3}, {2, 4}

    def frac(mask, f):
        return round(float(f[mask].float().mean()), 3) if int(mask.sum()) > 50 else None

    straight_miss = (targ != 0) & (polarg == 0)
    side = torch.tensor([0, -1, 1, -1, 1, 0, 0])
    flip = (side[targ] != 0) & (side[polarg] != 0) & (side[targ] != side[polarg])
    sharp = (targ > 0) & (polarg > 0) & ((targ > 2) != (polarg > 2))

    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 45), (45, 181)]
    err_rows = []
    for lo, hi in bins:
        m = sel & (err >= lo) & (err < hi)
        n = int(m.sum())
        if n < 100:
            continue
        err_rows.append(dict(
            bin=f"[{lo},{hi})", n=n,
            turn_kl=round(float(pol_kl[m].mean()), 4),
            straight_miss=frac(m, straight_miss),
            lr_flip=frac(m, flip),
            sharp_miss=frac(m, sharp),
        ))
        print(json.dumps(err_rows[-1]), flush=True)

    # Does the measured error explain the measured KL? Feed the probe's own
    # predicted bearing to the analytic teacher and score that against the truth.
    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    cfg = StochasticAgentConfig()

    def ramp(x, lo, hi, plo, phi):
        return plo + ((x - lo) / (hi - lo)).clamp(0, 1) * (phi - plo)

    def teacher_turn(rel):
        aa = rel.abs()
        pn = ramp(aa, *cfg.turn_angle_ramp, *cfg.turn_angle_prob)
        ps = ramp(aa, *cfg.sharp_turn_angle_ramp, *cfg.sharp_turn_angle_prob)
        r, l = (rel > 0).float(), (rel < 0).float()
        p = torch.stack([1 - pn, pn * l * (1 - ps), pn * r * (1 - ps), pn * l * ps, pn * r * ps,
                         torch.zeros_like(pn), torch.zeros_like(pn)], -1)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-8)

    pred_ang = torch.atan2(pred[:, 0], pred[:, 1])
    q = teacher_turn(pred_ang).clamp_min(1e-8)
    plug = float(kl(te["expert"][sel], q[sel].log()))
    real = float(kl(te["expert"][sel], te["logq"][sel]))
    summary = dict(
        tokens_alpha1=int(sel.sum()),
        probe_median_err_deg=round(float(err[sel].median()), 2),
        kl_from_plugging_probe_bearing_into_teacher=round(plug, 4),
        measured_policy_turn_kl=round(real, 4),
    )
    print(json.dumps(summary), flush=True)
    json.dump(dict(heads=rows, error_bins=err_rows, plugin=summary),
              open(f"{S}/exp8_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
