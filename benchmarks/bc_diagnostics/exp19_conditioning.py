"""Exp 19: is the bearing ill-conditioned rather than unrepresented?

The teacher's bearing is the *angle of a sum*:

    force = (1-recovery)*(objective_force + combat_force) + recovery*spawn + separation
    bearing = angle(force)

When those terms nearly cancel, |force| is small and the angle of the resultant
becomes arbitrarily sensitive to error in any term: d(angle) ~ |error| / |force|.
No amount of precision in the *terms* fixes that -- it is conditioning, not
representation, and it would limit any imitator including an exact one working
in finite precision.

This reconstructs the force from the recorded per-term (direction, magnitude)
pairs, validates the reconstruction against the teacher's own bearing, and then
asks whether probe error and turn KL track 1/|force|.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, fit_probe  # noqa: E402
from exp18_terms import load, sc  # noqa: E402


def polar(mag, ang):
    return torch.polar(mag.clamp_min(0), ang)


def main():
    tr, te = load("train"), load("heldout")

    def force(d):
        I = d["I"]
        rec = I["mag_recovery"]
        obj = polar(I["mag_objective"], I["dir_objective"])
        cf = polar(I["mag_combat_force"], I["dir_combat_force"])
        sep = polar(I["mag_separation"], I["dir_separation"])
        # spawn direction was not recorded; recovery is ~0 for healthy ships, so
        # the reconstruction is exact exactly where recovery vanishes -- which is
        # what the validation below checks rather than assumes.
        return (1 - rec) * (obj + cf) + sep, rec

    f_te, rec_te = force(te)
    mag = f_te.abs()
    ang_err = ang_err_deg(torch.stack([f_te.angle().sin(), f_te.angle().cos()], -1),
                          te["T"]["rel_front"])
    healthy = rec_te < 0.01
    print(f"reconstruction check (recovery<0.01, n={int(healthy.sum())}): "
          f"median angle error vs teacher bearing = {float(ang_err[healthy].median()):.3f} deg")
    print(f"  (all tokens: {float(ang_err.median()):.3f} deg; the gap is the missing spawn term)")

    # Probe the frozen latent for the bearing, then stratify by |force|.
    _, _, pred = fit_probe(tr["latent"], sc(tr["T"]["rel_front"]), te["latent"], epochs=40)
    perr = ang_err_deg(pred, te["T"]["rel_front"])
    exp, logq = te["expert"], te["logq"]
    kl = (exp.clamp_min(1e-8) * (exp.clamp_min(1e-8).log() - logq)).sum(-1)

    sel = healthy
    q = torch.quantile(mag[sel], torch.linspace(0, 1, 9))
    rows = []
    print("\nby |force| (recovery<0.01), octiles:")
    for i in range(8):
        lo, hi = float(q[i]), float(q[i + 1])
        m = sel & (mag >= lo) & (mag < hi if i < 7 else mag <= hi)
        if int(m.sum()) < 200:
            continue
        rows.append(dict(
            octile=i + 1, force_lo=round(lo, 4), force_hi=round(hi, 4), n=int(m.sum()),
            probe_err_median_deg=round(float(perr[m].median()), 2),
            probe_err_p90_deg=round(float(torch.quantile(perr[m], 0.9)), 2),
            turn_kl=round(float(kl[m].mean()), 4),
        ))
        print(json.dumps(rows[-1]), flush=True)

    # Teacher-side sensitivity: how much does the teacher's own bearing move for
    # a fixed perturbation of one term, as a function of |force|?
    print("\nteacher bearing sensitivity to a 0.02 perturbation of one term:")
    sens = []
    pert = f_te + 0.02
    dtheta = ((pert.angle() - f_te.angle() + torch.pi) % (2 * torch.pi) - torch.pi).abs() * 180 / torch.pi
    for i in range(8):
        lo, hi = float(q[i]), float(q[i + 1])
        m = sel & (mag >= lo) & (mag < hi if i < 7 else mag <= hi)
        if int(m.sum()) < 200:
            continue
        sens.append(dict(octile=i + 1, median_bearing_shift_deg=round(float(dtheta[m].median()), 2)))
        print(json.dumps(sens[-1]), flush=True)

    json.dump(dict(by_force=rows, sensitivity=sens), open(f"{S}/exp19_rows.json", "w"), indent=1)


main()
