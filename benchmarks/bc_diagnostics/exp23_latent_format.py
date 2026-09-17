"""Exp 23: is the trunk's internal representation of direction vector-like?

Exp 22 showed that *handing a head* direction terms in Cartesian form is worth
0.19 nats over polar form, and left open whether the trunk's own latent is
phase-like. This tests that with the gap between a linear and an MLP probe: a
quantity stored as a plain vector is linearly decodable, one stored as a phase
(or otherwise entangled) is not.

Targets, all read from the frozen final latent:

  velocity (vx, vy)   positive control -- fed in as SymlogVelocity, i.e. already
                      a Cartesian pair, so the latent has every reason to hold
                      it linearly;
  force (fx, fy)      the teacher's resultant, the quantity Exp 22 says matters;
  force unit vector   the same direction with magnitude divided out;
  attitude (cos, sin) the ship's own heading, fed in as Fourier phases.

Scored as held-out R^2 per component. A large linear/MLP gap on the force and a
small one on velocity says the trunk holds velocity as a vector and the force
direction as something else.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp14_probe import fit_scalar  # noqa: E402
from exp18_terms import load  # noqa: E402


def r2(pred, y):
    return 1 - float(((y - pred) ** 2).mean()) / max(float(y.var()), 1e-12)


def fit_linear(xtr, ytr, xte):
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    a = torch.cat([(xtr - mu) / sd, torch.ones(len(xtr), 1)], 1)
    b = torch.cat([(xte - mu) / sd, torch.ones(len(xte), 1)], 1)
    w = torch.linalg.lstsq(a, ytr[:, None]).solution
    return (b @ w).squeeze(-1)


def main():
    tr, te = load("train"), load("heldout")
    sel_tr = (tr["I"]["mag_n_visible_enemies"] == 0) & (tr["I"]["mag_recovery"] < 0.01)
    sel_te = (te["I"]["mag_n_visible_enemies"] == 0) & (te["I"]["mag_recovery"] < 0.01)
    print(f"tokens: {int(sel_tr.sum())} train / {int(sel_te.sum())} held-out")

    def build(d, sel):
        I = d["I"]
        f = torch.polar(I["mag_objective"].clamp_min(0), I["dir_objective"]) + torch.polar(
            I["mag_separation"].clamp_min(0), I["dir_separation"]
        )
        u = f / f.abs().clamp_min(1e-8)
        cols = {
            "force_x": f.real, "force_y": f.imag,
            "force_unit_x": u.real, "force_unit_y": u.imag,
        }
        return d["latent"][sel], {k: v[sel] for k, v in cols.items()}

    xtr, ytr = build(tr, sel_tr)
    xte, yte = build(te, sel_te)
    rows = []
    for name in ytr:
        lin = r2(fit_linear(xtr, ytr[name], xte), yte[name])
        mlp = r2(fit_scalar(xtr, ytr[name], xte), yte[name])
        rows.append(dict(target=name, linear_r2=round(lin, 3), mlp_r2=round(mlp, 3),
                         gap=round(mlp - lin, 3)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp23_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
