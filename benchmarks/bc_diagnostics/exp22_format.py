"""Exp 22: is the composition gap about information, or about format?

Exp 18 handed the frozen latent all three of the teacher's force terms as
(direction, magnitude) pairs and reached turn KL 0.338, against 0.034 for the
true bearing. Those two feature sets carry the *same information* -- the bearing
is a deterministic function of the terms -- so the gap is the head failing to
perform the composition, not a missing quantity.

The composition is a vector sum. In Cartesian form that is linear and a linear
layer can do it exactly; in polar form it needs sin/cos expansion and a product,
which a 2-layer MLP has to approximate. This re-runs Exp 18's cells with the
terms expressed as (x, y) instead of (angle, magnitude), and adds the
pre-summed resultant as an upper reference.

If Cartesian collapses the KL where polar did not, the lesson is about the
geometry of the representation, and it applies to what the trunk hands its own
action head.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402
from exp18_terms import load, sc  # noqa: E402


def main():
    tr, te = load("train"), load("heldout")
    # Zero visible enemies kills combat_force; recovery < 0.01 kills the spawn
    # term. In that sub-stratum force = objective + separation *exactly* -- Exp 19
    # validated the reconstruction at 0.000 deg there -- so the "resultant" cells
    # below are the real resultant and not an approximation of it.
    s_tr = (tr["I"]["mag_n_visible_enemies"] == 0) & (tr["I"]["mag_recovery"] < 0.01)
    s_te = (te["I"]["mag_n_visible_enemies"] == 0) & (te["I"]["mag_recovery"] < 0.01)
    print(f"zero-enemy AND recovery<0.01: {int(s_tr.sum())} / {int(s_te.sum())}")

    def terms(d):
        I = d["I"]
        names = (("dir_objective", "mag_objective"),
                 ("dir_separation", "mag_separation"))
        polar, cart = [], []
        total = torch.zeros(len(I["mag_objective"]), dtype=torch.cfloat)
        for da, ma in names:
            a, m = I[da], I[ma]
            polar += [sc(a), m[:, None]]
            v = torch.polar(m.clamp_min(0), a)
            cart += [v.real[:, None], v.imag[:, None]]
            total = total + v
        rec = I["mag_recovery"][:, None]
        polar.append(rec)
        cart.append(rec)
        return (torch.cat(polar, 1), torch.cat(cart, 1),
                torch.cat([total.real[:, None], total.imag[:, None]], 1),
                torch.cat([sc(total.angle()), total.abs()[:, None]], 1))

    p_tr, c_tr, r_tr, rp_tr = terms(tr)
    p_te, c_te, r_te, rp_te = terms(te)

    cells = {
        "latent only": (None, None),
        "+ terms, POLAR (sin,cos,mag)": (p_tr, p_te),
        "+ terms, CARTESIAN (x,y)": (c_tr, c_te),
        "+ resultant, CARTESIAN (x,y)": (r_tr, r_te),
        "+ resultant, POLAR (sin,cos,mag)": (rp_tr, rp_te),
        "+ true bearing (control)": (sc(tr["T"]["rel_front"]), sc(te["T"]["rel_front"])),
    }
    rows = [dict(features="frozen policy head",
                 turn_kl=round(float(kl(te["expert"][s_te], te["logq"][s_te])), 4))]
    print(json.dumps(rows[-1]), flush=True)
    for name, (a, b) in cells.items():
        xa = tr["latent"][s_tr] if a is None else torch.cat([tr["latent"][s_tr], a[s_tr]], 1)
        xb = te["latent"][s_te] if b is None else torch.cat([te["latent"][s_te], b[s_te]], 1)
        lq = fit_head(xa, tr["expert"][s_tr], xb)
        rows.append(dict(features=name, turn_kl=round(float(kl(te["expert"][s_te], lq)), 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp22_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
