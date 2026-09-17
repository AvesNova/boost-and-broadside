"""Exp 24: how much turn KL does sharpening the force vector actually buy?

Recommendation 3 says to supervise `frontline_strategy`'s force as an ego-frame
(x, y) vector, on Exp 23's finding that the frozen latent holds it to R^2
0.65-0.88 and that everything downstream follows from that imprecision. The
recommendation carries a falsification condition: if that R^2 is not the binding
constraint, sharpening it will move turn KL less than Exp 12's calibration
predicts.

That is testable now, with no training run. Interpolate the *probe's* estimate of
the force vector toward the true one, fit a turn head on each blend, and read off
held-out turn KL against the achieved R^2. The result is a dose-response curve:
what a given improvement in force-vector accuracy is worth in nats.

Same exactly-reconstructable sub-stratum as Exp 22/23, so the force is the real
resultant rather than an approximation of it.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402
from exp14_probe import fit_scalar  # noqa: E402
from exp18_terms import load  # noqa: E402


def r2(pred, y):
    return 1 - float(((y - pred) ** 2).mean()) / max(float(y.var()), 1e-12)


def main():
    tr, te = load("train"), load("heldout")
    m_tr = (tr["I"]["mag_n_visible_enemies"] == 0) & (tr["I"]["mag_recovery"] < 0.01)
    m_te = (te["I"]["mag_n_visible_enemies"] == 0) & (te["I"]["mag_recovery"] < 0.01)

    def force(d, m):
        I = d["I"]
        f = torch.polar(I["mag_objective"].clamp_min(0), I["dir_objective"]) + torch.polar(
            I["mag_separation"].clamp_min(0), I["dir_separation"]
        )
        return d["latent"][m], f.real[m], f.imag[m]

    xtr, fx_tr, fy_tr = force(tr, m_tr)
    xte, fx_te, fy_te = force(te, m_te)
    print(f"tokens: {len(xtr)} train / {len(xte)} held-out", flush=True)

    # The probe's own estimate, out of sample on the held-out half and in-sample
    # on the training half. The in-sample optimism inflates the *training*
    # features only; every reported number is scored on held-out tokens.
    px_te, py_te = fit_scalar(xtr, fx_tr, xte), fit_scalar(xtr, fy_tr, xte)
    px_tr, py_tr = fit_scalar(xtr, fx_tr, xtr), fit_scalar(xtr, fy_tr, xtr)
    print(json.dumps(dict(probe_r2_force_x=round(r2(px_te, fx_te), 3),
                          probe_r2_force_y=round(r2(py_te, fy_te), 3))), flush=True)

    rows = []
    base = float(kl(te["expert"][m_te], te["logq"][m_te]))
    rows.append(dict(blend="frozen policy head", turn_kl=round(base, 4)))
    print(json.dumps(rows[-1]), flush=True)
    for w in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        bx_tr = (1 - w) * px_tr + w * fx_tr
        by_tr = (1 - w) * py_tr + w * fy_tr
        bx_te = (1 - w) * px_te + w * fx_te
        by_te = (1 - w) * py_te + w * fy_te
        a = torch.cat([xtr, bx_tr[:, None], by_tr[:, None]], 1)
        b = torch.cat([xte, bx_te[:, None], by_te[:, None]], 1)
        lq = fit_head(a, tr["expert"][m_tr], b)
        rows.append(dict(
            blend=w,
            r2_x=round(r2(bx_te, fx_te), 3), r2_y=round(r2(by_te, fy_te), 3),
            turn_kl=round(float(kl(te["expert"][m_te], lq)), 4),
        ))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp24_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
