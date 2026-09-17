"""Exp 13: is the left/right gap a representation asymmetry or a head asymmetry?

If the bearing probes are equally accurate on both sides while the policy's turn
KL is not, the asymmetry is downstream of the geometry. If the probes are also
worse on the right, the trunk's own spatial representation is lopsided.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, fit_probe  # noqa: E402
from exp11_intervention import latents, sincos  # noqa: E402

LATENT_TAPS = ("b1.temporal0",)


def main():
    tr, te = latents("train"), latents("heldout2")
    d_tr = torch.load(f"{S}/probe_train.pt")
    d_te = torch.load(f"{S}/probe_heldout2.pt")
    ktr, kte = d_tr["keep"].reshape(-1), d_te["keep"].reshape(-1)
    rows = []
    for tap in LATENT_TAPS:
        xtr = d_tr["acts"][tap].reshape(-1, d_tr["acts"][tap].shape[-1])[ktr].float()
        xte = d_te["acts"][tap].reshape(-1, d_te["acts"][tap].shape[-1])[kte].float()
        for tgt in ("rel_front", "rel_personal"):
            _, _, pred = fit_probe(xtr, sincos(tr["teach"][tgt]), xte, epochs=40)
            true = te["teach"][tgt]
            err = ang_err_deg(pred, true)
            for side, sel in (("left", true < 0), ("right", true > 0)):
                # Match on |bearing| the same way Exp 12 does, so the two sides
                # are compared at the same difficulty.
                band = (true.abs() > 0.03) & (true.abs() < 0.7)
                m = sel & band
                rows.append(dict(tap=tap, target=tgt, side=side, n=int(m.sum()),
                                 median_err_deg=round(float(err[m].median()), 2),
                                 p90_err_deg=round(float(torch.quantile(err[m], 0.9)), 2)))
                print(json.dumps(rows[-1]), flush=True)
            # Signed error: a systematic rotation bias would show up here and
            # nowhere else.
            pa = torch.atan2(pred[:, 0], pred[:, 1])
            signed = ((pa - true + torch.pi) % (2 * torch.pi) - torch.pi) * 180 / torch.pi
            print(json.dumps(dict(tap=tap, target=tgt, signed_median_deg=round(float(signed.median()), 3),
                                  signed_mean_deg=round(float(signed.mean()), 3))), flush=True)
    json.dump(rows, open(f"{S}/exp13_rows.json", "w"), indent=1)


main()
