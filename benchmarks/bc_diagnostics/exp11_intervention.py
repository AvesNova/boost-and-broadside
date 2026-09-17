"""Exp 11: does a better geometric representation actually buy turn KL?

Chains Exp 10's result into the metric that matters. A sum-pooling bearing probe
over ego-relative tokens is fitted on one set of rollouts, its bearing estimate
is read out on two *other* sets, and a turn head is fitted on the second and
scored on the third. Every number is therefore an unseen-state number, and the
predicted bearing is out-of-sample everywhere it is used.

  frozen policy        the checkpoint's own action head, for reference
  latent only          a fresh head on the frozen final latent
  latent + predicted   the same head plus the probe's estimated bearings
  latent + true        the oracle ceiling from Exp 8
  true bearings only   the achievable floor
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, summarize  # noqa: E402
from exp8_oracle import fit_head, kl  # noqa: E402
from exp10_deepsets import build, fit  # noqa: E402

LATENT = "b1.temporal0"
TARGETS = ("rel_front", "rel_personal")


def latents(tag):
    d = torch.load(f"{S}/probe_{tag}.pt")
    keep = d["keep"].reshape(-1)
    return dict(
        latent=d["acts"][LATENT].reshape(-1, d["acts"][LATENT].shape[-1])[keep].float(),
        expert=d["expert"].reshape(-1, d["expert"].shape[-1])[keep].float(),
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[keep].float(),
        teach={k: v.reshape(-1)[keep] for k, v in d["teacher"].items() if v.dim() == 3},
    )


def sincos(a):
    return torch.stack([a.sin(), a.cos()], -1)


def main():
    fit_tag, eval_tag = "heldout", "heldout2"
    L = {t: latents(t) for t in ("train", fit_tag, eval_tag)}

    # 1. Fit the bearing probes on the train rollouts only.
    pred = {t: {} for t in (fit_tag, eval_tag)}
    rows_b = []
    xtr, mtr, ttr = build("train", "ego_relative", prefix="probe")
    xa, ma, _ = build(fit_tag, "ego_relative", prefix="probe")
    xb, mb, tb = build(eval_tag, "ego_relative", prefix="probe")
    for tgt in TARGETS:
        y = sincos(ttr[tgt])
        pa = fit(xtr, mtr, y, xa, ma)
        pb = fit(xtr, mtr, y, xb, mb)
        pred[fit_tag][tgt], pred[eval_tag][tgt] = pa, pb
        err = ang_err_deg(pb, tb[tgt])
        row = dict(target=tgt, probe="ego_relative sum-pool", split=eval_tag)
        row.update(summarize(err))
        rows_b.append(row)
        print(json.dumps(row), flush=True)
    del xtr, xa, xb

    # 2. Turn heads, fitted on `fit_tag`, scored on `eval_tag`.
    def feats(tag, mode):
        t = L[tag]["teach"]
        lat = L[tag]["latent"]
        true = torch.cat([sincos(t["rel_front"]), sincos(t["rel_personal"]),
                          t["alpha"][:, None], t["rel_front"].abs()[:, None],
                          t["rel_personal"].abs()[:, None]], 1)
        p = pred[tag]
        est = torch.cat([p["rel_front"], p["rel_personal"],
                         torch.atan2(p["rel_front"][:, :1], p["rel_front"][:, 1:2]).abs(),
                         torch.atan2(p["rel_personal"][:, :1], p["rel_personal"][:, 1:2]).abs()], 1)
        return {"latent": lat, "latent+pred": torch.cat([lat, est], 1),
                "latent+true": torch.cat([lat, true], 1), "true only": true,
                "pred only": est}[mode]

    rows_h = [dict(head="frozen policy action head",
                   turn_kl=round(float(kl(L[eval_tag]["expert"], L[eval_tag]["logq"])), 4))]
    print(json.dumps(rows_h[-1]), flush=True)
    for mode in ("latent", "latent+pred", "pred only", "latent+true", "true only"):
        lq = fit_head(feats(fit_tag, mode), L[fit_tag]["expert"], feats(eval_tag, mode))
        rows_h.append(dict(head=mode, turn_kl=round(float(kl(L[eval_tag]["expert"], lq)), 4)))
        print(json.dumps(rows_h[-1]), flush=True)
    json.dump(dict(bearing=rows_b, heads=rows_h), open(f"{S}/exp11_rows.json", "w"), indent=1)


main()
