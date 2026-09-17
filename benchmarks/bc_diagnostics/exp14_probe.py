"""Exp 14b: does the trunk represent directions but not magnitudes?

Two families of target, one probe architecture, one held-out split:

  dir_*  normalised weighted averages (softmax attention's natural output),
         scored as angular error in degrees;
  mag_*  unnormalised sums and counts (what softmax discards), scored as
         held-out R^2 -- the fraction of the quantity's variance the frozen
         latent can explain.

R^2 is the right scale here because these quantities have wildly different
units; a magnitude the trunk cannot represent at all scores ~0 however large it
is.
"""

import json
import os as _os
import sys

import torch
import torch.nn as nn

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, fit_probe  # noqa: E402

DEV = "cuda"
TAPS = ("encoder", "b0.temporal0", "b1.temporal0")


def load(tag):
    d = torch.load(f"{S}/probe2_{tag}.pt")
    k = d["keep"].reshape(-1)
    return dict(
        acts={n: v.reshape(-1, v.shape[-1])[k].float() for n, v in d["acts"].items()},
        internals={n: v.reshape(-1)[k] for n, v in d["internals"].items() if v.dim() == 3},
        teach={n: v.reshape(-1)[k] for n, v in d["teacher"].items() if v.dim() == 3},
    )


def fit_scalar(xtr, ytr, xte, hidden=512, epochs=40, bs=4096, lr=2e-3):
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    xtr_n, xte_n = ((xtr - mu) / sd).to(DEV), ((xte - mu) / sd).to(DEV)
    ym, ys = ytr.mean(), ytr.std().clamp_min(1e-6)
    y = ((ytr - ym) / ys).to(DEV)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(xtr_n) // bs)
    )
    for _ in range(epochs):
        perm = torch.randperm(len(xtr_n), device=DEV)
        for i in range(0, len(xtr_n) - bs + 1, bs):
            idx = perm[i : i + bs]
            loss = nn.functional.mse_loss(net(xtr_n[idx]).squeeze(-1), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        out = [net(xte_n[i : i + 65536]).squeeze(-1).cpu() for i in range(0, len(xte_n), 65536)]
    return torch.cat(out) * ys + ym


def main():
    tr, te = load("train"), load("heldout")
    rows = []
    names = sorted(tr["internals"]) + ["rel_front"]
    for tap in TAPS:
        xtr, xte = tr["acts"][tap], te["acts"][tap]
        for name in names:
            if name == "_drift":
                continue
            src_tr = tr["internals"] if name in tr["internals"] else tr["teach"]
            src_te = te["internals"] if name in te["internals"] else te["teach"]
            ytr, yte = src_tr[name], src_te[name]
            if name.startswith("dir_") or name == "rel_front":
                y = torch.stack([ytr.sin(), ytr.cos()], -1)
                _, _, pred = fit_probe(xtr, y, xte, epochs=40)
                err = ang_err_deg(pred, yte)
                row = dict(tap=tap, target=name, kind="direction",
                           median_deg=round(float(err.median()), 2),
                           p90_deg=round(float(torch.quantile(err, 0.9)), 2))
            else:
                pred = fit_scalar(xtr, ytr, xte)
                ss_res = float(((yte - pred) ** 2).mean())
                ss_tot = float(yte.var())
                row = dict(tap=tap, target=name, kind="magnitude",
                           r2=round(1 - ss_res / max(ss_tot, 1e-12), 3),
                           target_std=round(float(yte.std()), 3))
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(f"{S}/exp14_rows.json", "w"), indent=1)


main()
