"""Exp 18: give the head every term of the teacher's force sum, not just one.

Exp 16 handed the frozen latent `objective_force` alone and recovered only 0.115
of 0.655. But with no enemies visible the teacher's force is

    force = (1-recovery)*objective_force + recovery*spawn_dir + separation

so one term out of three was never going to close it. This gives them in
combination, inside the same zero-visible-enemy stratum, with the true bearing
as a positive control.

If the combination collapses the KL, the trunk holds the pieces and fails to
combine them. If it does not, the pieces are not what is missing.
"""

import json
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402


def load(tag):
    d = torch.load(f"{S}/probe2_{tag}.pt")
    k = d["keep"].reshape(-1)
    lat = d["acts"]["b1.temporal0"]
    return dict(
        latent=lat.reshape(-1, lat.shape[-1])[k].float(),
        expert=d["expert"].reshape(-1, d["expert"].shape[-1])[k].float(),
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[k].float(),
        I={n: v.reshape(-1)[k] for n, v in d["internals"].items() if v.dim() == 3},
        T={n: v.reshape(-1)[k] for n, v in d["teacher"].items() if v.dim() == 3},
    )


def sc(a):
    return torch.stack([a.sin(), a.cos()], -1)


def main():
    tr, te = load("train"), load("heldout")
    s_tr = tr["I"]["mag_n_visible_enemies"] == 0
    s_te = te["I"]["mag_n_visible_enemies"] == 0
    print(f"zero-visible-enemy stratum: {int(s_tr.sum())} train / {int(s_te.sum())} held-out")

    def feat(d, keys):
        cols = []
        for k in keys:
            if k == "objective":
                cols += [sc(d["I"]["dir_objective"]), d["I"]["mag_objective"][:, None]]
            elif k == "separation":
                cols += [sc(d["I"]["dir_separation"]), d["I"]["mag_separation"][:, None]]
            elif k == "recovery":
                cols += [d["I"]["mag_recovery"][:, None], d["I"]["mag_spawn_dist"][:, None] / 1000]
            elif k == "bearing":
                cols += [sc(d["T"]["rel_front"])]
        return torch.cat(cols, 1) if cols else None

    rows = [dict(extra="none (frozen policy head)",
                 turn_kl=round(float(kl(te["expert"][s_te], te["logq"][s_te])), 4))]
    print(json.dumps(rows[-1]), flush=True)
    cells = {
        "none (refit)": [],
        "+ objective": ["objective"],
        "+ separation": ["separation"],
        "+ recovery/spawn": ["recovery"],
        "+ objective + separation": ["objective", "separation"],
        "+ all three force terms": ["objective", "separation", "recovery"],
        "+ true bearing (control)": ["bearing"],
    }
    for name, keys in cells.items():
        f_tr, f_te = feat(tr, keys), feat(te, keys)
        a = tr["latent"][s_tr] if f_tr is None else torch.cat([tr["latent"][s_tr], f_tr[s_tr]], 1)
        b = te["latent"][s_te] if f_te is None else torch.cat([te["latent"][s_te], f_te[s_te]], 1)
        lq = fit_head(a, tr["expert"][s_tr], b)
        rows.append(dict(extra=name, turn_kl=round(float(kl(te["expert"][s_te], lq)), 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp18_rows.json", "w"), indent=1)


main()
