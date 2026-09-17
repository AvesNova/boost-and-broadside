"""Exp 7b: layerwise bearing probes.

Freeze the checkpoint, then fit small probes that read one trunk tap point and
predict (sin, cos) of a teacher bearing. Train on probe_train.pt, report on
probe_heldout.pt — independent rollouts, so every number below is a fresh-state
number. Angular error is reported in degrees because that is the unit the
teacher's ramps are steep in (turn_angle_ramp spans 1.7 deg to 6.9 deg).
"""

import json
import os as _os
import sys

import torch
import torch.nn as nn

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
DEV = "cuda"
TARGETS = {"frontline": "rel_front", "personal": "rel_personal", "combat": "rel_combat"}


def load(tag):
    d = torch.load(f"{S}/probe_{tag}.pt")
    keep = d["keep"].reshape(-1)
    acts = {k: v.reshape(-1, v.shape[-1])[keep].float() for k, v in d["acts"].items()}
    teach = {k: v.reshape(-1)[keep] if v.dim() == 3 else v for k, v in d["teacher"].items()}
    return dict(
        acts=acts,
        teach=teach,
        expert=d["expert"].reshape(-1, d["expert"].shape[-1])[keep],
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[keep],
    )


def fit_probe(xtr, ytr, xte, hidden=512, epochs=40, bs=4096, lr=3e-3, linear=False):
    """Fit x -> (sin, cos). Inputs standardized; output normalized to the circle."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-4)
    xtr_n, xte_n = ((xtr - mu) / sd).to(DEV), ((xte - mu) / sd).to(DEV)
    ytr = ytr.to(DEV)
    d = xtr.shape[1]
    net = (
        nn.Linear(d, 2)
        if linear
        else nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(xtr_n) // bs)
    )
    n = len(xtr_n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            p = net(xtr_n[idx])
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # 1 - cos(error): the natural circle loss, and monotone in |error|.
            loss = (1 - (p * ytr[idx]).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        out = []
        for i in range(0, len(xte_n), 65536):
            p = net(xte_n[i : i + 65536])
            out.append((p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)).cpu())
    return net, (mu, sd), torch.cat(out)


def ang_err_deg(pred_sc, true_ang):
    """Absolute angular error in degrees, wrapped to [0, 180]."""
    pred_ang = torch.atan2(pred_sc[:, 0], pred_sc[:, 1])
    e = (pred_ang - true_ang + torch.pi) % (2 * torch.pi) - torch.pi
    return e.abs() * 180 / torch.pi


def summarize(err):
    q = torch.quantile(err, torch.tensor([0.5, 0.9, 0.95]))
    return dict(
        median=round(float(q[0]), 2),
        mean=round(float(err.mean()), 2),
        p90=round(float(q[1]), 2),
        p95=round(float(q[2]), 2),
        **{f"within_{t}deg": round(float((err < t).float().mean()), 3) for t in (1, 3, 5, 10, 20)},
    )


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tr, te = load("train"), load("heldout")
    order = [
        "raw_input", "encoder", "b0.spatial0", "b0.spatial1", "b0.temporal0",
        "b1.spatial0", "b1.spatial1", "b1.temporal0",
    ]
    rows = []
    for tname, tkey in TARGETS.items():
        if which != "all" and which != tname:
            continue
        if tkey not in tr["teach"]:
            continue
        ytr = torch.stack([tr["teach"][tkey].sin(), tr["teach"][tkey].cos()], -1)
        true_te = te["teach"][tkey]
        for tap in order:
            for linear in (True, False):
                _, _, pred = fit_probe(
                    tr["acts"][tap], ytr, te["acts"][tap], linear=linear,
                    epochs=15 if linear else 40,
                )
                err = ang_err_deg(pred, true_te)
                row = dict(target=tname, tap=tap, probe="linear" if linear else "mlp", n=len(err))
                row.update(summarize(err))
                # The strata that matter: alpha says which bearing the teacher is
                # actually using, and the turn ramp is where a degree is expensive.
                a = te["teach"]["alpha"]
                inramp = (true_te.abs() >= 0.03) & (true_te.abs() < 0.12)
                for sname, sel in [
                    ("alpha~0", a < 0.01), ("alpha~1", a > 0.99), ("in_turn_ramp", inramp),
                ]:
                    if int(sel.sum()) > 200:
                        row[f"median@{sname}"] = round(float(err[sel].median()), 2)
                rows.append(row)
                print(json.dumps(row), flush=True)
    json.dump(rows, open(f"{S}/exp7_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
