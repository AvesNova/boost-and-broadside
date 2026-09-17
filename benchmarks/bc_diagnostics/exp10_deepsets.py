"""Exp 10: does a sum-pooling probe with ego-relative tokens recover the bearing?

The teacher's bearing is a masked weighted sum of per-token unit vectors:
  force = sum_j w_j * unit(p_j - p_i),   bearing = angle(force) - attitude.

Self-attention cannot form that sum. A value vector v_j depends on j alone, so
sum_j a_ij v_j can never contain unit(p_j - p_i), which depends on the pair.
The trunk must instead approximate the whole reduction inside the FFN acting on
a pooled vector. This probe gives a small network the pairwise term explicitly
and pools it, and is otherwise far smaller than the trunk -- so if it wins, the
gap is structural, not capacity.

Cells:
  abs_fourier   per-token absolute Fourier position (what the encoder sees)
  ego_relative  per-token ego-frame displacement + unit vector
  ego_rel_nounit  ego-frame displacement only, unit vector withheld
"""

import json
import math
import os as _os
import sys

import torch
import torch.nn as nn

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, summarize  # noqa: E402

from boost_and_broadside.train.rl.checkpoint_schema import position_fourier_frequencies  # noqa: E402

DEV = "cuda"
NSHIPS = 8


def onehot(x, n):
    return torch.nn.functional.one_hot(x.long().clamp(0, n - 1), n).float()


def fourier(x, period, n):
    f = (2 * math.pi / period) * (2.0 ** torch.arange(n, dtype=x.dtype))
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)


def torus(d, size):
    return (d + size / 2) % size - size / 2


def build(tag, cell, prefix="obs"):
    """Return (tokens (n, NM, F), token_mask (n, NM), labels dict)."""
    d = torch.load(f"{S}/{prefix}_{tag}.pt")
    T_keep = d["keep"].shape[0]
    o = {k: v[:T_keep] for k, v in d["obs"].items()}
    W, H = d["world_size"]
    T, B, NM = o["pos"].shape[:3]
    pos, att = o["pos"], o["att"]
    scal = torch.cat([
        o["health"].reshape(T, B, NM, -1),
        o["alive"].reshape(T, B, NM, 1),
        o["belief_valid"].reshape(T, B, NM, 1),
        o["radius"].reshape(T, B, NM, -1) / 1000.0,
        onehot(o["team_id"].reshape(T, B, NM), 3),
        onehot(o["object_type"].reshape(T, B, NM), 4),
        onehot(o["zone_role"].reshape(T, B, NM), 6),
        att.reshape(T, B, NM, -1),
    ], -1)
    nf = position_fourier_frequencies(float(W))

    toks, masks = [], []
    for s in range(NSHIPS):
        is_ego = torch.zeros(T, B, NM, 1)
        is_ego[:, :, s] = 1.0
        if cell == "abs_fourier":
            geo = torch.cat([fourier(pos[..., 0], W, nf), fourier(pos[..., 1], H, nf)], -1)
            # The ego's own absolute position, broadcast, is the only way an
            # absolute encoding can express "relative to me" -- give it freely.
            ego_pos = geo[:, :, s : s + 1].expand_as(geo)
            geo = torch.cat([geo, ego_pos], -1)
        else:
            dx = torus(pos[..., 0] - pos[:, :, s : s + 1, 0], W)
            dy = torus(pos[..., 1] - pos[:, :, s : s + 1, 1], H)
            ca, sa = att[:, :, s : s + 1, 0], att[:, :, s : s + 1, 1]
            rx, ry = dx * ca + dy * sa, -dx * sa + dy * ca
            dist = (rx.square() + ry.square()).sqrt().clamp_min(1e-6)
            cols = [rx / 1000.0, ry / 1000.0, dist / 1000.0, dist.log1p()]
            if cell == "ego_relative":
                cols += [rx / dist, ry / dist]
            geo = torch.stack(cols, -1)
        toks.append(torch.cat([geo, scal, is_ego], -1))
        masks.append(o["belief_valid"][:, :, :].reshape(T, B, NM))
    x = torch.stack(toks, 2)  # (T,B,N,NM,F)
    m = torch.stack(masks, 2)
    keep = d["keep"].reshape(-1)
    tk = {k: v.reshape(-1)[keep] for k, v in d["teacher"].items() if v.dim() == 3}
    return (
        x.reshape(-1, NM, x.shape[-1])[keep],
        m.reshape(-1, NM)[keep],
        tk,
    )


class SumProbe(nn.Module):
    """phi per token -> masked sum and max -> rho. ~200k params, far under the trunk."""

    def __init__(self, f, h=192):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(2 * h + f, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, x, m, ego):
        e = self.phi(x) * m.unsqueeze(-1)
        pooled = torch.cat([e.sum(1) / m.sum(1, keepdim=True).clamp_min(1), e.amax(1), ego], -1)
        p = self.rho(pooled)
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def fit(xtr, mtr, ytr, xte, mte, epochs=30, bs=2048, lr=2e-3):
    mu = xtr.reshape(-1, xtr.shape[-1]).mean(0)
    sd = xtr.reshape(-1, xtr.shape[-1]).std(0).clamp_min(1e-4)
    xtr, xte = ((xtr - mu) / sd).to(DEV), ((xte - mu) / sd).to(DEV)
    mtr, mte, ytr = mtr.float().to(DEV), mte.float().to(DEV), ytr.to(DEV)
    ego_tr = xtr[:, :NSHIPS].reshape(len(xtr), -1)[:, : xtr.shape[-1]]
    net = SumProbe(xtr.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(xtr) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    ego_of = lambda x: (x * (x[..., -1:] > 0).float()).sum(1)  # the is_ego-flagged token
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i : i + bs]
            xb = xtr[idx]
            p = net(xb, mtr[idx], ego_of(xb))
            loss = (1 - (p * ytr[idx]).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        out = []
        for i in range(0, len(xte), 16384):
            xb = xte[i : i + 16384]
            out.append(net(xb, mte[i : i + 16384], ego_of(xb)).cpu())
    return torch.cat(out)


def main():
    rows = []
    for cell in ("abs_fourier", "ego_relative", "ego_rel_nounit"):
        xtr, mtr, ttr = build("train", cell)
        xte, mte, tte = build("heldout", cell)
        for target in ("rel_front", "rel_personal"):
            ytr = torch.stack([ttr[target].sin(), ttr[target].cos()], -1)
            pred = fit(xtr, mtr, ytr, xte, mte)
            err = ang_err_deg(pred, tte[target])
            row = dict(cell=cell, target=target, tokdim=xtr.shape[-1])
            row.update(summarize(err))
            a = tte["alpha"]
            for sn, sel in (("alpha~0", a < 0.01), ("alpha~1", a > 0.99)):
                if int(sel.sum()) > 200:
                    row[f"median@{sn}"] = round(float(err[sel].median()), 2)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del xtr, xte
    json.dump(rows, open(f"{S}/exp10_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
