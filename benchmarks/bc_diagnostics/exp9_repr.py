"""Exp 9b: representation audit -- absolute Fourier position vs ego-relative geometry.

Same information, two encodings, one probe architecture, one held-out split:

  A "as encoded"   every token's absolute position through the model's own
                   8-frequency Fourier expansion, plus its scalar channels.
                   This is what the ship encoder actually receives.
  B "ego-relative" the same tokens, but with the toroidal displacement to the
                   ego ship rotated into the ego frame and handed over as
                   (dx, dy, distance, unit vector).

A probe that does well on B and badly on A says the geometry is present in the
observation but the encoding makes it expensive to recover. A probe that does
badly on both says the observation itself is the ceiling.
"""

import json
import math
import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_probe import ang_err_deg, fit_probe, summarize  # noqa: E402

from boost_and_broadside.train.rl.checkpoint_schema import position_fourier_frequencies  # noqa: E402

NSHIPS = 8


def onehot(x, n):
    return torch.nn.functional.one_hot(x.long().clamp(0, n - 1), n).float()


def fourier(x, period, n_freqs):
    k = torch.arange(n_freqs, dtype=x.dtype)
    f = (2 * math.pi / period) * (2.0**k)
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)


def torus(d, size):
    return (d + size / 2) % size - size / 2


def build(tag):
    d = torch.load(f"{S}/obs_{tag}.pt")
    # The obs buffer carries one extra bootstrap timestep that the masks do not;
    # trim to the mask length so states stay aligned with their labels.
    T_keep = d["keep"].shape[0]
    o = {k: v[:T_keep] for k, v in d["obs"].items()}
    W, H = d["world_size"]
    T, B, NM = o["pos"].shape[:3]
    pos, att = o["pos"], o["att"]  # (T,B,NM,2), (T,B,NM,2) as (cos,sin) or (x,y)

    scal = torch.cat([
        o["health"].reshape(T, B, NM, -1),
        o["alive"].reshape(T, B, NM, 1),
        o["belief_valid"].reshape(T, B, NM, 1),
        o["radius"].reshape(T, B, NM, -1) / 1000.0,
        onehot(o["team_id"].reshape(T, B, NM), 3),
        onehot(o["object_type"].reshape(T, B, NM), 4),
        onehot(o["zone_role"].reshape(T, B, NM), 6),
    ], -1)

    nf = position_fourier_frequencies(float(W))
    posenc = torch.cat([fourier(pos[..., 0], W, nf), fourier(pos[..., 1], H, nf)], -1)
    attenc = att.reshape(T, B, NM, -1)
    tok_abs = torch.cat([posenc, attenc, scal], -1)  # (T,B,NM,dA)

    feats_a, feats_b, targ, alpha = [], [], [], []
    ego_att = att[:, :, :NSHIPS]  # (T,B,N,2)
    for s in range(NSHIPS):
        # A: absolute encoding, with a one-hot marking which token is the ego.
        is_ego = torch.zeros(T, B, NM, 1)
        is_ego[:, :, s] = 1.0
        feats_a.append(torch.cat([tok_abs, is_ego], -1).reshape(T, B, -1))

        # B: the same tokens in the ego's own frame.
        dx = torus(pos[..., 0] - pos[:, :, s : s + 1, 0], W)
        dy = torus(pos[..., 1] - pos[:, :, s : s + 1, 1], H)
        ca, sa = ego_att[:, :, s : s + 1, 0], ego_att[:, :, s : s + 1, 1]
        rx, ry = dx * ca + dy * sa, -dx * sa + dy * ca  # rotate into ego heading
        dist = (rx.square() + ry.square()).sqrt().clamp_min(1e-6)
        feats_b.append(torch.cat([
            (rx / 1000.0).unsqueeze(-1), (ry / 1000.0).unsqueeze(-1),
            (dist / 1000.0).unsqueeze(-1), (dist.log1p()).unsqueeze(-1),
            (rx / dist).unsqueeze(-1), (ry / dist).unsqueeze(-1),
            attenc, scal, is_ego,
        ], -1).reshape(T, B, -1))

    keep = d["keep"]  # (T,B,N)
    out = {}
    for name, fl in (("A_absolute_fourier", feats_a), ("B_ego_relative", feats_b)):
        x = torch.stack(fl, 2)  # (T,B,N,dim)
        out[name] = x.reshape(-1, x.shape[-1])[keep.reshape(-1)]
    tk = {k: v.reshape(-1)[keep.reshape(-1)] for k, v in d["teacher"].items() if v.dim() == 3}
    return out, tk


def main():
    tr, ttr = build("train")
    te, tte = build("heldout")
    rows = []
    for target in ("rel_front", "rel_personal"):
        ytr = torch.stack([ttr[target].sin(), ttr[target].cos()], -1)
        for name in tr:
            for hidden in (512, 1024):
                _, _, pred = fit_probe(tr[name], ytr, te[name], hidden=hidden, epochs=60, lr=2e-3)
                err = ang_err_deg(pred, tte[target])
                row = dict(target=target, features=name, dim=tr[name].shape[1], hidden=hidden)
                row.update(summarize(err))
                a = tte["alpha"]
                for sn, sel in (("alpha~0", a < 0.01), ("alpha~1", a > 0.99)):
                    if int(sel.sum()) > 200:
                        row[f"median@{sn}"] = round(float(err[sel].median()), 2)
                rows.append(row)
                print(json.dumps(row), flush=True)
    json.dump(rows, open(f"{S}/exp9_rows.json", "w"), indent=1)


main()
