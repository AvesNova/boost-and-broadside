"""Exp 17: is the ceiling this trunk, or the observation?

Trains small transformers from scratch on (observation -> teacher turn
distribution), supervised, no RL. Every arm keeps the project's transfer
constraints: absolute Fourier positions in one shared global frame, O(N) encoder
cost, permutation-equivariant attention, no weight whose shape depends on N.

Arms:
  depth L in {1, 2, 4, 6}   how much relational depth the teacher's reduction
                            actually needs, holding the encoding fixed;
  +relbias at L=2           a scalar attention bias per (query, key) pair built
                            from the toroidal displacement. This is *not* the
                            ego-relative token set from Exp 10 -- it adds no
                            per-pair tokens and no per-pair values, so encoder
                            cost stays O(N) and the weights stay size-agnostic.

If KL keeps falling with depth, the trunk is depth-limited on a legible input.
If it plateaus well above zero, the observation is the ceiling and no
architecture change helps.
"""

import json
import math
import os as _os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
DEV = "cuda"
NSHIPS = 8
D = 128


def onehot(x, n):
    return F.one_hot(x.long().clamp(0, n - 1), n).float()


def fourier(x, period, n):
    f = (2 * math.pi / period) * (2.0 ** torch.arange(n, dtype=x.dtype))
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)


def build(tag):
    from boost_and_broadside.train.rl.checkpoint_schema import position_fourier_frequencies

    d = torch.load(f"{S}/probe2_{tag}.pt")
    T = d["keep"].shape[0]
    o = {k: v[:T] for k, v in d["obs"].items()}
    W, H = d["world_size"]
    Tn, B, NM = o["pos"].shape[:3]
    nf = position_fourier_frequencies(float(W))
    tok = torch.cat([
        fourier(o["pos"][..., 0], W, nf), fourier(o["pos"][..., 1], H, nf),
        o["att"].reshape(Tn, B, NM, -1), o["vel"].reshape(Tn, B, NM, -1) / 100.0,
        o["health"].reshape(Tn, B, NM, -1),
        o["alive"].reshape(Tn, B, NM, 1), o["belief_valid"].reshape(Tn, B, NM, 1),
        o["radius"].reshape(Tn, B, NM, -1) / 1000.0,
        onehot(o["team_id"].reshape(Tn, B, NM), 3),
        onehot(o["object_type"].reshape(Tn, B, NM), 4),
        onehot(o["zone_role"].reshape(Tn, B, NM), 6),
    ], -1)
    return dict(
        tok=tok.reshape(Tn * B, NM, -1),
        pos=o["pos"].reshape(Tn * B, NM, 2),
        mask=o["belief_valid"].reshape(Tn * B, NM).bool(),
        expert=d["expert"].reshape(Tn * B, NM if d["expert"].shape[2] == NM else NSHIPS, -1),
        keep=d["keep"].reshape(Tn * B, NSHIPS),
        world=(float(W), float(H)),
    )


class Block(nn.Module):
    def __init__(self, relbias):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.relbias = relbias
        if relbias:
            # One scalar bias per (query, key) head, from the toroidal
            # displacement only. No per-pair token, no per-pair value: encoder
            # stays O(N) and nothing here knows how many ships exist.
            self.rel = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, x, mask, relfeat):
        B, N, _ = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q, k, v = (t.reshape(B, N, 4, D // 4).transpose(1, 2) for t in (q, k, v))
        bias = (~mask)[:, None, None, :].float() * -1e9
        if self.relbias:
            bias = bias + self.rel(relfeat).permute(0, 3, 1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.ffn(self.n2(x))


class Net(nn.Module):
    def __init__(self, in_dim, depth, relbias=False):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 2 * D), nn.RMSNorm(2 * D), nn.GELU(),
                                 nn.Linear(2 * D, D), nn.RMSNorm(D))
        self.blocks = nn.ModuleList([Block(relbias) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(D, 2 * D), nn.RMSNorm(2 * D), nn.GELU(),
                                  nn.Linear(2 * D, 7))
        self.relbias = relbias

    def forward(self, tok, mask, relfeat):
        x = self.enc(tok)
        for b in self.blocks:
            x = b(x, mask, relfeat)
        return self.head(x[:, :NSHIPS])


def relfeats(pos, world):
    W, H = world
    dx = pos[:, None, :, 0] - pos[:, :, None, 0]
    dy = pos[:, None, :, 1] - pos[:, :, None, 1]
    dx = (dx + W / 2) % W - W / 2
    dy = (dy + H / 2) % H - H / 2
    r = (dx * dx + dy * dy).sqrt().clamp_min(1e-6)
    return torch.stack([dx / 1000, dy / 1000, r / 1000, r.log1p(),
                        dx / r, dy / r], -1)


def train_eval(tr, te, depth, relbias, epochs=25, bs=256, lr=1e-3):
    net = Net(tr["tok"].shape[-1], depth, relbias).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr["tok"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            tok, msk, p = (tr["tok"][idx].to(DEV), tr["mask"][idx].to(DEV),
                           tr["expert"][idx].to(DEV))
            rf = relfeats(tr["pos"][idx].to(DEV), tr["world"]) if relbias else None
            keep = tr["keep"][idx].to(DEV).float()
            logits = net(tok, msk, rf)
            ce = -(p * F.log_softmax(logits, -1)).sum(-1)
            loss = (ce * keep).sum() / keep.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
    num = den = 0.0
    with torch.no_grad():
        for i in range(0, len(te["tok"]), 512):
            tok, msk = te["tok"][i : i + 512].to(DEV), te["mask"][i : i + 512].to(DEV)
            p = te["expert"][i : i + 512].to(DEV)
            rf = relfeats(te["pos"][i : i + 512].to(DEV), te["world"]) if relbias else None
            keep = te["keep"][i : i + 512].to(DEV).float()
            lq = F.log_softmax(net(tok, msk, rf), -1)
            pc = p.clamp_min(1e-8)
            k = (pc * (pc.log() - lq)).sum(-1)
            num += float((k * keep).sum())
            den += float(keep.sum())
    return num / den, sum(p.numel() for p in net.parameters())


def main():
    tr, te = build("train"), build("heldout")
    print(f"scenes: {len(tr['tok'])} train / {len(te['tok'])} held-out; "
          f"token dim {tr['tok'].shape[-1]}", flush=True)
    rows = []
    for depth, rel in [(1, False), (2, False), (4, False), (6, False), (2, True), (4, True)]:
        kl, nparam = train_eval(tr, te, depth, rel)
        rows.append(dict(depth=depth, relbias=rel, params=nparam, heldout_turn_kl=round(kl, 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp17_rows.json", "w"), indent=1)


main()
