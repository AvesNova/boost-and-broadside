"""Exp 21: does extra relational depth help, starting from the frozen trunk?

Exp 17 tried to answer "is the ceiling this trunk or the observation?" by
training transformers from scratch, and failed: 12k scenes cannot stand in for
167.6M environment steps, and held-out KL rose with depth.

This asks the same question far more cheaply. Freeze the checkpoint, take its
*final latent for all 24 entity tokens*, and train only N additional spatial
layers plus a turn head on top. Nothing below is updated, so the experiment
needs no rollouts beyond collection and cannot be data-limited in the same way:
the hard representational work is already done and paid for.

  L=0   a turn head on the frozen latent -- the baseline (cf. Exp 8's 0.64)
  L=1,2,4   extra transformer layers over the 24 tokens, then the head
  L=2 +relbias   the same, with the transfer-safe relative-position bias

If KL falls steeply with L, the trunk is depth-limited on a representation it
already has, and more spatial sublayers are the fix. If it flattens immediately,
depth is not what is missing and the residual is upstream of the trunk's output.

Caveat stated up front: a frozen latent may already have discarded information
that end-to-end depth would have preserved, so this is a *lower* bound on what
depth buys, not an estimate of it.
"""

import json
import os as _os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp17_depth import Block, relfeats  # noqa: E402
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402
from boost_and_broadside.env.observation import ObsKey, YemongObservation  # noqa: E402

DEV = "cuda"
D = 128
NSHIPS = 8
TSTRIDE = 4


@torch.no_grad()
def latents_all_tokens(trainer, buf, envs_per_chunk=8):
    """Final-latent activations for every entity token, not just ships.

    Hooks the last YemongBlock rather than its temporal sublayer: the temporal
    path only ever sees ship tokens (fields take forward_nonrecurrent), and this
    experiment needs the zone tokens too.
    """
    policy = trainer.policy
    grab = []
    block = policy.yemong_layers[-1]
    # YemongBlock.sequence is called directly by evaluate_actions, so a forward
    # hook never fires on the block either -- wrap the method, as with the
    # temporal sublayers. Its x is already (T, B, N+M, D).
    orig_sequence = block.sequence

    def wrapped(*a, **kw):
        out = orig_sequence(*a, **kw)
        grab.append(out[0].detach().to("cpu", torch.float16))
        return out

    block.sequence = wrapped
    B, N, T = buf.num_envs, buf.num_ships, buf.num_steps
    # Packed hidden is (n_layers, B*N, CONV_KERNEL*D) -- read the width off the
    # tensor rather than assuming D, which silently mis-shapes the reshape.
    Dh = buf.initial_hidden.shape[-1]
    n_layers = buf.initial_hidden.shape[0]
    hid_full = buf.initial_hidden.reshape(n_layers, B, N, Dh)
    outs, tl = [], []
    for s in range(0, B, envs_per_chunk):
        e = min(s + envs_per_chunk, B)
        grab.clear()
        mb = YemongObservation(
            data={k: v[:, s:e] for k, v in buf.obs.items()},
            bullets=None
            if buf.bullet_obs is None
            else {k: v[:, s:e] for k, v in buf.bullet_obs.items()},
        )
        obs = mb.slice_time(0, T)
        hid = hid_full[:, s:e].reshape(n_layers, (e - s) * N, Dh).contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, logits, _, _, _ = policy.evaluate_actions(
                obs=obs, actions=buf.actions[:, s:e].long(), initial_hidden=hid,
                alive_mask=obs[ObsKey.BELIEF_VALID].bool(), done_mask=buf.terminated[:, s:e])
        tl.append(torch.log_softmax(logits[..., TURN_SLICE].float(), -1)[::TSTRIDE].cpu())
        outs.append(grab[-1][::TSTRIDE].contiguous())
    block.sequence = orig_sequence
    return torch.cat(outs, 1), torch.cat(tl, 1)


def collect(tag, seed, n_roll=3, burn=3):
    trainer, launch = build(num_envs=128, microbatch_tokens=12288, seed=seed)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    for _ in range(burn):
        trainer._collect_rollout(runtime, False)
    chunks = []
    for r in range(n_roll):
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        buf = trainer.buffer
        lat, tl = latents_all_tokens(trainer, buf)
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        T = m[::TSTRIDE].shape[0]
        chunks.append(dict(
            lat=lat, turn_logp=tl,
            expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
            keep=m[::TSTRIDE].cpu(),
            mask=buf.obs[ObsKey.BELIEF_VALID][::TSTRIDE][:T].bool().cpu(),
            pos=buf.obs[ObsKey.POS][::TSTRIDE][:T].float().cpu(),
        ))
        print(f"  {tag} {r}: {time.time()-t0:.0f}s lat={tuple(lat.shape)}", flush=True)
    out = {k: torch.cat([c[k] for c in chunks], 1) for k in chunks[0]}
    out["world"] = launch.resolved.ship_config.world_size
    return out


class Top(nn.Module):
    def __init__(self, depth, relbias):
        super().__init__()
        self.blocks = nn.ModuleList([Block(relbias) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(D, 2 * D), nn.RMSNorm(2 * D), nn.GELU(),
                                  nn.Linear(2 * D, 7))
        self.relbias = relbias

    def forward(self, x, mask, rf):
        for b in self.blocks:
            x = b(x, mask, rf)
        return self.head(x[:, :NSHIPS])


def train_eval(tr, te, depth, relbias, epochs=25, bs=256, lr=1e-3):
    net = Top(depth, relbias).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr["lat"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            x = tr["lat"][idx].to(DEV).float()
            msk = tr["mask"][idx].to(DEV)
            p = tr["expert"][idx].to(DEV)
            keep = tr["keep"][idx].to(DEV).float()
            rf = relfeats(tr["pos"][idx].to(DEV), tr["world"]) if relbias else None
            ce = -(p * F.log_softmax(net(x, msk, rf), -1)).sum(-1)
            loss = (ce * keep).sum() / keep.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
    num = den = 0.0
    with torch.no_grad():
        for i in range(0, len(te["lat"]), 512):
            x = te["lat"][i : i + 512].to(DEV).float()
            msk, p = te["mask"][i : i + 512].to(DEV), te["expert"][i : i + 512].to(DEV)
            keep = te["keep"][i : i + 512].to(DEV).float()
            rf = relfeats(te["pos"][i : i + 512].to(DEV), te["world"]) if relbias else None
            lq = F.log_softmax(net(x, msk, rf), -1)
            pc = p.clamp_min(1e-8)
            k = (pc * (pc.log() - lq)).sum(-1)
            num += float((k * keep).sum())
            den += float(keep.sum())
    return num / den, sum(q.numel() for q in net.parameters())


def flat(d):
    T, B = d["keep"].shape[:2]
    f = lambda t: t.reshape(T * B, *t.shape[2:])  # noqa: E731
    return dict(lat=f(d["lat"]), mask=f(d["mask"]), pos=f(d["pos"]),
                expert=f(d["expert"]), keep=f(d["keep"]),
                logq=f(d["turn_logp"]), world=d["world"])


def main():
    # More distinct scenes with proportionally fewer epochs. The first pass at 3
    # rollouts (12 288 scenes) was data-limited: held-out KL degraded
    # monotonically with depth -- 0.503 / 0.668 / 0.818 / 0.965 at 35k / 233k /
    # 430k / 825k parameters -- which measures overfitting, not capacity.
    n_roll = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    suffix = "" if n_roll == 3 else f"_{n_roll}"
    for tag, seed in (("train", 0), ("heldout", 1)):
        p = f"{S}/fd_{tag}{suffix}.pt"
        if not _os.path.exists(p):
            torch.save(collect(tag, seed, n_roll=n_roll), p)
    tr = flat(torch.load(f"{S}/fd_train{suffix}.pt"))
    te = flat(torch.load(f"{S}/fd_heldout{suffix}.pt"))
    pc = te["expert"].clamp_min(1e-8)
    base = float(((pc * (pc.log() - te["logq"])).sum(-1) * te["keep"]).sum() / te["keep"].sum())
    rows = [dict(depth="frozen policy head", relbias=False, heldout_turn_kl=round(base, 4))]
    print(json.dumps(rows[-1]), flush=True)
    print(f"scenes: {len(tr['lat'])} train / {len(te['lat'])} held-out", flush=True)
    for depth, rel in [(0, False), (1, False), (2, False), (2, True)]:
        kl, npar = train_eval(tr, te, depth, rel, epochs=epochs)
        rows.append(dict(depth=depth, relbias=rel, params=npar, heldout_turn_kl=round(kl, 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp21_rows{suffix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
