"""Exp 25: can this architecture fit the LEGACY combat teacher but not the frontline one?

The reported history is that behaviour cloning used to close against the older,
simpler scripted agent and stopped closing when the frontline strategy landed
(commit 2b7ba18). `stochastic_scripted.get_actions_and_probs` still contains
that older controller verbatim -- "a zero-length zone axis is the exact legacy
combat contract" -- so both teachers can be evaluated on *identical* states.

This records, per step, the turn distribution of

  frontline : what the checkpoint was actually trained on (num_zones > 0)
  legacy    : `_combat_probs`, the pre-frontline dogfighter, on the same state

then fits the same turn head on the same frozen latent to each, and compares
held-out KL. Same states, same representation, same head, same budget: the only
difference is which teacher is being imitated.

If legacy fits far better, the residual is specific to what frontline_strategy
added, and every session-3 conclusion about "the force vector" is a statement
about *that* addition rather than about BC in general.
"""

import json
import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402
from exp21_frozen_depth import latents_all_tokens  # noqa: E402
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.agents.scripted_utils import select_targets  # noqa: E402
from boost_and_broadside.constants import TURN_SLICE  # noqa: E402

TSTRIDE = 4
NSHIPS = 8


def collect(tag, seed, n_roll=3, burn=3):
    trainer, _ = build(num_envs=128, microbatch_tokens=12288, seed=seed)
    agent = trainer.scripted_agent
    legacy: list[torch.Tensor] = []
    recording = {"on": False}
    orig = agent.get_actions_and_probs

    def patched(state, team_visibility=None):
        out = orig(state, team_visibility)
        if recording["on"]:
            vis = team_visibility
            if vis is None:
                vis = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
            closest, tidx, has, _ = select_targets(state, agent.ship_config, vis)
            # The legacy dogfighter, unchanged, on this exact state.
            _, p_turn, _ = agent._combat_probs(state, closest, tidx, has, vis)
            legacy.append(p_turn.detach().to("cpu", torch.float32).clone())
        return out

    agent.get_actions_and_probs = patched
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    for _ in range(burn):
        trainer._collect_rollout(runtime, False)

    chunks = []
    for r in range(n_roll):
        legacy.clear()
        recording["on"] = True
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        recording["on"] = False
        buf = trainer.buffer
        lat, tl = latents_all_tokens(trainer, buf)
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        chunks.append(dict(
            lat=lat[:, :, :NSHIPS], turn_logp=tl,
            frontline=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
            legacy=torch.stack(legacy)[::TSTRIDE],
            keep=m[::TSTRIDE].cpu(),
        ))
        print(f"  {tag} {r}: {time.time()-t0:.0f}s", flush=True)
    return {k: torch.cat([c[k] for c in chunks], 1) for k in chunks[0]}


def flat(d):
    k = d["keep"].reshape(-1)
    return dict(
        latent=d["lat"].reshape(-1, d["lat"].shape[-1])[k].float(),
        frontline=d["frontline"].reshape(-1, d["frontline"].shape[-1])[k],
        legacy=d["legacy"].reshape(-1, d["legacy"].shape[-1])[k],
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[k],
    )


def main():
    for tag, seed in (("train", 0), ("heldout", 1)):
        p = f"{S}/oldnew_{tag}.pt"
        if not _os.path.exists(p):
            torch.save(collect(tag, seed), p)
    tr, te = flat(torch.load(f"{S}/oldnew_train.pt")), flat(torch.load(f"{S}/oldnew_heldout.pt"))
    print(f"tokens: {len(tr['latent'])} train / {len(te['latent'])} held-out", flush=True)

    # How different are the two teachers on these states at all?
    a, b = te["frontline"].clamp_min(1e-8), te["legacy"].clamp_min(1e-8)
    print(json.dumps(dict(
        kl_legacy_vs_frontline=round(float((a * (a.log() - b.log())).sum(-1).mean()), 4),
        frontline_entropy=round(float(-(a * a.log()).sum(-1).mean()), 4),
        legacy_entropy=round(float(-(b * b.log()).sum(-1).mean()), 4),
    )), flush=True)

    rows = [dict(target="frontline (deployed policy head)",
                 turn_kl=round(float(kl(te["frontline"], te["logq"])), 4)),
            dict(target="legacy (deployed policy head)",
                 turn_kl=round(float(kl(te["legacy"], te["logq"])), 4))]
    for r in rows:
        print(json.dumps(r), flush=True)
    for name in ("frontline", "legacy"):
        lq = fit_head(tr["latent"], tr[name], te["latent"])
        rows.append(dict(target=f"{name} (head refit on frozen latent)",
                         turn_kl=round(float(kl(te[name], lq)), 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp25_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
