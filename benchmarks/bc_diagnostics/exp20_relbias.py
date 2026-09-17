"""Exp 20: relative-position attention bias, matched, on more data.

Exp 17's absolute KL levels were meaningless -- 12k scenes cannot train a
transformer from scratch against a checkpoint that saw 167.6M environment steps,
and the give-away was KL *rising* with depth. The one contrast that survived was
within-experiment and matched: a relative-position attention bias beat plain
attention at equal depth, equal data and equal parameter count.

This re-runs only that contrast on ~4x the data. The bias is one scalar per
(query, key, head) built from the toroidal displacement: no per-pair token, no
per-pair value, encoder cost still O(N), no weight whose shape depends on N. It
is the transfer-safe form of the thing Exp 10 tested in a transfer-unsafe way.

The absolute numbers here are still far above the checkpoint's and still
data-limited; only the matched difference is being claimed.
"""

import json
import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_collect import OBS_KEYS, TSTRIDE  # noqa: E402
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402


def collect(tag, seed, n_roll, burn=3):
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
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        T = m[::TSTRIDE].shape[0]
        chunks.append(dict(
            obs={str(k): buf.obs[k][::TSTRIDE][:T].float().cpu() for k in OBS_KEYS},
            expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
            keep=m[::TSTRIDE].cpu(),
        ))
        print(f"  {tag} {r}: {time.time()-t0:.0f}s", flush=True)
    return dict(
        obs={k: torch.cat([c["obs"][k] for c in chunks], 1) for k in chunks[0]["obs"]},
        expert=torch.cat([c["expert"] for c in chunks], 1),
        keep=torch.cat([c["keep"] for c in chunks], 1),
        world_size=launch.resolved.ship_config.world_size,
    )


def main():
    n_roll = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for tag, seed in (("train", 0), ("heldout", 1)):
        p = f"{S}/rb_{tag}.pt"
        if not _os.path.exists(p):
            torch.save(collect(tag, seed, n_roll), p)

    import exp17_depth as E

    def load(tag):
        d = torch.load(f"{S}/rb_{tag}.pt")
        torch.save(d, f"{S}/probe2_tmp_{tag}.pt")  # reuse Exp 17's feature builder
        return d

    for tag in ("train", "heldout"):
        load(tag)
    _orig = E.build

    def build2(tag):
        return _orig(f"tmp_{tag}")

    E.build = build2
    tr, te = E.build("train"), E.build("heldout")
    print(f"scenes: {len(tr['tok'])} train / {len(te['tok'])} held-out", flush=True)
    rows = []
    for depth, rel in [(2, False), (2, True), (4, False), (4, True)]:
        kl, npar = E.train_eval(tr, te, depth, rel, epochs=20)
        rows.append(dict(depth=depth, relbias=rel, params=npar, heldout_turn_kl=round(kl, 4)))
        print(json.dumps(rows[-1]), flush=True)
    for tag in ("train", "heldout"):
        _os.remove(f"{S}/probe2_tmp_{tag}.pt")
    json.dump(rows, open(f"{S}/exp20_rows.json", "w"), indent=1)


main()
