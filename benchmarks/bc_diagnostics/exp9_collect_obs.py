"""Exp 9a: dump the policy's own observation, unencoded, for the representation audit.

Everything here is already in the policy's input -- POS, ATT, HEALTH, TEAM_ID,
BELIEF_VALID, OBJECT_TYPE, ZONE_ROLE, RADIUS. The point of storing it raw is to
ask whether the *information* suffices to compute the teacher's bearing, as
distinct from whether the *encoding* the trunk receives preserves it.
"""

import os as _os
import sys

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from harness import build, _NullSnap  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402
from boost_and_broadside.env.observation import ObsKey  # noqa: E402

TSTRIDE = 4
KEYS = [
    ObsKey.POS, ObsKey.ATT, ObsKey.VEL, ObsKey.HEALTH, ObsKey.TEAM_ID, ObsKey.ALIVE,
    ObsKey.BELIEF_VALID, ObsKey.OBJECT_TYPE, ObsKey.ZONE_ROLE, ObsKey.RADIUS,
    ObsKey.TIME_SINCE_OBSERVATION,
]


def main():
    tag, n_roll, burn, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    trainer, launch = build(num_envs=128, microbatch_tokens=12288, seed=seed, recording=True)
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    for _ in range(burn):
        trainer.scripted_agent.recording = False
        trainer._collect_rollout(runtime, False)

    chunks = []
    for _ in range(n_roll):
        trainer.scripted_agent.records = []
        trainer.scripted_agent.recording = True
        trainer._collect_rollout(runtime, False)
        trainer.scripted_agent.recording = False
        buf = trainer.buffer
        recs = trainer.scripted_agent.records
        obs = {str(k): buf.obs[k][::TSTRIDE].float().cpu() for k in KEYS}
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        chunks.append(dict(
            obs=obs,
            keep=m[::TSTRIDE].cpu(),
            expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
            teacher={k: torch.stack([x[k] for x in recs])[::TSTRIDE] for k in recs[0]},
        ))
        print("rollout done", flush=True)

    merged = dict(
        obs={k: torch.cat([c["obs"][k] for c in chunks], 1) for k in chunks[0]["obs"]},
        keep=torch.cat([c["keep"] for c in chunks], 1),
        expert=torch.cat([c["expert"] for c in chunks], 1),
        teacher={k: torch.cat([c["teacher"][k] for c in chunks], 1) for k in chunks[0]["teacher"]},
        world_size=launch.resolved.ship_config.world_size,
    )
    torch.save(merged, f"{S}/obs_{tag}.pt")
    print("wrote", f"{S}/obs_{tag}.pt", {k: tuple(v.shape) for k, v in merged["obs"].items()})


main()
