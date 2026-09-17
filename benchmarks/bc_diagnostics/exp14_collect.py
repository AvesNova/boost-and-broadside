"""Exp 14a: record the teacher's directional vs magnitude intermediates.

`frontline_strategy` builds the bearing as the angle of

    force = (1-recovery)*(objective_force + combat_force) + recovery*spawn_dir + separation

Every *direction* in that sum is a normalised weighted average -- softmax
attention's natural output. Every term setting their relative *magnitude*
(enemy_strength, allied_strength, the zone `need`) is an unnormalised sum, which
softmax discards by construction (see docs/architecture.md on bullet reads:
"conveys which ... but not how many").

This records both families so a probe can ask which one the trunk actually
represents. The internals are recomputed here rather than returned by
`frontline_strategy`, so `combat`, `need`, `separation` and `recovery` are
cross-checked against the values it does return -- a mismatch means this mirror
has drifted from the real thing.
"""

import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp7_collect import OBS_KEYS, TSTRIDE, run  # noqa: E402
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402


def frontline_internals(state, ship, cfg, visibility):
    """Mirror of frontline_strategy, keeping the intermediates it does not return."""
    import math

    import torch.nn.functional as F

    from boost_and_broadside.config import ZoneRole
    from boost_and_broadside.env.frontline import toroidal_displacement

    health = (state.ship_health / ship.max_health).clamp(0, 1)
    health = torch.where(state.ship_alive, health, 0.0)
    allied = state.ship_team_id[:, :, None] == state.ship_team_id[:, None, :]
    visible = visibility.gather(1, state.ship_team_id.long()[:, :, None].expand_as(allied))
    enemies = ~allied & visible & state.ship_alive[:, None, :]
    allies = allied & state.ship_alive[:, None, :]
    delta = toroidal_displacement(
        state.ship_pos[:, None, :] - state.ship_pos[:, :, None], ship.world_size
    )
    distance = delta.abs()
    unit = delta / distance.clamp_min(1e-8)
    kernel = torch.exp(-(distance / cfg.frontline_combat_radius).square())
    allied_strength = torch.where(allies, kernel * health[:, None, :], 0).sum(-1)
    enemy_weight = torch.where(enemies, kernel * health[:, None, :], 0)
    enemy_strength = enemy_weight.sum(-1)
    combat = torch.tanh(
        torch.log((allied_strength + 1e-6) / (enemy_strength + 1e-6)) + cfg.frontline_aggression
    )
    enemy_direction = (enemy_weight * unit).sum(-1) / enemy_strength.clamp_min(1e-8)
    combat_force = combat * enemy_direction * (1 - torch.exp(-enemy_strength))

    zone_delta = toroidal_displacement(
        state.zone_pos[:, None, :] - state.ship_pos[:, :, None], ship.world_size
    )
    zone_distance = zone_delta.abs()
    support_radius = (
        2 * state.zone_radius[:, None, :]
        if cfg.frontline_zone_radius is None
        else cfg.frontline_zone_radius
    )
    contribution = health[:, :, None] * torch.exp(-(zone_distance / support_radius).square())
    team_members = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
    team_allied_zone = torch.bmm(team_members.float(), contribution)
    team_enemy_zone = torch.bmm((~team_members & visibility).float(), contribution)
    observer_team = state.ship_team_id.long()[:, :, None].expand_as(contribution)
    allied_zone = team_allied_zone.gather(1, observer_team)
    enemy_zone = team_enemy_zone.gather(1, observer_team)
    without_self = (allied_zone - contribution).clamp_min(0)
    roles = state.zone_roles[:, None, :]
    team0 = state.ship_team_id[:, :, None] == 0
    own_defense = torch.where(
        team0, roles == int(ZoneRole.TEAM0_DEFENSE), roles == int(ZoneRole.TEAM1_DEFENSE)
    )
    offense = torch.where(
        team0, roles == int(ZoneRole.TEAM1_DEFENSE), roles == int(ZoneRole.TEAM0_DEFENSE)
    )
    margin = cfg.frontline_zone_margin * torch.where(
        offense, math.exp(cfg.frontline_aggression), math.exp(-cfg.frontline_aggression)
    )
    need = F.softplus(2 * (margin + enemy_zone - without_self)) / 2
    need = torch.where(own_defense | offense, need, 0)
    utility = need / (1 + (zone_distance / support_radius).square())
    preference = utility / utility.sum(-1, keepdim=True).clamp_min(1e-8)
    zone_unit = zone_delta / zone_distance.clamp_min(1e-8)
    objective_force = (preference * zone_unit).sum(-1)

    sep_radius = cfg.frontline_separation_radius or 4 * ship.collision_radius
    sep_weight = torch.where(
        allies & (distance > 0), torch.exp(-(distance / sep_radius).square()), 0
    )
    separation = -(sep_weight * delta / sep_radius).sum(-1)
    separation = separation / sep_weight.sum(-1).clamp_min(1)

    own_spawn = torch.where(
        team0, roles == int(ZoneRole.TEAM0_SPAWN), roles == int(ZoneRole.TEAM1_SPAWN)
    )
    spawn_delta = torch.where(own_spawn, zone_delta, 0).sum(-1)
    recovery = (1 - health).square() / (
        (1 - health).square() + (health / cfg.frontline_recovery_health).square() + 1e-8
    )
    att = state.ship_attitude
    ang = lambda z: torch.angle(z * torch.conj(att))  # noqa: E731

    return {
        # --- directions: normalised weighted averages, attention's natural output
        "dir_enemy": ang(enemy_direction),
        "dir_objective": ang(objective_force),
        "dir_separation": ang(separation),
        "dir_combat_force": ang(combat_force),
        # --- magnitudes: unnormalised sums, which softmax discards
        "mag_enemy_strength": enemy_strength,
        "mag_allied_strength": allied_strength,
        "mag_n_visible_enemies": enemies.float().sum(-1),
        "mag_n_allies": allies.float().sum(-1),
        "mag_combat": combat,
        "mag_need_total": need.sum(-1),
        "mag_objective": objective_force.abs(),
        "mag_combat_force": combat_force.abs(),
        "mag_separation": separation.abs(),
        "mag_recovery": recovery,
        "mag_spawn_dist": spawn_delta.abs(),
    }


def main():
    tag, n_roll, burn, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    trainer, launch = build(num_envs=128, microbatch_tokens=12288, seed=seed, recording=True)
    ship_cfg = launch.resolved.ship_config
    agent_cfg = trainer.scripted_agent.config

    from boost_and_broadside.agents.frontline_strategy import frontline_strategy

    extra: list[dict] = []
    orig = trainer.scripted_agent.get_actions_and_probs

    def patched(state, team_visibility=None):
        out = orig(state, team_visibility)
        if trainer.scripted_agent.recording:
            vis = team_visibility
            if vis is None:
                vis = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
            rec = frontline_internals(state, ship_cfg, agent_cfg, vis)
            ref = frontline_strategy(state, ship_cfg, agent_cfg, vis)
            # Cross-check the mirror against what the real function returns.
            drift = max(
                float((rec["mag_combat"] - ref.combat_score).abs().max()),
                float((rec["mag_recovery"] - ref.recovery).abs().max()),
                float((rec["mag_separation"] - ref.separation.abs()).abs().max()),
            )
            rec["_drift"] = torch.full_like(rec["mag_combat"], drift)
            extra.append({k: v.detach().to("cpu", torch.float32).clone() for k, v in rec.items()})
        return out

    trainer.scripted_agent.get_actions_and_probs = patched

    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    for _ in range(burn):
        trainer.scripted_agent.recording = False
        trainer._collect_rollout(runtime, False)

    chunks = []
    for r in range(n_roll):
        trainer.scripted_agent.records = []
        extra.clear()
        trainer.scripted_agent.recording = True
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        trainer.scripted_agent.recording = False
        buf = trainer.buffer
        acts, tlp = run(trainer, buf)
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        T = m[::TSTRIDE].shape[0]
        internals = {k: torch.stack([x[k] for x in extra])[::TSTRIDE] for k in extra[0]}
        chunks.append(
            dict(
                acts=acts,
                turn_logp=tlp,
                expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
                keep=m[::TSTRIDE].cpu(),
                teacher={k: torch.stack([x[k] for x in trainer.scripted_agent.records])[::TSTRIDE]
                         for k in trainer.scripted_agent.records[0]},
                internals=internals,
                obs={str(k): buf.obs[k][::TSTRIDE][:T].float().cpu() for k in OBS_KEYS},
            )
        )
        print(f"rollout {r}: {time.time()-t0:.0f}s drift={float(internals['_drift'].max()):.2e}",
              flush=True)

    merged = dict(
        acts={k: torch.cat([c["acts"][k] for c in chunks], 1) for k in chunks[0]["acts"]},
        turn_logp=torch.cat([c["turn_logp"] for c in chunks], 1),
        expert=torch.cat([c["expert"] for c in chunks], 1),
        keep=torch.cat([c["keep"] for c in chunks], 1),
        teacher={k: torch.cat([c["teacher"][k] for c in chunks], 1) for k in chunks[0]["teacher"]},
        internals={k: torch.cat([c["internals"][k] for c in chunks], 1) for k in chunks[0]["internals"]},
        obs={k: torch.cat([c["obs"][k] for c in chunks], 1) for k in chunks[0]["obs"]},
        world_size=ship_cfg.world_size,
    )
    torch.save(merged, f"{S}/probe2_{tag}.pt")
    print("wrote", f"{S}/probe2_{tag}.pt")


main()
