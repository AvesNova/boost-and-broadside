"""Exp 16: in pure zone navigation, which teacher quantity is the trunk missing?

With zero visible enemies the teacher reduces to three terms instead of four --
`combat_force` vanishes identically -- which makes it the cleanest stratum for
isolating the zone machinery. (It is *not* a harder stratum: its apparently
elevated KL is the `alpha = 1` effect, since no visible enemies implies
`alpha = 1`. See Exp 15.) There,

    force = (1-recovery)*objective_force + recovery*spawn_dir + separation
    objective_force = sum_z preference_z * unit(zone_z - me)
    preference  = normalise( need_z / (1 + (d_z/R)^2) )
    need_z      = softplus(2*(margin + enemy_zone_z - allies_without_self_z))

`enemy_zone` is zero with nothing visible, so `need` depends only on a
**team-wide sum** of allied contributions per zone -- `bmm(team_members,
contribution)` -- which is exactly the unbounded reduction softmax attention
replaces with a normalised average.

This appends each candidate teacher quantity to the frozen latent in turn and
measures held-out turn KL inside that stratum. Whichever one collapses the KL
is the quantity the trunk is failing to build.
"""

import json
import os as _os
import sys
import time

import torch

S = _os.environ.get("BC_DIAG_DIR", _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, S)
from exp8_oracle import fit_head, kl  # noqa: E402
from harness import _NullSnap, build  # noqa: E402

from boost_and_broadside.constants import TURN_SLICE  # noqa: E402
from exp7_collect import TSTRIDE, run  # noqa: E402


def zone_internals(state, ship, cfg, visibility):
    """Per-zone teacher intermediates, plus the ego-frame zone directions."""
    import math

    import torch.nn.functional as F

    from boost_and_broadside.config import ZoneRole
    from boost_and_broadside.env.frontline import toroidal_displacement

    health = (state.ship_health / ship.max_health).clamp(0, 1)
    health = torch.where(state.ship_alive, health, 0.0)
    zone_delta = toroidal_displacement(
        state.zone_pos[:, None, :] - state.ship_pos[:, :, None], ship.world_size
    )
    zd = zone_delta.abs()
    R = 2 * state.zone_radius[:, None, :] if cfg.frontline_zone_radius is None else cfg.frontline_zone_radius
    contribution = health[:, :, None] * torch.exp(-(zd / R).square())
    team_members = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
    ta = torch.bmm(team_members.float(), contribution)
    te = torch.bmm((~team_members & visibility).float(), contribution)
    obs_team = state.ship_team_id.long()[:, :, None].expand_as(contribution)
    allied_zone, enemy_zone = ta.gather(1, obs_team), te.gather(1, obs_team)
    without_self = (allied_zone - contribution).clamp_min(0)
    roles = state.zone_roles[:, None, :]
    team0 = state.ship_team_id[:, :, None] == 0
    own_def = torch.where(team0, roles == int(ZoneRole.TEAM0_DEFENSE), roles == int(ZoneRole.TEAM1_DEFENSE))
    off = torch.where(team0, roles == int(ZoneRole.TEAM1_DEFENSE), roles == int(ZoneRole.TEAM0_DEFENSE))
    margin = cfg.frontline_zone_margin * torch.where(
        off, math.exp(cfg.frontline_aggression), math.exp(-cfg.frontline_aggression)
    )
    need = F.softplus(2 * (margin + enemy_zone - without_self)) / 2
    need = torch.where(own_def | off, need, 0)
    utility = need / (1 + (zd / R).square())
    pref = utility / utility.sum(-1, keepdim=True).clamp_min(1e-8)
    zu = zone_delta / zd.clamp_min(1e-8)
    att = state.ship_attitude
    zang = torch.angle(zu * torch.conj(att[:, :, None]))  # ego-frame direction per zone
    obj = (pref * zu).sum(-1)
    return {
        "z_need": need,
        "z_pref": pref,
        "z_without_self": without_self,
        "z_dist": zd / 1000.0,
        "z_ang_sin": zang.sin(),
        "z_ang_cos": zang.cos(),
        "obj_sin": torch.angle(obj * torch.conj(att)).sin()[..., None],
        "obj_cos": torch.angle(obj * torch.conj(att)).cos()[..., None],
        "obj_mag": obj.abs()[..., None],
    }


def collect(tag, seed, n_roll=3, burn=3):
    trainer, launch = build(num_envs=128, microbatch_tokens=12288, seed=seed, recording=True)
    ship_cfg, agent_cfg = launch.resolved.ship_config, trainer.scripted_agent.config
    extra: list[dict] = []
    orig = trainer.scripted_agent.get_actions_and_probs

    def patched(state, team_visibility=None):
        out = orig(state, team_visibility)
        if trainer.scripted_agent.recording:
            vis = team_visibility
            if vis is None:
                vis = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
            allied = state.ship_team_id[:, :, None] == state.ship_team_id[:, None, :]
            seen = vis.gather(1, state.ship_team_id.long()[:, :, None].expand_as(allied))
            n_vis = (~allied & seen & state.ship_alive[:, None, :]).float().sum(-1)
            rec = zone_internals(state, ship_cfg, agent_cfg, vis)
            rec["n_vis_enemy"] = n_vis[..., None]
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
        extra.clear()
        trainer.scripted_agent.records = []
        trainer.scripted_agent.recording = True
        t0 = time.time()
        trainer._collect_rollout(runtime, False)
        trainer.scripted_agent.recording = False
        buf = trainer.buffer
        acts, tlp = run(trainer, buf)
        m = (buf.expert_probs.float().sum(-1) > 0) & buf.actor_masks & buf.alive_mask
        chunks.append(dict(
            latent=acts["b1.temporal0"], turn_logp=tlp,
            expert=buf.expert_probs[::TSTRIDE].float()[..., TURN_SLICE].cpu(),
            keep=m[::TSTRIDE].cpu(),
            internals={k: torch.stack([x[k] for x in extra])[::TSTRIDE] for k in extra[0]},
        ))
        print(f"  {tag} rollout {r}: {time.time()-t0:.0f}s", flush=True)
    return dict(
        latent=torch.cat([c["latent"] for c in chunks], 1),
        turn_logp=torch.cat([c["turn_logp"] for c in chunks], 1),
        expert=torch.cat([c["expert"] for c in chunks], 1),
        keep=torch.cat([c["keep"] for c in chunks], 1),
        internals={k: torch.cat([c["internals"][k] for c in chunks], 1) for k in chunks[0]["internals"]},
    )


def flat(d):
    k = d["keep"].reshape(-1)
    return dict(
        latent=d["latent"].reshape(-1, d["latent"].shape[-1])[k].float(),
        expert=d["expert"].reshape(-1, d["expert"].shape[-1])[k].float(),
        logq=d["turn_logp"].reshape(-1, d["turn_logp"].shape[-1])[k].float(),
        internals={n: v.reshape(-1, v.shape[-1])[k] for n, v in d["internals"].items()},
    )


def main():
    for tag, seed in (("train", 0), ("heldout", 1)):
        p = f"{S}/zone_{tag}.pt"
        if not _os.path.exists(p):
            torch.save(collect(tag, seed), p)
    tr, te = flat(torch.load(f"{S}/zone_train.pt")), flat(torch.load(f"{S}/zone_heldout.pt"))

    # The stratum Exp 15 flagged: nothing visible, so the teacher is pure zone
    # navigation and every input it uses is exactly present in the observation.
    str_tr = tr["internals"]["n_vis_enemy"].squeeze(-1) == 0
    str_te = te["internals"]["n_vis_enemy"].squeeze(-1) == 0
    print(f"stratum: {int(str_tr.sum())} train / {int(str_te.sum())} held-out tokens")

    def cell(keys):
        a = [tr["latent"][str_tr]] + [tr["internals"][k][str_tr] for k in keys]
        b = [te["latent"][str_te]] + [te["internals"][k][str_te] for k in keys]
        return torch.cat(a, 1), torch.cat(b, 1)

    rows = [dict(extra="none (frozen policy head)",
                 turn_kl=round(float(kl(te["expert"][str_te], te["logq"][str_te])), 4))]
    print(json.dumps(rows[-1]), flush=True)
    cells = {
        "none (refit head on latent)": [],
        "+ zone directions (ego frame)": ["z_ang_sin", "z_ang_cos", "z_dist"],
        "+ zone need (team-wide sum)": ["z_need"],
        "+ allied pressure without self": ["z_without_self"],
        "+ zone preference (normalised)": ["z_pref"],
        "+ need AND directions": ["z_need", "z_ang_sin", "z_ang_cos", "z_dist"],
        "+ objective_force direction": ["obj_sin", "obj_cos"],
        "+ objective direction AND magnitude": ["obj_sin", "obj_cos", "obj_mag"],
    }
    for name, keys in cells.items():
        a, b = cell(keys)
        lq = fit_head(a, tr["expert"][str_tr], b)
        rows.append(dict(extra=name, turn_kl=round(float(kl(te["expert"][str_te], lq)), 4)))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"{S}/exp16_rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
