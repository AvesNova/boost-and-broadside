"""Shared harness: build a BC trainer at the icy-energy-741 checkpoint and collect rollouts."""

from __future__ import annotations

import argparse
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.launch import resolve_training_launch
from boost_and_broadside.train.rl.ppo import PPOTrainer

CKPT = "checkpoints/icy-energy-741/step_000167608320.pt"


class RecordingScripted(StochasticScriptedAgent):
    """Teacher wrapper that stashes per-step state diagnostics."""

    def __init__(self, ship_config, agent_config):
        super().__init__(ship_config, agent_config)
        self.records: list[dict] = []
        self.recording = False

    def get_actions_and_probs(self, state, team_visibility=None):
        out = super().get_actions_and_probs(state, team_visibility)
        if self.recording:
            self.records.append(teacher_internals(self, state, team_visibility))
        return out


def teacher_internals(agent, state, team_visibility):
    """Recompute the teacher's turn-relevant intermediates for one env step."""
    from boost_and_broadside.agents.frontline_strategy import frontline_strategy
    from boost_and_broadside.agents.scripted_utils import (
        compute_team_target_bearings,
        predict_interception,
        select_targets,
    )

    cfg = agent.config
    if team_visibility is None:
        team_visibility = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
    closest_dist, target_idx, has_target, _ = select_targets(
        state, agent.ship_config, team_visibility
    )
    dir_pred = predict_interception(state, agent.ship_config, target_idx, closest_dist)
    dir_pred = torch.where(has_target, dir_pred, torch.zeros_like(dir_pred))
    team_bearing, _, team_idx, team_has = compute_team_target_bearings(
        state, agent.ship_config, team_visibility
    )
    p_team = (
        agent._linear_ramp(
            closest_dist,
            cfg.team_target_distance_ramp[0],
            cfg.team_target_distance_ramp[1],
            *cfg.team_target_distance_prob,
        )
        * team_has.float()
    )
    dir_turn_combat = (1.0 - p_team) * dir_pred + p_team * team_bearing
    dir_turn_combat = dir_turn_combat / (torch.abs(dir_turn_combat) + 1e-8)
    att = state.ship_attitude
    rel_combat = torch.angle(dir_turn_combat * torch.conj(att))

    rec = {
        "closest_dist": closest_dist,
        "target_idx": target_idx.int(),
        "has_target": has_target,
        "p_team": p_team,
        "team_target_idx": team_idx.int(),
        "team_has_target": team_has,
        "rel_combat": rel_combat,
        "rel_personal": torch.angle(dir_pred * torch.conj(att)),
        "rel_team": torch.angle(team_bearing * torch.conj(att)),
        "speed": state.ship_vel.abs(),
        "step_count": state.step_count.float()[:, None].expand_as(closest_dist).contiguous(),
        "n_alive": state.ship_alive.sum(-1, keepdim=True).float().expand_as(closest_dist).contiguous(),
    }
    if state.num_zones > 0:
        strat = frontline_strategy(state, agent.ship_config, cfg, team_visibility)
        rel_front = torch.angle(strat.bearing * torch.conj(att))
        r0 = cfg.shoot_distance_ramp[0]
        alpha = ((closest_dist - r0) / (cfg.frontline_combat_radius - r0)).clamp(0, 1)
        rec["alpha"] = alpha
        rec["rel_front"] = rel_front
        rec["front_dist"] = strat.distance
        # effective turn bearing used by the blended head is not a single angle
        # (the blend is over probabilities, not directions); keep both.
    return {k: v.detach().to("cpu", torch.float32).clone() for k, v in rec.items()}


def build(num_envs: int, microbatch_tokens: int = 16384, seed: int = 0, recording: bool = False):
    torch.manual_seed(seed)
    launch = resolve_training_launch(
        profile="bc",
        vram="off",
        device="cuda",
        seed=seed,
        compile_mode=None,
        wandb=False,
        num_envs=num_envs,
        microbatch_tokens=microbatch_tokens,
        allow_probe=False,
    )
    resolved = launch.resolved
    agent_cls = RecordingScripted if recording else StochasticScriptedAgent
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cuda",
        use_wandb=False,
        scripted_agent=agent_cls(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=None,
        resolved_config_document=None,
    )
    trainer.load_checkpoint(CKPT)
    return trainer, launch


def collect(trainer, n_rollouts: int = 1, record: bool = False):
    """Collect rollouts from a fresh runtime; returns list of (buffer-snapshot, records)."""
    runtime = trainer._initialize_rollout_runtime()
    runtime.elo_eval.step = lambda *a, **k: None  # diagnostics only: no ladder games
    runtime.elo_eval.flush = lambda *a, **k: _NullSnap()
    out = []
    for i in range(n_rollouts):
        if record:
            trainer.scripted_agent.records = []
            trainer.scripted_agent.recording = True
        trainer._collect_rollout(runtime, False)
        recs = trainer.scripted_agent.records if record else None
        if record:
            trainer.scripted_agent.recording = False
        out.append(recs)
        yield i, recs
    return out


class _NullSnap:
    live_elo = torch.tensor(0.0)
    avg_elo = torch.tensor(0.0)
    floating_games = 0
    match_counts = {}
    floating_label = None
    floating_elo = None
    ladder_counts = {}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()
