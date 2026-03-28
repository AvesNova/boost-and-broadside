"""Clean recurrent PPO trainer for the MVP policy.

Zero Mamba, zero auxiliary losses. One clean loop:
    collect rollout → compute GAE → update epochs → log async → repeat.

Logging is async (CPU-side via wandb) to avoid GPU sync on the hot path.
"""

import dataclasses
import time
import threading
from pathlib import Path
from queue import Queue, Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import TrainConfig, ModelConfig, ShipConfig, EnvConfig, RewardConfig
from boost_and_broadside.constants import POWER_SLICE, TURN_SLICE, SHOOT_SLICE
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.train.rl.buffer import RolloutBuffer, ReturnScaler, symlog
from boost_and_broadside.train.rl.roster import EloRoster, RosterEntry


# ------------------------------------------------------------------
# Opponent-management helpers (module-level, no class coupling)
# ------------------------------------------------------------------

def _slice_obs(obs: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    """Return a view of obs tensors for envs [start, end)."""
    return {k: v[start:end] for k, v in obs.items()}


def _slice_state(state: TensorState, start: int, end: int) -> TensorState:
    """Return a new TensorState containing only envs [start, end)."""
    return TensorState(
        step_count      = state.step_count     [start:end],
        ship_pos        = state.ship_pos        [start:end],
        ship_vel        = state.ship_vel        [start:end],
        ship_attitude   = state.ship_attitude   [start:end],
        ship_ang_vel    = state.ship_ang_vel    [start:end],
        ship_health     = state.ship_health     [start:end],
        ship_power      = state.ship_power      [start:end],
        ship_cooldown   = state.ship_cooldown   [start:end],
        ship_team_id    = state.ship_team_id    [start:end],
        ship_alive      = state.ship_alive      [start:end],
        ship_is_shooting= state.ship_is_shooting[start:end],
        prev_action     = state.prev_action     [start:end],
        bullet_pos      = state.bullet_pos      [start:end],
        bullet_vel      = state.bullet_vel      [start:end],
        bullet_time     = state.bullet_time     [start:end],
        bullet_active   = state.bullet_active   [start:end],
        bullet_cursor   = state.bullet_cursor   [start:end],
    )


def _override_opponent(
    action: torch.Tensor,
    team_id: torch.Tensor,
    opp_team_flag: torch.Tensor,
    start: int,
    end: int,
    opp_action: torch.Tensor,
) -> None:
    """Replace team-opp_team_flag actions in envs [start, end) with opp_action in-place.

    Args:
        action:        (B, N, 3) combined action tensor — modified in-place.
        team_id:       (B, N) int — team assignment per ship.
        opp_team_flag: (end-start,) int — which team_id is the opponent per env.
        start, end:    slice of envs to update.
        opp_action:    (end-start, N, 3) — opponent agent's actions.
    """
    opp_mask = (team_id[start:end] == opp_team_flag.unsqueeze(1))  # (slice, N)
    action[start:end] = torch.where(opp_mask.unsqueeze(-1), opp_action, action[start:end])


class _DecayScheduler:
    """Anneals reward shaping weights to zero once training metrics cross thresholds.

    Each target watches one W&B metric key via an EMA.  When the EMA stays above
    the threshold for `sustain` consecutive updates the weight is halved every
    `half_life` updates until it reaches zero.

    Missing metric keys (e.g. no episodes finished this update) leave the EMA
    unchanged — the scheduler simply waits.
    """

    def __init__(
        self,
        targets: list[dict],
        ema_alpha: float = 0.05,
    ) -> None:
        self._targets   = targets   # list of mutable dicts (see _build_decay_targets)
        self._alpha     = ema_alpha
        self._ema: dict[str, float] = {}

    def step(self, metrics: dict[str, float]) -> dict[str, float]:
        """Update EMAs, advance counters, and apply decay. Returns decay log dict."""
        for key, val in metrics.items():
            prev = self._ema.get(key)
            self._ema[key] = val if prev is None else (1.0 - self._alpha) * prev + self._alpha * val

        logged: dict[str, float] = {}
        for t in self._targets:
            ema = self._ema.get(t["watch_key"], 0.0)

            if ema >= t["threshold"]:
                t["count"] = min(t["count"] + 1, t["sustain"])
            else:
                t["count"] = max(t["count"] - 1, 0)

            if t["count"] >= t["sustain"]:
                factor = 0.5 ** (1.0 / t["half_life"])
                cur    = getattr(t["component"], t["weight_attr"])
                if cur > 0.0:
                    new_val = cur * factor
                    setattr(t["component"], t["weight_attr"], new_val)
                    logged[f"decay/{t['weight_attr']}"] = new_val

        return logged


class PPOTrainer:
    """Proximal Policy Optimization for the MVP multi-agent policy.

    Args:
        train_config:  PPO hyperparameters.
        model_config:  Policy architecture.
        ship_config:   Physics constants.
        env_config:    Environment sizing.
        reward_config: Reward weights.
        device:        Torch device.
        use_wandb:     Whether to log metrics to W&B.
    """

    def __init__(
        self,
        train_config:   TrainConfig,
        model_config:   ModelConfig,
        ship_config:    ShipConfig,
        env_config:     EnvConfig,
        reward_config:  RewardConfig,
        device:         str | torch.device,
        use_wandb:      bool = False,
        scripted_agent: StochasticScriptedAgent | None = None,
    ) -> None:
        self.cfg           = train_config
        self.model_config  = model_config
        self.ship_config   = ship_config
        self.env_config    = env_config
        self.reward_config = reward_config
        self.device        = torch.device(device)
        self.use_wandb     = use_wandb
        self.scripted_agent = scripted_agent

        # Env group sizes — contiguous slices of the B envs:
        #   [0, B_self)                          → pure self-play
        #   [B_self, B_self+B_sc)               → scripted opponent (+ BC targets)
        #   [B_self+B_sc, B_self+B_sc+B_avg)   → avg-model opponent
        #   [B_self+B_sc+B_avg, B)              → league roster opponent
        B = train_config.num_envs
        self.B_sc     = round(train_config.scripted_frac  * B)
        self.B_avg    = round(train_config.avg_model_frac * B)
        self.B_league = round(train_config.league_frac    * B)
        self.B_self   = B - self.B_sc - self.B_avg - self.B_league

        if self.B_sc > 0 and scripted_agent is None:
            raise ValueError(
                "scripted_frac > 0 requires a scripted_agent to be provided."
            )
        if self.B_league > 0 and scripted_agent is None and train_config.avg_model_frac == 0.0:
            # League can still work with only checkpoint entries (none yet at startup).
            pass

        self.wrapper = MVPEnvWrapper(
            num_envs       = train_config.num_envs,
            ship_config    = ship_config,
            env_config     = env_config,
            reward_config  = reward_config,
            device         = device,
            scripted_agent = scripted_agent,
        )
        K = self.wrapper.num_active_components
        self._active_names = self.wrapper.active_names  # stable ref used throughout

        self.policy = MVPPolicy(model_config, ship_config, num_value_components=K).to(self.device)
        self.optim  = optim.Adam(self.policy.parameters(), lr=train_config.learning_rate, eps=1e-5)

        # Linear LR warmup: ramp from 0 → learning_rate over lr_warmup_steps global steps.
        # Implemented as a per-update scheduler (steps once per rollout update).
        steps_per_update = train_config.num_envs * train_config.num_steps
        warmup_updates   = max(1, train_config.lr_warmup_steps // steps_per_update)
        if train_config.lr_warmup_steps > 0:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optim,
                start_factor = 1.0 / warmup_updates,
                end_factor   = 1.0,
                total_iters  = warmup_updates,
            )
        else:
            self.lr_scheduler = None

        N = env_config.num_ships

        # Build obs shapes for the buffer from a dummy reset
        sample_obs  = self.wrapper.reset()
        obs_shapes  = {
            k: v.shape[1:]   # strip batch dim → (N, ...) or (N,)
            for k, v in sample_obs.items()
            if v.dtype in (torch.float32, torch.int32, torch.int64, torch.bool)
        }

        self.buffer = RolloutBuffer(
            num_steps       = train_config.num_steps,
            num_envs        = train_config.num_envs,
            num_ships       = N,
            num_components  = K,
            obs_shapes      = obs_shapes,
            gamma           = train_config.gamma,
            gae_lambda      = train_config.gae_lambda,
            device          = self.device,
        )

        # Pre-compute lambda mask for active components only
        enemy_neg_set    = reward_config.enemy_neg_lambda_components
        self.enemy_neg_k = torch.tensor(
            [name in enemy_neg_set for name in self._active_names],
            dtype=torch.bool, device=self.device,
        )  # (K,)

        # Per-component return scaler: EMA of p5/p95 in symlog-reward space
        self.scaler = ReturnScaler(
            num_components = K,
            device         = self.device,
            ema_alpha      = train_config.return_ema_alpha,
            min_span       = train_config.return_min_span,
        )

        # --- Avg-model opponent (uniform mean of all post-warmup policy snapshots) ---
        # Weights initialized as a copy of the training policy.
        # Only updated after LR warmup completes; avg group envs use self-play until then.
        self.avg_policy = MVPPolicy(model_config, ship_config, num_value_components=K).to(self.device)
        self.avg_policy.load_state_dict(self.policy.state_dict())
        for p in self.avg_policy.parameters():
            p.requires_grad_(False)
        self._avg_param_cumsum: list[torch.Tensor] = [
            torch.zeros_like(p) for p in self.policy.parameters()
        ]
        self._avg_update_count: int = 0

        # Per-env flag: which team_id is the opponent in scripted/avg/league groups.
        # Randomised at init and re-randomised each episode reset.
        # Shape: (B_sc + B_avg + B_league,) — indexed relative to the non-self-play slice.
        #   [:B_sc]                → scripted group
        #   [B_sc : B_sc+B_avg]   → avg-model group
        #   [B_sc+B_avg :]         → league group
        n_opp_envs = self.B_sc + self.B_avg + self.B_league
        self._opp_team_flag = torch.randint(
            0, 2, (n_opp_envs,), device=self.device, dtype=torch.int32
        ) if n_opp_envs > 0 else torch.empty(0, device=self.device, dtype=torch.int32)

        # --- League play + ELO ---
        self.roster = EloRoster(
            max_size        = train_config.league_size,
            k_factor        = train_config.elo_k_factor,
            elo_temperature = train_config.elo_temperature,
        )
        # Random anchor is added by EloRoster.__init__ (ELO=0, fixed).
        # "avg" entry is added when _update_avg_model() is first called.
        # "scripted" entry is added lazily after scripted_roster_min_steps.

        # Training ELO starts at 0 (same as the random anchor) — reflecting that an
        # untrained policy behaves like random.  It will climb as eval progresses.
        self._training_elo:  float = 0.0
        self._elo_milestone: float = 0.0   # training ELO at the last milestone checkpoint
        self._last_checkpoint_path: Path | None = None

        # Current league opponent for the ongoing rollout (rotated each rollout).
        self._current_league_entry:  RosterEntry | None = None
        self._current_league_policy: MVPPolicy | None = None

        # Async logging queue
        self._log_queue: Queue = Queue()
        if use_wandb:
            self._init_wandb(train_config, model_config, ship_config, env_config, reward_config)
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        self._global_step = 0
        self._num_updates = train_config.total_timesteps // (train_config.num_envs * train_config.num_steps)

        # Run name used as checkpoint subdirectory (e.g. "checkpoints/good-spaceship-223/")
        if use_wandb:
            import wandb as _wandb
            self._run_name: str = _wandb.run.name
        else:
            from datetime import datetime
            self._run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._decay = self._build_decay_scheduler()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _update_avg_model(self) -> None:
        """Add the current training policy snapshot to the uniform running average."""
        first_update = (self._avg_update_count == 0)
        self._avg_update_count += 1
        for cum, p in zip(self._avg_param_cumsum, self.policy.parameters()):
            cum.add_(p.detach())
        for avg_p, cum in zip(self.avg_policy.parameters(), self._avg_param_cumsum):
            avg_p.data.copy_(cum / self._avg_update_count)
        # Register the avg model as a roster entry the first time it's ready,
        # seeded at current training ELO (it's a recent snapshot, so it's a
        # reasonable starting estimate that will quickly self-correct via eval).
        if first_update:
            self.roster.add_special("avg", self._global_step, 0, initial_elo=self._training_elo)

    def train(self) -> None:
        """Run the full PPO training loop."""
        B  = self.cfg.num_envs
        N  = self.wrapper.num_ships
        sc_start     = self.B_self
        sc_end       = self.B_self + self.B_sc
        avg_start    = sc_end
        avg_end      = avg_start + self.B_avg
        league_start = avg_end
        league_end   = B

        obs     = self.wrapper.reset()
        # Stagger initial step counts so envs don't all truncate simultaneously.
        # Uniformly distributed over [0, max_episode_steps) — after the first wave
        # of truncations they naturally desynchronize on their own.
        self.wrapper.env.state.step_count.random_(0, self.env_config.max_episode_steps)
        hidden  = self.policy.initial_hidden(B, N, self.device)

        # Avg-model hidden state — lives across the whole training run.
        avg_hidden: torch.Tensor | None = None
        if self.B_avg > 0:
            avg_hidden = self.avg_policy.initial_hidden(self.B_avg, N, self.device)

        # League hidden state — re-initialised each rollout when the entry changes.
        league_hidden: torch.Tensor | None = None

        start_time = time.time()

        for update in range(1, self._num_updates + 1):
            self.buffer.reset()
            self.buffer.store_initial_hidden(hidden)

            # Accumulate episode stats across the rollout — flushed once per update
            ep_rewards: list[torch.Tensor] = []
            ep_lengths: list[torch.Tensor] = []
            ep_components: dict[str, list[torch.Tensor]] = {}

            # Sample a league opponent for this rollout (rotated each update).
            # Evict the previous checkpoint's weights before loading the new one.
            if self.B_league > 0:
                self.roster.evict_all_checkpoint_policies()
                entry = self.roster.sample(self._training_elo)
                self._current_league_entry = entry
                if entry is None or (entry.kind == "avg" and self._avg_update_count == 0):
                    # No valid opponent yet — league group falls back to self-play this rollout.
                    self._current_league_entry  = None
                    self._current_league_policy = None
                elif entry.kind == "checkpoint":
                    self.roster.load_policy(
                        entry, self.model_config, self.ship_config,
                        self.wrapper.num_active_components, self.device,
                    )
                    self._current_league_policy = entry._policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, N, self.device
                    )
                elif entry.kind == "avg":
                    self._current_league_policy = self.avg_policy
                    league_hidden = self._current_league_policy.initial_hidden(
                        self.B_league, N, self.device
                    )
                else:  # "scripted" — no policy forward pass needed
                    self._current_league_policy = None
                    league_hidden = None

            # ----------------------------------------------------------------
            # Rollout collection
            # ----------------------------------------------------------------
            for _ in range(self.cfg.num_steps):
                # Training policy: all envs, all ships
                action, logprob, value_norm, hidden = self.policy.get_action_and_value(obs, hidden)

                # Whether the avg model is ready (at least one post-warmup snapshot)
                use_avg = self._avg_update_count > 0 and self.B_avg > 0

                # Avg-model opponent actions (when ready)
                if use_avg:
                    with torch.no_grad():
                        obs_avg = _slice_obs(obs, avg_start, avg_end)
                        action_avg, _, _, avg_hidden = self.avg_policy.get_action_and_value(
                            obs_avg, avg_hidden
                        )

                # League opponent actions (when a valid entry is loaded for this rollout)
                use_league = self.B_league > 0 and self._current_league_entry is not None
                if use_league:
                    if self._current_league_policy is not None:
                        with torch.no_grad():
                            obs_league = _slice_obs(obs, league_start, league_end)
                            action_league, _, _, league_hidden = \
                                self._current_league_policy.get_action_and_value(
                                    obs_league, league_hidden
                                )
                    else:  # scripted
                        state_league = _slice_state(
                            self.wrapper.env.state, league_start, league_end
                        )
                        with torch.no_grad():
                            action_league = self.scripted_agent.get_actions(state_league)

                # Team IDs — used for opponent overrides, actor mask, and BC expert masking
                team_id = obs["team_id"]  # (B, N) int

                # Scripted opponent actions + expert probs for BC loss
                expert_probs_step: torch.Tensor | None = None
                if self.B_sc > 0:
                    with torch.no_grad():
                        state_sc = _slice_state(self.wrapper.env.state, sc_start, sc_end)
                        action_scripted, expert_probs_sc = self.scripted_agent.get_actions_and_probs(state_sc)
                    # Only fill BC targets for training-policy ships (not the opponent team)
                    opp_flag_sc = self._opp_team_flag[:self.B_sc].unsqueeze(1)  # (B_sc, 1)
                    train_mask_sc = (team_id[sc_start:sc_end] != opp_flag_sc)   # (B_sc, N) bool
                    expert_probs_step = torch.zeros(
                        self.cfg.num_envs, self.env_config.num_ships, 12,
                        device=self.device,
                    )
                    expert_probs_step[sc_start:sc_end] = expert_probs_sc * train_mask_sc.unsqueeze(-1)

                # Override opponent team actions; training-policy actions are the base
                action  = action.clone()
                sc_flags  = self._opp_team_flag[:self.B_sc]
                avg_flags = self._opp_team_flag[self.B_sc : self.B_sc + self.B_avg]
                lg_flags  = self._opp_team_flag[self.B_sc + self.B_avg :]
                if self.B_sc > 0:
                    _override_opponent(action, team_id, sc_flags,  sc_start,     sc_end,     action_scripted)
                if use_avg:
                    _override_opponent(action, team_id, avg_flags, avg_start,    avg_end,    action_avg)
                if use_league:
                    _override_opponent(action, team_id, lg_flags,  league_start, league_end, action_league)

                # Actor mask: True for ships whose actions were chosen by training policy
                actor_mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                if self.B_sc > 0:
                    actor_mask[sc_start:sc_end] = (
                        team_id[sc_start:sc_end] != sc_flags.unsqueeze(1)
                    )
                if use_avg:
                    actor_mask[avg_start:avg_end] = (
                        team_id[avg_start:avg_end] != avg_flags.unsqueeze(1)
                    )
                if use_league:
                    actor_mask[league_start:league_end] = (
                        team_id[league_start:league_end] != lg_flags.unsqueeze(1)
                    )

                next_obs, reward, dones, truncated, info = self.wrapper.step(action)

                if info.get("ep_reward") is not None:
                    ep_rewards.append(info["ep_reward"])
                    ep_lengths.append(info["ep_length"].float())
                    for name, t in info["ep_reward_components"].items():
                        ep_components.setdefault(name, []).append(t)

                done_any = dones | truncated
                self.buffer.add(
                    obs          = obs,
                    action       = action,
                    logprob      = logprob,
                    reward       = reward,
                    done         = dones.float(),      # only true termination cuts GAE bootstrap
                    value        = self.scaler.denormalize(value_norm),  # symlog-reward space for GAE
                    alive        = obs["alive"].bool(),
                    actor_mask   = actor_mask,
                    expert_probs = expert_probs_step,
                )

                # Reset hidden states for terminated envs
                hidden = self.policy.reset_hidden_for_envs(hidden, done_any, N)
                if use_avg and self.B_avg > 0:
                    avg_hidden = self.avg_policy.reset_hidden_for_envs(
                        avg_hidden, done_any[avg_start:avg_end], N
                    )
                if use_league and self._current_league_policy is not None:
                    league_hidden = self._current_league_policy.reset_hidden_for_envs(
                        league_hidden, done_any[league_start:league_end], N
                    )

                # Re-randomise opponent team for envs that just ended an episode
                if self.B_sc + self.B_avg + self.B_league > 0:
                    done_non_self = done_any[self.B_self:]
                    new_flags = torch.randint(
                        0, 2, self._opp_team_flag.shape,
                        device=self.device, dtype=torch.int32,
                    )
                    self._opp_team_flag = torch.where(done_non_self, new_flags, self._opp_team_flag)

                obs = next_obs
                self._global_step += B

            # ----------------------------------------------------------------
            # GAE computation
            # ----------------------------------------------------------------
            with torch.no_grad():
                _, _, next_value_norm, _ = self.policy.get_action_and_value(obs, hidden)
            next_value = self.scaler.denormalize(next_value_norm)  # symlog-reward space
            self.buffer.compute_gae(next_value, dones.float())

            # Update per-component return percentiles from this rollout's returns
            self.scaler.update(self.buffer.returns)

            # ----------------------------------------------------------------
            # PPO update epochs
            # ----------------------------------------------------------------
            record_hist = (update % 10 == 0)
            metrics = self._update_epochs(record_histograms=record_hist)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update avg model after warmup completes
            warmup_done = (self.cfg.lr_warmup_steps == 0 or
                           self._global_step >= self.cfg.lr_warmup_steps)
            if self.B_avg > 0 and warmup_done:
                self._update_avg_model()

            # Scaler stats — one CPU transfer per component group
            p5_cpu   = self.scaler._p5.cpu()
            p95_cpu  = self.scaler._p95.cpu()
            span_cpu = p95_cpu - p5_cpu
            for i, name in enumerate(self._active_names):
                metrics[f"scaler/p5/{name}"]   = p5_cpu[i].item()
                metrics[f"scaler/p95/{name}"]  = p95_cpu[i].item()
                metrics[f"scaler/span/{name}"] = span_cpu[i].item()

            # Merge episode stats collected during rollout into the metrics dict
            if ep_rewards:
                all_rewards = torch.cat(ep_rewards)  # (num_finished_eps * N,)
                all_lengths = torch.cat(ep_lengths)
                metrics["ep/reward_mean"] = all_rewards.mean().item()
                metrics["ep/reward_min"]  = all_rewards.min().item()
                metrics["ep/reward_max"]  = all_rewards.max().item()
                metrics["ep/length_mean"] = all_lengths.mean().item()
                for name, tensors in ep_components.items():
                    metrics[f"ep/reward_{name}"] = torch.cat(tensors).mean().item()

            sps = int(self._global_step / (time.time() - start_time))
            metrics["train/lr"]          = self.optim.param_groups[0]["lr"]
            metrics["train/global_step"] = self._global_step
            metrics["train/sps"]         = sps

            # Decay shaping weights based on training progress
            metrics.update(self._decay.step(metrics))

            # ELO evaluation — runs sync matchups against all roster entries
            if self.cfg.elo_eval_interval > 0 and update % self.cfg.elo_eval_interval == 0:
                elo_metrics = self._run_elo_eval()
                metrics.update(elo_metrics)

            # Single log call per update — all metrics at the same step
            self._enqueue_log(metrics, step=self._global_step)

            if update % 10 == 0:
                elo_str = f"  elo={self._training_elo:.0f}" if self.cfg.elo_eval_interval > 0 else ""
                print(
                    f"update={update}/{self._num_updates}  "
                    f"step={self._global_step:,}  "
                    f"sps={sps:,}  "
                    f"loss={metrics.get('train/loss', 0.0):.4f}"
                    f"{elo_str}"
                )

            if self.cfg.checkpoint_interval > 0 and update % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(update)
                # Add to roster if training ELO has crossed the next milestone
                if (self._last_checkpoint_path is not None and
                        self._last_checkpoint_path.exists() and
                        self.cfg.elo_milestone_gap > 0 and
                        self._training_elo - self._elo_milestone >= self.cfg.elo_milestone_gap):
                    self.roster.add_checkpoint(
                        str(self._last_checkpoint_path), self._global_step, update,
                        initial_elo=self._training_elo,
                    )
                    self._elo_milestone = self._training_elo
                    self._save_roster_json()

        self._log_queue.put(None)  # signal logger thread to exit

    # ------------------------------------------------------------------
    # PPO update inner loop
    # ------------------------------------------------------------------

    def _update_epochs(self, record_histograms: bool = False) -> dict:
        """Run num_epochs × num_minibatches of PPO updates.

        Args:
            record_histograms: If True, capture return/logprob distributions from
                the last minibatch for async histogram logging.

        Returns:
            Dict of mean metric values over all minibatch updates.
            Per-component keys use the format ``critic/<metric>/<component_name>``.
            Histogram entries are raw numpy arrays; the log worker creates
            ``wandb.Histogram`` objects off the training thread.
        """
        cfg = self.cfg
        K   = self.buffer.num_components

        # Per-component lambda weights — active components only, may change via decay.
        comp_weights  = torch.tensor(
            [c.weight for c in self.wrapper._active_components],
            dtype=torch.float32, device=self.device,
        )  # (K,)

        accum_scalar: dict[str, list[float]] = {
            "train/loss":                      [],
            "train/policy_gradient_loss":      [],
            "train/value_loss":                [],
            "train/entropy_loss":              [],
            "train/behavioral_cloning_loss":   [],
            "train/approximate_kl":            [],
            "train/clip_fraction":             [],
            "train/gradient_norm":             [],
            "train/advantage_std":             [],
            "returns/aggregate":               [],  # λ-aggregated return, actor ships, symlog space
            "returns/aggregate_std":           [],
        }
        # Per-component (K,) CPU tensors accumulated across minibatches
        accum_k: dict[str, list[torch.Tensor]] = {
            "critic/value_loss":        [],
            "critic/explained_variance":[],
            "critic/return_mean":       [],
            "returns/component":        [],  # λ-weighted per-component return, actor ships
        }
        last_returns_np = None
        last_logprob_np = None

        for epoch_idx in range(cfg.num_epochs):
            for batch in self.buffer.get_minibatch_iterator(cfg.num_minibatches):
                (mb_obs, mb_actions, mb_old_logprobs, mb_advantages,
                 mb_returns, mb_values, mb_alive, mb_hidden, mb_actor_mask,
                 mb_expert_probs) = batch

                # Re-evaluate actions with current policy
                # new_value: (T, B_mb, N, K) in normalized space
                # policy_logits: (T, B_mb, N, 12) — needed for BC loss
                logprob, entropy, new_value, policy_logits = self.policy.evaluate_actions(
                    obs            = mb_obs,
                    actions        = mb_actions.long(),
                    initial_hidden = mb_hidden,
                    alive_mask     = mb_alive,
                )

                # Alive mask — used for critic loss (all alive ships, all groups)
                alive_f  = mb_alive.float()            # (T, B_mb, N)
                alive_k  = alive_f.unsqueeze(-1)       # (T, B_mb, N, 1) — broadcasts over K
                mask_sum = alive_f.sum().clamp(min=1.0)

                # Actor mask — subset of alive: only ships controlled by training policy
                # Used for actor loss, entropy, advantage normalization, and KL/clip diags.
                actor_f   = (mb_actor_mask & mb_alive).float()   # (T, B_mb, N)
                actor_sum = actor_f.sum().clamp(min=1.0)

                # ---- Lambda aggregation: per-ship aggregated advantage ----------
                # Build lambda matrix from team_id (teams are fixed within episode)
                team_id   = mb_obs["team_id"][0].long()                   # (B_mb, N)
                same_team = team_id.unsqueeze(2) == team_id.unsqueeze(1)  # (B_mb, N, N)

                # lambda_ij[b,i,j,k]: sign (+1 allied, -1 enemy outcome, 0 enemy shaping)
                #                    × component weight (from reward config, may decay)
                enemy_lam = torch.where(self.enemy_neg_k, -1.0, 0.0)     # (K,)
                lambda_ij = (same_team.float().unsqueeze(-1)
                             + (~same_team).float().unsqueeze(-1) * enemy_lam
                             ) * comp_weights                              # (B_mb, N, N, K)

                # Aggregate: A_i = Σ_j Σ_k lambda[b,i,j,k] * adv[t,b,j,k]
                adv_agg      = torch.einsum('bijk,tbjk->tbi',  lambda_ij, mb_advantages)  # (T, B_mb, N)
                ret_agg      = torch.einsum('bijk,tbjk->tbi',  lambda_ij, mb_returns)     # (T, B_mb, N)
                ret_per_comp = torch.einsum('bijk,tbjk->tbik', lambda_ij, mb_returns)     # (T, B_mb, N, K)

                # Normalise over training-controlled ships (actor_f)
                adv_mean = (adv_agg * actor_f).sum() / actor_sum
                adv_var  = ((adv_agg - adv_mean).pow(2) * actor_f).sum() / actor_sum
                adv_norm = (adv_agg - adv_mean) / (adv_var.sqrt() + 1e-8)

                # ---- Policy gradient loss (training ships only) ---------------
                log_ratio = logprob - mb_old_logprobs
                ratio     = log_ratio.exp()
                pg_loss1  = -adv_norm * ratio
                pg_loss2  = -adv_norm * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss   = (torch.max(pg_loss1, pg_loss2) * actor_f).sum() / actor_sum

                # ---- Value loss (MSE in normalized space, all alive ships) ----
                target_norm = self.scaler.normalize(mb_returns).detach()  # (T, B_mb, N, K)
                vf_loss_raw = (new_value - target_norm).pow(2)            # (T, B_mb, N, K)
                vf_loss     = (vf_loss_raw * alive_k).sum() / (mask_sum * K)

                # ---- Entropy bonus (training ships only) ---------------------
                ent_loss = -(entropy * actor_f).sum() / actor_sum

                # ---- Behavioral cloning loss (scripted-group training ships) -
                # bc_f: alive, training-policy, scripted-group ships
                # expert_probs > 0 exactly where scripted agent filled in targets.
                bc_valid = (mb_expert_probs.sum(-1) > 0)           # (T, B_mb, N)
                bc_f     = (bc_valid & mb_actor_mask & mb_alive).float()
                bc_sum   = bc_f.sum().clamp(min=1.0)
                ce = (
                    -(mb_expert_probs[..., POWER_SLICE] *
                      F.log_softmax(policy_logits[..., POWER_SLICE], dim=-1)).sum(-1)
                    -(mb_expert_probs[..., TURN_SLICE] *
                      F.log_softmax(policy_logits[..., TURN_SLICE],  dim=-1)).sum(-1)
                    -(mb_expert_probs[..., SHOOT_SLICE] *
                      F.log_softmax(policy_logits[..., SHOOT_SLICE], dim=-1)).sum(-1)
                )  # (T, B_mb, N)
                bc_loss = (ce * bc_f).sum() / bc_sum

                loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss + cfg.bc_coef * bc_loss

                self.optim.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (((ratio - 1) - log_ratio) * actor_f).sum().item() / actor_sum.item()
                    clip_frac = (((ratio - 1).abs() > cfg.clip_coef).float() * actor_f).sum().item() / actor_sum.item()

                    # λ-aggregated return stats (symlog-reward space, actor ships only)
                    ret_agg_mean = (ret_agg * actor_f).sum() / actor_sum
                    ret_agg_var  = ((ret_agg - ret_agg_mean).pow(2) * actor_f).sum() / actor_sum

                    # Per-component λ-weighted return (same scale as aggregate, actor ships)
                    actor_k             = actor_f.unsqueeze(-1)                                # (T, B_mb, N, 1)
                    ret_per_comp_mean_k = (ret_per_comp * actor_k).sum((0,1,2)) / actor_sum   # (K,)

                    # Per-component critic stats — single GPU→CPU transfer for all K dims
                    pred_k       = self.scaler.denormalize(new_value.detach())                     # (T, B_mb, N, K)
                    value_loss_k = (vf_loss_raw.detach() * alive_k).sum((0,1,2)) / mask_sum       # (K,)
                    ret_mean_k   = (mb_returns * alive_k).sum((0,1,2)) / mask_sum                 # (K,)
                    ret_var_k    = ((mb_returns - ret_mean_k) ** 2 * alive_k).sum((0,1,2)) / mask_sum
                    # Var(y - ŷ) not MSE: subtract residual mean to avoid penalising bias.
                    # MSE would give EV < 0 for a biased-constant predictor even though
                    # its predictive variance is the same as the naive mean predictor.
                    residuals_k = mb_returns - pred_k                                              # (T, B_mb, N, K)
                    res_mean_k  = (residuals_k * alive_k).sum((0,1,2)) / mask_sum                 # (K,)
                    res_var_k   = ((residuals_k - res_mean_k) ** 2 * alive_k).sum((0,1,2)) / mask_sum
                    ev_k        = 1.0 - res_var_k / (ret_var_k + 1e-8)                            # (K,)
                    # One transfer: stack → (4, K) → cpu
                    stats_k_cpu = torch.stack([value_loss_k, ev_k, ret_mean_k, ret_per_comp_mean_k]).cpu()

                accum_scalar["train/loss"]                    .append(loss.item())
                accum_scalar["train/policy_gradient_loss"]    .append(pg_loss.item())
                accum_scalar["train/value_loss"]              .append(vf_loss.item())
                accum_scalar["train/entropy_loss"]            .append(ent_loss.item())
                accum_scalar["train/behavioral_cloning_loss"] .append(bc_loss.item())
                accum_scalar["train/approximate_kl"]          .append(approx_kl)
                accum_scalar["train/clip_fraction"]           .append(clip_frac)
                accum_scalar["train/gradient_norm"]           .append(grad_norm.item())
                accum_scalar["train/advantage_std"]           .append(adv_var.sqrt().item())
                accum_scalar["returns/aggregate"]             .append(ret_agg_mean.item())
                accum_scalar["returns/aggregate_std"]         .append(ret_agg_var.sqrt().item())

                accum_k["critic/value_loss"]        .append(stats_k_cpu[0])
                accum_k["critic/return_mean"]       .append(stats_k_cpu[2])
                accum_k["returns/component"]        .append(stats_k_cpu[3])
                # Explained variance from last epoch only — reflects the value head after training,
                # not the average over all updates (which is dampened by early epochs).
                if epoch_idx == cfg.num_epochs - 1:
                    accum_k["critic/explained_variance"].append(stats_k_cpu[1])

                if record_histograms:
                    # Capture alive-ship returns/logprobs from last minibatch for histograms.
                    # .cpu().numpy() here is fine: only runs every hist_interval updates.
                    alive_flat      = mb_alive.reshape(-1).bool()
                    last_returns_np = mb_returns.reshape(-1, K)[alive_flat].cpu().numpy()
                    last_logprob_np = logprob.detach().reshape(-1)[alive_flat].cpu().numpy()

        # --- Build output metrics dict ---
        metrics: dict = {k: sum(v) / len(v) for k, v in accum_scalar.items() if v}

        # Expand per-component averages into named keys.
        # "returns/component" uses the section prefix directly so keys become "returns/{name}".
        for key, tensors in accum_k.items():
            avg = torch.stack(tensors).mean(0)   # (K,) CPU
            prefix = "returns" if key == "returns/component" else key
            for i, name in enumerate(self._active_names):
                metrics[f"{prefix}/{name}"] = avg[i].item()

        # Raw numpy arrays — worker thread converts to wandb.Histogram
        if last_returns_np is not None:
            metrics["hist/returns"] = last_returns_np   # (alive_count, K)
            metrics["hist/logprob"] = last_logprob_np   # (alive_count,)

        return metrics

    # ------------------------------------------------------------------
    # Reward decay
    # ------------------------------------------------------------------

    def _build_decay_scheduler(self) -> _DecayScheduler:
        """Register all shaping rewards for metric-triggered annealing.

        Thresholds are calibrated to training-curve observations:
          - kill-based triggers use ep/reward_kill (kill_weight=20 → ~9-10 at convergence)
          - self-metric triggers use the component's own ep/reward_X key

        Each weight decays with a half-life of 100 updates once armed.
        The "count" field tracks consecutive updates above threshold (hysteresis).
        """
        # Proxy for "agent is learning to fight": victory reward is +victory_weight
        # when winning. Threshold ~20 = agent wins >25% of the time (victory_weight=80).
        _victory = "ep/reward_victory"
        targets = []
        for comp in self.wrapper._all_components:
            if comp.name == "closing_speed":
                targets.append({"component": comp, "weight_attr": "closing_speed_weight",
                                 "watch_key": _victory, "threshold": 20.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "turn_rate":
                targets.append({"component": comp, "weight_attr": "turn_rate_weight",
                                 "watch_key": _victory, "threshold": 20.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "shoot_quality":
                targets.append({"component": comp, "weight_attr": "shoot_quality_weight",
                                 "watch_key": _victory, "threshold": 20.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "facing":
                targets.append({"component": comp, "weight_attr": "facing_weight",
                                 "watch_key": _victory, "threshold": 40.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "exposure":
                targets.append({"component": comp, "weight_attr": "exposure_weight",
                                 "watch_key": _victory, "threshold": 40.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "positioning":
                targets.append({"component": comp, "weight_attr": "positioning_weight",
                                 "watch_key": _victory, "threshold": 40.0,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "speed_range":
                targets.append({"component": comp, "weight_attr": "speed_range_weight",
                                 "watch_key": "ep/reward_speed_range", "threshold": 0.20,
                                 "sustain": 30, "half_life": 100, "count": 0})
            elif comp.name == "power_range":
                targets.append({"component": comp, "weight_attr": "power_range_weight",
                                 "watch_key": "ep/reward_power_range", "threshold": 0.08,
                                 "sustain": 30, "half_life": 100, "count": 0})
        return _DecayScheduler(targets)

    # ------------------------------------------------------------------
    # ELO evaluation
    # ------------------------------------------------------------------

    def _run_elo_eval(self) -> dict[str, float]:
        """Run sync matchups against every roster entry and update ELO ratings.

        Also lazily adds the scripted entry once scripted_roster_min_steps is reached.

        Creates a temporary TensorEnv for the matchups, then evicts all loaded
        checkpoint policies to reclaim memory.  The current league opponent is
        reloaded at the next rollout start.

        Returns:
            Dict of ELO metrics suitable for W&B logging.
        """
        from boost_and_broadside.modes.collect import _run_matchup

        # Lazily add the scripted agent once enough training has happened.
        if (self.scripted_agent is not None
                and self._global_step >= self.cfg.scripted_roster_min_steps
                and not any(e.kind == "scripted" for e in self.roster.entries)):
            self.roster.add_special(
                "scripted", self._global_step, 0, initial_elo=self._training_elo,
            )

        games = self.cfg.elo_eval_games
        env   = TensorEnv(games, self.ship_config, self.env_config, self.device)
        self.policy.eval()
        metrics: dict[str, float] = {}

        try:
            with torch.no_grad():
                for entry in list(self.roster.entries):  # copy: entries may grow
                    if entry.kind == "random":
                        opponent = None  # random agent
                    elif entry.kind == "checkpoint":
                        self.roster.load_policy(
                            entry, self.model_config, self.ship_config,
                            self.wrapper.num_active_components, self.device,
                        )
                        opponent = entry._policy
                    elif entry.kind == "avg":
                        if self._avg_update_count == 0:
                            continue  # avg not ready yet
                        opponent = self.avg_policy
                    else:  # "scripted"
                        if self.scripted_agent is None:
                            continue
                        opponent = self.scripted_agent

                    win_rate = _run_matchup(
                        self.policy, opponent, env,
                        self.ship_config, self.env_config, games, self.device,
                    )
                    self._training_elo = self.roster.update_elo(
                        self._training_elo, entry, win_rate,
                    )
                    metrics[f"elo/{entry.label}"] = entry.elo
        finally:
            self.policy.train()
            self.roster.evict_all_checkpoint_policies()

        metrics["elo/training"] = self._training_elo
        return metrics

    def _save_roster_json(self) -> None:
        """Persist roster metadata alongside the run's checkpoints."""
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.roster.save_json(ckpt_dir / "roster.json")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file.

        Written to cfg.checkpoint_dir/checkpoint_{update:06d}.pt.
        Directory is created if it does not exist.

        Args:
            update: Current update index (used as filename suffix).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self._run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"
        torch.save({
            "policy_state_dict":     self.policy.state_dict(),
            "optimizer_state_dict":  self.optim.state_dict(),
            "scaler_state_dict":     self.scaler.state_dict(),
            "avg_policy_state_dict": self.avg_policy.state_dict(),
            "avg_param_cumsum":      [c.cpu() for c in self._avg_param_cumsum],
            "avg_update_count":      self._avg_update_count,
            "update":                update,
            "global_step":           self._global_step,
            "training_elo":          self._training_elo,
            "train_config":          dataclasses.asdict(self.cfg),
            "model_config":          dataclasses.asdict(self.model_config),
            "env_config":            dataclasses.asdict(self.env_config),
            "reward_config":         dataclasses.asdict(self.reward_config),
        }, path)
        self._last_checkpoint_path = path
        print(f"Checkpoint saved: {path}")

        # Prune: keep only the latest checkpoint + all roster-referenced files
        kept = self.roster.kept_paths()
        kept.add(str(path))
        for old_path in ckpt_dir.glob("step_*.pt"):
            if str(old_path) not in kept:
                old_path.unlink(missing_ok=True)

    def load_checkpoint(self, path: str) -> int:
        """Load policy and optimizer weights from a checkpoint file.

        Args:
            path: Path to a .pt checkpoint file.

        Returns:
            The update index stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "avg_policy_state_dict" in ckpt:
            self.avg_policy.load_state_dict(ckpt["avg_policy_state_dict"])
            self._avg_param_cumsum = [c.to(self.device) for c in ckpt["avg_param_cumsum"]]
            self._avg_update_count = ckpt["avg_update_count"]
        if "training_elo" in ckpt:
            self._training_elo  = ckpt["training_elo"]
            self._elo_milestone = ckpt["training_elo"]

        # Restore roster if its JSON exists alongside the checkpoint
        roster_path = Path(path).parent / "roster.json"
        if roster_path.exists():
            self.roster.load_json(roster_path)

        return ckpt["update"]

    # ------------------------------------------------------------------
    # Async logging
    # ------------------------------------------------------------------

    def _init_wandb(
        self,
        train_config:  TrainConfig,
        model_config:  ModelConfig,
        ship_config:   ShipConfig,
        env_config:    EnvConfig,
        reward_config: RewardConfig,
    ) -> None:
        """Initialize W&B run with all configs serialized as the run config.

        Args:
            train_config:  PPO hyperparameters.
            model_config:  Policy architecture.
            ship_config:   Physics constants.
            env_config:    Environment sizing.
            reward_config: Reward weights.
        """
        import wandb
        config: dict = {}
        for prefix, cfg in [
            ("train",  train_config),
            ("model",  model_config),
            ("ship",   ship_config),
            ("env",    env_config),
            ("reward", reward_config),
        ]:
            for k, v in dataclasses.asdict(cfg).items():
                # frozenset is not JSON-serializable; convert to sorted list
                config[f"{prefix}/{k}"] = sorted(v) if isinstance(v, (frozenset, set)) else v

        wandb.init(project="boost-and-broadside", config=config)

    def _enqueue_log(self, metrics: dict, step: int) -> None:
        """Put metrics onto the async log queue (non-blocking)."""
        self._log_queue.put((metrics, step))

    def _log_worker(self) -> None:
        """Background thread: drains the log queue and calls wandb.log().

        Handles two special value types so the training thread stays off the
        W&B serialization path:
          - ``np.ndarray`` with key ``"hist/returns"`` → one ``wandb.Histogram``
            per reward component, keyed ``hist/returns/<name>``.
          - ``np.ndarray`` with any other key → ``wandb.Histogram`` directly.
        """
        import numpy as np
        import wandb
        while True:
            try:
                item = self._log_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            raw_metrics, step = item
            processed: dict = {}
            for k, v in raw_metrics.items():
                if isinstance(v, np.ndarray):
                    if k == "hist/returns":
                        # v shape: (alive_count, K) — one histogram per active component
                        for i, name in enumerate(self._active_names):
                            processed[f"hist/returns/{name}"] = wandb.Histogram(v[:, i])
                    else:
                        processed[k] = wandb.Histogram(v)
                else:
                    processed[k] = v
            wandb.log(processed, step=step)
