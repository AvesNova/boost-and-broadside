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
import torch.optim as optim

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import TrainConfig, ModelConfig, ShipConfig, EnvConfig, RewardConfig
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.train.rl.buffer import RolloutBuffer
from boost_and_broadside.train.rl.video_logger import VideoLogger


# ---------------------------------------------------------------------------
# Team-split helpers for two-pass self-play
# ---------------------------------------------------------------------------

def _team_sort_idx(team_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable-sort ship indices so team-0 ships precede team-1 ships.

    Args:
        team_id: (B, N) int32 or float32 — 0 for team 0, 1 for team 1.

    Returns:
        sort_idx:   (B, N) long — sorted indices (team 0 first).
        unsort_idx: (B, N) long — inverse permutation of sort_idx.
    """
    sort_idx   = team_id.float().argsort(dim=1, stable=True).long()
    unsort_idx = sort_idx.argsort(dim=1, stable=True).long()
    return sort_idx, unsort_idx


def _gather_ships(t: torch.Tensor, idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Select ships from tensor t along ship dimension `dim` using per-batch indices.

    B must be at dimension dim-1.

    Args:
        t:   Tensor with N ships at `dim`.
        idx: (B, N_t) long — per-batch ship indices to select.
        dim: Position of the N (ships) dimension in t.

    Returns:
        Same tensor with N replaced by N_t at `dim`.
    """
    B, N_t   = idx.shape
    tgt      = list(t.shape)
    tgt[dim] = N_t
    vs       = [1] * t.dim()
    vs[dim - 1] = B
    vs[dim]     = N_t
    return t.gather(dim, idx.view(*vs).expand(*tgt))


class _StepDecayScheduler:
    """Linearly anneals shaping reward weights to zero over a fixed step budget.

    weight(t) = weight_0 * max(0, 1 - t / decay_steps)

    decay_steps == 0 disables decay (weights remain at their initial values).
    """

    def __init__(self, targets: list[dict], decay_steps: int) -> None:
        self._targets     = targets     # list of {component, weight_attr, initial_value}
        self._decay_steps = decay_steps

    def step(self, global_step: int) -> dict[str, float]:
        """Apply linear decay for current global_step. Returns decay log dict."""
        if self._decay_steps <= 0:
            return {}

        factor  = max(0.0, 1.0 - global_step / self._decay_steps)
        logged: dict[str, float] = {}
        for t in self._targets:
            new_val = t["initial_value"] * factor
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
        self.cfg          = train_config
        self.model_config = model_config
        self.ship_config  = ship_config
        self.env_config   = env_config
        self.reward_config = reward_config
        self.device       = torch.device(device)
        self.use_wandb    = use_wandb

        self.wrapper = MVPEnvWrapper(
            num_envs       = train_config.num_envs,
            ship_config    = ship_config,
            env_config     = env_config,
            reward_config  = reward_config,
            device         = device,
            scripted_agent = scripted_agent,
        )
        self.policy = MVPPolicy(model_config, ship_config).to(self.device)
        self.optim  = optim.Adam(self.policy.parameters(), lr=train_config.learning_rate, eps=1e-5)

        # Uniform average policy — frozen opponent used in half of self-play envs.
        # Maintained as a running arithmetic mean of main policy weights.
        self.avg_policy: MVPPolicy = MVPPolicy(model_config, ship_config).to(self.device)
        self.avg_policy.load_state_dict(self.policy.state_dict())  # start identical
        self.avg_policy.requires_grad_(False)
        self._avg_count = 0  # number of updates included in the running mean

        # How many envs play against avg_policy (team 1 = avg_policy opponent)
        self._envs_vs_avg = int(train_config.num_envs * train_config.avg_policy_opp_fraction)

        N = env_config.num_ships

        # Build obs shapes for the buffer from a dummy reset
        sample_obs  = self.wrapper.reset()
        obs_shapes  = {
            k: v.shape[1:]   # strip batch dim → (N, ...) or (N,)
            for k, v in sample_obs.items()
            if v.dtype in (torch.float32, torch.int32, torch.int64, torch.bool)
        }

        self.buffer = RolloutBuffer(
            num_steps    = train_config.num_steps,
            num_envs     = train_config.num_envs,
            num_ships    = N,
            obs_shapes   = obs_shapes,
            gamma        = train_config.gamma,
            gae_lambda   = train_config.gae_lambda,
            device       = self.device,
        )

        # Async logging queue
        self._log_queue: Queue = Queue()
        if use_wandb:
            self._init_wandb(train_config, model_config, ship_config, env_config, reward_config)
            self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._log_thread.start()

        # Async video logger (only when W&B + interval configured)
        self._video_logger: VideoLogger | None = None
        if use_wandb and train_config.video_log_interval > 0:
            self._video_logger = VideoLogger(
                ship_config=ship_config,
                env_config=env_config,
                reward_config=reward_config,
                model_config=model_config,
            )

        self._global_step = 0
        self._num_updates = train_config.total_timesteps // (train_config.num_envs * train_config.num_steps)

        self._decay = self._build_step_decay_scheduler()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full PPO training loop."""
        B = self.cfg.num_envs
        N = self.wrapper.num_ships

        obs         = self.wrapper.reset()
        hidden      = self.policy.initial_hidden(B, N, self.device)
        # avg_policy hidden for pool-B team 1 (separate from main policy hidden)
        N_t         = N // 2
        avg_hidden  = self.avg_policy.initial_hidden(self._envs_vs_avg, N_t, self.device)

        start_time = time.time()

        for update in range(1, self._num_updates + 1):
            self.buffer.reset()
            self.buffer.store_initial_hidden(hidden)

            # Accumulate episode stats across the rollout — flushed once per update
            ep_rewards: list[torch.Tensor] = []
            ep_lengths: list[torch.Tensor] = []
            ep_components: dict[str, list[torch.Tensor]] = {}

            # ----------------------------------------------------------------
            # Rollout collection
            # ----------------------------------------------------------------
            for _ in range(self.cfg.num_steps):
                action, logprob, value, hidden, avg_hidden = self._two_team_forward(
                    obs, hidden, avg_hidden
                )

                next_obs, reward, dones, truncated, info = self.wrapper.step(action)

                if info.get("ep_reward") is not None:
                    ep_rewards.append(info["ep_reward"])
                    ep_lengths.append(info["ep_length"].float())
                    for name, t in info["ep_reward_components"].items():
                        ep_components.setdefault(name, []).append(t)

                done_any = dones | truncated
                self.buffer.add(
                    obs     = obs,
                    action  = action,
                    logprob = logprob,
                    reward  = reward,
                    done    = dones.float(),      # only true termination cuts GAE bootstrap
                    value   = value,
                    alive   = obs["alive"].bool(),
                )

                # Reset hidden for envs that terminated
                hidden = self.policy.reset_hidden_for_envs(hidden, done_any, N)
                # Reset avg_policy hidden for pool-B envs that terminated
                done_avg = done_any[B - self._envs_vs_avg:]  # pool-B done flags
                avg_hidden = self.avg_policy.reset_hidden_for_envs(
                    avg_hidden, done_avg, N_t
                )

                obs = next_obs
                self._global_step += B

            # ----------------------------------------------------------------
            # GAE computation
            # ----------------------------------------------------------------
            with torch.no_grad():
                _, _, next_value, _, _ = self._two_team_forward(obs, hidden, avg_hidden)
            self.buffer.compute_gae(next_value, dones.float())

            # ----------------------------------------------------------------
            # PPO update epochs
            # ----------------------------------------------------------------
            metrics = self._update_epochs()

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
            metrics["train/global_step"] = self._global_step
            metrics["train/sps"]         = sps

            # Update weight-averaged policy once past warmup
            if self._global_step >= self.cfg.avg_policy_warmup_steps:
                self._update_avg_policy()

            # Decay shaping weights based on training progress
            metrics.update(self._decay.step(self._global_step))

            # Single log call per update — all metrics at the same step
            self._enqueue_log(metrics, step=self._global_step)

            if update % 10 == 0:
                print(
                    f"update={update}/{self._num_updates}  "
                    f"step={self._global_step:,}  "
                    f"sps={sps:,}  "
                    f"loss={metrics.get('train/loss', 0.0):.4f}"
                )

            if self.cfg.checkpoint_interval > 0 and update % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(update)

            if (self._video_logger is not None
                    and update % self.cfg.video_log_interval == 0):
                self._video_logger.schedule(self.policy, self._global_step)

        self._log_queue.put(None)  # signal logger thread to exit

    # ------------------------------------------------------------------
    # PPO update inner loop
    # ------------------------------------------------------------------

    def _update_epochs(self) -> dict:
        """Run num_epochs × num_minibatches of PPO updates.

        Returns:
            Dict of mean metric values over all minibatch updates.
        """
        cfg = self.cfg
        accum: dict[str, list[float]] = {
            "train/loss":               [],
            "train/pg_loss":            [],
            "train/vf_loss":            [],
            "train/ent_loss":           [],
            "train/approx_kl":          [],
            "train/clip_frac":          [],
            "train/value_mean":         [],
            "train/return_mean":        [],
            "train/explained_variance": [],
        }

        for _ in range(cfg.num_epochs):
            for batch in self.buffer.get_minibatch_iterator(cfg.num_minibatches):
                (mb_obs, mb_actions, mb_old_logprobs, mb_advantages,
                 mb_returns, mb_values, mb_alive, mb_hidden) = batch

                # Re-evaluate actions with current policy (two separate team passes)
                logprob, entropy, new_value = self._two_team_evaluate(
                    mb_obs     = mb_obs,
                    mb_actions = mb_actions,
                    mb_hidden  = mb_hidden,
                    mb_alive   = mb_alive,
                )

                # Mask dead ships out of all loss and metric computations
                alive_f  = mb_alive.float()                        # (T, B_mb, N)
                mask_sum = alive_f.sum().clamp(min=1.0)

                # Normalise advantages per minibatch — only over alive ships
                adv      = mb_advantages
                adv_mean = (adv * alive_f).sum() / mask_sum
                adv_var  = ((adv - adv_mean).pow(2) * alive_f).sum() / mask_sum
                adv      = (adv - adv_mean) / (adv_var.sqrt() + 1e-8)

                # Policy gradient loss
                log_ratio  = logprob - mb_old_logprobs
                ratio      = log_ratio.exp()
                pg_loss1   = -adv * ratio
                pg_loss2   = -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss    = (torch.max(pg_loss1, pg_loss2) * alive_f).sum() / mask_sum

                # Value loss (clipped)
                vf_loss    = 0.5 * ((new_value - mb_returns).pow(2) * alive_f).sum() / mask_sum

                # Entropy bonus
                ent_loss   = -(entropy * alive_f).sum() / mask_sum

                loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl        = (((ratio - 1) - log_ratio) * alive_f).sum().item() / mask_sum.item()
                    clip_frac        = (((ratio - 1).abs() > cfg.clip_coef).float() * alive_f).sum().item() / mask_sum.item()
                    var_returns      = mb_returns.var().item()
                    explained_var    = (1.0 - (mb_returns - mb_values).var().item() / (var_returns + 1e-8))

                accum["train/loss"]              .append(loss.item())
                accum["train/pg_loss"]           .append(pg_loss.item())
                accum["train/vf_loss"]           .append(vf_loss.item())
                accum["train/ent_loss"]          .append(ent_loss.item())
                accum["train/approx_kl"]         .append(approx_kl)
                accum["train/clip_frac"]         .append(clip_frac)
                accum["train/value_mean"]        .append(mb_values.mean().item())
                accum["train/return_mean"]       .append(mb_returns.mean().item())
                accum["train/explained_variance"].append(explained_var)

        return {k: sum(v) / len(v) for k, v in accum.items() if v}

    # ------------------------------------------------------------------
    # Two-team forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _two_team_forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        avg_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run two per-team forward passes; each team sees itself as team 0.

        Pool A (first B - envs_vs_avg envs): team 1 uses main policy.
        Pool B (last envs_vs_avg envs):      team 1 uses avg_policy as frozen opponent.

        Args:
            obs:        Full (B, N, ...) obs for all ships.
            hidden:     (1, B*N, D) main policy hidden in original ship-slot order.
            avg_hidden: (1, B_avg*N_t, D) avg_policy hidden for pool-B team 1.

        Returns:
            action:         (B, N, 3) int.
            logprob:        (B, N) float.
            value:          (B, N) float.
            new_hidden:     (1, B*N, D) updated main hidden in original slot order.
            new_avg_hidden: (1, B_avg*N_t, D) updated avg_policy hidden.
        """
        B, N   = obs["team_id"].shape
        N_t    = N // 2
        D      = self.policy.gru.hidden_size
        B_avg  = self._envs_vs_avg
        B_self = B - B_avg

        sort_idx, unsort_idx = _team_sort_idx(obs["team_id"])
        t0_idx = sort_idx[:, :N_t]   # (B, N_t)
        t1_idx = sort_idx[:, N_t:]   # (B, N_t)

        t0_obs = {k: _gather_ships(v, t0_idx, 1) for k, v in obs.items()}
        t1_obs = {k: _gather_ships(v, t1_idx, 1) for k, v in obs.items()}

        # Sort hidden then split into per-team halves
        sort_exp = sort_idx.unsqueeze(-1).expand(B, N, D)
        h_sorted = hidden.squeeze(0).reshape(B, N, D).gather(1, sort_exp)
        h_t0     = h_sorted[:, :N_t].reshape(1, B * N_t, D)

        # Pool A team 1: main policy (first B_self envs)
        t1_obs_self = {k: v[:B_self] for k, v in t1_obs.items()}
        t1_obs_self["team_id"] = torch.zeros_like(t1_obs_self["team_id"])
        h_t1_self = h_sorted[:B_self, N_t:].reshape(1, B_self * N_t, D)

        # Pool B team 1: avg_policy (last B_avg envs)
        t1_obs_avg = {k: v[B_self:] for k, v in t1_obs.items()}
        t1_obs_avg["team_id"] = torch.zeros_like(t1_obs_avg["team_id"])

        # Team 0 needs team_id = 0 (it already is, but be explicit)
        t0_obs_relabeled = dict(t0_obs)
        t0_obs_relabeled["team_id"] = torch.zeros_like(t0_obs["team_id"])

        a_t0,  lp_t0,  v_t0,  new_h_t0    = self.policy.get_action_and_value(t0_obs_relabeled, h_t0)
        a_t1s, lp_t1s, v_t1s, new_h_t1s   = self.policy.get_action_and_value(t1_obs_self, h_t1_self)
        a_t1a, lp_t1a, v_t1a, new_avg_h   = self.avg_policy.get_action_and_value(t1_obs_avg, avg_hidden)

        # Merge main-policy hidden back to original slot order
        new_h_t1 = torch.cat([new_h_t1s.reshape(B_self, N_t, D),
                               h_sorted[B_self:, N_t:]], dim=0)  # keep avg envs' main hidden unchanged
        new_h_sorted = torch.cat([new_h_t0.reshape(B, N_t, D), new_h_t1], dim=1)
        unsort_exp   = unsort_idx.unsqueeze(-1).expand(B, N, D)
        new_hidden   = new_h_sorted.gather(1, unsort_exp).reshape(1, B * N, D)

        # Combine team 1 outputs (pool A + pool B)
        a_t1  = torch.cat([a_t1s,  a_t1a],  dim=0)
        lp_t1 = torch.cat([lp_t1s, lp_t1a], dim=0)
        v_t1  = torch.cat([v_t1s,  v_t1a],  dim=0)

        # Scatter per-team outputs back to original slot order
        action  = torch.zeros(B, N, 3, dtype=a_t0.dtype, device=self.device)
        logprob = torch.zeros(B, N, device=self.device)
        value   = torch.zeros(B, N, device=self.device)
        action .scatter_(1, t0_idx.unsqueeze(-1).expand(B, N_t, 3), a_t0)
        action .scatter_(1, t1_idx.unsqueeze(-1).expand(B, N_t, 3), a_t1)
        logprob.scatter_(1, t0_idx, lp_t0)
        logprob.scatter_(1, t1_idx, lp_t1)
        value  .scatter_(1, t0_idx, v_t0)
        value  .scatter_(1, t1_idx, v_t1)

        return action, logprob, value, new_hidden, new_avg_h

    def _two_team_evaluate(
        self,
        mb_obs: dict[str, torch.Tensor],
        mb_actions: torch.Tensor,
        mb_hidden: torch.Tensor,
        mb_alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions for both teams separately for the PPO update.

        Uses first-timestep team IDs to compute sort order; valid because team
        assignments are fixed within each episode and hidden is zeroed at
        episode boundaries.

        Args:
            mb_obs:     Dict with (T, B, N, ...) tensors.
            mb_actions: (T, B, N, 3) int.
            mb_hidden:  (1, B*N, D) initial hidden state in original slot order.
            mb_alive:   (T, B, N) bool.

        Returns:
            logprob: (T, B, N) float.
            entropy: (T, B, N) float.
            value:   (T, B, N) float.
        """
        T, B, N, _ = mb_actions.shape
        N_t = N // 2
        D   = self.policy.gru.hidden_size

        sort_idx, _ = _team_sort_idx(mb_obs["team_id"][0])  # sort by first-step team IDs
        t0_idx = sort_idx[:, :N_t]
        t1_idx = sort_idx[:, N_t:]

        t0_obs = {k: _gather_ships(v, t0_idx, 2) for k, v in mb_obs.items()}
        t1_obs = {k: _gather_ships(v, t1_idx, 2) for k, v in mb_obs.items()}
        t1_obs["team_id"] = torch.zeros_like(t1_obs["team_id"])

        t0_acts  = _gather_ships(mb_actions, t0_idx, 2)   # (T, B, N_t, 3)
        t1_acts  = _gather_ships(mb_actions, t1_idx, 2)
        t0_alive = _gather_ships(mb_alive,   t0_idx, 2)   # (T, B, N_t)
        t1_alive = _gather_ships(mb_alive,   t1_idx, 2)

        # Split initial hidden
        sort_exp  = sort_idx.unsqueeze(-1).expand(B, N, D)
        h_sorted  = mb_hidden.squeeze(0).reshape(B, N, D).gather(1, sort_exp)
        h_t0      = h_sorted[:, :N_t].reshape(1, B * N_t, D)
        h_t1      = h_sorted[:, N_t:].reshape(1, B * N_t, D)

        lp_t0, ent_t0, val_t0 = self.policy.evaluate_actions(t0_obs, t0_acts.long(), h_t0, t0_alive)
        lp_t1, ent_t1, val_t1 = self.policy.evaluate_actions(t1_obs, t1_acts.long(), h_t1, t1_alive)

        # Scatter back to (T, B, N)
        t0_exp = t0_idx.unsqueeze(0).expand(T, B, N_t)
        t1_exp = t1_idx.unsqueeze(0).expand(T, B, N_t)

        logprob = torch.zeros(T, B, N, device=self.device)
        entropy = torch.zeros(T, B, N, device=self.device)
        value   = torch.zeros(T, B, N, device=self.device)

        logprob.scatter_(2, t0_exp, lp_t0)
        logprob.scatter_(2, t1_exp, lp_t1)
        entropy.scatter_(2, t0_exp, ent_t0)
        entropy.scatter_(2, t1_exp, ent_t1)
        value  .scatter_(2, t0_exp, val_t0)
        value  .scatter_(2, t1_exp, val_t1)

        return logprob, entropy, value

    # ------------------------------------------------------------------
    # Avg-policy update
    # ------------------------------------------------------------------

    def _update_avg_policy(self) -> None:
        """Update avg_policy as the uniform running arithmetic mean of main policy.

        avg_θ = avg_θ + (θ - avg_θ) / count   (true running mean)

        Only called once past avg_policy_warmup_steps.
        """
        self._avg_count += 1
        n = self._avg_count
        with torch.no_grad():
            for avg_p, main_p in zip(self.avg_policy.parameters(), self.policy.parameters()):
                avg_p.data.add_((main_p.data - avg_p.data) / n)

    # ------------------------------------------------------------------
    # Reward decay
    # ------------------------------------------------------------------

    def _build_step_decay_scheduler(self) -> _StepDecayScheduler:
        """Register all shaping rewards for step-based linear annealing.

        Objective rewards (damage, kill, death, victory) are excluded — only
        shaping components have their weights decayed to zero.
        """
        _OBJECTIVE_NAMES = {"damage", "kill", "death", "victory"}
        _WEIGHT_ATTR: dict[str, str] = {
            "closing_speed": "closing_speed_weight",
            "turn_rate":     "turn_rate_weight",
            "facing":        "facing_weight",
            "exposure":      "exposure_weight",
            "positioning":   "positioning_weight",
            "proximity":     "proximity_weight",
            "speed_range":   "speed_range_weight",
            "power_range":   "power_range_weight",
            "shoot_quality": "shoot_quality_weight",
            "scripted_agent": "weight",
        }
        targets = []
        for comp in self.wrapper._components:
            if comp.name in _OBJECTIVE_NAMES:
                continue
            attr = _WEIGHT_ATTR.get(comp.name)
            if attr is None:
                continue
            initial = getattr(comp, attr)
            targets.append({"component": comp, "weight_attr": attr, "initial_value": initial})
        return _StepDecayScheduler(targets, self.cfg.shaping_decay_steps)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file.

        Written to cfg.checkpoint_dir/{run_name}/step_{global_step:012d}.pt.
        The run_name is taken from wandb.run.name when W&B is active,
        otherwise falls back to "local".

        Args:
            update: Current update index (used for logging only).
        """
        run_name = "local"
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    run_name = wandb.run.name
            except Exception:
                pass

        ckpt_dir = Path(self.cfg.checkpoint_dir) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"
        torch.save({
            "policy_state_dict":     self.policy.state_dict(),
            "avg_policy_state_dict": self.avg_policy.state_dict(),
            "avg_count":             self._avg_count,
            "optimizer_state_dict":  self.optim.state_dict(),
            "update":                update,
            "global_step":           self._global_step,
            "train_config":          dataclasses.asdict(self.cfg),
            "model_config":          dataclasses.asdict(self.model_config),
            "env_config":            dataclasses.asdict(self.env_config),
            "reward_config":         dataclasses.asdict(self.reward_config),
        }, path)
        print(f"Checkpoint saved: {path}")

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
        if "avg_policy_state_dict" in ckpt:
            self.avg_policy.load_state_dict(ckpt["avg_policy_state_dict"])
            self._avg_count = ckpt.get("avg_count", 0)
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
        """Background thread: drains the log queue and calls wandb.log()."""
        import wandb
        while True:
            try:
                item = self._log_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            metrics, step = item
            wandb.log(metrics, step=step)
