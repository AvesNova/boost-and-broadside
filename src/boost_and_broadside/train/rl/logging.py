"""Training metric assembly and asynchronous W&B logging."""

import dataclasses
import statistics
import time
from queue import Empty

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig, TrainConfig
from boost_and_broadside.train.rl.rating import DRAW, LOSS, WIN, MatchCounts, solve_bt


class LoggingMixin:
    """Metric and logging behavior mixed into PPOTrainer."""

    def _assemble_metrics(
        self,
        metrics: dict,
        update: int,
        ship_tokens_per_update: int,
    ) -> tuple[int, int]:
        # Scaler stats — one CPU transfer per component group
        p5, p95 = self.scaler.percentiles
        p5_cpu = p5.cpu()
        p95_cpu = p95.cpu()
        span_cpu = p95_cpu - p5_cpu
        adv_rms_cpu = self.adv_scaler.rms.cpu()
        for i, name in enumerate(self._active_names):
            metrics[f"scaler/p5/{name}"] = p5_cpu[i].item()
            metrics[f"scaler/p95/{name}"] = p95_cpu[i].item()
            metrics[f"scaler/span/{name}"] = span_cpu[i].item()
            metrics[f"scaler/adv_rms/{name}"] = adv_rms_cpu[i].item()

        # Scaler span minimum — flags components where normalization may be degenerate
        metrics["scaler/span_min"] = span_cpu.min().item()

        # Merge episode stats accumulated on-GPU by the wrapper — one sync per update
        ep_stats = self.wrapper.pop_episode_stats()
        for aux_w in self.aux_wrappers:
            aux_w.pop_episode_stats()  # discarded, but keeps accumulators bounded
        n_eps = ep_stats["episodes"].item()
        if n_eps > 0:
            n_ship_eps = n_eps * self.wrapper.num_ships
            comp_sum = ep_stats["comp_sum"].cpu()
            comp_scaled_sum = ep_stats["comp_scaled_sum"].cpu()
            metrics["episode/reward_mean"] = ep_stats["reward_sum"].item() / n_ship_eps
            metrics["episode/reward_min"] = ep_stats["reward_min"].item()
            metrics["episode/reward_max"] = ep_stats["reward_max"].item()
            metrics["episode/length_mean"] = ep_stats["length_sum"].item() / n_eps
            for i, name in enumerate(self._active_names):
                metrics[f"episode/reward_{name}"] = comp_sum[i].item() / n_ship_eps
                metrics[f"episode/scaled_{name}"] = comp_scaled_sum[i].item() / n_ship_eps
            metrics["episode/win_rate"] = ep_stats["wins_sum"].item() / n_ship_eps
            metrics["episode/lifespan_mean"] = ep_stats["lifespan_sum"].item() / n_ship_eps

        self._ship_steps += ship_tokens_per_update
        self._grad_tokens += int(metrics["train/epochs_completed"] * self._entity_tokens_per_epoch)
        # Cumulative work / cumulative training time — spans checkpoint
        # resumes, as if the run never stopped.
        elapsed = self._elapsed_train_time + (time.time() - self._train_start_time)
        sps = int(self._global_step / elapsed)
        ship_tps = int(self._ship_steps / elapsed)
        metrics["train/global_step"] = self._global_step
        metrics["train/sps"] = sps
        metrics["train/ship_tokens_per_sec"] = ship_tps
        # Alternative x-axes — log as metrics so any chart can be re-plotted
        # against data volume, optimizer progress, compute, or wall clock.
        metrics["counters/env_steps"] = self._global_step
        metrics["counters/ship_tokens"] = self._ship_steps
        metrics["counters/updates"] = update
        metrics["counters/grad_tokens"] = self._grad_tokens
        metrics["counters/train_hours"] = elapsed / 3600.0

        metrics["elo/live"] = self._training_elo
        metrics["elo/live_se"] = self.roster.standard_errors.get("live", float("inf"))
        metrics["elo/scripted"] = self.roster.ratings.get("scripted", 0.0)
        if self._avg_update_count > 0:
            metrics["elo/avg"] = self._avg_training_elo
        metrics.update(self._rating_metrics)
        roster_elos = [entry.elo for entry in self.roster.entries]
        if roster_elos:
            metrics["elo/roster_min"] = min(roster_elos)
            metrics["elo/roster_median"] = statistics.median(roster_elos)
            metrics["elo/roster_max"] = max(roster_elos)

        if self._training_elo > self._best_training_elo:
            self._best_training_elo = self._training_elo
            self._save_best_checkpoint("best_training.pt")
        if self._avg_update_count > 0 and self._avg_training_elo > self._best_avg_elo:
            self._best_avg_elo = self._avg_training_elo
            self._save_best_checkpoint("best_avg.pt", self._avg_checkpoint_payload(update))

        # Overview — redundant copies of the most important global metrics
        for src, dst in [
            ("elo/live", "overview/elo"),
            ("elo/live_vs_scripted", "overview/win_rate_vs_scripted"),
            ("elo/live_vs_random", "overview/win_rate_vs_random"),
            ("elo/live_vs_avg", "overview/win_rate_vs_avg"),
            ("loss/total", "overview/loss_total"),
            ("loss_proxy/policy_gradient", "overview/loss_proxy_pg"),
            ("loss_proxy/behavioral_cloning", "overview/loss_proxy_bc"),
            ("policy/kl", "overview/kl"),
            ("policy/clip_fraction", "overview/clip_fraction"),
            ("episode/win_rate", "overview/win_rate"),
            ("episode/reward_mean", "overview/reward_mean"),
            ("train/gradient_norm", "overview/gradient_norm"),
            ("schedule/behavior_cloning_coef", "overview/bc_coef"),
        ]:
            if src in metrics:
                metrics[dst] = metrics[src]
        ev_vals = [v for k, v in metrics.items() if k.startswith("critic/explained_variance/")]
        if ev_vals:
            metrics["overview/explained_variance"] = sum(ev_vals) / len(ev_vals)

        return sps, ship_tps

    def _log_training_update(self, metrics: dict, update: int, sps: int, ship_tps: int) -> None:
        """Enqueue one metric batch and optionally print the terminal summary."""
        self._enqueue_log(metrics, step=self._global_step)
        if update % self.cfg.log_interval != 0:
            return
        lifespan = (
            f"  lifespan={metrics['episode/lifespan_mean']:.1f}"
            if "episode/lifespan_mean" in metrics
            else ""
        )
        print(
            f"update={update}/{self._num_updates}  "
            f"step={self._global_step:,}  "
            f"sps={sps:,}  "
            f"ship_tps={ship_tps:,}  "
            f"loss={metrics.get('loss/total', 0.0):.4f}"
            f"  elo={self._training_elo:.0f}"
            f"{lifespan}"
        )

    def _init_wandb(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        resume_run_id: str | None = None,
    ) -> None:
        """Initialize W&B run with all configs serialized as the run config."""
        import wandb

        def _sanitize(obj: object) -> object:
            """Recursively convert frozenset/set → sorted list for JSON serialization."""
            if isinstance(obj, (frozenset, set)):
                return sorted(_sanitize(x) for x in obj)  # type: ignore[misc]
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(x) for x in obj]
            return obj

        config: dict = {}
        for prefix, cfg in [
            ("train", train_config),
            ("model", model_config),
            ("ship", ship_config),
            ("env", env_config),
        ]:
            for k, v in dataclasses.asdict(cfg).items():
                if k == "schedule":
                    continue  # TrainingSchedule contains callables — not serializable
                config[f"{prefix}/{k}"] = _sanitize(v)

        if resume_run_id is not None:
            wandb.init(
                project="boost-and-broadside", config=config, id=resume_run_id, resume="must"
            )
        else:
            wandb.init(project="boost-and-broadside", config=config)

    def _enqueue_log(self, metrics: dict, step: int) -> None:
        """Put metrics onto the async log queue (non-blocking)."""
        self._log_queue.put(("metrics", metrics, step))

    def _enqueue_rating_solve(self, counts: MatchCounts) -> None:
        """Queue one CPU Bradley-Terry solve without blocking training."""
        self._log_queue.put(("ratings", counts, dict(self.roster.ratings)))

    def _consume_rating_result(self) -> None:
        """Apply the newest completed rating solve at a rollout boundary."""
        latest = None
        while True:
            try:
                latest = self._rating_results.get_nowait()
            except Empty:
                break
        if latest is None:
            return
        ratings, standard_errors, rating_metrics = latest
        merged_ratings = dict(self.roster.ratings)
        merged_errors = dict(self.roster.standard_errors)
        merged_ratings.update(ratings)
        merged_errors.update(standard_errors)
        current_ids = set(self.roster.counts.agent_ids)
        merged_ratings = {key: value for key, value in merged_ratings.items() if key in current_ids}
        merged_errors = {key: value for key, value in merged_errors.items() if key in current_ids}
        self.roster.refresh_ratings(merged_ratings, merged_errors)
        self._training_elo = self.roster.ratings["live"]
        self._avg_training_elo = self.roster.ratings.get("avg", 0.0)
        self._rating_metrics = rating_metrics

    def _log_worker(self) -> None:
        """Background thread: drains the log queue and calls wandb.log().

        Handles two special value types so the training thread stays off the
        W&B serialization path:
          - ``np.ndarray`` with key ``"hist/returns"`` → one ``wandb.Histogram``
            per reward component, keyed ``hist/returns/<name>``.
          - ``np.ndarray`` with any other key → ``wandb.Histogram`` directly.
        """
        import numpy as np

        wandb = None
        if self.use_wandb:
            import wandb as wandb_module

            wandb = wandb_module

        previous_counts: MatchCounts | None = None

        while True:
            try:
                item = self._log_queue.get(timeout=1.0)
            except Empty:
                continue
            if item is None:
                break
            kind, payload, context = item
            if kind == "ratings":
                counts = payload
                ratings, standard_errors = solve_bt(
                    counts,
                    "random",
                    context,
                    prior_draws=self.cfg.bt_prior_draws,
                )
                rating_metrics = self._recent_rating_metrics(counts, previous_counts)
                previous_counts = counts
                self._rating_results.put((ratings, standard_errors, rating_metrics))
                continue

            raw_metrics, step = payload, context
            if wandb is None:
                continue
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

    @staticmethod
    def _recent_rating_metrics(
        counts: MatchCounts,
        previous: MatchCounts | None,
    ) -> dict[str, float]:
        """Derive recent live matchup scores from consecutive count snapshots."""
        delta = counts.tensor.clone()
        if previous is not None and previous.agent_ids == counts.agent_ids:
            delta.sub_(previous.tensor)
            delta.clamp_(min=0.0)
        metrics = {}
        live_index = counts.index("live")
        for opponent_id in ("random", "scripted", "avg"):
            if opponent_id not in counts.agent_ids:
                continue
            opponent_index = counts.index(opponent_id)
            wins = delta[live_index, opponent_index, WIN] + delta[opponent_index, live_index, LOSS]
            losses = (
                delta[live_index, opponent_index, LOSS] + delta[opponent_index, live_index, WIN]
            )
            draws = (
                delta[live_index, opponent_index, DRAW] + delta[opponent_index, live_index, DRAW]
            )
            games = wins + losses + draws
            if games > 0.0:
                metrics[f"elo/live_vs_{opponent_id}"] = float((wins + 0.5 * draws) / games)
        return metrics
