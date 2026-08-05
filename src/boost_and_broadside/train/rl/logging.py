"""Training metric assembly and asynchronous W&B logging."""

import dataclasses
import json
import time
from pathlib import Path
from queue import Empty

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig, TrainConfig


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
        # A floor that binds on an active component decouples that component's
        # scale from its own statistics: the critic target (ReturnScaler) or the
        # policy-gradient share (AdvantageScaler) is then set by the guard.
        # Tracked per scaler because the two have different failure modes — see
        # the return_min_span note in runs/rl.py.
        span_bound_cpu = self.scaler.floor_bound.cpu()
        rms_bound_cpu = self.adv_scaler.floor_bound.cpu()
        for i, name in enumerate(self._active_names):
            metrics[f"scaler/p5/{name}"] = p5_cpu[i].item()
            metrics[f"scaler/p95/{name}"] = p95_cpu[i].item()
            metrics[f"scaler/span/{name}"] = span_cpu[i].item()
            metrics[f"scaler/adv_rms/{name}"] = adv_rms_cpu[i].item()
            metrics[f"scaler/floor_bound_span/{name}"] = float(span_bound_cpu[i].item())
            metrics[f"scaler/floor_bound_rms/{name}"] = float(rms_bound_cpu[i].item())
            if bool(rms_bound_cpu[i].item()) and name not in self._floor_warned:
                self._floor_warned.add(name)
                print(
                    f"[PPOTrainer] WARNING: advantage_min_rms binds on active component "
                    f"{name!r} (adv_rms={adv_rms_cpu[i].item():.3g} vs "
                    f"{self.cfg.advantage_min_rms:g}). Its policy-gradient share is set "
                    "by the guard, not by its own statistics."
                )

        # Scaler span minimum — flags components where normalization may be degenerate
        if self._field_map is not None:
            # Sustained non-zero means the radius/width/count combination is
            # too tight to place, and those maps are repeats of the last bank.
            metrics["physics/field_map_generation_failures"] = (
                self._field_map.generation_failures.item()
            )

        metrics["scaler/span_min"] = span_cpu.min().item()
        metrics["scaler/floor_bound_span_count"] = float(span_bound_cpu.sum().item())
        metrics["scaler/floor_bound_rms_count"] = float(rms_bound_cpu.sum().item())

        # Merge episode stats accumulated on-GPU by the wrapper — one sync per update
        ep_stats = self.wrapper.pop_episode_stats()
        for aux_w in self.aux_wrappers:
            aux_w.pop_episode_stats()  # discarded, but keeps accumulators bounded
        n_eps = ep_stats["episodes"].item()
        source_stats = ep_stats["source_stats"].cpu()
        live_ship_steps = source_stats[6].item()
        if live_ship_steps > 0:
            field_damage = source_stats[0].item()
            combat_damage = source_stats[1].item()
            total_damage = field_damage + combat_damage
            metrics["physics/field_damage_per_live_ship_step"] = field_damage / live_ship_steps
            metrics["physics/combat_damage_per_live_ship_step"] = combat_damage / live_ship_steps
            metrics["physics/field_deaths_per_million_live_ship_steps"] = (
                source_stats[2].item() * 1_000_000.0 / live_ship_steps
            )
            metrics["physics/combat_deaths_per_million_live_ship_steps"] = (
                source_stats[3].item() * 1_000_000.0 / live_ship_steps
            )
            metrics["physics/field_damage_step_fraction"] = source_stats[4].item() / live_ship_steps
            metrics["physics/nonambient_live_ship_fraction"] = (
                source_stats[5].item() / live_ship_steps
            )
            metrics["physics/field_damage_fraction"] = (
                field_damage / total_damage if total_damage > 0.0 else 0.0
            )
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

        # Elo evaluation — continuous live statistics from parallel slots.
        # Avg-model metrics only exist once the avg model has been initialized.
        metrics["elo/training"] = self._training_elo
        metrics["elo/scripted"] = self._scripted_elo
        metrics["ladder/frozen_count"] = sum(
            1 for e in self.roster.entries if e.kind == "checkpoint" and e.fixed
        )
        floating_entry = self.roster.floating_checkpoint()
        if floating_entry is not None:
            metrics["elo/floating"] = floating_entry.elo
            metrics["ladder/floating_games"] = self._floating_games
        if self._eval_window_rand:
            metrics["elo/training_vs_random"] = sum(self._eval_window_rand) / len(
                self._eval_window_rand
            )
        if self._eval_window_sc:
            metrics["elo/training_vs_scripted"] = sum(self._eval_window_sc) / len(
                self._eval_window_sc
            )
        if self._eval_window_ladder:
            metrics["elo/training_vs_ladder"] = sum(self._eval_window_ladder) / len(
                self._eval_window_ladder
            )
        if self._eval_window_floating:
            metrics["elo/training_vs_floating"] = sum(self._eval_window_floating) / len(
                self._eval_window_floating
            )
        if self._avg_update_count > 0:
            metrics["elo/avg"] = self._avg_training_elo
            if self._eval_window_live_vs_avg:
                metrics["elo/training_vs_avg"] = sum(self._eval_window_live_vs_avg) / len(
                    self._eval_window_live_vs_avg
                )
        self._append_elo_history(update)
        # Per-opponent outcome mix. tie_rate is the diagnostic that decides how
        # the post-hoc suite should handle draws: this game's draw frequency is
        # level-dependent rather than gap-dependent, which is exactly what the
        # standard tie models (Davidson, Rao-Kupper) cannot represent.
        total_games = 0
        for label, (win, loss, tie) in self._match_counts.items():
            games = win + loss + tie
            total_games += games
            if games == 0:
                continue
            metrics[f"matches/{label}/games"] = games
            metrics[f"matches/{label}/win_rate"] = win / games
            metrics[f"matches/{label}/tie_rate"] = tie / games
            decisive = win + loss
            if decisive > 0:
                metrics[f"matches/{label}/decisive_win_rate"] = win / decisive
        metrics["matches/games_total"] = total_games
        # One scalar per ladder entry, keyed under a shared prefix so a single
        # line-plot panel with y = `ladder/elo/*` overlays them all natively.
        # Frozen entries log a constant (flat line); live rides along as the
        # reference the floating rating is chasing.
        metrics["ladder/elo/live"] = self._training_elo
        for entry in self.roster.entries:
            metrics[f"ladder/elo/{entry.label}"] = entry.elo

        # Save overwriting best-model checkpoints (live, then avg) when the
        # rating improves. Absolute gauge, so no re-basing.
        self._maybe_save_best_checkpoints()

        # Overview — redundant copies of the most important global metrics
        for src, dst in [
            ("elo/training", "overview/elo"),
            ("elo/training_vs_scripted", "overview/win_rate_vs_scripted"),
            ("elo/training_vs_random", "overview/win_rate_vs_random"),
            ("elo/training_vs_ladder", "overview/win_rate_vs_ladder"),
            ("elo/training_vs_avg", "overview/win_rate_vs_avg"),
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

    def _append_elo_history(self, update: int) -> None:
        """Append one line of rating state to the run's elo_history.jsonl.

        The file is append-only across resumes, so the complete rating history
        of every ladder entry survives training for post-hoc analysis
        (calibration drift, cycling checks against old anchors).

        ``counts`` carries this update's raw win/loss/tie totals per opponent
        from the live policy's perspective. These are the run's only irreplaceable
        measurements: the live and avg policies exist in one form for exactly one
        update and can never be replayed, whereas every frozen ladder entry can
        be re-measured from disk afterwards at whatever precision is wanted.
        """
        path = Path(self.cfg.checkpoint_dir) / self.run_name / "elo_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "update": update,
            "global_step": self._global_step,
            "live": self._training_elo,
            "avg": self._avg_training_elo,
            "scripted": self._scripted_elo,
            "counts": {label: list(wlt) for label, wlt in self._match_counts.items()},
            "entries": [
                {"label": e.label, "elo": e.elo, "fixed": e.fixed} for e in self.roster.entries
            ],
        }
        with path.open("a") as history_file:
            history_file.write(json.dumps(record) + "\n")

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
        field_sources = (
            "  "
            f"field_dmg={metrics['physics/field_damage_per_live_ship_step']:.4f}  "
            "field_deaths/M="
            f"{metrics['physics/field_deaths_per_million_live_ship_steps']:.1f}"
            if "physics/field_damage_per_live_ship_step" in metrics
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
            f"{field_sources}"
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
