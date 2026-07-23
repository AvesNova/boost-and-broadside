"""Checkpoint serialization, asynchronous saves, ladder milestones, and resume support."""

import copy
import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from boost_and_broadside.train.rl.elo_eval import EloEvaluator

# Rolling window of full-resume (step_*.pt) and avg (avg_step_*.pt) checkpoints
# kept per run. Ladder snapshots (ladder_step_*.pt) and named best checkpoints
# (best_*.pt) use separate filename families and are never subject to this cap.
_KEEP_LAST_N_CHECKPOINTS = 3


def _prune_checkpoint_family(
    ckpt_dir: Path, glob_pattern: str, protected: set[str], keep_last_n: int
) -> None:
    """Delete files matching glob_pattern beyond the newest keep_last_n, minus protected paths.

    Filenames in every pruned family embed the zero-padded global step, so
    lexicographic sort order is chronological order.
    """
    candidates = sorted(ckpt_dir.glob(glob_pattern))
    kept = {str(p) for p in candidates[-keep_last_n:]} | protected
    for old_path in candidates:
        if str(old_path) not in kept:
            old_path.unlink(missing_ok=True)


def clone_to_cpu(obj: Any) -> Any:
    """Recursively copy all tensors to CPU, detached from live training memory.

    The device-to-host copy must block. ``non_blocking=True`` issues a
    ``cudaMemcpyAsync`` and returns immediately, so any read of the destination
    races the DMA — and because the caching allocator recycles the pageable
    destination buffer, the read silently returns a *previous* tensor's bytes
    rather than anything obviously wrong. A blocking copy of a CUDA tensor
    already yields a fresh tensor, so only CPU tensors need an explicit clone.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to("cpu") if obj.is_cuda else obj.clone()
    if isinstance(obj, dict):
        return {key: clone_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_to_cpu(value) for value in obj)
    if isinstance(obj, (set, frozenset)):
        return type(obj)(clone_to_cpu(value) for value in obj)
    return obj


class CheckpointMixin:
    """Checkpoint behavior mixed into PPOTrainer to keep trainer state colocated."""

    def _save_roster_json(self) -> None:
        """Persist roster metadata alongside the run's checkpoints."""
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.roster.save_json(ckpt_dir / "roster.json")

    def _maybe_save_checkpoint(self, update: int) -> None:
        """Save the resumable checkpoint on schedule."""
        interval = self._schedule_state.checkpoint_interval
        if interval <= 0 or update % interval != 0:
            return
        self._save_checkpoint(update)
        self._save_roster_json()

    def _maybe_save_best_checkpoints(self, random_elo: float) -> None:
        """Overwrite the best-model checkpoints (live, then avg) when normalized ELO improves.

        Each family (live/avg) tracks its own high-water mark independently, so
        the two files can lag different updates — e.g. the avg policy may still
        be rising after the live policy has plateaued.
        """
        training_elo_norm = self._training_elo - random_elo
        if training_elo_norm > self._best_training_elo_norm:
            self._best_training_elo_norm = training_elo_norm
            self._save_best_checkpoint("best_training.pt")
        if self._avg_update_count > 0:
            avg_elo_norm = self._avg_training_elo - random_elo
            if avg_elo_norm > self._best_avg_elo_norm:
                self._best_avg_elo_norm = avg_elo_norm
                self._save_best_checkpoint(
                    "best_avg.pt", payload=self._avg_checkpoint_payload_lightweight(update=0)
                )

    # ------------------------------------------------------------------
    # ELO measurement ladder
    # ------------------------------------------------------------------

    def _save_ladder_snapshot(self) -> Path:
        """Synchronously save a policy-only ladder snapshot.

        Ladder files are small (no optimizer or averaging state) and are never
        pruned — the full ladder is kept for post-hoc analysis.
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ladder_step_{self._global_step:012d}.pt"
        payload = clone_to_cpu(
            {
                "policy_state_dict": self._policy_module.state_dict(),
                "team_pma_k": self._win_k,
                "num_value_components": self.wrapper.num_active_components,
                "global_step": self._global_step,
                "training_elo": self._training_elo,
            }
        )
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def _maybe_advance_ladder(self, update: int, elo_eval: EloEvaluator) -> None:
        """Advance the measurement ladder when the live rating crosses a milestone.

        Freezes the floating checkpoint at its settled (measured) rating,
        snapshots the live policy as the new floating checkpoint, and promotes
        both inside the continuous evaluator. Deferred while the floating
        checkpoint has fewer than ``min_games_to_freeze`` rated games.

        Milestones sit on an absolute grid — multiples of ``elo_milestone_gap``,
        so snapshots land near 200, 400, 600 ELO and so on. Measuring the gap
        from the previous snapshot's actual rating instead would let the grid
        ratchet upward: a snapshot deferred to 250 pushes the next to 450, and
        the drift compounds for the rest of the run, leaving the ladder's rungs
        at run-dependent heights that no two runs share.
        """
        if self._policy_gradient_coef <= 0.0 or self.cfg.elo_milestone_gap <= 0:
            return
        gap = self.cfg.elo_milestone_gap
        elo_norm = self._training_elo - self._random_elo()
        if elo_norm < self._elo_milestone + gap:
            return
        floating = self.roster.floating_checkpoint()
        if floating is not None and self._floating_games < self.cfg.elo_eval.min_games_to_freeze:
            return

        self.roster.freeze_floating()
        path = self._save_ladder_snapshot()
        entry = self.roster.add_checkpoint(
            str(path), self._global_step, update, initial_elo=self._training_elo
        )
        snapshot_policy = copy.deepcopy(self._policy_module).eval()
        snapshot_policy.requires_grad_(False)
        elo_eval.promote_floating(snapshot_policy, entry.label)
        self._floating_games = 0
        # Claim every grid point at or below the current rating, not just the one
        # that fired. A rating that jumps several gaps in one update takes a
        # single snapshot rather than queueing one per crossed point, and a dip
        # back below a claimed point cannot re-trigger on the way up.
        self._elo_milestone = elo_norm // gap * gap
        self._save_roster_json()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint_payload(self, update: int) -> dict:
        """Build the data dict shared by all checkpoint saves."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
            "num_value_components": self.wrapper.num_active_components,
            "optimizer_state_dict": self.optim.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "avg_policy_state_dict": self._avg_policy_module.state_dict(),
            # Left on device; the clone_to_cpu walk over this payload copies it.
            "avg_param_cumsum": list(self._avg_param_cumsum),
            "avg_update_count": self._avg_update_count,
            "update": update,
            "global_step": self._global_step,
            "ship_steps": self._ship_steps,
            "grad_tokens": self._grad_tokens,
            "elapsed_train_time": self._elapsed_train_time + (time.time() - self._train_start_time),
            "training_elo": self._training_elo,
            "avg_training_elo": self._avg_training_elo,
            "scripted_elo": self._scripted_elo,
            "floating_games": self._floating_games,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "eval_window_ladder": list(self._eval_window_ladder),
            "eval_window_floating": list(self._eval_window_floating),
            "eval_window_live_vs_avg": list(self._eval_window_live_vs_avg),
            "elo_milestone": self._elo_milestone,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _run_async_save(self, thread_attr: str, label: str, target: Callable[[], None]) -> None:
        """Spawn an async save thread, skipping if the previous save of this kind is still running.

        Args:
            thread_attr: Name of the instance attribute tracking this save kind's thread.
            label:       Describes the in-flight save in the skip warning.
            target:      The save function to run on the background thread.
        """
        active = getattr(self, thread_attr, None)
        if active is not None and active.is_alive():
            print(
                f"[PPOTrainer] Warning: Previous {label} is still in progress. "
                "Skipping this save to prevent disk/GIL congestion."
            )
            return
        thread = threading.Thread(target=target, daemon=True)
        setattr(self, thread_attr, thread)
        thread.start()

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file asynchronously.

        Written to cfg.checkpoint_dir/checkpoint_{update:06d}.pt.
        Directory is created if it does not exist.

        Args:
            update: Current update index (used as filename suffix).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"

        cpu_payload = clone_to_cpu(self.checkpoint_payload(update))

        avg_path = None
        avg_cpu_payload = None
        if self._avg_update_count > 0:
            avg_path = ckpt_dir / f"avg_step_{self._global_step:012d}.pt"
            avg_cpu_payload = clone_to_cpu(self._avg_checkpoint_payload(update))

        self._last_checkpoint_path = path

        def _async_save():
            # Write to a temp file then rename atomically so .exists() only
            # returns True once the file is complete (avoids partial-read crashes).
            tmp = path.with_suffix(".tmp")
            torch.save(cpu_payload, tmp)
            tmp.replace(path)
            print(f"Checkpoint saved asynchronously: {path}")

            if avg_cpu_payload is not None and avg_path is not None:
                tmp_avg = avg_path.with_suffix(".tmp")
                torch.save(avg_cpu_payload, tmp_avg)
                tmp_avg.replace(avg_path)
                print(f"Avg checkpoint saved asynchronously: {avg_path}")

            # Prune each family (live step_*.pt, avg avg_step_*.pt) down to the
            # newest _KEEP_LAST_N_CHECKPOINTS. best_*.pt and ladder_step_*.pt live
            # under different filename prefixes, so neither glob below ever
            # touches them regardless of this cap. roster.kept_paths() is unioned
            # in as defense in depth for any roster entry that ever names a path
            # in one of these two families — today ladder snapshots always use
            # the ladder_step_ prefix, so in practice this set never intersects
            # either glob.
            protected = self.roster.kept_paths()
            _prune_checkpoint_family(ckpt_dir, "step_*.pt", protected, _KEEP_LAST_N_CHECKPOINTS)
            _prune_checkpoint_family(ckpt_dir, "avg_step_*.pt", protected, _KEEP_LAST_N_CHECKPOINTS)

        self._run_async_save("_active_save_thread", "standard checkpoint saving", _async_save)

    def _avg_checkpoint_payload(self, update: int) -> dict:
        """Build checkpoint payload with avg_policy as the primary policy_state_dict.

        Allows best_avg.pt / avg_step_*.pt to be loaded by _load_checkpoint_agent
        in elo_stats.py, which reads ``ckpt["policy_state_dict"]``.
        """
        payload = self.checkpoint_payload(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _checkpoint_payload_lightweight(self, update: int) -> dict:
        """Build a best-model payload without heavy optimizer and average states."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
            "num_value_components": self.wrapper.num_active_components,
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "update": update,
            "global_step": self._global_step,
            "training_elo": self._training_elo,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "elo_milestone": self._elo_milestone,
            "team_pma_k": self._win_k,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _avg_checkpoint_payload_lightweight(self, update: int) -> dict:
        """Build a best-avg-model payload: lightweight payload with the avg policy's weights."""
        payload = self._checkpoint_payload_lightweight(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _save_best_checkpoint(self, name: str, payload: dict | None = None) -> None:
        """Save a named best-model checkpoint asynchronously, overwriting any previous version.

        Args:
            name:    Filename, e.g. "best_training.pt" or "best_avg.pt".
            payload: Custom payload dict; defaults to _checkpoint_payload_lightweight(update=0).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name

        raw_payload = (
            payload if payload is not None else self._checkpoint_payload_lightweight(update=0)
        )
        cpu_payload = clone_to_cpu(raw_payload)

        def _async_save():
            tmp = path.with_suffix(".tmp")
            torch.save(cpu_payload, tmp)
            tmp.replace(path)
            print(f"Best checkpoint saved asynchronously: {path}")

        self._run_async_save(
            "_active_best_thread", f"best checkpoint save for '{name}'", _async_save
        )

    def load_pretrained_weights(self, path: str) -> None:
        """Load policy and scaler from a pretrained checkpoint, discarding optimizer state.

        Use this when starting an RL run from a BC-pretrained policy. The optimizer
        is left in its freshly-initialised state so Adam calibrates to RL gradients
        from scratch — avoiding contamination from BC gradient statistics.

        The avg_policy is synced to the loaded weights so that if avg-model opponents
        are used, they start from the same pretrained base rather than random init.

        Args:
            path: Path to any .pt checkpoint (step_*.pt or best_*.pt).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._policy_module.load_state_dict(ckpt["policy_state_dict"])
        self._avg_policy_module.load_state_dict(ckpt["policy_state_dict"])
        # fp32 regardless of parameter dtype: this is a running sum over every
        # snapshot, so accumulating it in a narrower dtype lets each += round
        # away and the mean drifts without bound. Mirrors the fresh-init path.
        self._avg_param_cumsum = [
            torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for p in self._policy_module.parameters()
        ]
        self._avg_update_count = 0
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        print(f"Pretrained weights loaded from: {path} (optimizer state discarded)")

    def load_checkpoint(self, path: str) -> int:
        """Load policy and optimizer weights from a checkpoint file.

        Args:
            path: Path to a .pt checkpoint file.

        Returns:
            The update index stored in the checkpoint.

        Raises:
            ValueError: If the checkpoint was trained under a different paradigm —
                a policy trained in one paradigm misbehaves when resumed in the
                other (ego_pass policies only ever act as team 0).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_paradigm = ckpt.get("train_config", {}).get("paradigm")
        if ckpt_paradigm is not None and ckpt_paradigm != self.cfg.paradigm:
            raise ValueError(
                f"Checkpoint was trained with paradigm={ckpt_paradigm!r} but this "
                f"run uses paradigm={self.cfg.paradigm!r}. Resuming across "
                f"paradigms is not supported."
            )
        self._policy_module.load_state_dict(ckpt["policy_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        if "avg_policy_state_dict" in ckpt:
            self._avg_policy_module.load_state_dict(ckpt["avg_policy_state_dict"])
            self._avg_param_cumsum = [
                c.to(self.device, torch.float32) for c in ckpt["avg_param_cumsum"]
            ]
            self._avg_update_count = ckpt["avg_update_count"]
        if "training_elo" in ckpt:
            self._training_elo = ckpt["training_elo"]
            self._elo_milestone = ckpt.get("elo_milestone", 0.0)
        self._avg_training_elo = ckpt.get("avg_training_elo", 0.0)
        self._scripted_elo = ckpt.get("scripted_elo", self.cfg.elo_eval.scripted_elo_init)
        self._floating_games = ckpt.get("floating_games", 0)
        if "eval_window_rand" in ckpt:
            self._eval_window_rand = deque(
                ckpt["eval_window_rand"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_sc" in ckpt:
            self._eval_window_sc = deque(
                ckpt["eval_window_sc"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_ladder" in ckpt:
            self._eval_window_ladder = deque(
                ckpt["eval_window_ladder"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_floating" in ckpt:
            self._eval_window_floating = deque(
                ckpt["eval_window_floating"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_live_vs_avg" in ckpt:
            self._eval_window_live_vs_avg = deque(
                ckpt["eval_window_live_vs_avg"], maxlen=self.cfg.elo_eval.window_size
            )
        if "global_step" in ckpt:
            self._global_step = ckpt["global_step"]
            self._start_update = ckpt["update"] + 1
        self._elapsed_train_time = ckpt.get("elapsed_train_time", 0.0)
        # Older checkpoints lack ship_steps — reconstruct from update count,
        # exact as long as the scale config hasn't changed between runs.
        ship_tokens_per_update = self.cfg.num_steps * sum(
            sc.num_envs * sc.env_config.num_ships for sc in self.cfg.scales
        )
        self._ship_steps = ckpt.get("ship_steps", ckpt.get("update", 0) * ship_tokens_per_update)
        # Older checkpoints lack grad_tokens — reconstruct from the update count
        # and the current schedule's num_epochs (approximate: ignores target_kl
        # early stops and epoch-schedule changes before the checkpoint).
        self._grad_tokens = ckpt.get(
            "grad_tokens",
            ckpt.get("update", 0) * self._schedule_state.num_epochs * self._entity_tokens_per_epoch,
        )

        # Restore roster if its JSON exists alongside the checkpoint
        roster_path = Path(path).parent / "roster.json"
        if roster_path.exists():
            self.roster.load_json(roster_path)

        print(
            f"Checkpoint loaded from: {path} (resuming from update {self._start_update}, "
            f"step {self._global_step:,})"
        )
        return ckpt["update"]

    # ------------------------------------------------------------------
