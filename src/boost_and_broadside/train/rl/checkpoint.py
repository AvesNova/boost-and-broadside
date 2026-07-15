"""Checkpoint serialization, asynchronous saves, and resume support."""

import dataclasses
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def cast_norms_bf16(module: nn.Module) -> None:
    """Cast CUDA RMSNorm weights to bf16 for the fused policy path."""
    for submodule in module.modules():
        if isinstance(submodule, nn.RMSNorm) and submodule.weight.is_cuda:
            submodule.weight.data = submodule.weight.data.bfloat16()


def clone_to_cpu(obj: Any) -> Any:
    """Recursively copy all tensors to CPU and clone them non-blockingly."""
    if isinstance(obj, torch.Tensor):
        return obj.to("cpu", non_blocking=True).clone()
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
        """Save on schedule and add milestone checkpoints to the league roster."""
        interval = self._schedule_state.checkpoint_interval
        if interval <= 0 or update % interval != 0:
            return
        self._save_checkpoint(update)
        training_elo_norm = self._training_elo - self._random_elo()
        if (
            self._policy_gradient_coef > 0.0
            and self._last_checkpoint_path is not None
            and self._last_checkpoint_path.exists()
            and self.cfg.elo_milestone_gap > 0
            and training_elo_norm - self._elo_milestone >= self.cfg.elo_milestone_gap
        ):
            self.roster.add_checkpoint(
                str(self._last_checkpoint_path),
                self._global_step,
                update,
                initial_elo=self._training_elo,
            )
            self._elo_milestone = training_elo_norm
            self._save_roster_json()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint_payload(self, update: int) -> dict:
        """Build the data dict shared by all checkpoint saves."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "avg_policy_state_dict": self._avg_policy_module.state_dict(),
            "avg_param_cumsum": [c.to("cpu", non_blocking=True) for c in self._avg_param_cumsum],
            "avg_update_count": self._avg_update_count,
            "update": update,
            "global_step": self._global_step,
            "ship_steps": self._ship_steps,
            "grad_tokens": self._grad_tokens,
            "elapsed_train_time": self._elapsed_train_time + (time.time() - self._train_start_time),
            "training_elo": self._training_elo,
            "avg_training_elo": self._avg_training_elo,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "eval_window_avg_vs_sc": list(self._eval_window_avg_vs_sc),
            "eval_window_live_vs_avg": list(self._eval_window_live_vs_avg),
            "elo_milestone": self._elo_milestone,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
            "model_config": dataclasses.asdict(self.model_config),
            "env_config": dataclasses.asdict(self.env_config),
        }

    def _save_checkpoint(self, update: int) -> None:
        """Save policy and optimizer state to a .pt file asynchronously.

        Written to cfg.checkpoint_dir/checkpoint_{update:06d}.pt.
        Directory is created if it does not exist.

        Args:
            update: Current update index (used as filename suffix).
        """
        # Check if the previous standard saving thread is still running
        if (
            hasattr(self, "_active_save_thread")
            and self._active_save_thread is not None
            and self._active_save_thread.is_alive()
        ):
            print(
                "[PPOTrainer] Warning: Previous standard checkpoint saving is still in "
                "progress. Skipping this save to prevent disk/GIL congestion."
            )
            return

        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self._global_step:012d}.pt"

        cpu_payload = clone_to_cpu(self.checkpoint_payload(update))

        avg_path = None
        avg_cpu_payload = None
        if self._avg_update_count > 0:
            avg_path = ckpt_dir / "recent_avg.pt"
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
                print(f"Recent avg checkpoint saved asynchronously: {avg_path}")

            # Prune: keep only the latest checkpoint + all roster-referenced files.
            # best_*.pt files are not touched (they don't match the step_*.pt glob).
            kept = self.roster.kept_paths()
            kept.add(str(path))
            for old_path in ckpt_dir.glob("step_*.pt"):
                if str(old_path) not in kept:
                    old_path.unlink(missing_ok=True)

        self._active_save_thread = threading.Thread(target=_async_save, daemon=True)
        self._active_save_thread.start()

    def _avg_checkpoint_payload(self, update: int) -> dict:
        """Build checkpoint payload with avg_policy as the primary policy_state_dict.

        Allows best_avg.pt / recent_avg.pt to be loaded by _load_checkpoint_agent
        in elo_stats.py, which reads ``ckpt["policy_state_dict"]``.
        """
        payload = self.checkpoint_payload(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _checkpoint_payload_lightweight(self, update: int) -> dict:
        """Build a best-model payload without heavy optimizer and average states."""
        return {
            "policy_state_dict": self._policy_module.state_dict(),
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

    def _save_best_checkpoint(self, name: str, payload: dict | None = None) -> None:
        """Save a named best-model checkpoint asynchronously, overwriting any previous version.

        Args:
            name:    Filename, e.g. "best_training.pt" or "best_avg.pt".
            payload: Custom payload dict; defaults to _checkpoint_payload_lightweight(update=0).
        """
        # Check if the previous best saving thread is still running
        if (
            hasattr(self, "_active_best_thread")
            and self._active_best_thread is not None
            and self._active_best_thread.is_alive()
        ):
            print(
                f"[PPOTrainer] Warning: Previous best checkpoint save for '{name}' is still "
                "in progress. Skipping this save to prevent disk/GIL congestion."
            )
            return

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

        self._active_best_thread = threading.Thread(target=_async_save, daemon=True)
        self._active_best_thread.start()

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
        cast_norms_bf16(self._policy_module)
        cast_norms_bf16(self._avg_policy_module)
        self._avg_param_cumsum = [torch.zeros_like(p) for p in self._policy_module.parameters()]
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
        cast_norms_bf16(self._policy_module)
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "adv_scaler_state_dict" in ckpt:
            self.adv_scaler.load_state_dict(ckpt["adv_scaler_state_dict"])
        if "avg_policy_state_dict" in ckpt:
            self._avg_policy_module.load_state_dict(ckpt["avg_policy_state_dict"])
            cast_norms_bf16(self._avg_policy_module)
            self._avg_param_cumsum = [c.to(self.device) for c in ckpt["avg_param_cumsum"]]
            self._avg_update_count = ckpt["avg_update_count"]
        if "training_elo" in ckpt:
            self._training_elo = ckpt["training_elo"]
            self._elo_milestone = ckpt.get("elo_milestone", 0.0)
        self._avg_training_elo = ckpt.get("avg_training_elo", 0.0)
        if "eval_window_rand" in ckpt:
            self._eval_window_rand = deque(
                ckpt["eval_window_rand"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_sc" in ckpt:
            self._eval_window_sc = deque(
                ckpt["eval_window_sc"], maxlen=self.cfg.elo_eval.window_size
            )
        if "eval_window_avg_vs_sc" in ckpt:
            self._eval_window_avg_vs_sc = deque(
                ckpt["eval_window_avg_vs_sc"], maxlen=self.cfg.elo_eval.window_size
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
