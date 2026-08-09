"""Checkpoint serialization, asynchronous saves, ladder milestones, and resume support."""

import copy
import dataclasses
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from boost_and_broadside.train.rl.checkpoint_schema import (
    OBSERVATION_SCHEMA,
    load_checkpoint_payload,
    require_observation_schema,
)
from boost_and_broadside.train.rl.elo_eval import EloEvaluator

# Rolling window of full-resume (step_*.pt) and avg (avg_step_*.pt) checkpoints
# kept per run. Ladder snapshots (ladder_step_*.pt) and named best checkpoints
# (best_*.pt) use separate filename families and are never subject to this cap.
_KEEP_LAST_N_CHECKPOINTS = 3


def build_policy_checkpoint_payload(
    *,
    policy_state_dict: Mapping[str, Any],
    num_value_components: int,
    team_pma_k: tuple[int, ...],
    global_step: int,
    training_elo: float,
    model_config: Any,
    env_config: Any,
    ship_config: Any,
    paradigm: str,
    resolved_config: Mapping[str, object] | None = None,
    launch: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the current policy-provenance block without constructing a trainer.

    Every production checkpoint family starts with this block. Keeping its
    construction pure also lets hermetic tools create a randomly initialized,
    current-schema policy fixture without fabricating optimizer state or
    allocating the full training engine.
    """

    payload: dict[str, Any] = {
        "observation_schema": OBSERVATION_SCHEMA,
        "policy_state_dict": policy_state_dict,
        "num_value_components": num_value_components,
        "team_pma_k": team_pma_k,
        "global_step": global_step,
        "training_elo": training_elo,
        "model_config": dataclasses.asdict(model_config),
        "env_config": dataclasses.asdict(env_config),
        "ship_config": dataclasses.asdict(ship_config),
        "paradigm": paradigm,
    }
    if resolved_config is not None:
        payload["resolved_config"] = copy.deepcopy(dict(resolved_config))
    if launch is not None:
        payload["launch"] = copy.deepcopy(dict(launch))
    return payload


def write_checkpoint_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically serialize one already-snapshotted checkpoint payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)
    return destination


def build_training_checkpoint_payload(
    *,
    policy_payload: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    scaler_state_dict: Mapping[str, Any],
    adv_scaler_state_dict: Mapping[str, Any],
    avg_policy_state_dict: Mapping[str, Any],
    avg_param_cumsum: list[torch.Tensor],
    avg_update_count: int,
    update: int,
    ship_steps: int,
    grad_tokens: int,
    elapsed_train_time: float,
    avg_training_elo: float,
    scripted_elo: float,
    floating_games: int,
    eval_window_rand: list[Any],
    eval_window_sc: list[Any],
    eval_window_ladder: list[Any],
    eval_window_floating: list[Any],
    eval_window_live_vs_avg: list[Any],
    elo_milestone: float,
    train_config: Any,
) -> dict[str, Any]:
    """Build the complete resumable ``step_*.pt`` payload as plain data."""

    return {
        **policy_payload,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "adv_scaler_state_dict": adv_scaler_state_dict,
        "avg_policy_state_dict": avg_policy_state_dict,
        "avg_param_cumsum": avg_param_cumsum,
        "avg_update_count": avg_update_count,
        "update": update,
        "ship_steps": ship_steps,
        "grad_tokens": grad_tokens,
        "elapsed_train_time": elapsed_train_time,
        "avg_training_elo": avg_training_elo,
        "scripted_elo": scripted_elo,
        "floating_games": floating_games,
        "eval_window_rand": eval_window_rand,
        "eval_window_sc": eval_window_sc,
        "eval_window_ladder": eval_window_ladder,
        "eval_window_floating": eval_window_floating,
        "eval_window_live_vs_avg": eval_window_live_vs_avg,
        "elo_milestone": elo_milestone,
        "train_config": {
            key: value
            for key, value in dataclasses.asdict(train_config).items()
            if key != "schedule"
        },
    }


def _check_resolved_config_provenance(
    checkpoint: Mapping[str, Any],
    current: Mapping[str, object] | None,
    *,
    allow_config_drift: bool,
) -> None:
    """Reject a resume whose complete recorded launch config changed."""
    recorded = checkpoint.get("resolved_config")
    if current is None or not isinstance(recorded, Mapping):
        return
    recorded_fingerprint = recorded.get("resolved_config_fingerprint")
    current_fingerprint = current.get("resolved_config_fingerprint")
    if recorded_fingerprint == current_fingerprint:
        return
    message = (
        "Checkpoint resolved configuration does not match this launch: "
        f"recorded={recorded_fingerprint!r}, current={current_fingerprint!r}"
    )
    if not allow_config_drift:
        raise ValueError(f"{message}. Pass --allow-config-drift to override explicitly.")
    warnings.warn(f"{message}; continuing because config drift is allowed", stacklevel=2)


def _load_checkpoint_state(
    target: Any, state: Mapping[str, Any], path: str, component: str
) -> None:
    """Normalize state incompatibility at the checkpoint input boundary."""

    if not isinstance(state, Mapping):
        raise ValueError(
            f"checkpoint {path!r} has invalid {component}: expected a mapping, "
            f"got {type(state).__name__}"
        )
    try:
        target.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError(f"checkpoint {path!r} has incompatible {component}: {error}") from None


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
    if isinstance(obj, Mapping):
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

    def _wait_for_checkpoint_saves(self) -> None:
        """Finish every in-flight writer before trainer shutdown returns."""

        for attr in (
            "_active_save_thread",
            "_active_best_thread",
            "_active_best_avg_thread",
        ):
            thread = getattr(self, attr, None)
            if thread is not None and thread.is_alive():
                thread.join()

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

    def _maybe_save_best_checkpoints(self) -> None:
        """Overwrite the best-model checkpoints (live, then avg) when the rating improves.

        Each family (live/avg) tracks its own high-water mark independently, so
        the two files can lag different updates — e.g. the avg policy may still
        be rising after the live policy has plateaued. The mark is advanced only
        when the save is actually dispatched: a save skipped for an in-flight
        write must be retried on the next improving update, not silently recorded
        as captured. The two families use separate thread slots so they never
        contend within a single update.
        """
        if self._training_elo > self._best_training_elo_norm and self._save_best_checkpoint(
            "best_training.pt"
        ):
            self._best_training_elo_norm = self._training_elo
        if self._avg_update_count > 0:
            avg_elo_norm = self._avg_training_elo
            if avg_elo_norm > self._best_avg_elo_norm and self._save_best_checkpoint(
                "best_avg.pt",
                payload=self._avg_checkpoint_payload_lightweight(update=0),
                thread_attr="_active_best_avg_thread",
            ):
                self._best_avg_elo_norm = avg_elo_norm

    # ------------------------------------------------------------------
    # Elo measurement ladder
    # ------------------------------------------------------------------

    def _save_ladder_snapshot(self) -> Path:
        """Synchronously save a policy-only ladder snapshot.

        Ladder files are small (no optimizer or averaging state) and are never
        pruned — the full ladder is kept for post-hoc analysis.
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ladder_step_{self._global_step:012d}.pt"
        payload = clone_to_cpu(self._provenance())
        write_checkpoint_payload(path, payload)
        return path

    def _maybe_advance_ladder(self, update: int, elo_eval: EloEvaluator) -> None:
        """Advance the measurement ladder when the live rating crosses a milestone.

        Freezes the floating checkpoint at its settled (measured) rating,
        snapshots the live policy as the new floating checkpoint, and promotes
        both inside the continuous evaluator. Deferred while the floating
        checkpoint has fewer than ``min_games_to_freeze`` rated games.

        Milestones sit on an absolute grid — multiples of ``elo_milestone_gap``,
        so snapshots land near 200, 400, 600 Elo and so on. Measuring the gap
        from the previous snapshot's actual rating instead would let the grid
        ratchet upward: a snapshot deferred to 250 pushes the next to 450, and
        the drift compounds for the rest of the run, leaving the ladder's rungs
        at run-dependent heights that no two runs share.
        """
        if self._policy_gradient_coef <= 0.0 or self.cfg.elo_milestone_gap <= 0:
            return
        gap = self.cfg.elo_milestone_gap
        # Grid points are absolute on the scripted=1000 gauge, so snapshots
        # from different runs land at comparable heights.
        elo_norm = self._training_elo
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

    def _provenance(self) -> dict:
        """Everything a loader needs to rebuild this policy as it exists now.

        Every payload family carries this block, including the ladder snapshots —
        those are the files the league and the post-hoc calibrator actually reload,
        so they are the ones that most need to say what they are. Without the
        configs, a loader has no choice but to assume the reader's own, and an
        assumed physics constant is indistinguishable from a correct one until the
        policy quietly underperforms its rating.
        """
        return build_policy_checkpoint_payload(
            policy_state_dict=self._policy_module.state_dict(),
            num_value_components=self.wrapper.num_active_components,
            team_pma_k=self._win_k,
            global_step=self._global_step,
            training_elo=self._training_elo,
            model_config=self.model_config,
            env_config=self.env_config,
            ship_config=self.ship_config,
            # An ego_pass policy only ever acted as team 0, so whoever replays it
            # has to know to hand it the mirrored view when it plays team 1.
            paradigm=self.cfg.paradigm,
            resolved_config=self.resolved_config_document,
            launch=self.launch_provenance,
        )

    def checkpoint_payload(self, update: int) -> dict:
        """Build the data dict shared by all checkpoint saves."""
        return build_training_checkpoint_payload(
            policy_payload=self._provenance(),
            optimizer_state_dict=self.optim.state_dict(),
            scaler_state_dict=self.scaler.state_dict(),
            adv_scaler_state_dict=self.adv_scaler.state_dict(),
            avg_policy_state_dict=self._avg_policy_module.state_dict(),
            # Left on device; the clone_to_cpu walk over this payload copies it.
            avg_param_cumsum=list(self._avg_param_cumsum),
            avg_update_count=self._avg_update_count,
            update=update,
            ship_steps=self._ship_steps,
            grad_tokens=self._grad_tokens,
            elapsed_train_time=self._elapsed_train_time
            + (time.time() - self._train_start_time),
            avg_training_elo=self._avg_training_elo,
            scripted_elo=self._scripted_elo,
            floating_games=self._floating_games,
            eval_window_rand=list(self._eval_window_rand),
            eval_window_sc=list(self._eval_window_sc),
            eval_window_ladder=list(self._eval_window_ladder),
            eval_window_floating=list(self._eval_window_floating),
            eval_window_live_vs_avg=list(self._eval_window_live_vs_avg),
            elo_milestone=self._elo_milestone,
            train_config=self.cfg,
        )

    def _run_async_save(self, thread_attr: str, label: str, target: Callable[[], None]) -> bool:
        """Spawn an async save thread, skipping if the previous save of this kind is still running.

        Args:
            thread_attr: Name of the instance attribute tracking this save kind's thread.
            label:       Describes the in-flight save in the skip warning.
            target:      The save function to run on the background thread.

        Returns:
            True if the save was dispatched, False if it was skipped because the
            previous save on this thread slot is still running. Callers that gate a
            high-water mark on the save must only advance it when this returns True,
            or a skipped save would raise the bar without persisting the checkpoint.
        """
        active = getattr(self, thread_attr, None)
        if active is not None and active.is_alive():
            print(
                f"[PPOTrainer] Warning: Previous {label} is still in progress. "
                "Skipping this save to prevent disk/GIL congestion."
            )
            return False
        thread = threading.Thread(target=target, daemon=True)
        setattr(self, thread_attr, thread)
        thread.start()
        return True

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
            write_checkpoint_payload(path, cpu_payload)
            print(f"Checkpoint saved asynchronously: {path}")

            if avg_cpu_payload is not None and avg_path is not None:
                write_checkpoint_payload(avg_path, avg_cpu_payload)
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
            **self._provenance(),
            "scaler_state_dict": self.scaler.state_dict(),
            "adv_scaler_state_dict": self.adv_scaler.state_dict(),
            "update": update,
            "eval_window_rand": list(self._eval_window_rand),
            "eval_window_sc": list(self._eval_window_sc),
            "elo_milestone": self._elo_milestone,
            "train_config": {
                k: v for k, v in dataclasses.asdict(self.cfg).items() if k != "schedule"
            },
        }

    def _avg_checkpoint_payload_lightweight(self, update: int) -> dict:
        """Build a best-avg-model payload: lightweight payload with the avg policy's weights."""
        payload = self._checkpoint_payload_lightweight(update)
        payload["policy_state_dict"] = self._avg_policy_module.state_dict()
        return payload

    def _save_best_checkpoint(
        self, name: str, payload: dict | None = None, thread_attr: str = "_active_best_thread"
    ) -> bool:
        """Save a named best-model checkpoint asynchronously, overwriting any previous version.

        Args:
            name:        Filename, e.g. "best_training.pt" or "best_avg.pt".
            payload:     Custom payload dict; defaults to _checkpoint_payload_lightweight(update=0).
            thread_attr: Instance attribute tracking this save kind's thread. Each best-model
                family gets its own slot so a best_training save in flight cannot cause a
                same-update best_avg save to be skipped (and vice versa).

        Returns:
            True if the save was dispatched, False if skipped (previous same-family
            save still running).
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name

        raw_payload = (
            payload if payload is not None else self._checkpoint_payload_lightweight(update=0)
        )
        cpu_payload = clone_to_cpu(raw_payload)

        def _async_save():
            write_checkpoint_payload(path, cpu_payload)
            print(f"Best checkpoint saved asynchronously: {path}")

        return self._run_async_save(thread_attr, f"best checkpoint save for '{name}'", _async_save)

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
        ckpt = load_checkpoint_payload(path, map_location=self.device)
        require_observation_schema(ckpt, path)
        _load_checkpoint_state(
            self._policy_module, ckpt["policy_state_dict"], path, "policy weights"
        )
        _load_checkpoint_state(
            self._avg_policy_module, ckpt["policy_state_dict"], path, "averaged policy weights"
        )
        # fp32 regardless of parameter dtype: this is a running sum over every
        # snapshot, so accumulating it in a narrower dtype lets each += round
        # away and the mean drifts without bound. Mirrors the fresh-init path.
        self._avg_param_cumsum = [
            torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for p in self._policy_module.parameters()
        ]
        self._avg_update_count = 0
        if "scaler_state_dict" in ckpt:
            _load_checkpoint_state(self.scaler, ckpt["scaler_state_dict"], path, "scaler state")
        if "adv_scaler_state_dict" in ckpt:
            _load_checkpoint_state(
                self.adv_scaler, ckpt["adv_scaler_state_dict"], path, "advantage-scaler state"
            )
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
        ckpt = load_checkpoint_payload(path, map_location=self.device)
        require_observation_schema(ckpt, path)
        _check_resolved_config_provenance(
            ckpt,
            self.resolved_config_document,
            allow_config_drift=bool(
                self.launch_provenance and self.launch_provenance.get("allow_config_drift")
            ),
        )
        ckpt_paradigm = ckpt.get("train_config", {}).get("paradigm")
        if ckpt_paradigm is not None and ckpt_paradigm != self.cfg.paradigm:
            raise ValueError(
                f"Checkpoint was trained with paradigm={ckpt_paradigm!r} but this "
                f"run uses paradigm={self.cfg.paradigm!r}. Resuming across "
                f"paradigms is not supported."
            )
        _load_checkpoint_state(
            self._policy_module, ckpt["policy_state_dict"], path, "policy weights"
        )
        _load_checkpoint_state(self.optim, ckpt["optimizer_state_dict"], path, "optimizer state")
        if "scaler_state_dict" in ckpt:
            _load_checkpoint_state(self.scaler, ckpt["scaler_state_dict"], path, "scaler state")
        if "adv_scaler_state_dict" in ckpt:
            _load_checkpoint_state(
                self.adv_scaler, ckpt["adv_scaler_state_dict"], path, "advantage-scaler state"
            )
        if "avg_policy_state_dict" in ckpt:
            _load_checkpoint_state(
                self._avg_policy_module,
                ckpt["avg_policy_state_dict"],
                path,
                "averaged policy weights",
            )
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

        # Restore roster if its JSON exists alongside the checkpoint. load_json
        # replaces the entry list wholesale, so the special entries are
        # re-registered afterwards — a roster written before the scripted agent
        # was a league entry would otherwise resume without it.
        roster_path = Path(path).parent / "roster.json"
        if roster_path.exists():
            self.roster.load_json(roster_path)
            self._register_special_opponents()

        print(
            f"Checkpoint loaded from: {path} (resuming from update {self._start_update}, "
            f"step {self._global_step:,})"
        )
        return ckpt["update"]

    # ------------------------------------------------------------------
