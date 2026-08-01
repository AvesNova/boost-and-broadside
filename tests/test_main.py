"""Tests for the CLI mode dispatch in main.py.

These exercise the real argument dispatch and the real _apply_smoke, stubbing
only the heavy trainer construction/training so the test stays fast and
deterministic.
"""

import sys
from types import SimpleNamespace

import pytest

import main
from runs.bc_warmstart import BC_WARMSTART_PRETRAIN_CONFIG, BC_WARMSTART_RL_CONFIG
from runs.rl import RL_TRAIN_CONFIG


class _StubTrainer:
    """Stand-in for PPOTrainer that records nothing but satisfies the flow."""

    run_name = "stub-run"

    def __init__(self) -> None:
        self.loaded_checkpoint: str | None = None
        self.loaded_pretrained: str | None = None

    def checkpoint_payload(self, update: int) -> dict:
        return {}

    def shutdown(self) -> None:
        pass

    def train(self) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        self.loaded_checkpoint = path

    def load_pretrained_weights(self, path: str) -> None:
        self.loaded_pretrained = path


def _run_bc_warmstart(monkeypatch, tmp_path, smoke: bool) -> list[SimpleNamespace]:
    """Invoke `main()` in bc_warmstart mode, capturing every _make_trainer call."""
    captured: list[SimpleNamespace] = []

    def fake_make_trainer(train_config, args, *, resume_wandb_run_id=None):
        captured.append(SimpleNamespace(config=train_config, args=args))
        return _StubTrainer()

    argv = ["main.py", "--mode", "bc_warmstart"]
    if smoke:
        argv.append("--smoke")

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(main, "_make_trainer", fake_make_trainer)
    monkeypatch.setattr(main, "_run_trainer", lambda trainer: None)
    # Stub the checkpoint write so no real file is produced (paths are still built).
    monkeypatch.setattr(main.torch, "save", lambda payload, path: None)

    main.main()
    return captured


class TestTrainingModeDispatch:
    """`rl` and `rl_fields` share `_run_training_mode`, so resume and
    pretrain handling must behave identically for both modes."""

    @pytest.mark.parametrize("mode", ["rl", "rl_fields"])
    def test_resume_flag_loads_checkpoint(self, monkeypatch, mode):
        captured: list[_StubTrainer] = []

        def fake_make_trainer(train_config, args, *, resume_wandb_run_id=None):
            trainer = _StubTrainer()
            captured.append(trainer)
            return trainer

        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", mode, "--resume", "ckpt.pt"])
        monkeypatch.setattr(main, "_make_trainer", fake_make_trainer)
        monkeypatch.setattr(main, "_find_resume_checkpoint", lambda hint: ("ckpt.pt", None))

        main.main()

        assert captured[0].loaded_checkpoint == "ckpt.pt"
        assert captured[0].loaded_pretrained is None

    @pytest.mark.parametrize("mode", ["rl", "rl_fields"])
    def test_pretrain_from_flag_loads_pretrained_weights(self, monkeypatch, mode):
        captured: list[_StubTrainer] = []

        def fake_make_trainer(train_config, args, *, resume_wandb_run_id=None):
            trainer = _StubTrainer()
            captured.append(trainer)
            return trainer

        monkeypatch.setattr(
            sys, "argv", ["main.py", "--mode", mode, "--pretrain_from", "pretrained.pt"]
        )
        monkeypatch.setattr(main, "_make_trainer", fake_make_trainer)

        main.main()

        assert captured[0].loaded_pretrained == "pretrained.pt"
        assert captured[0].loaded_checkpoint is None

    def test_no_wandb_flag_threads_through_without_shrinking(self, monkeypatch):
        """--no-wandb must reach _make_trainer (which derives use_wandb from it)
        while leaving the full config untouched — unlike --smoke."""
        captured: list[SimpleNamespace] = []

        def fake_make_trainer(train_config, args, *, resume_wandb_run_id=None):
            captured.append(SimpleNamespace(config=train_config, args=args))
            return _StubTrainer()

        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "rl", "--no-wandb"])
        monkeypatch.setattr(main, "_make_trainer", fake_make_trainer)

        main.main()

        assert captured[0].args.no_wandb is True
        assert captured[0].args.smoke is False
        assert captured[0].config is RL_TRAIN_CONFIG


def test_play_mode_dispatches_fixed_interactive_preset(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "play"])
    monkeypatch.setattr(main, "run_play_mode", lambda **kwargs: captured.update(kwargs))

    main.main()

    assert captured["ship_config"] is main.SHIP_CONFIG
    assert captured["rewards"] is main.REWARDS
    assert captured["model_config"] is main.MODEL_CONFIG
    assert captured["checkpoint_dir"] == "checkpoints"


class TestBcWarmstartSmoke:
    def test_smoke_shrinks_both_stages(self, monkeypatch, tmp_path):
        """--smoke must reach both warmstart stages (AUDIT-001): each config is
        shrunk to the crash-test size instead of the full multi-day run."""
        captured = _run_bc_warmstart(monkeypatch, tmp_path, smoke=True)

        assert len(captured) == 2
        pretrain_cfg, rl_cfg = captured[0].config, captured[1].config
        assert pretrain_cfg.total_timesteps == 5_000
        assert rl_cfg.total_timesteps == 5_000
        assert pretrain_cfg.num_minibatches == 1
        assert rl_cfg.num_minibatches == 1

    def test_smoke_flag_reaches_trainer_args(self, monkeypatch, tmp_path):
        """The real --smoke value must thread through to _make_trainer, which
        derives use_wandb/compile from it (an earlier version forced smoke=False)."""
        captured = _run_bc_warmstart(monkeypatch, tmp_path, smoke=True)

        assert all(call.args.smoke is True for call in captured)

    def test_without_smoke_uses_full_configs(self, monkeypatch, tmp_path):
        """Absent --smoke, the full production stage configs pass through unchanged."""
        captured = _run_bc_warmstart(monkeypatch, tmp_path, smoke=False)

        assert captured[0].config is BC_WARMSTART_PRETRAIN_CONFIG
        assert captured[1].config is BC_WARMSTART_RL_CONFIG
        assert all(call.args.smoke is False for call in captured)
