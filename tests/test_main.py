"""Tests for the CLI mode dispatch in main.py.

These exercise the real argument dispatch and the real _apply_smoke, stubbing
only the heavy trainer construction/training so the test stays fast and
deterministic.
"""

import sys
from types import SimpleNamespace

import main
from runs.bc_warmstart import BC_WARMSTART_PRETRAIN_CONFIG, BC_WARMSTART_RL_CONFIG


class _StubTrainer:
    """Stand-in for PPOTrainer that records nothing but satisfies the flow."""

    run_name = "stub-run"

    def checkpoint_payload(self, update: int) -> dict:
        return {}

    def shutdown(self) -> None:
        pass

    def load_pretrained_weights(self, path: str) -> None:
        pass


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
