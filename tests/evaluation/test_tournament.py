"""Shared tournament allocation and outcome-scoring contracts."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.run_catalog import InvalidCheckpointError
from boost_and_broadside.evaluation.tournament import (
    Player,
    Tournament,
    effective_wins,
    load_ladder_policy,
    load_run_config,
)


def _tournament(num_envs: int = 6) -> Tournament:
    players = [
        Player("a", ResolvedAgent("random", None), None, None),
        Player("b", ResolvedAgent("random", None), None, None),
    ]
    env_config = EnvConfig(
        num_ships=8,
        max_bullets=4,
        max_episode_steps=8,
        num_fields=0,
    )
    return Tournament(players, ShipConfig(), env_config, "ego_pass", num_envs, "cpu")


def test_pair_allocation_balances_team_sides():
    tournament = _tournament()
    allocation = np.array([[0, 6], [6, 0]], dtype=np.int64)
    team0, team1 = tournament._assign(allocation)
    assignments = list(zip(team0.tolist(), team1.tolist()))
    assert assignments.count((0, 1)) == 3
    assert assignments.count((1, 0)) == 3


def test_half_win_scoring_is_symmetric_while_decisive_drops_ties():
    wins = np.array([[0.0, 7.0], [3.0, 0.0]])
    ties = np.array([[0.0, 4.0], [0.0, 0.0]])
    assert effective_wins(wins, ties, "half_win").tolist() == [[0.0, 9.0], [5.0, 0.0]]
    assert effective_wins(wins, ties, "decisive") is wins


def test_ladder_loader_rejects_checkpoint_content_from_another_step(monkeypatch):
    monkeypatch.setattr(
        "boost_and_broadside.evaluation.tournament.load_policy_bundle",
        lambda *args, **kwargs: SimpleNamespace(policy=object(), global_step=21),
    )

    with pytest.raises(InvalidCheckpointError, match="roster records 20"):
        load_ladder_policy(
            Path("ladder_step_000000000020.pt"),
            None,
            ShipConfig(),
            8,
            "cpu",
            expected_global_step=20,
        )


def _write_run_checkpoint(run_dir: Path, *, num_fields: int, field_map: dict | None) -> Path:
    """A minimal resumable payload -- only the fields load_run_config reads."""
    import torch

    from boost_and_broadside.config import ModelConfig
    from boost_and_broadside.train.rl.checkpoint_schema import OBSERVATION_SCHEMA

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "step_000000000064.pt"
    torch.save(
        {
            "observation_schema": OBSERVATION_SCHEMA,
            "env_config": dataclasses.asdict(
                EnvConfig(num_ships=8, max_bullets=4, max_episode_steps=8, num_fields=num_fields)
            ),
            "model_config": dataclasses.asdict(
                ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1)
            ),
            "train_config": {"paradigm": "ego_pass", "field_map": field_map},
        },
        path,
    )
    return path


def test_a_fields_run_reports_the_map_distribution_it_trained_on(tmp_path):
    """Rating a fields policy on a different map distribution rates a different task."""
    from boost_and_broadside.config import FieldMapConfig

    _write_run_checkpoint(
        tmp_path / "run",
        num_fields=4,
        field_map={"cache_size": 512, "max_generation_attempts": 256, "nesting_probability": 0.35},
    )

    env_config, _, paradigm, field_map = load_run_config(tmp_path / "run")

    assert env_config.num_fields == 4
    assert paradigm == "ego_pass"
    assert field_map == FieldMapConfig(
        cache_size=512, max_generation_attempts=256, nesting_probability=0.35
    )


def test_a_fields_run_without_recorded_map_intent_is_refused(tmp_path):
    """Silently evaluating on a default distribution would be a wrong measurement
    reported as a right one."""
    _write_run_checkpoint(tmp_path / "run", num_fields=4, field_map=None)

    with pytest.raises(InvalidCheckpointError, match="no field-map intent"):
        load_run_config(tmp_path / "run")


def test_a_field_free_run_reports_no_map(tmp_path):
    _write_run_checkpoint(tmp_path / "run", num_fields=0, field_map=None)

    assert load_run_config(tmp_path / "run")[3] is None
