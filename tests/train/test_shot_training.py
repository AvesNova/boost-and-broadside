"""The counterfactual shot heads: targets, buffer filing, and a training update."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env import shot_labels as labels
from boost_and_broadside.train.rl.buffer import RolloutBuffer
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.train.rl.shot_codes import (
    ALLY_HIT,
    ALLY_NEAR,
    CLEAR_MISS,
    ENEMY_HIT,
    ENEMY_NEAR,
    OUTCOME_CLASSES,
    SHOT_LABEL_DIM,
    TRAJECTORY_DIM,
    TRAJECTORY_GROUPS,
    ShotCodec,
)
from tests.train.test_ppo import _make_train_config

CONFIG = ShipConfig(dt=1.0 / 30.0)
CODEC = ShotCodec.from_ship_config(CONFIG)
DELAY = labels.shot_label_delay(CONFIG)


def _rows(hit: int, distance: float, side: int) -> torch.Tensor:
    rows = torch.zeros(SHOT_LABEL_DIM)
    rows[0] = 1.0
    example = rows[1:]
    example[labels.EX_HIT] = hit
    example[labels.EX_CLOSEST_DSQ] = distance**2
    example[labels.EX_CLOSEST_SIDE] = side
    example[labels.EX_TAU1] = 4
    example[labels.EX_TAU_FALLBACK] = 17
    example[labels.EX_EVENT_TAU] = 8
    example[labels.EX_SNAP1] = torch.tensor([60.0, 2.0, 480.0, 8.0, 1.0])
    example[labels.EX_SNAP_FALLBACK] = torch.tensor([260.0, -3.0, 470.0, -4.0, 0.7])
    example[labels.EX_SNAP_EVENT] = torch.tensor([130.0, 1.0, 475.0, 3.0, 1.4])
    return rows


class TestTargets:
    def test_hits_are_hard_and_take_the_encounter_age(self):
        for hit, index in ((labels.ENEMY, ENEMY_HIT), (labels.ALLY, ALLY_HIT)):
            target = CODEC.targets(_rows(hit, 3.0, hit))
            assert target.outcome[index] == 1.0
            assert target.ages.tolist() == [4.0, 8.0]

    def test_near_miss_spreads_on_its_own_side(self):
        enemy = CODEC.targets(_rows(labels.NO_HIT, 25.0, labels.ENEMY)).outcome
        ally = CODEC.targets(_rows(labels.NO_HIT, 25.0, labels.ALLY)).outcome
        assert enemy[ENEMY_NEAR].sum() > 0.99
        assert ally[ALLY_NEAR].sum() > 0.99
        # 25 px sits between the third and fourth geometric bins from 10 px.
        coordinate = math.log(2.5) / CODEC.near.spacing
        assert int(enemy[ENEMY_NEAR].argmax()) in (math.floor(coordinate), math.ceil(coordinate))
        assert enemy[ENEMY_NEAR].max() < 0.9  # HL-Gauss, not one-hot

    def test_far_miss_is_clear_alone_and_takes_the_fallback_age(self):
        target = CODEC.targets(_rows(labels.NO_HIT, 900.0, labels.ENEMY))
        assert target.outcome[CLEAR_MISS] == 1.0
        assert target.ages.tolist() == [4.0, 17.0]

    def test_nothing_in_range_is_clear(self):
        rows = _rows(labels.NO_HIT, 0.0, labels.NO_HIT)
        rows[1 + labels.EX_CLOSEST_DSQ] = labels.NO_ENCOUNTER_DSQ
        assert CODEC.targets(rows).outcome[CLEAR_MISS] == 1.0

    def test_every_target_is_a_distribution(self):
        target = CODEC.targets(torch.stack([_rows(0, d, 1) for d in (12.0, 40.0, 70.0, 77.0)]))
        assert torch.allclose(target.outcome.sum(-1), torch.ones(4))
        # One unit of probability per softmax group per query.
        assert torch.allclose(
            target.trajectory.sum(-1), torch.full((4, 2), float(TRAJECTORY_GROUPS))
        )


def test_buffer_resimulates_its_stored_launches():
    from boost_and_broadside.env.observation import YemongObservation

    T, B, N = 40, 1, 2
    buffer = RolloutBuffer(
        num_steps=T,
        num_envs=B,
        num_ships=N,
        num_components=1,
        obs_sample=YemongObservation(data={"pos": torch.zeros(B, N, 2)}),
        gamma=torch.ones(1),
        gae_lambda=torch.ones(1),
        device=torch.device("cpu"),
        shot_label_dim=SHOT_LABEL_DIM,
        num_fields=0,
    )
    # Ship 0 faces an enemy 100 px ahead on every row; the episode ends on row 5.
    launch = labels.ShotLaunchState(
        position=torch.tensor([[0j, 100 + 0j]], dtype=torch.complex64),
        velocity=torch.zeros(B, N, dtype=torch.complex64),
        attitude=torch.ones(B, N, dtype=torch.complex64),
        local_index=torch.ones(B, N),
        field_gradient=torch.zeros(B, N, dtype=torch.complex64),
        alive=torch.ones(B, N, dtype=torch.bool),
        team_id=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    empty = torch.zeros(B, 0)
    fields = labels.FieldLayout(empty.to(torch.complex64), empty, empty, empty)
    for row in range(T):
        buffer.ptr = row + 1  # as if ``add`` had just stored this row
        buffer.terminated[row] = row == 5
        buffer.add_shot_launch(launch, fields)
    buffer.resimulate_shots(dataclasses.replace(CONFIG, bullet_spread=0.0), None)

    valid = buffer.shot_labels[..., 0] > 0.5  # (T, B, N)
    rows = torch.arange(T)
    expected = (rows > 5) & (rows <= T - DELAY)
    assert torch.equal(valid[:, 0, 0], expected)
    hits = buffer.shot_labels[valid[:, 0, 0], 0, 0, 1 + labels.EX_HIT]
    assert (hits == labels.ENEMY).all()
    buffer.reset()
    assert buffer.shot_labels[..., 0].sum() == 0
    assert not buffer.shot_launch.alive.any()


def test_heads_answer_every_query():
    model = ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1)
    policy = build_policy(
        model, CONFIG, num_value_components=1, num_ships=4, global_value_k=(), predict_shots=True
    )
    x = torch.randn(3, 2, 4, 32)
    ages = torch.randint(0, DELAY + 1, (3, 2, 4, 2)).float()
    assert policy.shot_heads.trajectory(x, ages).shape == (3, 2, 4, 2, TRAJECTORY_DIM)
    assert policy.shot_heads.outcome(x).shape == (3, 2, 4, OUTCOME_CLASSES)
    headless = build_policy(model, CONFIG, num_value_components=1, num_ships=4, global_value_k=())
    assert headless.shot_heads is None
    assert not any(key.startswith("shot_heads.") for key in headless.state_dict())


def _shot_trainer(tmp_path, shot_labels_on: bool = True, coef: float = 0.1) -> PPOTrainer:
    config = dataclasses.replace(
        _make_train_config(
            checkpoint_dir=str(tmp_path),
            env_config=EnvConfig(
                num_ships=4, max_bullets=10, max_episode_steps=500, shot_labels=shot_labels_on
            ),
        ),
        num_steps=40,
        total_timesteps=4 * 40,
        shot_trajectory_coef=coef,
        shot_outcome_coef=coef,
    )
    torch.manual_seed(2)
    return PPOTrainer(
        train_config=config,
        model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        ship_config=CONFIG,
        device="cpu",
        use_wandb=False,
        scripted_agent=None,
        # Eager: compiling on CPU costs minutes and tests nothing here.
        compile_mode=None,
    )


def test_heads_and_labels_come_together(tmp_path):
    with pytest.raises(ValueError, match="shot"):
        _shot_trainer(tmp_path, shot_labels_on=False)
    with pytest.raises(ValueError, match="shot"):
        _shot_trainer(tmp_path, shot_labels_on=True, coef=0.0)


def test_a_rollout_files_labels_and_an_update_trains_the_heads(tmp_path):
    trainer = _shot_trainer(tmp_path)
    before = {
        name: value.detach().clone()
        for name, value in trainer._policy_module.shot_heads.named_parameters()
    }
    trainer.train()

    valid = trainer.buffer.shot_labels[..., 0] > 0.5
    completed_rows = trainer.buffer.num_steps - DELAY + 1
    assert valid[:completed_rows].any(), "no shot completed inside the rollout"
    assert not valid[completed_rows:].any(), "a row past the delay claimed a label"
    for name, value in trainer._policy_module.shot_heads.named_parameters():
        assert not torch.equal(value, before[name]), f"{name} never trained"
