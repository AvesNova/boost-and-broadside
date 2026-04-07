import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.train.evolve.trainer import (
    EvoTrainer,
    BoostAndBroadsideProblem,
)
from evotorch import SolutionBatch


@pytest.fixture
def mock_cfg():
    cfg = OmegaConf.create(
        {
            "project_name": "test_evolve",
            "mode": "train_evolve",
            "wandb": {"enabled": False},
            "environment": {
                "world_size": [1024, 1024],
                "max_ships": 8,
                "max_bullets": 20,
            },
            "model": {
                "max_ships": 8,
            },
            "train": {
                "evolve": {
                    "pop_size": 10,
                    "num_elites": 2,
                    "mutation_std": 0.05,
                    "num_generations": 1,
                    "max_episode_steps": 10,  # Short episode for testing
                }
            },
        }
    )
    return cfg


@pytest.fixture
def mock_env(mock_cfg):
    valid_keys = ShipConfig.__annotations__.keys()
    env_cfg = OmegaConf.to_container(mock_cfg.environment, resolve=True)
    ship_cfg_dict = {k: v for k, v in env_cfg.items() if k in valid_keys}
    ship_config = ShipConfig(**ship_cfg_dict)

    env = TensorEnv(
        num_envs=1,
        config=ship_config,
        device=torch.device("cpu"),  # use CPU for fast local tests
        max_ships=mock_cfg.model.max_ships,
        max_bullets=mock_cfg.environment.max_bullets,
        max_episode_steps=mock_cfg.train.evolve.max_episode_steps,
    )
    return env


def test_problem_evaluate_batch(mock_cfg, mock_env):
    problem = BoostAndBroadsideProblem(mock_cfg, mock_env)

    # Create a batch of solutions
    batch_size = 5
    solutions = problem.generate_batch(batch_size)
    assert solutions.values.shape == (batch_size, 96)

    # Evaluate
    problem.evaluate(solutions)

    # Check fitness is set
    assert solutions.evals.shape == (batch_size, 1)

    # Ensure ramp sorting happened on values
    pop = solutions.values.view(batch_size, 4, 24)
    for idx1, idx2 in problem.ramp_indices:
        assert torch.all(pop[..., idx1] <= pop[..., idx2])

    assert torch.all(pop >= 0.0)
    assert torch.all(pop <= 1.0)


def test_evo_trainer_train_step(mock_cfg, mock_env):
    trainer = EvoTrainer(mock_cfg, mock_env)
    # Just run 1 generation explicitly or let train() run
    trainer.train()
    # If no exceptions, it passes. EvoTorch manages the internal population tensor.
