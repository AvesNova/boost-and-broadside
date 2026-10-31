"""Toy learning: the whole loop (compiled rollout, CUDA graphs, PPO) improves a policy.

The stub MLP learns to stop dying against the chaser well within this budget; it does not
learn to hunt (that is the learning sections' job).
"""

import pytest
import torch

from bnb.contracts import GameConfig, Pool
from bnb.runtime.train import LEARNER, TrainConfig, Trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_learner_improves_against_the_chaser():
    torch.manual_seed(0)
    cfg = TrainConfig(game=GameConfig(2, 4, 1200.0), num_envs=256,
                      pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),))
    trainer = Trainer(cfg, torch.device("cuda"))
    rates = [trainer.update()["reward_per_s"]["chaser"] for _ in range(150)]
    early, late = sum(rates[:10]) / 10, sum(rates[-20:]) / 20
    assert late > early + 0.01, (early, late)
