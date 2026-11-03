"""Toy learning (learner spec 9): the whole loop (compiled rollout, CUDA graphs, the learner)
trains the relational model from scratch to beat the chaser at 2v2 team deathmatch.

The budget is fixed: 400 updates of 256 environments x 128 ticks (13.1M environment steps),
about three minutes on the dev GPU (learner-notes.md has the measured curve).
"""

import pytest
import torch

from bnb.contracts import GameConfig, Pool
from bnb.runtime.train import LEARNER, TrainConfig, Trainer, win_rate

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
TOY = TrainConfig(
    game=GameConfig(2, 4, 1200.0),
    environments=256,
    model="relational",
    pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),),
)
UPDATES = 400


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_toy_learner_beats_the_chaser():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer = Trainer(TOY, torch.device("cuda"))
        for _ in range(UPDATES):
            trainer.update()
        result = win_rate(trainer, games=256)
    assert result["wins"] >= 0.6, result
