"""The toy BC test (scripted spec B6.5): the toy model (the relational MLP) cloned from the
scripted agent at 2v2 team deathmatch scores at least 0.40 against its teacher over 512
games (within two standard errors, see SCORE_BAR), and its score against the chaser is
within 0.10 of the teacher's.

The labels are at the default floor, and the teacher is the one they describe: BC's
scripted seat (`bnb.learn.bc.TeacherOpponent`), the same opponent the stop rule scores
against. BC runs to its score stop (the live score over 1000 games, less two standard
errors, reaches 0.40) within BUDGET environment steps of 256 environments x 128 ticks
(the section notes, "Stop rule (2026-11-02)", have the curve).
"""

import pytest
import torch

from bnb.contracts import GameConfig, Pool
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.learn import bc
from bnb.runtime import cloning
from bnb.runtime.evaluate import first_results, score
from bnb.runtime.train import LEARNER, SCRIPTED, TrainConfig, Trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
GAME = GameConfig(2, 4, 1200.0)
BUDGET = 100_000_000
GAMES = 512
# "At least 0.40" is read as not significantly below it: two standard errors of a GAMES-game
# score (at most 0.5 / sqrt(GAMES), about 0.022).
SCORE_BAR = bc.StopConfig().score_target - 2 * 0.5 / GAMES**0.5
TOY = TrainConfig(
    game=GAME,
    pools=cloning.LINEUP,
    environments=256,
    model="relational",
    learner=bc.config(),
    objective="bc",
    stop=bc.StopConfig(budget_steps=BUDGET),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_toy_clone_scores_against_its_teacher_and_the_chaser():
    device = torch.device("cuda")
    with torch.random.fork_rng():
        torch.manual_seed(0)
        student = Trainer(TOY, device)
        reason = cloning.clone(student, log=lambda row: None)
        teacher = student.controllers[SCRIPTED]
        assert isinstance(teacher, bc.TeacherOpponent) and teacher.teacher.floor == bc.FLOOR
        chaser = Chaser(GAME.ships_per_team)

        def against(player, opponent) -> float:
            controllers = {"player": player, "opponent": opponent}
            pool = Pool("evaluation", 1.0, ("player", "opponent"))
            return score(
                first_results(
                    GAME, api.ship_types(device), (pool,), controllers, "player", GAMES, device
                )
            )

        clone = student.controllers[LEARNER]
        versus_teacher = against(clone, teacher)
        versus_chaser = against(clone, chaser)
        teacher_versus_chaser = against(teacher, chaser)
    assert reason == "score", reason
    assert versus_teacher >= SCORE_BAR, (versus_teacher, SCORE_BAR)
    assert abs(versus_chaser - teacher_versus_chaser) <= 0.10, (
        versus_chaser,
        teacher_versus_chaser,
    )
