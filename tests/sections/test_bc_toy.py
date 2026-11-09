"""The toy BC test (scripted spec B6.5): the toy model (the relational MLP) cloned from the
scripted agent at 2v2 team deathmatch (the average of its weights its score stop read)
scores at least 0.40 against its teacher over 512 games (within two standard errors, see
SCORE_BAR), and its score against the chaser is within 0.10 of the teacher's.

The labels are the default teacher's (the scripted agent at T = 1, floor 0), and the
teacher is the one they describe: BC's scripted seat (`bnb.learn.bc.TeacherOpponent`). BC
runs to its score stop, here the toy's own target (TARGET: the better average's lane score
against the raw scripted agent over 1000 games reaches 0.40, not the default parity), within BUDGET
environment steps of 256 environments x 128 ticks, one shard, no lookahead, with 256 rating
lanes (the section notes, "Stop rule (2026-11-02)", have the curve of the earlier rule).
"""

import dataclasses

import pytest
import torch

from bnb.contracts import GameConfig, Pool
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.league.lanes import LaneHooks
from bnb.learn import bc
from bnb.runs.session import policy_player
from bnb.runtime import cloning
from bnb.runtime.evaluate import first_results, score
from bnb.runtime.train import LEARNER, SCRIPTED, TrainConfig, Trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
GAME = GameConfig(2, 4, 1200.0)
BUDGET = 100_000_000
GAMES = 512
TARGET = 0.40
# "At least 0.40" is read as not significantly below it: two standard errors of a GAMES-game
# score (at most 0.5 / sqrt(GAMES), about 0.022).
SCORE_BAR = TARGET - 2 * 0.5 / GAMES**0.5
TOY = TrainConfig(
    game=GAME,
    pools=cloning.LINEUP,
    environments=256,
    shards=1,
    lookahead=0,
    lanes=dataclasses.replace(cloning.LANES, environments=256),
    model="relational",
    learner=bc.config(),
    objective="bc",
    stop=bc.StopConfig(score_target=TARGET, budget_steps=BUDGET),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_toy_clone_scores_against_its_teacher_and_the_chaser():
    device = torch.device("cuda")
    with torch.random.fork_rng():
        torch.manual_seed(0)
        student = Trainer(TOY, device)
        hooks = LaneHooks(student)  # the lanes' ratings and the averages the stop reads
        reason = cloning.clone(student, log=lambda row: None)
        teacher = student.controllers[SCRIPTED]
        assert isinstance(teacher, bc.TeacherOpponent) and teacher.teacher == bc.TeacherConfig()
        chaser = Chaser(characteristics(api.ship_types(), api.rules_of(GAME)), GAME.ships_per_team)

        def against(player, opponent) -> float:
            controllers = {"player": player, "opponent": opponent}
            pool = Pool("evaluation", 1.0, ("player", "opponent"))
            return score(
                first_results(
                    GAME, api.ship_types(device), (pool,), controllers, "player", GAMES, device
                )
            )

        # the product: the average the score stop read, in the learner's controller
        student.model.load_state_dict(hooks.averages.weights[policy_player(student)])
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
