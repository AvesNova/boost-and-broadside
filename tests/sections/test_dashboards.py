"""The dashboards against what training logs (diagnostics spec 4): every panel pattern of a
run's views matches at least one series that a short CPU run of its objective logs, PPO and
BC (a BC run's views leave out reinforcement learning's panels), so a renamed or retired
series cannot leave a dead panel behind.

The runs are the smallest that log every group: tiny Yemong on the training mix (frontline
and the map scales), episodes of half a second (so games end, the league scores its
entries, the lanes rate their players and the outcome head is scored), one lane per
matchup and mode, and, for BC, a score window of two games.
"""

import dataclasses
import pathlib
import re
import sys

import pytest
import torch

from bnb.runtime import dashboards

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "runs"))
from runs_cases import tiny_yemong_config  # noqa: E402

CPU = torch.device("cpu")
# Patterns a short CPU run cannot log, each for its reason: the allocator's peak and the
# budget are measured on a GPU only (`bnb.runs.recording`); the outcome head's Brier skill
# is absent while a time-to-end bucket's games are all won or all not won
# (`bnb.runtime.analytics.outcome_series`), as random play's half-second games are.
EXEMPT = (
    "train/memory_peak_gigabytes",
    "train/memory_budget_gigabytes",
    "outcome/brier_skill/*",
)


def _config(objective: str):
    from bnb.learn import bc
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime import cloning
    from bnb.runtime.lanes import LaneConfig
    from bnb.runtime.train import LINEUP, TrainConfig, mixed

    game, pools = mixed(cloning.LINEUP if objective == "bc" else LINEUP)
    pools = tuple(
        dataclasses.replace(pool, episode_seconds=0.5, respawn_seconds=0.2) for pool in pools
    )
    settings = dict(epochs=1, minibatches=2, micro_batch_rows=64)
    matchups = cloning.LANES.matchups if objective == "bc" else LaneConfig().matchups
    return TrainConfig(
        game=game,
        pools=pools,
        environments=16,
        length=16,
        shards=1,
        lookahead=16,
        lanes=LaneConfig(environments=3 * len(matchups), matchups=matchups),
        yemong=tiny_yemong_config(),
        learner=bc.config(**settings) if objective == "bc" else LearnerConfig(**settings),
        compiled=False,
        host_store=False,
        objective=objective,
        stop=bc.StopConfig(score_window=2),
    )


def logged(objective: str, directory: pathlib.Path, updates: int = 2) -> set[str]:
    """The series names a short CPU run of `objective` logs: the trainer's rows as the
    run's recorder writes them (the trigger rules' flags included), the league attached
    to a PPO run."""
    from bnb.league.lanes import LaneHooks
    from bnb.league.league import League
    from bnb.runs.recording import Recorder
    from bnb.runs.triggers import Triggers
    from bnb.runtime import cloning
    from bnb.runtime.train import Trainer

    trainer = Trainer(_config(objective), CPU)
    if objective == "ppo":
        League(trainer, directory / "league")
    else:
        LaneHooks(trainer)
    recorder = Recorder(None, trainer, None, Triggers(objective=objective))
    names = set()
    for _ in range(updates):
        row = cloning.update(trainer)[0] if objective == "bc" else trainer.update()
        names |= set(recorder.record(row))
    return names


def dead_panels(objective: str, names: set[str]) -> list[str]:
    """The panels of `objective`'s views with a pattern that matches none of `names`, as
    "view / section / panel: pattern"."""
    out = []
    for view, sections in dashboards.VIEWS.items():
        for section in dashboards.sections_for(sections, objective):
            for item in section.panels:
                for pattern in item.series:
                    if pattern in EXEMPT:
                        continue
                    expression = re.compile(dashboards.regex((pattern,)))
                    if not any(expression.match(name) for name in names):
                        out.append(f"{view} / {section.name} / {item.title}: {pattern}")
    return out


@pytest.mark.slow
@pytest.mark.parametrize("objective", ["ppo", "bc"])
def test_every_panel_shows_a_series_a_short_run_logs(objective, tmp_path, global_generator):
    names = logged(objective, tmp_path)
    dead = dead_panels(objective, names)
    assert not dead, "panels that match no logged series:\n" + "\n".join(dead)


def test_every_panel_is_in_some_view():
    """A panel whose objectives name no objective would be in no view at all."""
    for sections in dashboards.VIEWS.values():
        for section in sections:
            for item in section.panels:
                assert set(item.objectives) & set(dashboards.OBJECTIVES), item.title


def test_a_pattern_that_matches_nothing_is_named():
    names = {"policy/entropy/joint"}
    dead = dead_panels("ppo", names)
    assert "headline / Is it learning? / policy/kl and clip fraction: policy/kl" in dead
    assert not any(line.endswith(": policy/entropy/joint") for line in dead)
