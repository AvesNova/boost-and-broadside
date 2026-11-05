"""The scripted agent's win rates (scripted spec A7.2, A7.3): fresh 5v5 games to their end at
every training preset, fog on, the seat drawn per game, with the default parameters.

- Against random: at most 2 non-wins (losses or draws) in 2048 games per preset.
- Against the chaser: above 0.5 over 1024 games per preset; 0.70 is the target, and a score
  below it warns rather than fails.

The games are seeded at or above `bench.scripted_search.SEED_LIMIT`, so none replays a game
the parameter search tuned on. Each match plays in a fresh process, so its compiled
rollout does not depend on what the suite compiled before it.
"""

import json
import os
import pathlib
import subprocess
import sys
import warnings

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))

from bench.scripted_search import SEED_LIMIT  # noqa: E402
from bnb.contracts import Pool  # noqa: E402
from bnb.controllers.scripted import Chaser, Random, Scripted  # noqa: E402
from bnb.game import api  # noqa: E402
from bnb.game.handling.characteristics import characteristics  # noqa: E402
from bnb.game.modes.rules import PRESETS  # noqa: E402
from bnb.runtime.evaluate import first_results  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
PRESET_NAMES = ("deathmatch", "deathmatch_wide", "frontline")
SEEDS = {
    (preset, opponent): 20_261_102 + 10 * index + offset
    for index, preset in enumerate(PRESET_NAMES)
    for offset, opponent in enumerate(("random", "chaser"))
}


def _results(preset: str, opponent: str, games: int) -> torch.Tensor:
    """The scripted agent's results over fresh games, in this process."""
    device = torch.device("cuda")
    game = PRESETS[preset]
    card = characteristics(api.ship_types(device), api.rules_of(game))
    rival = Random(game.ships_per_team) if opponent == "random" else Chaser(game.ships_per_team)
    torch.manual_seed(SEEDS[preset, opponent])
    return first_results(
        game,
        api.ship_types(device),
        (Pool("acceptance", 1.0, ("scripted", "opponent")),),
        {"scripted": Scripted(card, game.ships_per_team), "opponent": rival},
        "scripted",
        games,
        device,
    )


def _counts(preset: str, opponent: str, games: int) -> dict[str, int]:
    """(wins, draws, losses) of `_results`, played in a fresh process: one compiled rollout
    with nothing compiled before it, whatever the suite ran."""
    finished = subprocess.run(
        [sys.executable, __file__, preset, opponent, str(games)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_the_seeds_are_not_the_searchs():
    assert min(SEEDS.values()) >= SEED_LIMIT
    assert len(set(SEEDS.values())) == len(SEEDS)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_it_beats_random_play_in_essentially_every_game(preset):
    counts = _counts(preset, "random", 2048)
    assert sum(counts.values()) == 2048
    non_wins = counts["draws"] + counts["losses"]
    assert non_wins <= 2, f"{non_wins} non-wins in 2048 games"


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_it_beats_the_chaser(preset):
    counts = _counts(preset, "chaser", 1024)
    result = (counts["wins"] + 0.5 * counts["draws"]) / sum(counts.values())
    assert result > 0.5, result
    if result < 0.70:
        warnings.warn(
            f"{preset}: {result:.3f} against the chaser, below the 0.70 target", stacklevel=1
        )


if __name__ == "__main__":  # one match: preset, opponent, games; prints the counts as JSON
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    results = _results(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    print(
        json.dumps(
            dict(
                wins=int((results > 0).sum()),
                draws=int((results == 0).sum()),
                losses=int((results < 0).sum()),
            )
        )
    )
