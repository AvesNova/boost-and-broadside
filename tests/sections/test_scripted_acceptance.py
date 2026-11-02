"""The scripted agent's win rates (scripted spec A7.2, A7.3): fresh 5v5 games to their end at
every training preset, fog on, the seat drawn per game, with the default parameters.

- Against random: at most 2 non-wins (losses or draws) in 2048 games per preset.
- Against the chaser: above 0.5 over 1024 games per preset; 0.70 is the target, and a score
  below it warns rather than fails.

The games are seeded at or above `bench.scripted_search.SEED_LIMIT`, so none replays a game
the parameter search tuned on.
"""

import pathlib
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
from bnb.runtime.evaluate import first_results, score  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
PRESET_NAMES = ("deathmatch", "deathmatch_wide", "frontline")
SEEDS = {(preset, opponent): 20_261_102 + 10 * index + offset
         for index, preset in enumerate(PRESET_NAMES)
         for offset, opponent in enumerate(("random", "chaser"))}


def _results(preset: str, opponent: str, games: int) -> torch.Tensor:
    device = torch.device("cuda")
    game = PRESETS[preset]
    card = characteristics(api.ship_types(device), api.rules_of(game))
    rival = Random(game.ships_per_team) if opponent == "random" else Chaser(game.ships_per_team)
    torch._dynamo.reset()  # one compiled rollout per match, under the recompile limit
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(SEEDS[preset, opponent])
        return first_results(game, (Pool("acceptance", 1.0, ("scripted", "opponent")),),
                             {"scripted": Scripted(card), "opponent": rival}, "scripted",
                             games, device)


def test_the_seeds_are_not_the_searchs():
    assert min(SEEDS.values()) >= SEED_LIMIT
    assert len(set(SEEDS.values())) == len(SEEDS)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_it_beats_random_play_in_essentially_every_game(preset):
    results = _results(preset, "random", 2048)
    assert results.shape == (2048,)
    non_wins = int((results <= 0).sum())
    assert non_wins <= 2, f"{non_wins} non-wins in 2048 games"


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_it_beats_the_chaser(preset):
    result = score(_results(preset, "chaser", 1024))
    assert result > 0.5, result
    if result < 0.70:
        warnings.warn(f"{preset}: {result:.3f} against the chaser, below the 0.70 target",
                      stacklevel=1)
