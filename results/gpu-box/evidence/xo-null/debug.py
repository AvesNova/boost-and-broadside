import dataclasses, sys, torch, pathlib
from bnb.cli import _players
from bnb.game.api import PRESETS
from bnb.league import crossover as fights
from bnb.league.matches import ArenaConfig
torch.manual_seed(0)
first, second = _players(["runs/2026-11-09-rl-absolute/policies/step_180355072/uniform_average", "scripted"], pathlib.Path("runs"))
game = dataclasses.replace(PRESETS["frontline"], ships_per_team=10, episode_seconds=float(sys.argv[2]))
rows = fights.crossover(first, second, [(5, 10), (5, 6)], 4, game, torch.device(sys.argv[1]),
                        ArenaConfig(blocks=2, block_environments=2, compiled=False, length=32))
for r in rows: print(r["first_active"], r["second_active"], r["wins"], r["draws"], r["losses"])
