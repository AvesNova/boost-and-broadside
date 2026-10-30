"""``head-to-head`` mode: score chosen pairs of agents in one batch of games.

A full calibration plays every pair until every rating is pinned, which is the
right tool for publishing ratings and far too slow for a question about two
runs. Most questions name their pairs: a checkpoint against the scripted
controller, two seeds at the same step, a live policy against its own average.
This plays exactly those pairs, the same number of games each, side-balanced,
every game to completion, and reports each pair's score with its standard error.

No rating is fitted. A score between two specific agents is what the question
asks, and it carries no assumption that strength is transitive.
"""

import json
import math
from pathlib import Path

import numpy as np

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import agents_read_bullets
from boost_and_broadside.evaluation.environment import resolve_evaluation_environment
from boost_and_broadside.evaluation.tournament import Progress, Tournament
from boost_and_broadside.modes.elo_calibrate import build_arbitrary_players


def run_head_to_head_mode(
    pairs: list[tuple[str, str]],
    games_per_pair: int,
    ship_config: ShipConfig,
    model_config: ModelConfig,
    env_config: EnvConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    out: str | Path | None = None,
) -> list[dict]:
    """Play every pair ``games_per_pair`` times in one batch and score it.

    Returns one record per pair, in the order given, scored from the first
    agent's side: ``score`` counts a draw as half a win, and ``elo_diff`` is the
    rating gap that score implies.
    """
    specs = list(dict.fromkeys(spec for pair in pairs for spec in pair))
    players = build_arbitrary_players(
        specs, ship_config, model_config, env_config.num_ships, device, checkpoint_dir
    )
    evaluation_config = resolve_evaluation_environment(
        env_config, [player.agent for player in players], ship_config=ship_config
    )
    index = {spec: i for i, spec in enumerate(specs)}
    allocation = np.zeros((len(specs), len(specs)))
    for a, b in pairs:
        if a == b:
            raise ValueError(f"a pair needs two different agents, got {a!r} twice")
        i, j = sorted((index[a], index[b]))
        allocation[i, j] = games_per_pair
    tournament = Tournament(
        players,
        ship_config,
        evaluation_config,
        "ego_pass",
        int(allocation.sum()),
        device,
        include_bullets=agents_read_bullets(*(player.agent for player in players)),
    )
    print(f"\n=== head-to-head: {len(pairs)} pairs x {games_per_pair} games ===")
    tournament.play_batch(allocation, Progress())
    print()

    records = []
    for a, b in pairs:
        i, j = index[a], index[b]
        wins, losses = tournament.wins[i, j], tournament.wins[j, i]
        ties = tournament.ties[i, j] + tournament.ties[j, i]
        games = wins + losses + ties
        score = (wins + 0.5 * ties) / games
        clamped = min(max(score, 1e-3), 1.0 - 1e-3)
        record = {
            "a": players[i].label,
            "b": players[j].label,
            "games": int(games),
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
            "score": score,
            "score_se": math.sqrt(max(score * (1.0 - score), 1e-9) / games),
            "elo_diff": 400.0 * math.log10(clamped / (1.0 - clamped)),
        }
        records.append(record)
        print(
            f"  {record['a']:>28} vs {record['b']:<28} W/L/T "
            f"{record['wins']:4d}/{record['losses']:4d}/{record['ties']:4d}  "
            f"score {score:.3f} ± {record['score_se']:.3f}  Δelo {record['elo_diff']:+5.0f}",
            flush=True,
        )
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(records, indent=1))
    return records
