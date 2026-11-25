"""Sight ellipse: scripted-vs-scripted games at one vision eccentricity, where first sightings
and first merges happen, whether the first spotter wins, game length and results.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src uv run --no-sync python \
        docs/rewrite/sections/vision-ellipse/measure.py --eccentricity 0.5 --games 256 --out DIR \
        [--vision-range 850] [--tracking off]

Every ship is the yardstick (ScriptedConfig()), both teams in one call; the core game
(GameConfig(): its spawn, fog and clock) with fields or none (field_count 0), 1v1 and 4v4. The
same seed per condition at every eccentricity, vision range and tracking setting, so the deals
are paired. --tracking on|off sets GameConfig.fog_tracking (the tracking circle, default on);
--vision-range p sets every type's vision range (default the card's). Each condition's games are
written to DIR/[p{P}_]e{E}[_track{on|off}]_{N}v{N}_{fields|nofields}.json when it finishes.

- First sighting, per team: the first tick (the deal counts as tick 0) any of its ships has an
  enemy in sight (lean.game.core.sight_pairs; reveals by shots and hits are not sightings). Of
  the pairs that start seeing on that tick, the closest: its distance, the angle of the target
  off the observer's nose and of the observer off the target's nose (0 nose-on).
- First sighting of each ship after the deal: per ship not in its enemies' sight at the deal,
  the first tick it comes into it (the closest new observer; the same three numbers). In 4v4
  most first sightings per team happen at the deal (ships spawn in sight), so these are where
  later sightings happen.
- First merge: the first tick an enemy pair's distance has a local minimum under the gun range
  (closing, then opening): the angle between the two noses then (180 head-on, 0 same way) and
  the distance (the lateral miss, at tick resolution).
- First spotter: the team whose first sighting is strictly earlier; its result.
- Visibility (the game's own: an enemy is visible when its ticks since seen is 0, by sight,
  tracking or a reveal), per ship from the deal to the game's end: a contact starts when the
  ship turns visible (or is visible at the deal); a contact loss is a visible ship turning hidden
  while it and some ship of the other team are alive; a hidden spell runs from a loss to the
  next tick the ship is visible (spells cut by the ship's death or the game's end are counted
  apart, not in the mean).
- Kills per minute: deaths over all games' minutes.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from lean import game, scripted
from lean.game import card, core, tuning


def set_vision(eccentricity: float, vision_range: float | None) -> None:
    rows = tuning.load_card()
    changed = {"vision_eccentricity": tuple(eccentricity for _ in rows["vision_eccentricity"])}
    if vision_range is not None:
        scale = vision_range / card.VISION_RANGE
        changed["vision_range_scale"] = tuple(scale for _ in rows["vision_range_scale"])
    card.CARD = {**rows, **changed}


def angle_off(heading_cos, heading_sin, line_x, line_y):
    """Degrees between a nose and a line (any shapes that broadcast)."""
    length = torch.sqrt(line_x * line_x + line_y * line_y).clamp_min(1e-6)
    cosine = (heading_cos * line_x + heading_sin * line_y) / length
    return torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))


@torch.no_grad()
def play(config: core.GameConfig, size: int, games: int, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    slots = 2 * size
    state = game.deal(config, games, slots, torch.tensor([[size, size]] * games), generator, "cpu")
    memory = scripted.fresh_memory(games, slots, generator, "cpu")
    agent = scripted.ScriptedConfig()
    gun_range = float(torch.as_tensor(state.constants["gun_range"]).flatten()[0])
    team = state.team
    enemy = team[:, :, None] != team[:, None, :]  # (E, target, observer)
    rows = [
        {
            "spot_tick": [None, None],
            "spot": [None, None],
            "later": [],
            "merge": None,
            "result": None,
            "contacts": 0,
            "losses": 0,
            "hidden_spells": [],
            "hidden_cut": 0,
            "deaths": 0,
        }
        for _ in range(games)
    ]
    running = torch.ones(games, dtype=torch.bool)
    spotted = torch.zeros(games, 2, dtype=torch.bool)
    merged = torch.zeros(games, dtype=torch.bool)
    previous_pairs = torch.zeros(games, slots, slots, dtype=torch.bool)
    ever_seen = None  # (E, N): the ship has been in its enemies' sight (from the deal on)
    distance_before = distance_last = None
    heading_last = None
    tick = 0
    visible_last = state.ticks_since_seen == 0  # the deal's sightings (tick 0)
    visible_last &= state.exists
    for e, count in enumerate((visible_last.sum(-1)).tolist()):
        rows[e]["contacts"] += count
    hidden_from = torch.full((games, slots), -1, dtype=torch.long)  # the tick of the loss
    while True:
        # Sightings on this state (the deal is tick 0).
        pairs = core.sight_pairs(state)
        line_x = state.x[:, :, None] - state.x[:, None, :]  # target minus observer
        line_y = state.y[:, :, None] - state.y[:, None, :]
        distance = torch.sqrt(line_x * line_x + line_y * line_y)
        new = pairs & ~previous_pairs

        def geometry(e, target, observer):
            return {
                "distance": float(distance[e, target, observer]),
                "observer_off_nose": float(
                    angle_off(
                        state.heading_cos[e, observer],
                        state.heading_sin[e, observer],
                        line_x[e, target, observer],
                        line_y[e, target, observer],
                    )
                ),
                "target_off_nose": float(
                    angle_off(
                        state.heading_cos[e, target],
                        state.heading_sin[e, target],
                        -line_x[e, target, observer],
                        -line_y[e, target, observer],
                    )
                ),
            }

        in_sight = pairs.any(-1)
        if ever_seen is None:
            ever_seen = in_sight.clone()
        else:
            fresh = in_sight & ~ever_seen & running[:, None] & state.alive
            for e, target in torch.nonzero(fresh).tolist():
                masked = torch.where(new[e, target], distance[e, target], torch.inf)
                observer = int(masked.argmin())
                rows[e]["later"].append({"tick": tick, **geometry(e, target, observer)})
            ever_seen |= in_sight
        for observer_team in (0, 1):
            by_team = pairs & (team[:, None, :] == observer_team)
            first = by_team.flatten(1).any(-1) & ~spotted[:, observer_team] & running
            for e in torch.nonzero(first).flatten().tolist():
                candidates = new[e] & (team[e][None] == observer_team)
                masked = torch.where(candidates, distance[e], torch.inf)
                target, observer = divmod(int(masked.argmin()), slots)
                rows[e]["spot_tick"][observer_team] = tick
                rows[e]["spot"][observer_team] = geometry(e, target, observer)
            spotted[:, observer_team] |= first
        previous_pairs = pairs
        # First merge: a local minimum of an enemy pair's distance, under the gun range.
        both = state.alive[:, :, None] & state.alive[:, None, :] & enemy
        if distance_before is not None:
            minimum = (
                both
                & (distance_last < distance_before)
                & (distance >= distance_last)
                & (distance_last < gun_range)
            )
            first = minimum.flatten(1).any(-1) & ~merged & running
            for e in torch.nonzero(first).flatten().tolist():
                masked = torch.where(minimum[e], distance_last[e], torch.inf)
                i, j = divmod(int(masked.argmin()), slots)
                cosine = (
                    heading_last[0][e, i] * heading_last[0][e, j]
                    + heading_last[1][e, i] * heading_last[1][e, j]
                )
                rows[e]["merge"] = {
                    "tick": tick - 1,
                    "noses_angle": math.degrees(math.acos(max(-1.0, min(1.0, float(cosine))))),
                    "miss": float(distance_last[e, i, j]),
                }
            merged |= first
        distance_before, distance_last = distance_last, torch.where(both, distance, torch.inf)
        heading_last = (state.heading_cos.clone(), state.heading_sin.clone())
        if not running.any():
            break
        uniforms = scripted.draw_uniforms(games, slots, generator, "cpu")
        action, memory, _ = scripted.decide(agent, state, None, None, uniforms, memory)
        state, out = game.tick(config, state, action, generator)
        tick += 1
        # Visibility on this tick, for the games that ran it.
        visible = (state.ticks_since_seen == 0) & state.exists
        live = running[:, None] & state.exists
        team_alive = torch.stack([(state.alive & (team == k)).any(-1) for k in (0, 1)], -1)
        enemy_alive = torch.where(team == 0, team_alive[:, 1:2], team_alive[:, 0:1])
        started = visible & ~visible_last & live
        lost = visible_last & ~visible & live & state.alive & enemy_alive
        found = visible & (hidden_from >= 0) & live
        ended = (hidden_from >= 0) & live & ~visible & ~state.alive
        for e, n in torch.nonzero(found).tolist():
            rows[e]["hidden_spells"].append(tick - int(hidden_from[e, n]))
        for e, count in enumerate((started.sum(-1)).tolist()):
            rows[e]["contacts"] += count
        for e, count in enumerate((lost.sum(-1)).tolist()):
            rows[e]["losses"] += count
        for e, count in enumerate(((out.died & live).sum(-1)).tolist()):
            rows[e]["deaths"] += count
        hidden_from = torch.where(found | ended, -1, hidden_from)
        hidden_from = torch.where(lost, tick, hidden_from)
        for e in torch.nonzero(out.done & running).flatten().tolist():
            rows[e]["hidden_cut"] += int((hidden_from[e] >= 0).sum())
        for e, count in enumerate((ended.sum(-1)).tolist()):
            rows[e]["hidden_cut"] += count
        visible_last = visible
        for e in torch.nonzero(out.done & running).flatten().tolist():
            alive = state.alive[e] & state.exists[e]
            left = [int((alive & (team[e] == k)).sum()) for k in (0, 1)]
            rows[e]["result"] = int(out.result[e])
            rows[e]["ticks"] = tick
            rows[e]["wiped"] = min(left) == 0
        running &= ~out.done
    return rows


def summary(rows: list[dict], tick_length: float) -> dict:
    def quartiles(values):
        if not values:
            return None
        ordered = torch.tensor(values, dtype=torch.float64)
        q = torch.quantile(ordered, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        return [round(float(v), 1) for v in q]

    spots = [s for r in rows for s in r["spot"] if s is not None]
    spot_ticks = [t for r in rows for t in r["spot_tick"] if t is not None]
    later = [s for r in rows for s in r["later"]]
    merges = [r["merge"] for r in rows if r["merge"] is not None]
    first_spotter = []
    for r in rows:
        a, b = r["spot_tick"]
        a = math.inf if a is None else a
        b = math.inf if b is None else b
        if a != b:
            first_spotter.append((0 if a < b else 1, r["result"]))
    decisive = [(team, result) for team, result in first_spotter if result in (0, 1)]
    games = len(rows)
    return {
        "games": games,
        "first_sighting_seconds": quartiles([t * tick_length for t in spot_ticks]),
        "sighted_at_deal": sum(t == 0 for t in spot_ticks) / max(1, len(spot_ticks)),
        "first_sighting_distance": quartiles([s["distance"] for s in spots]),
        "first_sighting_observer_off_nose": quartiles([s["observer_off_nose"] for s in spots]),
        "first_sighting_target_off_nose": quartiles([s["target_off_nose"] for s in spots]),
        "first_sighting_observer_within_60": sum(s["observer_off_nose"] < 60.0 for s in spots)
        / max(1, len(spots)),
        "later_sightings": len(later),
        "later_sighting_distance": quartiles([s["distance"] for s in later]),
        "later_sighting_observer_off_nose": quartiles([s["observer_off_nose"] for s in later]),
        "later_sighting_target_off_nose": quartiles([s["target_off_nose"] for s in later]),
        "later_sighting_observer_within_60": sum(s["observer_off_nose"] < 60.0 for s in later)
        / max(1, len(later)),
        "merge_share": len(merges) / games,
        "merge_seconds": quartiles([m["tick"] * tick_length for m in merges]),
        "merge_noses_angle": quartiles([m["noses_angle"] for m in merges]),
        "merge_head_on_share_over_135": sum(m["noses_angle"] > 135.0 for m in merges)
        / max(1, len(merges)),
        "merge_miss": quartiles([m["miss"] for m in merges]),
        "games_with_a_first_spotter": len(first_spotter),
        "first_spotter_decisive_games": len(decisive),
        "first_spotter_win_share_of_decisive": (
            sum(team == result for team, result in decisive) / len(decisive) if decisive else None
        ),
        "first_spotter_draw_share": (
            sum(result == 2 for _, result in first_spotter) / len(first_spotter)
            if first_spotter
            else None
        ),
        "seconds_mean": sum(r["ticks"] for r in rows) / games * tick_length,
        "contacts_per_game": sum(r["contacts"] for r in rows) / games,
        "contact_losses_per_game": sum(r["losses"] for r in rows) / games,
        "contact_losses_per_contact": sum(r["losses"] for r in rows)
        / max(1, sum(r["contacts"] for r in rows)),
        "hidden_spells": sum(len(r["hidden_spells"]) for r in rows),
        "hidden_seconds_mean": (
            sum(t for r in rows for t in r["hidden_spells"])
            / max(1, sum(len(r["hidden_spells"]) for r in rows))
            * tick_length
        ),
        "hidden_seconds": quartiles([t * tick_length for r in rows for t in r["hidden_spells"]]),
        "hidden_spells_cut": sum(r["hidden_cut"] for r in rows),
        "kills_per_minute": sum(r["deaths"] for r in rows)
        / (sum(r["ticks"] for r in rows) * tick_length / 60.0),
        "team0_wins": sum(r["result"] == 0 for r in rows),
        "team1_wins": sum(r["result"] == 1 for r in rows),
        "draws": sum(r["result"] == 2 for r in rows),
        "wipes": sum(r["wiped"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eccentricity", type=float, required=True)
    parser.add_argument("--vision-range", type=float, default=None, help="metres, every type")
    parser.add_argument("--tracking", choices=("on", "off"), default="on")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4], help="ships per team")
    parser.add_argument(
        "--fields", choices=("both", "none", "on"), default="both", help="which field conditions"
    )
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    arguments.out.mkdir(parents=True, exist_ok=True)
    set_vision(arguments.eccentricity, arguments.vision_range)
    tracking = arguments.tracking == "on"
    field_choices = {"both": (False, True), "none": (False,), "on": (True,)}[arguments.fields]
    for size in arguments.sizes:
        for fields in field_choices:
            config = core.GameConfig(fog_tracking=tracking)
            if not fields:
                config = core.GameConfig(field_count=0, fog_tracking=tracking)
            prefix = "" if arguments.vision_range is None else f"p{arguments.vision_range:g}_"
            name = (
                f"{prefix}e{arguments.eccentricity:g}_track{arguments.tracking}"
                f"_{size}v{size}_{'fields' if fields else 'nofields'}"
            )
            started = time.perf_counter()
            rows = play(config, size, arguments.games, arguments.seed + 1000 * size + fields)
            result = summary(rows, config.tick_length)
            result["condition"] = name
            result["seconds_to_run"] = round(time.perf_counter() - started, 1)
            (arguments.out / f"{name}.json").write_text(
                json.dumps({"summary": result, "games": rows}, indent=1)
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
