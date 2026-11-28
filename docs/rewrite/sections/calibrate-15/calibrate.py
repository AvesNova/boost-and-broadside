"""Queue item 15: the learner's targets and the observation under the core rules of 2026-11-28
(nose armour, circle deflector, sight egg; the shrinking zone, no clock).

    OMP_NUM_THREADS=4 PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/calibrate-15/calibrate.py TEAM_SIZE ENVS TICKS OUT.npz

What BC sees: the default scripted agent ScriptedConfig() on both teams (persona spread 0,
lean.learn.rollout.scripted_actions), the training game (GameConfig(training_spawn=True):
drawn fields, fog, the zone), TEAM_SIZE v TEAM_SIZE, ENVS environments for TICKS ticks,
finished environments re-dealt (as target_ranges.py). Collected:

- value returns per level that trains (damage, kill_death, outcome): each living ship's
  discounted Monte Carlo return to its episode's end, every tick of the episodes that ended
  inside the run. With no clock every game ends, so this is the target of a perfect critic
  for any GAE lambda (the lambda-return of the true values is the Monte Carlo return); the
  outcome continues through a ship's death (it is paid to the dead too), which is option C's
  target on the outcome (the team-mates' mean value). Each level's discount is the learner's.
- next-state targets (next_state_target) of every ship alive before and after the step in an
  unfinished game;
- every bullet outcome that ended (the hit head's labels);
- every GEOMETRY_EVERY ticks, on the first GEOMETRY_ENVS environments, both teams' views: the
  geometry targets the learner trains (lean.learn.geometry: rows the team sees), ship pairs
  (symmetric columns on first < second, the order-dependent ones on both orders), ship-field
  signed distance, map edge; and the observation's scaled features on the attended rows
  (key_mask), the globals and the fields.

Writes OUT.npz (subsamples of every quantity, at most KEEP values each) and OUT.json (exact
percentiles over every row, counts). Progress lines with a clock-time ETA to stdout.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.game.core import FIELD_RADIUS_SCALE, GEOMETRY_COLUMNS
from lean.learn.config import LearnConfig
from lean.learn.geometry import observed_rows
from lean.learn.rewards import blank_ledger, clear_ledger, reward_step, tick_events
from lean.learn.rollout import draw_team_sizes, scripted_actions
from lean.model.policy import ModelConfig, next_state_target

PERCENTILES = (0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0)
KEEP = 4_000_000
GEOMETRY_EVERY = 8
GEOMETRY_ENVS = 256
PROGRESS_EVERY = 500


def quantiles(values: torch.Tensor) -> dict[str, float]:
    ordered = torch.sort(values.float().flatten()).values
    if ordered.numel() == 0:
        return {}
    last = ordered.numel() - 1
    return {str(share): float(ordered[round(share * last)]) for share in PERCENTILES}


def keep(values: torch.Tensor, generator: torch.Generator) -> np.ndarray:
    values = values.flatten()
    if values.numel() > KEEP:
        index = torch.randperm(values.numel(), generator=generator, device="cpu")[:KEEP]
        values = values.cpu()[index]
    return values.cpu().numpy().astype(np.float32)


def main() -> None:
    team_size, envs, ticks = (int(argument) for argument in sys.argv[1:4])
    out = Path(sys.argv[4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    learn = LearnConfig(team_size_minimum=team_size, team_size_maximum=team_size)
    model = ModelConfig()
    game_config = game.GameConfig(training_spawn=True)
    scripted_config = scripted.ScriptedConfig()
    sizes = draw_team_sizes(learn, envs, generator)
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    levels = learn.level_names
    rewards = torch.zeros(ticks, envs, slots, len(levels), device=device)
    valued = torch.zeros(ticks, envs, slots, dtype=torch.bool, device=device)
    done_at = torch.zeros(ticks, envs, dtype=torch.bool, device=device)
    ledger = blank_ledger(envs, slots, device)
    outcomes = torch.zeros(3, dtype=torch.long, device=device)
    next_rows: list[torch.Tensor] = []
    parts: dict[str, list[torch.Tensor]] = {}
    started = torch.zeros(envs, dtype=torch.long, device=device)
    lengths: list[torch.Tensor] = []
    upper = torch.triu(torch.ones(slots, slots, dtype=torch.bool, device=device), 1)
    other = ~torch.eye(slots, dtype=torch.bool, device=device)
    g = GEOMETRY_ENVS

    def add(name: str, values: torch.Tensor) -> None:
        parts.setdefault(name, []).append(values.float().cpu())

    began = time.time()
    with torch.no_grad():
        for tick in range(ticks):
            if tick % GEOMETRY_EVERY == 0:
                record = game.geometry_record(game_config, state)[:g]
                for team in (0, 1):
                    obs = game.observe(game_config, state, team)
                    seen = observed_rows(obs)[:g]
                    pos, heading = obs.pos[:g], obs.heading[:g]
                    pairs = game.pair_geometry(
                        pos[:, :, None],
                        heading[:, :, None],
                        record[:, :, None],
                        pos[:, None, :],
                        record[:, None, :],
                    )
                    both = seen[:, :, None] & seen[:, None, :]
                    for name, values in pairs.items():
                        mask = both & (other if name in ("train_angle", "gun_error") else upper)
                        add(f"pair_{name}", values[mask])
                    field_pos = obs.field_pos[:g]
                    radius = obs.field[:g, :, 0] * FIELD_RADIUS_SCALE
                    distance = game.field_edge_distance(
                        field_pos[:, None, :, 0],
                        field_pos[:, None, :, 1],
                        radius[:, None, :],
                        pos[:, :, None, 0],
                        pos[:, :, None, 1],
                    )
                    add("field_edge", distance[seen].flatten())
                    add("map_edge", record[..., GEOMETRY_COLUMNS.index("edge")][seen])
                    attended = obs.key_mask[:g]
                    add("ship_features", obs.ship[:g][attended])
                    add("since_seen", obs.since_seen[:g][attended])
                    add("globals", obs.globals[:g])
                    add("field_features", obs.field[:g].flatten(0, 1))
            move = scripted_actions(scripted_config, state, None, generator, team_size)
            next_state, step_out = game.tick(game_config, state, move.action, generator)
            reward, ledger = reward_step(learn, ledger, tick_events(step_out, state), state.team)
            rewards[tick] = reward
            valued[tick] = state.alive
            done = step_out.done
            done_at[tick] = done
            valid = state.alive & next_state.alive & ~done[:, None]
            target = next_state_target(model, state, next_state, step_out, game_config.tick_length)
            next_rows.append(target[valid].cpu())
            ended = step_out.shot_outcome[step_out.shot_outcome >= 0]
            outcomes += torch.bincount(ended.flatten(), minlength=3)
            lengths.append((tick + 1 - started)[done].cpu())
            started = torch.where(done, tick + 1, started)
            ledger = clear_ledger(ledger, done)
            fresh = game.deal(
                game_config, envs, slots, draw_team_sizes(learn, envs, generator), generator, device
            )
            state = game.take_deal(next_state, done, fresh)
            if (tick + 1) % PROGRESS_EVERY == 0 or tick + 1 == ticks:
                elapsed = time.time() - began
                eta = datetime.now() + timedelta(seconds=elapsed / (tick + 1) * (ticks - tick - 1))
                print(
                    f"{team_size}v{team_size} tick {tick + 1} of {ticks}, {elapsed:.0f} s, "
                    f"ETA {eta:%H:%M:%S}, episodes ended {int(done_at[: tick + 1].sum())}",
                    flush=True,
                )

        discounts = torch.tensor(learn.per_level(learn.discounts), device=device)
        running = torch.zeros(envs, slots, len(levels), device=device)
        returns = torch.empty_like(rewards)
        for tick in reversed(range(ticks)):
            keep_running = (~done_at[tick]).float()[:, None, None]
            running = rewards[tick] + discounts * keep_running * running
            returns[tick] = running
        tick_index = torch.arange(ticks, device=device)[:, None]
        last_end = torch.where(done_at, tick_index, -1).amax(0)
        complete = tick_index <= last_end[None, :]
        value_rows = returns[valued & complete[..., None]]
        del rewards, returns

    cpu = torch.Generator().manual_seed(1)
    stats: dict = {
        "team_size": team_size,
        "envs": envs,
        "ticks": ticks,
        "episodes": int(done_at.sum()),
        "seconds": time.time() - began,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "hit_counts": outcomes.tolist(),
        "quantiles": {},
        "counts": {},
    }
    arrays: dict[str, np.ndarray] = {}
    episode_lengths = torch.cat(lengths).float()
    stats["episode_ticks"] = quantiles(episode_lengths)
    arrays["episode_ticks"] = episode_lengths.numpy()
    for level, name in enumerate(levels):
        values = value_rows[:, level]
        stats["quantiles"][f"value_{name}"] = quantiles(values)
        stats["counts"][f"value_{name}"] = values.numel()
        arrays[f"value_{name}"] = keep(values, cpu)
    targets = torch.cat(next_rows)
    for column, spec in enumerate(model.next_state_columns):
        values = targets[:, column]
        stats["quantiles"][f"next_{spec.name}"] = quantiles(values)
        stats["counts"][f"next_{spec.name}"] = values.numel()
        arrays[f"next_{spec.name}"] = keep(values, cpu)
    for name, pieces in parts.items():
        values = torch.cat(pieces)
        if values.dim() == 2:  # features: per column
            for column in range(values.shape[1]):
                key = f"{name}_{column}"
                stats["quantiles"][key] = quantiles(values[:, column])
                stats["counts"][key] = values.shape[0]
                arrays[key] = keep(values[:, column], cpu)
        else:
            stats["quantiles"][name] = quantiles(values)
            stats["counts"][name] = values.numel()
            arrays[name] = keep(values, cpu)
    np.savez(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out} at {datetime.now():%H:%M:%S}, {time.time() - began:.0f} s", flush=True)


if __name__ == "__main__":
    main()
