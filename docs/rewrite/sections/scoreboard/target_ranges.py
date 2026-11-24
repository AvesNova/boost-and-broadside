"""The learner's measured target ranges: the value supports, the next-state ranges and the hit
head's shares (lean/model/NOTES.md "Next-state head", lean/learn/config.py value_low/high,
ModelConfig.hit_class_frequencies). Re-measure when the game, the clock or the scripted agent
changes.

    OMP_NUM_THREADS=4 PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/target_ranges.py [ENVS] [TICKS]

What BC sees: the scripted yardstick on both teams (lean.learn.rollout.scripted_actions, persona
spread 0), the training game (GameConfig(training_spawn=True): drawn fields, fog, ghost
bullets, the clock) and the learner's team sizes (LearnConfig), E environments for T ticks,
finished environments re-dealt. Per tick: next_state_target of every ship alive before and
after the step in an unfinished game (the rollout's next_valid), every bullet outcome that
ended (ghosts included), and the rewards (lean.learn.rewards). After the run, each living
ship's discounted return per level (the level's discount, Monte Carlo to its episode's end),
for every tick of the episodes that ended inside the run.
"""

import sys
import time

import torch

from lean import game, scripted
from lean.contract import LEVEL_NAMES
from lean.learn.config import LearnConfig
from lean.learn.rewards import blank_ledger, clear_ledger, reward_step, tick_events
from lean.learn.rollout import draw_team_sizes, scripted_actions
from lean.model.policy import ModelConfig, next_state_target

PERCENTILES = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)


def quantiles(values: torch.Tensor) -> list[float]:
    """PERCENTILES of a 1-d tensor by sorting (torch.quantile caps its input size)."""
    ordered = torch.sort(values.float()).values
    last = ordered.numel() - 1
    return [float(ordered[round(share * last)]) for share in PERCENTILES]


def main() -> None:
    envs = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    learn = LearnConfig()
    model = ModelConfig()
    game_config = game.GameConfig(training_spawn=True)
    sizes = draw_team_sizes(learn, envs, generator)
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    scripted_config = scripted.ScriptedConfig()
    persona = scripted.personas(
        scripted_config, envs, slots, generator, device, learn.persona_spread
    )
    memory = scripted.fresh_memory(envs, slots, generator, device)
    ledger = blank_ledger(envs, slots, device)
    rewards = torch.zeros(ticks, envs, slots, len(LEVEL_NAMES), device=device)
    valued = torch.zeros(ticks, envs, slots, dtype=torch.bool, device=device)
    done_at = torch.zeros(ticks, envs, dtype=torch.bool, device=device)
    next_rows, outcomes = [], torch.zeros(3, dtype=torch.long, device=device)
    began = time.time()
    with torch.no_grad():
        for tick in range(ticks):
            move = scripted_actions(scripted_config, state, persona, memory, generator)
            next_state, out = game.tick(game_config, state, move.action, generator)
            reward, ledger = reward_step(learn, ledger, tick_events(out, state), state.team)
            rewards[tick] = reward
            valued[tick] = state.alive
            done_at[tick] = out.done
            valid = state.alive & next_state.alive & ~out.done[:, None]
            target = next_state_target(model, state, next_state, out, game_config.tick_length)
            next_rows.append(target[valid])
            ended = out.shot_outcome[out.shot_outcome >= 0]
            outcomes += torch.bincount(ended.flatten(), minlength=3)
            done = out.done
            ledger = clear_ledger(ledger, done)
            fresh_sizes = draw_team_sizes(learn, envs, generator)
            fresh = game.deal(game_config, envs, slots, fresh_sizes, generator, device)
            state = game.take_deal(next_state, done, fresh)
            new_memory = scripted.fresh_memory(envs, slots, generator, device)
            memory = {
                name: torch.where(
                    done.view(-1, *[1] * (move.memory[name].dim() - 1)),
                    new_memory[name],
                    move.memory[name],
                )
                for name in move.memory
            }
            if (tick + 1) % 200 == 0:
                elapsed = time.time() - began
                print(
                    f"tick {tick + 1} of {ticks}, {elapsed:.0f} s, "
                    f"ETA {elapsed / (tick + 1) * (ticks - tick - 1):.0f} s more, "
                    f"episodes ended {int(done_at[: tick + 1].sum())}",
                    flush=True,
                )

        # Monte Carlo returns per level, backward, cut at episode ends.
        discounts = torch.tensor(learn.discounts, device=device)
        running = torch.zeros(envs, slots, len(LEVEL_NAMES), device=device)
        returns = torch.empty_like(rewards)
        for tick in reversed(range(ticks)):
            keep = (~done_at[tick]).float()[:, None, None]
            running = rewards[tick] + discounts * keep * running
            returns[tick] = running
        # Only episodes that ended inside the run: ticks up to each environment's last end.
        tick_index = torch.arange(ticks, device=device)[:, None]
        last_end = torch.where(done_at, tick_index, -1).amax(0)
        complete = tick_index <= last_end[None, :]
        rows = returns[valued & complete[..., None]]

    episodes = int(done_at.sum())
    print(
        f"\n{envs} environments x {ticks} ticks, {episodes} episodes ended, team sizes "
        f"{learn.team_size_minimum} to {learn.team_size_maximum}, clock {game_config.clock_ticks}"
    )
    header = "| quantity | rows | min | 0.1% | 1% | median | 99% | 99.9% | max |"
    print("\nValue returns (discounted, per living ship and tick)\n" + header)
    print("|---|---|---|---|---|---|---|---|---|")
    for level, name in enumerate(LEVEL_NAMES):
        q = quantiles(rows[:, level])
        print(f"| {name} | {rows.shape[0]} | " + " | ".join(f"{v:.4g}" for v in q) + " |")
    targets = torch.cat(next_rows)
    print("\nNext-state targets (target units)\n" + header)
    print("|---|---|---|---|---|---|---|---|---|")
    for column, spec in enumerate(model.next_state_columns):
        values = targets[:, column]
        q = quantiles(values)
        print(f"| {spec.name} | {values.numel()} | " + " | ".join(f"{v:.4g}" for v in q) + " |")
        if not spec.symlog:
            half = max(abs(q[1]), abs(q[5]))
            outside = float((values.abs() > half).float().mean())
            print(
                f"  {spec.name}: symmetric half-range at the 0.1/99.9 percentiles {half:.4g}, "
                f"outside {outside:.4%}; current {spec.low}..{spec.high} outside "
                f"{float(((values < spec.low) | (values > spec.high)).float().mean()):.4%}"
            )
        else:
            print(
                f"  {spec.name}: share > 0 {float((values > 0).float().mean()):.4%}, "
                f"share < 0 {float((values < 0).float().mean()):.4%}, "
                f"below {spec.low}: {float((values < spec.low).float().mean()):.6%}"
            )
    total = int(outcomes.sum())
    shares = (outcomes.double() / total).tolist()
    print(
        f"\nHit outcomes: {total} bullets ended; missed {shares[0]:.4%}, enemy {shares[1]:.4%}, "
        f"team-mate {shares[2]:.4%}"
    )


if __name__ == "__main__":
    main()
