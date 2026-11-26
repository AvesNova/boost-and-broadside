"""Scripted games for the look-ahead agent's checks: action usage, the teacher's entropy, kills
per minute, and outcomes against the rule-based agent it replaces.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/measure.py \
        --mode selfplay --size 4 --fields 10 --envs 256 --ticks 5400 --out DIR

Modes: "selfplay" (the look-ahead yardstick on both teams: the 42 actions' counts over living
ship-ticks, the mean entropy of its joint distribution, kills per environment-minute, mean
speed and store, games ended); "versus" (look-ahead on team 0 in the first half of the
environments and team 1 in the second, the rule-based agent on the other team: wins, draws,
losses by the look-ahead's side); "old" (the rule-based agent on both teams, for the kills and
usage numbers). Fog on, the default game otherwise (`--fields` fields). Finished games are
re-dealt. `--set name=value ...` changes look-ahead rows (diagnostics). Every `--report`
ticks it writes the totals so far to --out (JSON) and prints them with the ETA as a clock
time. `--compile` compiles the agent's call.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lean import game, scripted
from lean.contract import JOINT, compile_dynamic, join_action
from lean.scripted.rule_based import opponent as rule_based
from lean.scripted.rule_based.parameters import ScriptedConfig as RuleBasedConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selfplay", "versus", "old"), default="selfplay")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--fields", type=int, default=10)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=5400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=int, default=600)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--set", nargs="*", default=[], help="look-ahead parameters name=value (diagnostics)"
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = game.GameConfig(field_count=arguments.fields)
    envs, size = arguments.envs, arguments.size
    generator = torch.Generator(device).manual_seed(arguments.seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    changes = {item.split("=")[0]: float(item.split("=")[1]) for item in arguments.set}
    new_config, old_config = scripted.ScriptedConfig().replace(**changes), RuleBasedConfig()
    old_memory = rule_based.fresh_memory(envs, slots, generator, device)

    def new_call(state, uniforms):
        return scripted.decide(new_config, state, None, None, uniforms)

    def old_call(state, uniforms, memory):
        action, memory, _ = rule_based.decide(old_config, state, None, None, uniforms, memory)
        return action, memory

    if arguments.compile:
        new_call, old_call = compile_dynamic(new_call), compile_dynamic(old_call)
    # Versus: which team the look-ahead agent flies, per environment (seats alternated).
    new_team = (torch.arange(envs, device=device) >= envs // 2).long()
    counts = torch.zeros(JOINT, dtype=torch.long, device=device)
    entropy_sum, entropy_count = 0.0, 0
    kills = speed_sum = store_sum = living_ticks = 0.0
    outcome = {"wins": 0, "draws": 0, "losses": 0, "ended": 0, "team_0_wins": 0, "team_1_wins": 0}
    started = time.time()
    tick_length = config.tick_length
    with torch.no_grad():
        for tick in range(1, arguments.ticks + 1):
            living = state.exists & state.alive
            if arguments.mode != "old":
                uniforms = scripted.draw_uniforms(envs, slots, generator, device)
                action, choice = new_call(state, uniforms)
                joint = choice.joint
                entropy = -(joint * joint.clamp_min(1e-30).log()).sum(-1)
                mine = (
                    living
                    if arguments.mode == "selfplay"
                    else living & (state.team == new_team[:, None])
                )
                entropy_sum += float(entropy[mine].sum())
                entropy_count += int(mine.sum())
            if arguments.mode != "selfplay":
                old_uniforms = rule_based.draw_uniforms(envs, slots, generator, device)
                old_action, old_memory = old_call(state, old_uniforms, old_memory)
                if arguments.mode == "old":
                    action = old_action
                else:
                    use_new = (state.team == new_team[:, None])[..., None]
                    action = torch.where(use_new, action, old_action)
            counts += torch.bincount(join_action(action)[living], minlength=JOINT)
            speed_sum += float((state.speed / state.constants["corner_speed"])[living].sum())
            store_sum += float((state.energy * state.constants["inverse_max_energy"])[living].sum())
            living_ticks += float(living.sum())
            state, out = game.step(
                config, state, action, game.draw_noise(config, envs, slots, generator, device)
            )
            kills += float(out.died.sum())
            done = out.done
            if bool(done.any()):
                result = out.result[done]
                outcome["ended"] += int(done.sum())
                outcome["draws"] += int((result == 2).sum())
                outcome["team_0_wins"] += int((result == 0).sum())
                outcome["team_1_wins"] += int((result == 1).sum())
                winner_new = result == new_team[done]
                outcome["wins"] += int(((result < 2) & winner_new).sum())
                outcome["losses"] += int(((result < 2) & ~winner_new).sum())
                fresh = rule_based.fresh_memory(envs, slots, generator, device)
                old_memory = {
                    name: torch.where(done[:, None], fresh[name], old_memory[name])
                    for name in old_memory
                }
                state = game.take_deal(
                    state, done, game.deal(config, envs, slots, sizes, generator, device)
                )
            if tick % arguments.report == 0 or tick == arguments.ticks:
                minutes = envs * tick * tick_length / 60.0
                total = counts.sum().clamp_min(1)
                summary = {
                    "mode": arguments.mode,
                    "set": changes,
                    "size": size,
                    "fields": arguments.fields,
                    "envs": envs,
                    "ticks": tick,
                    "usage": (counts.float() / total).tolist(),
                    "counts": counts.tolist(),
                    "entropy": entropy_sum / max(entropy_count, 1),
                    "kills_per_environment_minute": kills / minutes,
                    "mean_speed_corner": speed_sum / max(living_ticks, 1.0),
                    "mean_store": store_sum / max(living_ticks, 1.0),
                    **outcome,
                }
                arguments.out.parent.mkdir(parents=True, exist_ok=True)
                arguments.out.write_text(json.dumps(summary, indent=1))
                elapsed = time.time() - started
                eta = time.strftime(
                    "%H:%M", time.localtime(started + elapsed * arguments.ticks / tick)
                )
                print(
                    f"tick {tick}: kills/env-min {summary['kills_per_environment_minute']:.2f}"
                    f" entropy {summary['entropy']:.3f} speed {summary['mean_speed_corner']:.2f}"
                    f" store {summary['mean_store']:.2f} {outcome} ETA {eta}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
