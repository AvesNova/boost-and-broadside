"""Scripted games for the look-ahead agent's checks: action usage, the teacher's entropy, kills
per minute, and duels between two look-ahead agents (the rule-based agent it replaced was
removed in round 3).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/measure.py \
        --mode selfplay --size 4 --fields 10 --envs 256 --ticks 5400 --out DIR

Modes: "selfplay" (the look-ahead yardstick on both teams: the 42 actions' counts over living
ship-ticks, the mean entropy of its joint distribution, kills per environment-minute, mean
speed and store, games ended); "duel" (this agent on team 0 in the first half of the
environments and team 1 in the second, another look-ahead agent (`--other-set`,
`--other-look-ahead`) on the other team: wins, draws, losses by this agent's side). Fog on,
the default game otherwise (`--fields` fields). Finished games are re-dealt. `--set
name=value ...` changes look-ahead rows. Every `--report` ticks it writes the totals so far
to --out (JSON) and prints them with the ETA as a clock time. `--compile` compiles the
agent's call.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from lean import game, scripted
from lean.contract import FIRES, JOINT, POWERS, STICKS, compile_dynamic, join_action
from lean.game import card
from lean.game.core import steady_lookup
from lean.scripted import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selfplay", "duel"), default="selfplay")
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
    parser.add_argument(
        "--spread", type=float, default=0.0, help="persona spread (0: the yardstick on all)"
    )
    parser.add_argument("--no-fog", action="store_true", help="fog off (author: 1v1 runs)")
    parser.add_argument("--look-ahead", default=None, help="points, e.g. 0.25,0.5,1,2")
    parser.add_argument(
        "--other-look-ahead", default=None, help="duel: the other look-ahead agent's points"
    )
    parser.add_argument(
        "--other-set",
        nargs="*",
        default=None,
        help="duel: the other agent's rows name=value (default: the same rows as --set)",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    def points(text):
        return tuple(float(t) for t in text.split(","))

    main_points = points(arguments.look_ahead) if arguments.look_ahead else paths.LOOK_AHEAD
    other_points = points(arguments.other_look_ahead) if arguments.other_look_ahead else None
    paths.LOOK_AHEAD = main_points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = game.GameConfig(field_count=arguments.fields, fog=not arguments.no_fog)
    envs, size = arguments.envs, arguments.size
    generator = torch.Generator(device).manual_seed(arguments.seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    changes = {item.split("=")[0]: float(item.split("=")[1]) for item in arguments.set}
    new_config = scripted.ScriptedConfig().replace(**changes)

    # Personas drawn per ship at each deal (training's mechanism); None: the yardstick.
    persona = scripted.personas(new_config, envs, slots, generator, device, arguments.spread)

    def new_call(state, uniforms, persona):
        return scripted.decide(new_config, state, None, persona, uniforms)

    other_config = new_config
    if arguments.other_set is not None:
        other_config = scripted.ScriptedConfig().replace(
            **{item.split("=")[0]: float(item.split("=")[1]) for item in arguments.other_set}
        )

    def other_call(state, uniforms):
        return scripted.decide(other_config, state, None, None, uniforms)[0]

    if arguments.compile:
        new_call = compile_dynamic(new_call)
        other_call = compile_dynamic(other_call)
    # Duel: which team the look-ahead agent flies, per environment (seats alternated).
    new_team = (torch.arange(envs, device=device) >= envs // 2).long()
    counts = torch.zeros(JOINT, dtype=torch.long, device=device)
    entropy_sum, entropy_count = 0.0, 0
    kills = speed_sum = store_sum = living_ticks = 0.0
    # Action changes from one tick to the next of a living ship (the power flicker).
    previous = torch.full((envs, slots, 3), -1, dtype=torch.long, device=device)
    power_switches = stick_switches = compared = 0.0
    # The distribution's own movement: total variation between a ship's consecutive-tick
    # distributions (joint, stick marginal, power marginal), look-ahead ships only.
    previous_joint = previous_mine = None
    variation = {"joint": 0.0, "stick": 0.0, "power": 0.0}
    # Grouped by effect (round 3): the stick distribution over the side (left / straight /
    # right) and over the signed steady turn rate (the 1-Wasserstein distance between the two
    # ticks' stick distributions on the 7 sticks' rates at the ship's speed, store and pending
    # power now, from the game's steady table; in deg/s, and over the span of the 7 rates: a
    # mass fraction times a rate fraction, comparable to the total variation).
    grouped = {"side": 0.0, "rate_w1_deg_s": 0.0, "rate_w1_of_span": 0.0}
    side_of_stick = torch.sign(torch.tensor(card.STICK_LEVELS, device=device)).long() + 1
    level_of_stick = torch.tensor(paths.LEVEL_OF_STICK, device=device)
    stick_sign = torch.sign(torch.tensor(card.STICK_LEVELS, device=device))
    variation_count = 0.0
    outcome = {"wins": 0, "draws": 0, "losses": 0, "ended": 0, "team_0_wins": 0, "team_1_wins": 0}
    started = time.time()
    tick_length = config.tick_length

    def steady_rates_now(state):
        """Each stick's signed steady turn rate (E, N, STICKS) at the ship's speed, store and
        pending power now (the game's steady-flight table), no field."""
        c = state.constants

        def laid(value):
            return value[..., None] if isinstance(value, torch.Tensor) else value

        p = SimpleNamespace(
            **{
                name: laid(c[name])
                for name in (
                    "steady_low_speed",
                    "inverse_steady_step",
                    "inverse_fade",
                    "max_energy",
                )
            }
        )
        rate = steady_lookup(
            p,
            c["steady_table"],
            state.type[..., None],
            state.speed[..., None],
            level_of_stick,
            state.pending[..., 1:2],
            state.energy[..., None],
        )[0]
        return rate * stick_sign

    with torch.no_grad():
        for tick in range(1, arguments.ticks + 1):
            living = state.exists & state.alive
            uniforms = scripted.draw_uniforms(envs, slots, generator, device)
            action, choice = new_call(state, uniforms, persona)
            joint = choice.joint
            entropy = -(joint * joint.clamp_min(1e-30).log()).sum(-1)
            mine = (
                living
                if arguments.mode == "selfplay"
                else living & (state.team == new_team[:, None])
            )
            entropy_sum += float(entropy[mine].sum())
            entropy_count += int(mine.sum())
            if previous_joint is not None:
                both = mine & previous_mine
                now_parts = joint.view(envs, slots, STICKS, POWERS, FIRES)
                then_parts = previous_joint.view(envs, slots, STICKS, POWERS, FIRES)
                for name, now_p, then_p in (
                    ("joint", joint, previous_joint),
                    ("stick", now_parts.sum((-1, -2)), then_parts.sum((-1, -2))),
                    ("power", now_parts.sum((-1, -3)), then_parts.sum((-1, -3))),
                ):
                    distance = 0.5 * (now_p - then_p).abs().sum(-1)
                    variation[name] += float(distance[both].sum())
                variation_count += float(both.sum())
                stick_now, stick_then = now_parts.sum((-1, -2)), then_parts.sum((-1, -2))

                def by_side(stick):
                    return torch.zeros(stick.shape[:-1] + (3,), device=device).index_add_(
                        -1, side_of_stick, stick
                    )

                side = 0.5 * (by_side(stick_now) - by_side(stick_then)).abs().sum(-1)
                grouped["side"] += float(side[both].sum())
                rates = steady_rates_now(state)  # (E, N, STICKS) rad/s, signed
                order = rates.argsort(-1)
                sorted_rates = rates.gather(-1, order)
                gaps = sorted_rates.diff(dim=-1)
                cumulative = (stick_now - stick_then).gather(-1, order).cumsum(-1)[..., :-1]
                w1 = (cumulative.abs() * gaps).sum(-1)
                span = (sorted_rates[..., -1] - sorted_rates[..., 0]).clamp_min(1e-6)
                grouped["rate_w1_deg_s"] += float(torch.rad2deg(w1)[both].sum())
                grouped["rate_w1_of_span"] += float((w1 / span)[both].sum())
            previous_joint, previous_mine = joint, mine
            if arguments.mode == "duel":  # the other look-ahead agent on the other team
                paths.LOOK_AHEAD = other_points or main_points
                other_uniforms = scripted.draw_uniforms(envs, slots, generator, device)
                other_action = other_call(state, other_uniforms)
                paths.LOOK_AHEAD = main_points
                use_new = (state.team == new_team[:, None])[..., None]
                action = torch.where(use_new, action, other_action)
            counts += torch.bincount(join_action(action)[living], minlength=JOINT)
            measured = living & (previous[..., 0] >= 0)
            if arguments.mode == "duel":
                measured = measured & (state.team == new_team[:, None])
            power_switches += float((measured & (action[..., 1] != previous[..., 1])).sum())
            stick_switches += float((measured & (action[..., 0] != previous[..., 0])).sum())
            compared += float(measured.sum())
            previous = torch.where(living[..., None], action, -1)
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
                previous = torch.where(done[:, None, None], -1, previous)
                if previous_joint is not None:
                    previous_mine = previous_mine & ~done[:, None]
                if persona is not None:
                    fresh_persona = scripted.personas(
                        new_config, envs, slots, generator, device, arguments.spread
                    )
                    persona = {
                        name: torch.where(done[:, None], fresh_persona[name], persona[name])
                        for name in persona
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
                    "look_ahead": main_points,
                    "other_look_ahead": other_points,
                    "other_set": arguments.other_set,
                    "size": size,
                    "fields": arguments.fields,
                    "fog": not arguments.no_fog,
                    "spread": arguments.spread,
                    "envs": envs,
                    "ticks": tick,
                    "usage": (counts.float() / total).tolist(),
                    "counts": counts.tolist(),
                    "entropy": entropy_sum / max(entropy_count, 1),
                    "kills_per_environment_minute": kills / minutes,
                    "mean_speed_corner": speed_sum / max(living_ticks, 1.0),
                    "mean_store": store_sum / max(living_ticks, 1.0),
                    "total_variation_per_tick": {
                        name: total / max(variation_count, 1.0) for name, total in variation.items()
                    },
                    "grouped_stick_change_per_tick": {
                        name: total / max(variation_count, 1.0) for name, total in grouped.items()
                    },
                    "power_switches_per_ship_second": power_switches
                    / max(compared, 1.0)
                    / tick_length,
                    "stick_switches_per_ship_second": stick_switches
                    / max(compared, 1.0)
                    / tick_length,
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
                    f" store {summary['mean_store']:.2f}"
                    f" power switches/s {summary['power_switches_per_ship_second']:.2f}"
                    f" TV {summary['total_variation_per_tick']}"
                    f" grouped {summary['grouped_stick_change_per_tick']}"
                    f" {outcome} ETA {eta}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
