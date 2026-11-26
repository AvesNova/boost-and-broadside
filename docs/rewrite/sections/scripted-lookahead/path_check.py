"""How well the look-ahead's candidate paths (lean.scripted.paths.predict) match the game: ships
flying straight at 0.6, 1.0 and 1.5 x corner speed (half store, random places and headings in
the default game, with and without fields) hold each of the 21 flight actions in the real tick
(lean.game.step, no fire) for 2 s; the predicted position, flight-path heading and nose at each
look-ahead point against the flown ones (the flown state interpolated linearly between the two
ticks around the point). Prints median / 90th percentile errors by stick level (both sides and
all powers pooled) and speed, as Markdown.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/path_check.py
"""

import argparse
import math

import torch

from lean import game
from lean.contract import POWERS, STICKS
from lean.game import card
from lean.scripted import paths as paths_module
from lean.scripted.paths import LEVEL_OF_STICK, predict

SPEEDS = (0.6, 1.0, 1.5)


def wrapped(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def run(fields: int, envs: int, seed: int) -> dict:
    config = game.GameConfig(field_count=fields, fog=False)
    generator = torch.Generator().manual_seed(seed)
    state = game.reset(config, envs, torch.ones(envs, 2, dtype=torch.long), generator, "cpu")
    corner = state.constants["corner_speed"]
    errors = {}
    for factor in SPEEDS:
        zeros = torch.zeros_like(state.x)
        start = state.replace(
            speed=torch.full_like(state.x, factor * corner),
            angle=zeros,
            angle_cos=torch.ones_like(state.x),
            angle_sin=zeros,
            nose_rate=zeros,
        )
        if fields:  # the cached medium at the ships' places (the tick reads it)
            index, gradient_x, gradient_y = game.core.medium(start.fields, start.x, start.y)
            start = start.replace(log_index=index, gradient_x=gradient_x, gradient_y=gradient_y)
        start = game.core.with_velocity(start)
        LOOK_AHEAD = paths_module.LOOK_AHEAD
        at_power = [
            predict(start.constants, start.fields, start, torch.full_like(start.team, power))
            for power in range(POWERS)
        ]
        ticks = [t / config.tick_length for t in LOOK_AHEAD]
        last = math.ceil(ticks[-1])
        for flight in range(STICKS * POWERS):
            stick, power = divmod(flight, POWERS)
            paths, candidate = at_power[power], stick
            action = torch.zeros(envs, 2, 3, dtype=torch.long)
            action[..., 0], action[..., 1] = stick, power
            flown = start.replace(pending=action.clone())
            track = [flown]
            for _ in range(last):
                noise = game.draw_noise(config, envs, 2, generator, "cpu")
                flown, _ = game.step(config, flown, action, noise)
                track.append(flown)
            for k, count in enumerate(ticks):
                low = math.floor(count)
                share = count - low
                a, b = track[low], track[min(low + 1, last)]
                x = (1 - share) * a.x + share * b.x
                y = (1 - share) * a.y + share * b.y

                def angle_at(cos, sin, a=a, b=b, share=share):
                    first = torch.atan2(getattr(a, sin), getattr(a, cos))
                    second = torch.atan2(getattr(b, sin), getattr(b, cos))
                    return first + share * wrapped(second - first)

                nose = angle_at("heading_cos", "heading_sin")
                velocity = angle_at("velocity_x", "velocity_y")
                alive = a.alive & b.alive
                position = torch.hypot(
                    paths.x[..., candidate, k] - x, paths.y[..., candidate, k] - y
                )
                predicted_nose = torch.atan2(
                    paths.nose_sin[..., candidate, k], paths.nose_cos[..., candidate, k]
                )
                predicted_path = torch.atan2(
                    paths.path_sin[..., candidate, k], paths.path_cos[..., candidate, k]
                )
                rows = errors.setdefault((factor, LEVEL_OF_STICK[stick], k), ([], [], []))
                rows[0].append(position[alive])
                rows[1].append(wrapped(predicted_path - velocity).abs()[alive])
                rows[2].append(wrapped(predicted_nose - nose).abs()[alive])
    return errors


def cell(values: list[torch.Tensor], scale: float = 1.0) -> str:
    joined = torch.cat(values) * scale
    return f"{joined.median():.1f} / {joined.quantile(0.9):.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--look-ahead", default=None, help="points, e.g. 0.25,0.5,1,2")
    parser.add_argument("--retarget", type=int, default=None, help="1 / 0: paths.RETARGET")
    parser.add_argument("--fields", default="0,10")
    arguments = parser.parse_args()
    if arguments.look_ahead:
        paths_module.LOOK_AHEAD = tuple(float(t) for t in arguments.look_ahead.split(","))
    if arguments.retarget is not None:
        paths_module.RETARGET = bool(arguments.retarget)
    LOOK_AHEAD = paths_module.LOOK_AHEAD
    torch.set_num_threads(4)
    names = ("straight",) + card.LEVEL_NAMES
    quantities = (("position m", 1.0), ("path deg", 180 / math.pi), ("nose deg", 180 / math.pi))
    with torch.no_grad():
        for fields in (int(item) for item in arguments.fields.split(",")):
            errors = run(fields, arguments.envs, arguments.seed)
            print(f"\n### {'With' if fields else 'Without'} fields\n")
            heads = " | ".join(f"{t} s" for t in LOOK_AHEAD)
            print(f"| speed / corner | level | error | {heads} |")
            print("|---|---|---|" + "---|" * len(LOOK_AHEAD))
            for factor in SPEEDS:
                for level, name in enumerate(names):
                    for index, (quantity, scale) in enumerate(quantities):
                        cells = " | ".join(
                            cell(errors[(factor, level, k)][index], scale)
                            for k in range(len(LOOK_AHEAD))
                        )
                        print(f"| {factor} | {name} | {quantity} | {cells} |")


if __name__ == "__main__":
    main()
