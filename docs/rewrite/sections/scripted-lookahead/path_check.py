"""How well the look-ahead's candidate paths (lean.scripted.paths.predict) match the game: every
ship of a few dealt states holds each of the 21 flight actions for 2 s in the real tick
(lean.game.step, no fire), and the predicted position, nose and speed at each look-ahead point
are compared with the flown ones.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/path_check.py
"""

import argparse
import math

import torch

from lean import game
from lean.contract import POWERS, STICKS
from lean.scripted.paths import LOOK_AHEAD, predict


def run(fields: int, envs: int, seed: int):
    config = game.GameConfig(field_count=fields, training_spawn=True, fog=False)
    generator = torch.Generator().manual_seed(seed)
    sizes = torch.full((envs, 2), 1, dtype=torch.long)
    state = game.reset(config, envs, sizes, generator, "cpu")
    # Some turning already: random nose rates and angles from a few ticks of random sticks.
    for _ in range(10):
        action = torch.stack(
            [
                torch.randint(0, STICKS, (envs, 2), generator=generator),
                torch.randint(0, POWERS, (envs, 2), generator=generator),
                torch.zeros(envs, 2, dtype=torch.long),
            ],
            -1,
        )
        state, _ = game.step(
            config, state, action, game.draw_noise(config, envs, 2, generator, "cpu")
        )
    paths = predict(state.constants, state.fields, state)
    errors = {"position": [], "nose": [], "speed": []}
    ticks = [round(t / config.tick_length) for t in LOOK_AHEAD]
    for flight in range(STICKS * POWERS):
        stick, power = divmod(flight, POWERS)
        flown = state
        action = torch.zeros(envs, 2, 3, dtype=torch.long)
        action[..., 0], action[..., 1] = stick, power
        # The pending action applies next tick: set it now so the hold starts at once.
        flown = flown.replace(pending=action.clone())
        for tick in range(1, ticks[-1] + 1):
            flown, _ = game.step(
                config, flown, action, game.draw_noise(config, envs, 2, generator, "cpu")
            )
            if tick in ticks:
                k = ticks.index(tick)
                alive = flown.alive
                dx = paths.x[..., flight, k] - flown.x
                dy = paths.y[..., flight, k] - flown.y
                nose = torch.atan2(
                    paths.nose_cos[..., flight, k] * flown.heading_sin
                    - paths.nose_sin[..., flight, k] * flown.heading_cos,
                    paths.nose_cos[..., flight, k] * flown.heading_cos
                    + paths.nose_sin[..., flight, k] * flown.heading_sin,
                )
                errors["position"].append((k, flight, torch.sqrt(dx * dx + dy * dy)[alive]))
                errors["nose"].append((k, flight, nose.abs()[alive]))
                errors["speed"].append(
                    (k, flight, (paths.speed[..., flight, k] - flown.speed)[alive])
                )
    print(f"fields {fields}: median / 90th percentile error by look-ahead point")
    for k, time in enumerate(LOOK_AHEAD):
        pos = torch.cat([e for kk, _, e in errors["position"] if kk == k])
        nose = torch.cat([e for kk, _, e in errors["nose"] if kk == k])
        speed = torch.cat([e for kk, _, e in errors["speed"] if kk == k]).abs()
        print(
            f"  t={time:4.2f} s position {pos.median():6.1f} / {pos.quantile(0.9):6.1f} m"
            f"  nose {math.degrees(nose.median()):5.1f}"
            f" / {math.degrees(nose.quantile(0.9)):5.1f} deg"
            f"  speed {speed.median():5.1f} / {speed.quantile(0.9):5.1f} m/s"
        )
    print("  by stick level at t=1 s (position median m, nose median deg):")
    for stick in range(STICKS):
        pos = torch.cat([e for kk, f, e in errors["position"] if kk == 2 and f // POWERS == stick])
        nose = torch.cat([e for kk, f, e in errors["nose"] if kk == 2 and f // POWERS == stick])
        print(f"    stick {stick}: {pos.median():6.1f} m  {math.degrees(nose.median()):5.1f} deg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():
        for fields in (0, 10):
            run(fields, arguments.envs, arguments.seed)
