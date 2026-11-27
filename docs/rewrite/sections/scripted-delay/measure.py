"""The decision-to-shot aim error and the hit rate per shot of the scripted agent in 1v1 self-play
(correctness review 2026-11-28, finding 4), for the agent before the pending-tick fix
(old_opponent.py, core 3f9769db) and after it (lean.scripted).

    measure.py old|new [--envs 128] [--ticks 900] [--rows JSON] [--device cpu] [--out JSON]

Both seats play the chosen variant. At each fire decision (p_fire > 0.5, a living controlled
ship, the game not finished on the next tick) the nose's error against the lead point (the
trigger's own aim, `opponent.aim`, true enemy state) is computed three ways: predicted from
the decision's pose (the old trigger's view), predicted from the pose after the pending tick
with the enemy one tick on (the new trigger's view), and actual at the shot's tick (the next
state's pose against the enemy's next state). Each predicted-minus-actual difference is divided
by the target's half angle side-on (atan(half the game's side-on hit width / distance)). Hit
rate: real bullets that hit an enemy over real shots fired.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import old_opponent  # noqa: E402

from lean import game  # noqa: E402
from lean.game.core import REAL, hit_width  # noqa: E402
from lean.scripted import ScriptedConfig, draw_uniforms, opponent  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("old", "new"))
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--ticks", type=int, default=900)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    module = old_opponent if args.variant == "old" else opponent
    rows = json.loads(Path(args.rows).read_text()) if args.rows else {}
    config = ScriptedConfig().replace(**rows)
    device = torch.device(args.device)
    generator = torch.Generator(device).manual_seed(args.seed)
    game_config = game.GameConfig()
    envs = args.envs
    sizes = torch.full((envs, 2), 1, dtype=torch.long, device=device)
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    tick = game_config.tick_length

    def read(name, dims=0):
        return opponent._lay(config.values[name], dims)

    old_d, new_d = [], []
    shots = hits = 0
    decisions = 0
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.ticks):
            uniforms = draw_uniforms(envs, slots, generator, device)
            action, choice = module.decide(config, state, None, None, uniforms)
            noise = game.draw_noise(game_config, envs, slots, generator, device)
            after, out = game.step(game_config, state, action, noise)
            # Shots and hits of this tick (the pending action's shot).
            K = state.bullets.shape[-1]
            written = (torch.arange(K, device=device) == (state.tick % K)[:, None])[:, None]
            bullet_state = torch.where(
                written, after.fired[..., None].to(torch.uint8), state.bullet_state
            )
            shots += int(after.fired.sum())
            hits += int(((out.shot_outcome == 1) & (bullet_state == REAL)).sum())
            # Fire decisions: the shot leaves on the next tick, from `after`'s pose.
            enemy_slot = (1 - torch.arange(slots, device=device)).expand(envs, slots)[..., None]
            enemy_slot = enemy_slot.clamp(0, slots - 1)
            firing = (choice.fire > 0.5) & choice.controlled & ~out.done[:, None] & after.alive
            if firing.any():
                _, _, error_now = opponent.aim(state, state, enemy_slot, read, 0.0)
                ahead = opponent.after_pending(state)
                _, _, error_ahead = opponent.aim(ahead, state, enemy_slot, read, tick)
                distance, _, error_shot = opponent.aim(after, after, enemy_slot, read, 0.0)
                widths = opponent._widths_at(after.constants, enemy_slot)
                half = 0.5 * hit_width(widths, torch.zeros_like(distance))
                half_angle = torch.atan(half / distance)
                mask = firing[..., None]
                old_d.append(((error_now - error_shot).abs() / half_angle)[mask].cpu())
                new_d.append(((error_ahead - error_shot).abs() / half_angle)[mask].cpu())
                decisions += int(mask.sum())
            fresh = game.deal(game_config, envs, slots, sizes, generator, device)
            state = game.take_deal(after, out.done, fresh)
    seconds = time.perf_counter() - started

    def summary(values):
        values = torch.cat(values).double()
        q = torch.quantile(values, torch.tensor([0.5, 0.75, 0.9], dtype=torch.double))
        return {
            "median": float(q[0]),
            "p75": float(q[1]),
            "p90": float(q[2]),
            "over_half_width": float((values > 1.0).double().mean()),
        }

    minutes = args.ticks * envs * 2 * tick / 60.0
    result = {
        "variant": args.variant,
        "rows": args.rows,
        "envs": envs,
        "ticks": args.ticks,
        "fire_decisions": decisions,
        "from_decision_pose": summary(old_d),
        "from_pose_after_pending": summary(new_d),
        "shots": shots,
        "hits": hits,
        "hit_rate": hits / max(shots, 1),
        "shots_per_ship_minute": shots / minutes,
        "seconds": seconds,
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
