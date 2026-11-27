"""Round 2, step 3: the crowd measures per team size, from the yardstick mirror (fields, fog).

    crowd.py OUT.json --sizes 1 2 4 8 16 --ticks 3600

Per size, every 15 ticks for every living ship: the three candidate measures of
lean.scripted (`crowd_measure`): 0 the scored enemies within 2 gun ranges (weighted, as
store_crowd), 1 the known enemies, 2 my team's living ships; their mean, quantiles and the
crowd weight w = (m - 1) / (m - 1 + 1) each gives. Games are redealt when done (the eval's
study clock per size, skill2.CLOCKS). Uncompiled; bounded by `--ticks` per size.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402

from lean import game, scripted  # noqa: E402
from lean.game import core  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.opponent import (  # noqa: E402
    CROWD_RANGE,
    _at,
    dead_reckon,
    known_enemies,
    nearest_enemies,
)


@torch.no_grad()
def measures(state, roam_cutoff):
    c = state.constants
    enemies = core.enemy_view(state)
    if enemies is not state:
        enemies = dead_reckon(enemies, state.ticks_since_seen.to(state.x.dtype) * c["tick_length"])
    slot, weight, _ = nearest_enemies(state, enemies, roam_cutoff)
    dx = _at(enemies.x, slot) - state.x[..., None]
    dy = _at(enemies.y, slot) - state.y[..., None]
    near = ((dx * dx + dy * dy) < (CROWD_RANGE * c["gun_range"]) ** 2).float()
    near = (weight * near).sum(-1)
    known = known_enemies(state, enemies, roam_cutoff)[0].sum(-1).float()
    alive = state.exists & state.alive
    own = ((state.team[:, :, None] == state.team[:, None, :]) & alive[:, None, :]).sum(-1).float()
    return alive, near, known, own


def run(size, ticks, envs, seed):
    here = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = skill2.config_of(size)
    generator = torch.Generator(device=here).manual_seed(seed)
    sizes = torch.full((envs, 2), size, dtype=torch.long, device=here)
    state = game.reset(config, envs, sizes, generator, here)
    slots = state.x.shape[1]
    yard = ScriptedConfig()
    gathered = {"near": [], "known": [], "own": []}
    started = time.perf_counter()
    for tick in range(ticks):
        if tick % 15 == 0:
            alive, near, known, own = measures(state, yard.values["roam_cutoff"])
            for name, value in (("near", near), ("known", known), ("own", own)):
                gathered[name].append(value[alive].cpu().numpy())
        uniforms = scripted.draw_uniforms(envs, slots, generator, here)
        action = scripted.act(yard, state, None, None, uniforms)
        state, out = game.tick(config, state, action, generator)
        if bool(out.done.any()):
            state = game.take_deal(
                state, out.done, game.deal(config, envs, slots, sizes, generator, here)
            )
        if tick % 600 == 0:
            print(f"  {size}v{size} tick {tick}, {time.perf_counter() - started:.0f} s",
                  file=sys.stderr, flush=True)
    result = {}
    for name, items in gathered.items():
        values = np.concatenate(items)
        excess = np.clip(values - 1.0, 0.0, None)
        w = excess / (excess + 1.0)
        result[name] = {
            "mean": float(values.mean()),
            "quantiles": np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "weight_mean": float(w.mean()),
            "weight_quantiles": np.quantile(w, [0.1, 0.5, 0.9]).tolist(),
            "share_duel": float((values <= 1.0).mean()),
            "samples": int(values.size),
            # Histogram on quarter-ship bins (bin centres, shares): any half's mean weight.
            "histogram": [
                [float(centre), float(share)]
                for centre, share in zip(
                    *np.unique(np.round(values * 4.0) / 4.0, return_counts=True), strict=True
                )
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--envs", type=int, default=256)
    arguments = parser.parse_args()
    out = Path(arguments.out)
    result = json.loads(out.read_text()) if out.exists() else {}
    for size in arguments.sizes:
        key = f"{size}v{size}"
        if key in result:
            continue
        result[key] = run(size, arguments.ticks, arguments.envs, 5000 + size)
        out.write_text(json.dumps(result, indent=1))
        print(key, json.dumps({k: round(v["mean"], 2) for k, v in result[key].items()}),
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
