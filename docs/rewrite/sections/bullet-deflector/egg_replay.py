"""Replay videos (docs/rewrite/sections/replay/replay.py, unchanged) under a hit model: the
game's ship card is read through core.with_hit_model, so every deal of the render flies the
given rows (the deflector's turn, the spread, ...).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/bullet-deflector/egg_replay.py --model deflector_turn=0 \
        bullet_spread_mrad=10 -- --seeds 0 --ships-per-team 1 --no-fields --no-fog --out DIR

`bullet_spread_mrad` is the spread in milliradians at the bullet launch speed; other names are
core.HIT_MODEL's. Everything after `--` goes to replay.py.
"""

import sys
from pathlib import Path

import torch

from lean.game import core

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "replay"))
import replay  # noqa: E402


def main() -> None:
    split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    items = [item for item in sys.argv[1:split] if item != "--model"]
    model = {item.split("=")[0]: float(item.split("=")[1]) for item in items}
    original = core.ship_card
    if "bullet_spread_mrad" in model:
        launch = original(core.GameConfig())["bullet_launch_speed"]
        launch = float(torch.as_tensor(launch).flatten()[0])
        model["bullet_spread"] = model.pop("bullet_spread_mrad") * 1e-3 * launch

    built = {}  # by the original card (one per card: the cards are built once)

    def patched(config):
        card = original(config)
        if id(card) not in built:
            built[id(card)] = core.with_hit_model(card, **model)
        return built[id(card)]

    core.ship_card = patched
    print("hit model:", core.hit_model_of(patched(core.GameConfig())), flush=True)
    sys.argv = [sys.argv[0]] + sys.argv[split + 1 :]
    replay.main()


if __name__ == "__main__":
    main()
