"""The match engine's seat step alone (`MatchSeat.step`, its lanes' network forwards and the
scripted pass), compiled: its compile time, its traced ops and its step time, at a small
row count (compile time does not depend on it). Prints one JSON line.

Usage: `python bench/lane_probe.py policy=DIR lanes=16 block_environments=8 [cpu=1]`
"""

import json
import sys
import time

import torch
import torch._dynamo.utils as dynamo_utils
from torch.utils import _pytree

from bnb.contracts import Obs
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.league.matches import MatchSeat
from bnb.runs import checkpoint
from bnb.runtime.realtime import Realtime
from bnb.runtime.train import model_controller


def _top_times(count: int = 14) -> dict:
    names, values = dynamo_utils.compile_times(repr="csv", aggregate=True)
    pairs = sorted(zip(names, map(float, values)), key=lambda pair: -pair[1])
    return {name: round(value, 1) for name, value in pairs[:count]}


def main() -> None:
    """Compile the seat's step and print its compile time, step time and op count."""
    options = dict(a.split("=", 1) for a in sys.argv[1:])
    device = torch.device("cpu" if options.get("cpu") else "cuda")
    lanes = int(options.get("lanes", 16))
    block_environments = int(options.get("block_environments", 8))
    game = api.PRESETS[options.get("preset", "frontline")]
    model = checkpoint.load_policy(options["policy"], device).requires_grad_(False)
    sim = Realtime(game, compiled=False)
    agent = characteristics(sim.types, sim.rules)
    agent = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), agent)
    blocks = max(1, lanes // 2)
    seat = MatchSeat(
        [model_controller(model, game)],
        blocks,
        block_environments,
        agent,
        game.ships_per_team,
        device,
        lanes=(lanes,),
    )
    rows = seat.rows
    obs = Obs(
        **{
            part: getattr(sim.obs, part)[0]
            .repeat(rows // 2 + 1, *([1] * (getattr(sim.obs, part).dim() - 2)))[:rows]
            .to(device)
            for part in ("ships", "fields", "globals", "zones")
        }
    )
    reset = torch.zeros(rows, dtype=torch.bool, device=device)
    seat.lane_active[0].fill_(True)
    seat.scripted.fill_(False)
    carry = seat.init_carry(rows, device)
    step = torch.compile(seat.step, fullgraph=True, dynamic=False)

    def run():
        with torch.no_grad():  # the model applies its own autocast
            torch.manual_seed(0)
            action, _, _ = step(carry, obs, reset)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return action.packed()

    started = time.perf_counter()
    first = run()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(20):
        again = run()
    step_ms = (time.perf_counter() - started) / 20 * 1e3
    print(
        json.dumps(
            dict(
                lanes=lanes,
                rows=rows,
                device=device.type,
                compile_s=compile_seconds,
                step_ms=step_ms,
                ops=dynamo_utils.counters["stats"].get("calls_captured", 0),
                graphs=dynamo_utils.counters["stats"].get("unique_graphs", 0),
                repeatable=bool(torch.equal(first, again)),
                action_sum=int(first.sum()),
                compile_times=_top_times(),
            )
        )
    )
    if options.get("save"):
        torch.save(first.cpu(), options["save"])


if __name__ == "__main__":
    main()
