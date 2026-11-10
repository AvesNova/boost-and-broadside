"""The Yemong step's kernels at a match engine's rows: the compiled controller step on
`rows` views, profiled (torch.profiler, CUDA time by kernel). Prints the top kernels and the
total. Usage: python bench/network_kernels.py policy=DIR [rows=2048] [top=25]
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

from bnb.contracts import Obs
from bnb.game import api
from bnb.runs import checkpoint
from bnb.runtime.realtime import Realtime
from bnb.runtime.train import model_controller


def main() -> None:
    options = dict(a.split("=", 1) for a in sys.argv[1:])
    device = torch.device("cuda")
    rows, top = int(options.get("rows", 2048)), int(options.get("top", 25))
    game = api.PRESETS[options.get("preset", "frontline")]
    model = checkpoint.load_policy(options["policy"], device).requires_grad_(False)
    controller = model_controller(model, game)
    sim = Realtime(game, compiled=False)
    obs = Obs(
        **{
            part: getattr(sim.obs, part)[0]
            .repeat(rows // 2, *([1] * (getattr(sim.obs, part).dim() - 2)))
            .to(device)
            for part in ("ships", "fields", "globals", "zones")
        }
    )
    reset = torch.zeros(rows, dtype=torch.bool, device=device)
    carry = controller.init_carry(rows, device)
    step = torch.compile(controller.step, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for _ in range(3):
            step(carry, obs, reset)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                step(carry, obs, reset)
            torch.cuda.synchronize()
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M; "
          f"config width {getattr(model.config, 'width', None)}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=top, max_name_column_width=70))


if __name__ == "__main__":
    main()
