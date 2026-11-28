"""Times RolloutStore.end_segment alone at the training sizes (queue item 10): 4 segments of
128 ticks, 3,072 seats (2,048 environments, half self-play), 8 slots, on the GPU. The hit
labels and episode ends are random (about one end per seat per 512 ticks).

    PYTHONPATH=. python docs/rewrite/sections/fixes-10/end_segment_timing.py
"""

import time

import torch

from lean.learn.store import RolloutStore

segments, ticks, seats, slots = 4, 128, 3072, 8
device = torch.device("cuda")
generator = torch.Generator(device=device).manual_seed(0)
store = RolloutStore(segments, stream=True)
for repeat in range(6):
    times = []
    for segment in range(segments):
        store.begin_segment(
            segment,
            ticks,
            torch.zeros(1, seats, 1, device=device),
            {"position_offset": torch.zeros(seats, 2, device=device)},
        )
        if "done" not in store.resident:
            store.resident["done"] = torch.zeros(
                ticks, store.views + 1, dtype=torch.bool, device=device
            )
        columns = store._columns(segment)
        store.resident["done"][:, columns] = (
            torch.rand(ticks, seats, device=device, generator=generator) < 1 / 512
        )
        label = torch.randint(-1, 3, (ticks, seats, slots), device=device, generator=generator)
        previous = torch.randint(-1, 3, (ticks, seats, slots), device=device, generator=generator)
        bootstrap = torch.zeros(seats, slots, 3, device=device)
        centres = torch.zeros(3, 63, device=device)
        torch.cuda.synchronize()
        began = time.perf_counter()
        store.end_segment(segment, label, bootstrap, centres, previous if segment else None)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    print(
        f"repeat {repeat}: end_segment ms per segment " + " ".join(f"{t * 1e3:.2f}" for t in times)
    )
