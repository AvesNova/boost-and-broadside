"""The diagnostics' cost (diagnostics spec 6.6) at a 5v5 training cell with Yemong: whole
updates (collect, the rollout's series, learn, the one readback and the record) with the
learner's diagnostics on and off, alternating in blocks after a warm-up, so drift in the
machine's clocks hits both alike. Prints `RESULT {json}`; run it under the benchmark lock.

    uv run --no-sync python -m bench.diagnostics_cost [--preset mixed] [--environments N]
"""

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402


def timed_update(trainer, recorder) -> tuple[float, float]:
    """(the whole update's seconds, the record's seconds), synchronised."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    row = trainer.update()
    recorded = time.perf_counter()
    recorder.record(row)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    return finished - started, finished - recorded


def main() -> None:
    """Time the updates and print the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="mixed")
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--per-block", type=int, default=3)
    options = parser.parse_args()

    from bench.cell import train_config
    from bnb.runs.recording import Recorder
    from bnb.runs.triggers import Triggers
    from bnb.runtime.train import Trainer

    torch.manual_seed(0)
    config = train_config(None, options.environments, options.preset)
    trainer = Trainer(config, torch.device("cuda"))
    recorder = Recorder(None, trainer, None, Triggers())
    for diagnostics in (True, False, True):  # compile both variants, then warm up
        trainer.ppo.diagnostics = diagnostics
        timed_update(trainer, recorder)
    times = {True: [], False: []}
    records = {True: [], False: []}
    for block in range(options.blocks):
        for diagnostics in (True, False) if block % 2 == 0 else (False, True):
            trainer.ppo.diagnostics = diagnostics
            for _ in range(options.per_block):
                update, record = timed_update(trainer, recorder)
                times[diagnostics].append(update)
                records[diagnostics].append(record)
    on, off = statistics.median(times[True]), statistics.median(times[False])
    result = dict(
        preset=options.preset,
        environments=options.environments,
        updates_each=len(times[True]),
        update_seconds_on=on,
        update_seconds_off=off,
        per_update_overhead=on / off - 1.0,
        record_seconds=statistics.median(records[True]),
        load_average=os.getloadavg(),
        device=torch.cuda.get_device_name(),
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
