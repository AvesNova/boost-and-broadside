"""The diagnostics' cost (diagnostics spec 6.6) at a 5v5 training cell with Yemong: whole
updates (collect, the rollout's series, learn, the one readback and the record) with the
learner's diagnostics on and off, alternating in blocks after a warm-up, so drift in the
machine's clocks hits both alike; then the cadence groups on their own (their stopwatch,
amortised over `CADENCE` updates); then the rollout's per-tick game sums, which run in both
("off" keeps them), timed alone on the rollout's buffers and scaled to an update's ticks.
Prints `RESULT {json}`; run it under the benchmark lock.

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


def game_sums_seconds(trainer, repeats: int = 50) -> float:
    """Seconds per tick of the rollout's game sums (`analytics.resolve_tick`,
    `perceive_tick` and the outcome ledger) on the rollout's own buffers, with zero
    events."""
    from bnb.contracts import REWARD_LEVELS, Events
    from bnb.game import api
    from bnb.learn.rewards import REWARD_EVENTS
    from bnb.runtime import analytics

    rollout, carry = trainer.rollout, trainer.rollout.carry
    B, N, device = rollout.environments, rollout.ship_count, rollout.device
    flags = dict(dtype=torch.bool, device=device)
    events = Events(
        damage=torch.zeros((B, N, N), device=device),
        boundary_damage=torch.zeros((B, N), device=device),
        died=torch.zeros((B, N), **flags),
        fired=torch.zeros((B, N), **flags),
        spawned=torch.zeros((B, N), **flags),
        killer=torch.zeros((B, N), dtype=torch.int32, device=device),
        done=torch.zeros(B, **flags),
        shield_lost=torch.zeros((B, N), device=device),
        shield_restored=torch.zeros((B, N), device=device),
        objective_progress=torch.zeros((B, N), device=device),
        objective=torch.zeros((B, N), device=device),
        result=torch.zeros(B, dtype=torch.int32, device=device),
    )
    reward = torch.zeros((B, N, len(REWARD_LEVELS)), device=device)
    event_rewards = torch.zeros((B, N, len(REWARD_EVENTS)), device=device)
    ships, globals_ = carry["obs"]["ships"], carry["obs"]["globals"]
    views = rollout.lineup.environments[rollout.learner]
    outcome = torch.full((views.numel(), 4), 0.25, device=device)
    elapsed = torch.zeros(views.numel(), device=device)

    def tick() -> None:
        analytics.resolve_tick(
            carry["game"], carry["game_sums"], events, reward, event_rewards,
            carry["truth"][:, 0], rollout.rules.tick_length,
        )  # fmt: skip
        analytics.perceive_tick(
            carry["game_sums"], ships, ships, carry["truth"], globals_, globals_,
            rollout.ships_per_team, rollout.rules.tick_length, api.MAP_SCALE_RADIUS,
        )  # fmt: skip
        if "outcome" in carry:
            analytics.outcome_record(carry["outcome"], outcome, elapsed)
            analytics.outcome_score(
                carry["outcome"], carry["outcome_tables"], events.done[views],
                events.done[views], elapsed,
            )  # fmt: skip

    saved = {name: _clone(carry[name]) for name in ("game", "outcome") if name in carry}
    tick()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        tick()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / repeats
    for name, tree in saved.items():  # in place: the captured graph reads these buffers
        for kept, value in zip(_leaves(carry[name]), _leaves(tree), strict=True):
            kept.copy_(value)
    rollout.take_games()
    return seconds


def _clone(tree):
    from torch.utils import _pytree

    return _pytree.tree_map_only(torch.Tensor, torch.clone, tree)


def _leaves(tree) -> list[torch.Tensor]:
    from torch.utils import _pytree

    return [leaf for leaf in _pytree.tree_leaves(tree) if isinstance(leaf, torch.Tensor)]


def main() -> None:
    """Time the updates and print the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="mixed")
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--per-block", type=int, default=3)
    parser.add_argument("--cadences", type=int, default=3)
    options = parser.parse_args()

    from bench.cell import train_config
    from bnb.runs.recording import Recorder
    from bnb.runs.triggers import Triggers
    from bnb.runtime import metrics
    from bnb.runtime.train import Trainer

    torch.manual_seed(0)
    config = train_config(None, options.environments, options.preset)
    trainer = Trainer(config, torch.device("cuda"))
    recorder = Recorder(None, trainer, None, Triggers())
    for diagnostics in (True, False, True):  # compile both variants, then warm up
        trainer.ppo.diagnostics = diagnostics
        print(f"warm-up, diagnostics {diagnostics}: {timed_update(trainer, recorder)[0]:.2f} s")
    times = {True: [], False: []}
    records = {True: [], False: []}
    for block in range(options.blocks):
        for diagnostics in (True, False) if block % 2 == 0 else (False, True):
            trainer.ppo.diagnostics = diagnostics
            for _ in range(options.per_block):
                if diagnostics and metrics.due(trainer.updates_done + 1):
                    trainer.updates_done += 1  # the cadence's updates are measured below
                update, record = timed_update(trainer, recorder)
                print(f"diagnostics {diagnostics}: {update:.2f} s", flush=True)
                times[diagnostics].append(update)
                records[diagnostics].append(record)
    on, off = statistics.median(times[True]), statistics.median(times[False])
    trainer.ppo.diagnostics = True
    cadence_seconds, cadence_updates = [], []
    for _ in range(options.cadences):
        trainer.updates_done = metrics.CADENCE - 1  # the next update measures the cadence
        cadence_updates.append(timed_update(trainer, recorder)[0])
        cadence_seconds.append(trainer.ppo.cadence_stopwatch.seconds())
        print(
            f"cadence: {cadence_updates[-1]:.2f} s, {cadence_seconds[-1]:.2f} s of it", flush=True
        )
    cadence = statistics.median(cadence_seconds)
    tick = game_sums_seconds(trainer)
    ticks = trainer.config.length * trainer.config.shards
    result = dict(
        preset=options.preset,
        environments=options.environments,
        updates_each=len(times[True]),
        update_seconds_on=on,
        update_seconds_off=off,
        per_update_overhead=on / off - 1.0,
        record_seconds=statistics.median(records[True]),
        cadence_seconds=cadence,
        cadence_update_seconds=statistics.median(cadence_updates),
        cadence_amortised_overhead=cadence / metrics.CADENCE / off,
        game_sums_seconds_per_tick=tick,
        game_sums_per_update_overhead=tick * ticks / off,
        load_average=os.getloadavg(),
        device=torch.cuda.get_device_name(),
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
