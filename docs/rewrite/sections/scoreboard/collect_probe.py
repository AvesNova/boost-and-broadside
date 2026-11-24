"""Scoreboard, collect alone at parity (review item V3), the league full or empty;
then one counted segment checking that no work is skipped. Setup as in train_gpu.py.

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/collect_probe.py \\
        --snapshots 8|0 [--warm 2] [--timed 8] [--out rows.jsonl]
"""

import argparse
import json
import time

import torch

import lean
from lean import game, model, scripted
from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    RolloutStore,
    WeightAverages,
    collect,
    rollout,
    start,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--warm", type=int, default=2)
    parser.add_argument("--timed", type=int, default=8)
    parser.add_argument("--envs", type=int, default=LearnConfig().environments)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    print("lean from", lean.__file__, flush=True)

    device = torch.device("cuda")
    config = LearnConfig()
    envs, ticks, segments = arguments.envs, config.segment_ticks, config.segments_per_update
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig()).to(device)
    league = League(config, device)
    averages = WeightAverages(policy, config)
    averages.update(policy, 10**9)
    nobody = torch.zeros(0, dtype=torch.long, device=device)
    for index in range(arguments.snapshots):
        step = league.schedule.points(10**12)[index]
        league.after_update(policy, averages, step - 1, step, nobody, nobody)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(segments, config.stream_rollout)
    setting = {
        "snapshots": league.snapshots,
        "envs": envs,
        "ticks": ticks,
        "present": league.present_count(),
        "league_envs": int(carry.member.shape[0]),
        "gpu": torch.cuda.get_device_name(),
    }
    print(setting, flush=True)

    def out(row):
        print(row, flush=True)
        if arguments.out:
            with open(arguments.out, "a") as file:
                file.write(json.dumps({**setting, **row}) + "\n")

    def segment_run(number):
        nonlocal state, carry
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = collect(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            ticks,
            generator,
            compiled=True,
            store=store,
            segment=number % segments,
        )
        state, carry = result.state, result.carry
        torch.cuda.synchronize()
        return time.perf_counter() - began

    for number in range(arguments.warm):
        seconds = segment_run(number)
        out(
            {
                "phase": "warm",
                "segment": number,
                "s": seconds,
                "steps_per_s": envs * ticks / seconds,
            }
        )
    timed = []
    for number in range(arguments.timed):
        seconds = segment_run(arguments.warm + number)
        timed.append(seconds)
        out(
            {
                "phase": "timed",
                "segment": number,
                "s": seconds,
                "steps_per_s": envs * ticks / seconds,
            }
        )
    out(
        {
            "phase": "summary",
            "collect_steps_per_s": envs * ticks * len(timed) / sum(timed),
            "best": envs * ticks / min(timed),
            "worst": envs * ticks / max(timed),
        }
    )

    # Counted segment: per tick, who moved each league environment's opponent side.
    counts = {
        "league_calls": 0,
        "slot_calls": 0,
        "advance_calls": 0,
        "act_calls": 0,
        "write_tick": 0,
    }
    gpu = {
        name: torch.zeros((), dtype=torch.long, device=device)
        for name in (
            "scripted_rows",
            "network_rows",
            "over_width",
            "scripted_mismatch",
            "env_ticks",
            "live_seats",
            "unrouted",
        )
    }
    original_league_actions = rollout.league_actions
    from lean.learn.compiled import graphs

    compiled = graphs()
    original_act, original_advance = compiled.act, compiled.advance

    def counted_league_actions(
        config_, opponent_, views, carry_, scripted_action, generator_, step
    ):
        counts["league_calls"] += 1
        league_envs = carry_.member.shape[0]
        width = (
            opponent_.league.table.positions.shape[0] if opponent_.league.table is not None else 0
        )
        routed = torch.zeros(league_envs, dtype=torch.long, device=device)

        def counted_step(*args):
            counts["slot_calls"] += 1
            slot = args[4]
            mine = carry_.slot == slot
            gpu["network_rows"] += mine.sum()
            gpu["over_width"] += (mine.sum() > width).long()
            routed.add_(mine.long())
            return step(*args)

        action, opponent_carry = original_league_actions(
            config_, opponent_, views, carry_, scripted_action, generator_, counted_step
        )
        scripted_rows = carry_.slot == -1
        gpu["scripted_rows"] += scripted_rows.sum()
        routed.add_(scripted_rows.long())
        gpu["unrouted"] += (routed != 1).sum()
        tail = scripted_action[scripted_action.shape[0] - league_envs :]
        gpu["scripted_mismatch"] += ((action != tail).flatten(1).any(1) & scripted_rows).sum()
        return action, opponent_carry

    def counted_act(*args):
        counts["act_calls"] += 1
        result = original_act(*args)
        gpu["live_seats"] += result[3].shape[0]
        return result

    def counted_advance(*args):
        counts["advance_calls"] += 1
        gpu["env_ticks"] += args[4].exists.shape[0]
        return original_advance(*args)

    original_write = store.write_tick

    def counted_write(*args):
        counts["write_tick"] += 1
        return original_write(*args)

    rollout.league_actions = counted_league_actions
    compiled.act, compiled.advance = counted_act, counted_advance
    store.write_tick = counted_write
    segment_run(0)
    out(
        {
            "phase": "counted",
            **counts,
            **{name: int(value) for name, value in gpu.items()},
            "table_slots": len(league.table.slots) if league.table is not None else 0,
            "table_width": int(league.table.positions.shape[0]) if league.table is not None else 0,
            "capacity": int(league.capacity.max())
            if hasattr(league.capacity, "max")
            else league.capacity,
        }
    )


if __name__ == "__main__":
    main()
