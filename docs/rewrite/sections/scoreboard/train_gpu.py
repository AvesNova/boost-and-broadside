"""Scoreboard, training on the GPU: collect alone, update alone and the whole loop, eager
against compiled, with peak memory and the first update's time (the compile).

    PYTHONPATH=.:src python docs/rewrite/sections/scoreboard/train_gpu.py \
        [--compile | --no-compile] [--envs 256] [--updates 3] [--game v2|core] \
        [--minibatches 8] [--snapshots 8] [--cached-counts]

The loop is lean/train.py's (collect, update, league addition and abandon) with the
league filled with `--snapshots` copies of the initial weights before the first rollout,
so every league member is present as in a run's steady state (snapshot_every is set
beyond the run, so nothing is added or abandoned during it). `--cached-counts` reads the
league's member counts once per rollout instead of every tick (wrong groups once a game
re-deals, same work): the difference is what the per-tick read-back costs.

Prints one line per update (collect s, update s, loop s, environment steps per second of
each) and a summary: the first update (the compile, cold or warm by the inductor cache
in TORCHINDUCTOR_CACHE_DIR), the steady mean over the later updates, peak memory
allocated and reserved, and the seconds from process start to the end of the first update.
"""

import argparse
import dataclasses
import importlib
import statistics
import time

PROCESS_START = time.perf_counter()

import torch  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import League, LearnConfig, Opponent, collect, make_optimiser, start, update  # noqa: E402
collect_module = importlib.import_module("lean.learn.collect")


def cached_counts_league_actions():
    """league_actions with the member counts read once per rollout (see the docstring)."""
    original = collect_module.league_actions
    cache = {}

    def patched(opponent, views, state, carry, scripted_action, generator, move=collect_module.snapshot_move):
        league = opponent.league
        if league is None or league.snapshots == 0:
            return original(opponent, views, state, carry, scripted_action, generator, move)
        if "counts" not in cache:
            cache["counts"] = torch.bincount(carry.member, minlength=league.members).tolist()
        counts = cache["counts"]
        envs = state.exists.shape[0]
        first = envs - carry.member.shape[0]
        action, opponent_carry = scripted_action[first:], carry.opponent
        order = torch.argsort(carry.member, stable=True)
        team = 1 - carry.learner_team[first:]
        offset = carry.position_offset[first:]
        begin = counts[0]
        for index in range(1, league.members):
            count = counts[index]
            if count == 0:
                continue
            chosen = order[begin:begin + count]
            begin += count
            member_action, new_carry = move(
                league.frozen[index - 1], views, chosen + first, team.index_select(0, chosen),
                collect_module.index_carry(opponent_carry, chosen), offset.index_select(0, chosen), generator,
            )
            opponent_carry = opponent_carry.index_copy(collect_module.carry_axis(opponent_carry), chosen, new_carry)
            action = action.index_copy(0, chosen, member_action)
        return action, opponent_carry

    collect_module.league_actions = patched
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--game", choices=("v2", "core"), default="v2")
    parser.add_argument("--minibatches", type=int, default=None, help="default: LearnConfig's")
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--cached-counts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    device = torch.device("cuda")
    config = LearnConfig()
    if arguments.minibatches is not None:
        config = dataclasses.replace(config, minibatches=arguments.minibatches)
    config = dataclasses.replace(config, snapshot_every=10**9)
    envs, ticks = arguments.envs, config.segment_ticks
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    game_config = game.GameConfig(ghost_bullets=True) if arguments.game == "v2" else game.GameConfig.core()
    policy = model.Policy(model.ModelConfig()).to(device)
    optimiser = make_optimiser(config, policy)
    league = League(1, config.snapshot_capacity, device, config.league_floor, config.league_decay)
    for index in range(arguments.snapshots):
        league.maybe_add(policy, index + 1)
    league.every = config.snapshot_every
    opponent = Opponent(scripted.ScriptedConfig(), league, config.self_play_share, config.persona_spread)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    cache = cached_counts_league_actions() if arguments.cached_counts else None
    compiled = arguments.compile
    print(
        f"compile={compiled} envs={envs} ticks={ticks} game={arguments.game} minibatches={config.minibatches} "
        f"epochs={config.epochs} snapshots={league.snapshots} cached_counts={arguments.cached_counts} "
        f"torch {torch.__version__} {torch.cuda.get_device_name()}",
        flush=True,
    )

    rows = []
    first_update_end = None
    for index in range(arguments.updates):
        if cache is not None:
            cache.clear()
        torch.cuda.synchronize()
        began = time.perf_counter()
        rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator, compiled=compiled)
        state, carry = rollout.state, rollout.carry
        torch.cuda.synchronize()
        collected = time.perf_counter()
        update(
            config, policy, optimiser, rollout, scripted_win_rate=league.scripted_win_rate(), update_index=index,
            generator=generator, compiled=compiled,
        )
        replaced = league.maybe_add(policy, index + 1)
        if replaced is not None:
            raise RuntimeError("the league should not change during the measurement")
        del rollout
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if first_update_end is None:
            first_update_end = ended
        steps = envs * ticks
        row = (collected - began, ended - collected, ended - began)
        rows.append(row)
        print(
            f"update {index}: collect {row[0]:.2f} s ({steps / row[0]:.0f}/s) update {row[1]:.2f} s ({steps / row[1]:.0f}/s) "
            f"loop {row[2]:.2f} s ({steps / row[2]:.0f}/s)",
            flush=True,
        )
    steady = rows[1:] or rows
    mean = [statistics.mean(row[k] for row in steady) for k in range(3)]
    steps = envs * ticks
    print(
        f"SUMMARY compile={compiled} envs={envs} game={arguments.game} minibatches={config.minibatches} "
        f"cached_counts={arguments.cached_counts} | first update: collect {rows[0][0]:.1f} s update {rows[0][1]:.1f} s "
        f"| steady steps/s: collect {steps / mean[0]:.0f} update {steps / mean[1]:.0f} loop {steps / mean[2]:.0f} "
        f"| peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB reserved "
        f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB | start to first update end "
        f"{first_update_end - PROCESS_START:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
