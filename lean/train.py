"""Train the lean core: python -m lean.train [--smoke].

Wires the game, the scripted opponent, the model, collect, update and a periodic
crossover. `--seed` fixes the initial weights, the scenario draws and the data order
(compiled, the rollout's draws come from torch's default generator, seeded by `--seed`
too). `--compile` (the default) compiles the rollout tick and the update's loss.
"""

import argparse
import time

import torch

from lean import game, model, scripted
from lean.eval import crossover
from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    abandon,
    collect,
    make_optimiser,
    start,
    update,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="a few tiny updates on the CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--crossover-every", type=int, default=50)
    parser.add_argument("--crossover-games", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for torch; 0 keeps the default")
    parser.add_argument(
        "--game", choices=("v2", "core"), default="v2",
        help="v2: lean.game.GameConfig() with ghost bullets (labels for the hit head); core: GameConfig.core()",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="compile the rollout tick and the update's loss (lean.learn.compiled); --no-compile runs eagerly",
    )
    arguments = parser.parse_args()

    config = LearnConfig()
    envs, updates, ticks = arguments.envs, arguments.updates, config.segment_ticks
    crossover_every, crossover_games, crossover_sizes = arguments.crossover_every, arguments.crossover_games, [4]
    if arguments.smoke:
        config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1, snapshot_every=1)
        envs, updates, ticks = 4, 2, 8
        crossover_every, crossover_games, crossover_sizes = 2, 2, [1]

    device = torch.device(arguments.device)
    # The smoke run stays eager: on the CPU a compile costs more than the run.
    compiled = arguments.compile and not arguments.smoke
    if arguments.threads > 0:
        torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    game_config = game.GameConfig(ghost_bullets=True) if arguments.game == "v2" else game.GameConfig.core()
    scripted_config = scripted.ScriptedConfig()
    policy = model.Policy(model.ModelConfig()).to(device)
    optimiser = make_optimiser(config, policy)
    league = League(config.snapshot_every, config.snapshot_capacity, device, config.league_floor, config.league_decay)
    opponent = Opponent(scripted_config, league, config.self_play_share, config.persona_spread)
    state, carry = start(config, game_config, policy, opponent, envs, generator)

    steps = 0
    for index in range(updates):
        began = time.perf_counter()
        rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator, compiled=compiled)
        state, carry = rollout.state, rollout.carry
        steps += envs * ticks
        win_rate = league.scripted_win_rate()
        metrics = update(
            config, policy, optimiser, rollout, scripted_win_rate=win_rate, update_index=index, generator=generator,
            compiled=compiled,
        )
        replaced = league.maybe_add(policy, index + 1)
        if replaced is not None:
            state, carry = abandon(config, game_config, policy, opponent, state, carry, replaced, generator)
        rate = envs * ticks / (time.perf_counter() - began)
        warnings = [name.removeprefix("range_warning_") for name, value in metrics.items() if name.startswith("range_warning_") and value > 0]
        shown = " ".join(f"{name}={value:.4g}" for name, value in metrics.items() if not name.startswith("range_warning_"))
        rates = " ".join(f"{value:.2f}" for value in league.win_rates()[league.present].tolist())
        print(
            f"update {index} steps {steps} scripted_win_rate {win_rate:.3f} league_win_rates [{rates}] "
            f"steps_per_second {rate:.0f} {shown}",
            flush=True,
        )
        if warnings:
            print(f"update {index} WARNING value support: {' '.join(warnings)}", flush=True)
        if (index + 1) % crossover_every == 0:
            report = crossover(game_config, policy, scripted_config, crossover_sizes, crossover_games, generator)
            for size, entry in report["sizes"].items():
                print(f"crossover {size}: beats_up_to {entry['beats_up_to']} ratio {entry['ratio']:.2f}", flush=True)


if __name__ == "__main__":
    main()
