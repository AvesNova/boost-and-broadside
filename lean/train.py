"""Train the lean core: python -m lean.train [--smoke].

Wires the game, the scripted opponent, the model, collect, update and a periodic
crossover. `--seed` fixes the initial weights, the scenario draws and the data order.
"""

import argparse
import time

import torch

from lean import game, model, scripted
from lean.eval import crossover
from lean.learn import LearnConfig, Opponent, SnapshotPool, collect, make_optimiser, shaping_scale, start, update


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
    arguments = parser.parse_args()

    config = LearnConfig()
    envs, updates, ticks = arguments.envs, arguments.updates, config.segment_ticks
    crossover_every, crossover_games, crossover_sizes = arguments.crossover_every, arguments.crossover_games, [4]
    if arguments.smoke:
        config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1, snapshot_every=1)
        envs, updates, ticks = 4, 2, 8
        crossover_every, crossover_games, crossover_sizes = 2, 2, [1]

    device = torch.device(arguments.device)
    if arguments.threads > 0:
        torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    game_config = game.GameConfig()
    scripted_config = scripted.ScriptedConfig()
    policy = model.Policy(model.ModelConfig()).to(device)
    optimiser = make_optimiser(config, policy)
    pool = SnapshotPool(policy, config.snapshot_every, config.snapshot_capacity)
    opponent = Opponent(scripted_config, None, config.scripted_share, config.persona_spread)
    state, carry = start(config, game_config, policy, opponent, envs, generator)

    win_rate, steps = 0.0, 0
    for index in range(updates):
        began = time.perf_counter()
        opponent.snapshot = pool.sample(generator)
        rollout = collect(
            config, game_config, policy, opponent, state, carry, ticks, generator, shaping=shaping_scale(config, steps)
        )
        state, carry = rollout.state, rollout.carry
        steps += envs * ticks
        games = int(rollout.scripted_games)
        if games > 0:
            momentum = config.win_rate_momentum
            win_rate = momentum * win_rate + (1.0 - momentum) * int(rollout.scripted_wins) / games
        metrics = update(
            config, policy, optimiser, rollout, scripted_win_rate=win_rate, update_index=index, generator=generator
        )
        pool.maybe_add(policy, index + 1)
        rate = envs * ticks / (time.perf_counter() - began)
        shown = " ".join(f"{name}={value:.4g}" for name, value in metrics.items())
        print(f"update {index} steps {steps} win_rate {win_rate:.3f} steps_per_second {rate:.0f} {shown}", flush=True)
        if (index + 1) % crossover_every == 0:
            report = crossover(game_config, policy, scripted_config, crossover_sizes, crossover_games, generator)
            for size, entry in report["sizes"].items():
                print(f"crossover {size}: beats_up_to {entry['beats_up_to']} ratio {entry['ratio']:.2f}", flush=True)


if __name__ == "__main__":
    main()
