"""Evaluate a checkpoint against the scripted agent: v1's crossover walk.

    python -m lean.eval DIR/rl.pt [--weights ema|uniform|live] [--sizes 4 8] [--games 64]

The network plays team 0 at each size in `--sizes`; the scripted team's size walks from
parity (lean.eval.crossover). `--games` is the games per match, or 0 for v1's formula
(lean.eval.games_for). Games deal the standard spawn of the core game.
"""

import argparse
from pathlib import Path

import torch

from lean import game, model, scripted
from lean.eval import crossover, games_for
from lean.learn import load_checkpoint, policy_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weights", choices=("ema", "uniform", "live"), default="ema")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--games", type=int, default=64, help="games per match; 0 for v1's games_for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    checkpoint = load_checkpoint(arguments.checkpoint, device)
    policy = model.Policy(checkpoint["model_config"]).to(device)
    policy.load_state_dict(policy_weights(checkpoint, arguments.weights))
    policy.eval()
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    games = arguments.games if arguments.games > 0 else games_for
    report = crossover(game.GameConfig(), policy, scripted.ScriptedConfig(), arguments.sizes, games, generator)
    print(f"{arguments.checkpoint} ({arguments.weights} weights, stage {checkpoint.get('stage')}, steps {checkpoint.get('steps')})")
    for size, entry in report["sizes"].items():
        matches = " ".join(f"{scripted_size}:{match['wins']}/{match['games']}" for scripted_size, match in entry["matches"].items())
        print(
            f"crossover {size}: beats_up_to {entry['beats_up_to']} ratio {entry['ratio']:.2f} "
            f"(rounds {report['rounds']}; wins by scripted size {matches})",
            flush=True,
        )


if __name__ == "__main__":
    main()
