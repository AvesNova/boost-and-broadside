"""crossover mode: how many scripted agents does it take to beat the trained team?

For each trained-team size T, the trained policy (team 0) plays the scripted agent
(team 1) at growing scripted counts S, and the trained win rate is measured over a
batch of parallel games. As S rises the win rate falls; the crossover S* is the
smallest scripted count at which the trained team wins fewer than half its games.
The headline per T is "beats up to S*-1 scripted".

The same trained weights play every size — the model is token-based and scale-
invariant — so this is one checkpoint measured across a grid of matchups. An
exponential-then-bisection search over S keeps it to ~log(range) batches per T.

Writes ``<output_dir>/crossover.json`` (full win-rate curves) and prints a table.
"""

import json
from pathlib import Path

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.modes.agent_factory import resolve_agent_spec
from boost_and_broadside.modes.capture import _final_checkpoint, _find_run_dir
from boost_and_broadside.modes.collect import evaluate_matchup

# Collision physics allocates a (B, N*bullets, N) tensor, so peak memory grows as
# B*N^2. Hold B*N^2 under this budget (tuned for an 8 GB GPU) by shrinking the
# batch for big battles; a floor keeps the win-rate estimate meaningful.
_COLLISION_BUDGET = 4_000_000
_MIN_ENVS = 48


def _envs_for(n_ships: int, max_envs: int) -> int:
    """Parallel games to run at this ship count, shrunk to fit the memory budget."""
    return max(_MIN_ENVS, min(max_envs, _COLLISION_BUDGET // (n_ships * n_ships)))


def run_crossover_mode(
    run_spec: str,
    trained_counts: list[int],
    ship_config: ShipConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    num_envs: int = 256,
    max_total_ships: int = 320,
    output_dir: str = "docs/crossover",
) -> dict:
    """Find, per trained-team size, the scripted count that tips wins below 50%."""
    run_dir = _find_run_dir(run_spec, checkpoint_dir)
    checkpoint = _final_checkpoint(run_dir)
    base_env = EnvConfig(**torch.load(str(checkpoint), map_location="cpu", weights_only=False)[
        "env_config"
    ])

    trained = resolve_agent_spec(str(checkpoint), ship_config, model_config, device, num_ships=2)
    scripted = resolve_agent_spec("scripted", ship_config, model_config, device)
    print(f"\n=== crossover: {run_dir.name}  ({num_envs} games/matchup, {device}) ===\n")

    rows: list[dict] = []
    for trained_n in trained_counts:
        curve: dict[int, float] = {}

        def win_rate(scripted_n: int, trained_n: int = trained_n, curve=curve) -> float:
            """Trained-team win fraction at T vs S (ties count against trained)."""
            if scripted_n not in curve:
                # The policy only slices the first (T+S) tokens as ships; nothing in
                # it is sized by ship count, so one loaded module plays every matchup.
                total = trained_n + scripted_n
                trained.agent._num_ships = total
                games = _envs_for(total, num_envs)
                t0_wins, _, _, _ = evaluate_matchup(
                    trained, scripted, trained_n, scripted_n, games, ship_config,
                    base_env, device,
                )
                curve[scripted_n] = t0_wins / games
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  {trained_n:>3}v{scripted_n:<4} trained wins {curve[scripted_n]:6.1%}"
                      f"  ({games} games)")
            return curve[scripted_n]

        crossover = _find_crossover(win_rate, trained_n, max_total_ships)
        beats_up_to = None if crossover is None else crossover - 1
        rows.append(
            {
                "trained": trained_n,
                "crossover": crossover,  # first scripted count with trained < 50%
                "beats_up_to": beats_up_to,  # largest scripted team still beaten
                "capped": crossover is None,
                "win_rate_at_beats_up_to": curve.get(beats_up_to) if beats_up_to else None,
                "win_rate_at_crossover": curve.get(crossover) if crossover else None,
                "curve": {str(k): v for k, v in sorted(curve.items())},
            }
        )

    result = {
        "run": run_dir.name,
        "num_envs": num_envs,
        "max_total_ships": max_total_ships,
        "rows": rows,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crossover.json").write_text(json.dumps(result, indent=2))
    _print_table(rows, max_total_ships)
    print(f"\n  wrote {out / 'crossover.json'}")
    return result


def _find_crossover(win_rate, trained_n: int, max_total_ships: int) -> int | None:
    """Smallest scripted S with trained win rate < 50%, by exponential + bisection.

    Returns None if no crossover is found before the total-ship cap.
    """
    cap = max_total_ships - trained_n  # largest scripted count that fits
    if cap < 1:
        return None

    # Bracket: lo = a scripted count still won (>=50%), hi = one lost (<50%).
    lo: int | None = None
    hi: int | None = None
    probe = max(1, trained_n)  # trained is expected to win at parity
    while probe <= cap:
        if win_rate(probe) >= 0.5:
            lo = probe
            probe = max(probe + 1, probe * 2)
        else:
            hi = probe
            break
    if hi is None:
        return None  # never dropped below 50% within the cap

    if lo is None:
        # Lost even at the first probe; walk down to the true crossover.
        while probe > 1 and win_rate(probe - 1) < 0.5:
            probe -= 1
        return probe

    # Bisect for the smallest S in (lo, hi] with win rate < 0.5.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if win_rate(mid) < 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def _print_table(rows: list[dict], max_total_ships: int) -> None:
    print(f"\n  {'trained':>8} {'beats up to':>12} {'crossover':>10} "
          f"{'win@beats':>10} {'win@cross':>10}")
    print(f"  {'-' * 54}")
    for row in rows:
        if row["capped"]:
            beats = f">{max_total_ships - row['trained']}"
            cross = wb = wc = "—"
        else:
            beats = f"{row['beats_up_to']} scripted"
            cross = str(row["crossover"])
            wb = f"{row['win_rate_at_beats_up_to']:.0%}" if row["win_rate_at_beats_up_to"] else "—"
            wc = f"{row['win_rate_at_crossover']:.0%}" if row["win_rate_at_crossover"] else "—"
        print(f"  {row['trained']:>8} {beats:>12} {cross:>10} {wb:>10} {wc:>10}")
