"""crossover mode: how many scripted-controlled ships beat the learned team?

For each trained-team size T, the trained policy (team 0) plays the scripted agent
(team 1) at growing scripted counts S, and the trained win rate is measured over a
batch of parallel games. As S rises the win rate falls; the crossover S* is the
smallest scripted count at which the trained team wins fewer than half its games.
The headline per T is "beats up to S*-1 scripted".

The same trained weights play every size — the model is token-based and scale-
invariant — so this is one checkpoint measured across a grid of matchups. An
exponential-then-bisection search over S keeps it to ~log(range) batches per T.

Writes a run-owned ``crossover`` artifact holding the full outcome-count curves,
and prints a table. The figures built from it are rendered by ``bnb publish``.
"""

import torch

from boost_and_broadside.artifacts import Artifact, ArtifactRecipe, ArtifactStore, file_sha256
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import resolve_agent_spec
from boost_and_broadside.evaluation.match import evaluate_matchup
from boost_and_broadside.evaluation.run_catalog import (
    resolve_exact_run,
    select_final_training_checkpoint,
)
from boost_and_broadside.evaluation.subjects import describe_agent, describe_environment
from boost_and_broadside.train.rl.checkpoint_schema import (
    load_checkpoint_payload,
    require_observation_schema,
)

# Collision physics allocates a (B, N*bullets, N) tensor, so peak memory grows as
# B*N^2. Hold B*N^2 under this budget (tuned for an 8 GB GPU) by shrinking the
# batch for big battles; a floor keeps the win-rate estimate meaningful.
_COLLISION_BUDGET = 4_000_000
_MIN_ENVS = 48
_SCHEMA_VERSION = 2


def _envs_for(n_ships: int, max_envs: int) -> int:
    """Parallel games to run at this ship count, shrunk to fit the memory budget."""
    return max(_MIN_ENVS, min(max_envs, _COLLISION_BUDGET // (n_ships * n_ships)))


def parse_counts(spec: str) -> list[int]:
    """Trained-team sizes from a spec: '1,2,4' or ranges like '1-64' (mixable)."""
    out: list[int] = []
    for token in spec.split(","):
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif token:
            out.append(int(token))
    return out


def _outcome_record(
    wins: int, losses: int, ties: int, games: int, mean_episode_steps: float
) -> dict[str, int | float | bool]:
    """Lossless per-matchup result stored in the crossover curve."""
    if wins + losses + ties != games:
        raise ValueError(f"outcome counts ({wins}+{losses}+{ties}) do not match games={games}")
    return {
        "counts_available": True,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "games": games,
        "win_rate": wins / games,
        "mean_episode_steps": mean_episode_steps,
    }


def _curve_rate(record: float | dict) -> float:
    """Read a win rate from current count records or the legacy rate-only artifact."""
    return float(record["win_rate"] if isinstance(record, dict) else record)


def run_crossover_mode(
    run_spec: str,
    trained_counts: list[int],
    ship_config: ShipConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    num_envs: int = 256,
    max_total_ships: int = 320,
    store: ArtifactStore | None = None,
) -> dict:
    """Find, per trained-team size, the scripted count that tips wins below 50%."""
    run_dir = resolve_exact_run(run_spec, checkpoint_dir).path
    checkpoint = select_final_training_checkpoint(run_dir).path
    checkpoint_data = load_checkpoint_payload(checkpoint, map_location="cpu")
    require_observation_schema(checkpoint_data, str(checkpoint))
    base_env = EnvConfig(**checkpoint_data["env_config"])

    store = store or ArtifactStore(checkpoint_root=checkpoint_dir)
    recipe = ArtifactRecipe(
        artifact_type="crossover",
        result_schema_version=_SCHEMA_VERSION,
        subjects={
            "run": run_dir.name,
            "trained": {
                "kind": "policy",
                "checkpoint": checkpoint.name,
                "sha256": file_sha256(checkpoint),
                "global_step": int(checkpoint_data.get("global_step", 0)),
            },
            "scripted": describe_agent("scripted"),
        },
        parameters={
            "trained_counts": sorted(set(trained_counts)),
            "num_envs": num_envs,
            "max_total_ships": max_total_ships,
            "environment": describe_environment(base_env),
        },
    )
    # Resume: keep any rows already computed so a crash mid-sweep loses nothing.
    artifact, resumed = store.open_resumable(recipe, store.run_owner(run_dir.name))
    by_trained = _load_progress(artifact) if resumed else {}

    trained = resolve_agent_spec(str(checkpoint), ship_config, model_config, device, num_ships=2)
    scripted = resolve_agent_spec("scripted", ship_config, model_config, device)
    print(f"\n=== crossover: {run_dir.name}  ({num_envs} games/matchup, {device}) ===\n")

    # Ascending order makes the crossover monotonic, so each size warm-starts from
    # the previous one's result (whether just computed or loaded from a prior run).
    hint: int | None = None
    for trained_n in sorted(trained_counts):
        if trained_n in by_trained:
            hint = by_trained[trained_n]["crossover"] or hint
            continue
        curve: dict[int, dict[str, int | float | bool]] = {}

        def win_rate(scripted_n: int, trained_n: int = trained_n, curve=curve) -> float:
            """Trained-team win fraction at T vs S (ties count against trained)."""
            if scripted_n not in curve:
                # The policy only slices the first (T+S) tokens as ships; nothing in
                # it is sized by ship count, so one loaded module plays every matchup.
                total = trained_n + scripted_n
                trained.agent._num_ships = total
                games = _envs_for(total, num_envs)
                t0_wins, t1_wins, ties, mean_episode_steps = evaluate_matchup(
                    trained,
                    scripted,
                    trained_n,
                    scripted_n,
                    games,
                    ship_config,
                    base_env,
                    device,
                )
                curve[scripted_n] = _outcome_record(
                    t0_wins, t1_wins, ties, games, mean_episode_steps
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"  {trained_n:>3}v{scripted_n:<4} policy wins "
                    f"{_curve_rate(curve[scripted_n]):6.1%}  "
                    f"({t0_wins}W/{t1_wins}L/{ties}T, {games} games)"
                )
            return _curve_rate(curve[scripted_n])

        # Warm start from the previous size's crossover (monotonic in T); the very
        # first size has no hint and falls back to an exponential bracket.
        crossover = _find_crossover(win_rate, trained_n, max_total_ships, hint)
        beats_up_to = None if crossover is None else crossover - 1
        by_trained[trained_n] = {
            "trained": trained_n,
            "crossover": crossover,  # first scripted count with trained < 50%
            "beats_up_to": beats_up_to,  # largest scripted team still beaten
            "capped": crossover is None,
            "win_rate_at_beats_up_to": (
                _curve_rate(curve[beats_up_to]) if beats_up_to in curve else None
            ),
            "win_rate_at_crossover": (
                _curve_rate(curve[crossover]) if crossover in curve else None
            ),
            "curve": {str(k): v for k, v in sorted(curve.items())},
        }
        if crossover is not None:
            hint = crossover
        _save(artifact, run_dir.name, num_envs, max_total_ships, by_trained)  # after each size

    result = _save(artifact, run_dir.name, num_envs, max_total_ships, by_trained)
    artifact.complete()
    _print_table(result["rows"], max_total_ships)
    print(f"\n  wrote {artifact.path}")
    return result


def _load_progress(artifact: Artifact) -> dict[int, dict]:
    """Rows already computed, keyed by trained-team size (empty if none)."""
    if not artifact.has("result.json"):
        return {}
    rows = artifact.read_json().get("rows", [])
    for row in rows:
        row["curve"] = {
            scripted_n: (
                record
                if isinstance(record, dict)
                else {"counts_available": False, "win_rate": float(record)}
            )
            for scripted_n, record in row.get("curve", {}).items()
        }
    return {r["trained"]: r for r in rows}


def _save(
    artifact: Artifact, run: str, num_envs: int, max_total_ships: int, by_trained: dict
) -> dict:
    result = {
        "schema_version": _SCHEMA_VERSION,
        "run": run,
        "num_envs": num_envs,
        "max_total_ships": max_total_ships,
        "rows": [by_trained[t] for t in sorted(by_trained)],
    }
    artifact.write_json(result)
    return result


def _find_crossover(
    win_rate, trained_n: int, max_total_ships: int, hint: int | None = None
) -> int | None:
    """Smallest scripted S with trained win rate < 50%.

    With a ``hint`` (the previous size's crossover), walk locally from it — win
    rate is monotonic in S, so the boundary is a step or two away. Without one,
    fall back to an exponential bracket. Returns None if no crossover is found
    before the total-ship cap.
    """
    cap = max_total_ships - trained_n  # largest scripted count that fits
    if cap < 1:
        return None

    if hint is not None:
        probe = max(1, min(hint, cap))
        if win_rate(probe) >= 0.5:  # still winning at the hint — climb until it loses
            while probe < cap and win_rate(probe + 1) >= 0.5:
                probe += 1
            return probe + 1 if probe < cap else None
        while probe > 1 and win_rate(probe - 1) < 0.5:  # losing — descend to the boundary
            probe -= 1
        return probe

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
    print(
        f"\n  {'policy':>8} {'beats up to':>12} {'crossover':>10} "
        f"{'win@beats':>10} {'win@cross':>10}"
    )
    print(f"  {'-' * 54}")
    for row in rows:
        if row["capped"]:
            beats = f">{max_total_ships - row['trained']}"
            cross = wb = wc = "—"
        else:
            beats = f"{row['beats_up_to']} scripted"
            cross = str(row["crossover"])
            wb = (
                f"{row['win_rate_at_beats_up_to']:.0%}"
                if row["win_rate_at_beats_up_to"] is not None
                else "—"
            )
            wc = (
                f"{row['win_rate_at_crossover']:.0%}"
                if row["win_rate_at_crossover"] is not None
                else "—"
            )
        print(f"  {row['trained']:>8} {beats:>12} {cross:>10} {wb:>10} {wc:>10}")
