"""elo_calibrate mode: re-rate a finished training run from raw match counts.

A training run leaves behind two kinds of evidence. Its frozen ladder
checkpoints are files on disk that can be replayed against each other as many
times as wanted, so their ratings are limited only by how much compute is spent.
Its live and averaged policies are gone — each existed in one form for a single
update — so all that survives of them is the win/loss/tie record captured at the
time.

This mode uses the first to calibrate the second. It plays an adaptive
tournament among the stationary players until every rating is pinned to a target
standard error, then replays each update's stored record against those now-known
opponents to recover what the live policy was actually worth at that moment.
The result is a training curve that no longer depends on where the in-training
K-factor filter happened to drift.

Every rating is fit twice, under both draw conventions (see TIE_MODES). The two
differ only where draws are common, which in this game means the weak end of the
ladder, and the half-win fit is the one directly comparable to the in-training
curve. Both come from the same match counts, so carrying both is free.

Reported ratings are shifted so the scripted controller reads
SCRIPTED_ANCHOR_ELO (see that constant for why). Because the raw win/tie
matrices are persisted, ``refit=True`` reruns every fit and artifact from the
stored counts without playing a single game.

Writes to the run's checkpoint directory:
    elo_calibrated.json      ratings, per-update curve, batch stats, and the raw
                             win/tie matrices, so any later refit needs no replay
    elo_calibration/*.png    live curve vs in-training, the two draw conventions,
                             per-checkpoint ratings, convergence, and draw rates
    elo_calibration/         the calibrated ratings in the W&B export format
      history.jsonl,           (see modes/elo_calibrate_history.py), so one loader
      summary.json             reads them alongside the run's in-training history
"""

import json
from pathlib import Path

import numpy as np

from boost_and_broadside.config import EloCalibrateConfig, ShipConfig
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.run_catalog import resolve_legacy_elo_run
from boost_and_broadside.evaluation.tournament import (
    TIE_MODES,
    BatchStat,
    Player,
    Progress,
    Tournament,
    build_players,
    effective_wins,
    load_run_config,
    run_tournament,
)
from boost_and_broadside.train.rl.bradley_terry import (
    fit_bradley_terry,
    fit_single_rating,
)

# Opponents that are not stationary and so cannot be tournament players. "avg"
# changes every update, exactly like the live policy it is measured against.
_NON_STATIONARY = frozenset({"avg"})

# How draws enter the likelihood.
#
#   "half_win" — a draw is half a win to each side. The default, and the
#                convention Elo has always used. It treats drawing as evidence
#                of parity, which is the whole content of a draw: tying every
#                game against a 3600-rated opponent implies a rating of 3600.
#   "decisive" — draws dropped; the rating answers the narrower question "who
#                wins, given that somebody wins".
#
# Note that neither introduces a draw *parameter*. Davidson and Rao-Kupper do,
# and both are scale-invariant in the strengths, so they can only express draw
# frequency as a function of the rating gap — which measurement here contradicts,
# since draws track the absolute level of a matchup instead. That rules those two
# models out; it says nothing against half-win scoring, which models no draw
# process at all and simply fits the expected score.
#
# Dropping draws is the costly option in this game. Against the random anchor the
# live policy's whole-run record is 2794W/10L/1120T: decisive-only extracts a
# Fisher information of 10 from it, half-win extracts 487 — a 49x difference,
# because a 99.6% win rate sits where p(1-p) has almost nothing left to give.
# "decisive" is kept as a diagnostic rather than a default: it is the check for a
# policy farming draws, which would earn parity under half-win scoring without
# ever having to win. A large disagreement between the two is that signature.
#
# Both are fit from the same match counts, so carrying both costs no extra play.

# Reporting anchor: every rating is shifted so the scripted controller reads
# 1000. Scripted is the one opponent shared across runs and fleet scales, so
# pinning it makes ratings comparable between them, and its link to the field
# is far tighter than random's — every trained agent beats random so decisively
# that those games barely constrain the scale. Random still plays in the
# tournament and is reported like any other player; it lands below zero here.
SCRIPTED_ANCHOR_ELO = 1000.0


def calibrate_live_curve(
    history: list[dict], ratings: dict[str, float], tie_mode: str = "decisive"
) -> list[dict]:
    """Recover the live and averaged policies' ratings, update by update.

    Each update's record is refit against opponents whose ratings are now known,
    which is what makes this independent of the in-training filter. The averaged
    policy is a second stage: its only opponent is the live policy, so it can
    only be placed once the live rating for that same update is known.
    """
    curve = []
    for record in history:
        counts = record.get("counts") or {}
        opponents, wins, losses = [], [], []
        for label, (win, loss, tie) in counts.items():
            if label in _NON_STATIONARY or label not in ratings:
                continue
            share = 0.0 if tie_mode == "decisive" else 0.5 * tie
            opponents.append(ratings[label])
            wins.append(win + share)
            losses.append(loss + share)
        if not opponents:
            continue
        live, live_stderr = fit_single_rating(
            np.array(opponents), np.array(wins, dtype=float), np.array(losses, dtype=float)
        )
        point = {
            "update": record["update"],
            "global_step": record["global_step"],
            "live_training": record["live"],
            "live_calibrated": live,
            "live_stderr": live_stderr,
            "games": int(sum(wins) + sum(losses)),
        }
        avg_counts = counts.get("avg")
        if avg_counts is not None and np.isfinite(live):
            # Stored from the live policy's perspective, so invert for the avg.
            live_win, live_loss, live_tie = avg_counts
            share = 0.0 if tie_mode == "decisive" else 0.5 * live_tie
            avg, avg_stderr = fit_single_rating(
                np.array([live]),
                np.array([float(live_loss) + share]),
                np.array([float(live_win) + share]),
            )
            point["avg_training"] = record["avg"]
            point["avg_calibrated"] = avg
            point["avg_stderr"] = avg_stderr
        curve.append(point)
    return curve


def training_tie_rates(
    history: list[dict], curve: list[dict], ratings: dict[str, float]
) -> list[dict]:
    """Per-update draw rates from the training record, placed by rating level.

    The tournament only ever pits trained agents against each other, so its
    draw rates all come from the strong end of the scale. The training record is
    the only evidence covering the weak end — where the live policy was still
    near-random and stalemates were common — and that is exactly the range that
    decides whether draws track a matchup's level or its gap.
    """
    live_by_update = {
        point["update"]: point["live_calibrated"]
        for point in curve
        if np.isfinite(point["live_calibrated"])
    }
    rows = []
    for record in history:
        live = live_by_update.get(record["update"])
        if live is None:
            continue
        for label, (win, loss, tie) in (record.get("counts") or {}).items():
            total = win + loss + tie
            if label not in ratings or total < 20:  # too few games to be a rate
                continue
            rows.append(
                {
                    "a": "live",
                    "b": label,
                    "games": int(total),
                    "tie_rate": float(tie / total),
                    "mean_rating": float((live + ratings[label]) / 2.0),
                    "rating_gap": float(abs(live - ratings[label])),
                }
            )
    return rows


def tie_rate_table(
    labels: list[str], wins: np.ndarray, ties: np.ndarray, ratings: np.ndarray
) -> list[dict]:
    """Per-pair tie rates against the pair's rating level, for the draw model."""
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            decisive = wins[i, j] + wins[j, i]
            tied = ties[i, j] + ties[j, i]
            total = decisive + tied
            if total <= 0:
                continue
            rows.append(
                {
                    "a": labels[i],
                    "b": labels[j],
                    "games": int(total),
                    "tie_rate": float(tied / total),
                    "mean_rating": float((ratings[i] + ratings[j]) / 2.0),
                    "rating_gap": float(abs(ratings[i] - ratings[j])),
                }
            )
    return rows


def _load_stored_result(run_dir: Path) -> dict:
    """Read a previous calibration's persisted result."""
    path = run_dir / "elo_calibrated.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no elo_calibrated.json in {run_dir}; run without refit first"
        )
    return json.loads(path.read_text())


def run_elo_calibrate_mode(
    run_spec: str,
    ship_config: ShipConfig,
    device: str,
    config: EloCalibrateConfig,
    checkpoint_dir: str = "checkpoints",
    plot: bool = True,
    refit: bool = False,
) -> dict:
    """Re-rate a finished run and write calibrated ratings, curve, and plots.

    With ``refit=True`` no game is played: the raw win/tie matrices persisted by
    a previous calibration are loaded and refit under the current reporting
    conventions. Refitting reuses the stored reference gauge, so the underlying
    fit reproduces the original; only downstream reporting can differ. This is
    the cheap path for a change of anchor or draw convention.
    """
    progress = Progress()
    prior_games = config.prior_games
    run_dir = resolve_legacy_elo_run(run_spec, checkpoint_dir).path
    history_path = run_dir / "elo_history.jsonl"

    if refit:
        stored = _load_stored_result(run_dir)
        print(f"\n=== Elo refit from stored matrices (no play): {run_dir.name} ===")
        players = [
            Player(p["label"], ResolvedAgent("stored", None), p["training_elo"], p["global_step"])
            for p in stored["players"]
        ]
        wins = np.asarray(stored["wins_matrix"], dtype=np.float64)
        ties = np.asarray(stored["ties_matrix"], dtype=np.float64)
        directed_outcomes = stored.get("directed_outcomes")
        stats = [BatchStat(**batch) for batch in stored["batches"]]
        target_stderr = float(stored["target_stderr"])
        reference = stored["player_labels"].index(stored["reference"])
        fit = fit_bradley_terry(
            effective_wins(wins, ties, config.tie_mode),
            anchor=reference,
            prior_games=prior_games,
        )
        progress.done(f"refit {len(players)} players from stored matrices")
    else:
        num_envs = config.num_envs
        roster_path = run_dir / "roster.json"
        if not roster_path.exists():
            raise FileNotFoundError(f"no roster.json in {run_dir}; nothing to calibrate")

        roster = json.loads(roster_path.read_text())
        env_config, model_config, paradigm = load_run_config(run_dir)
        print(f"\n=== Elo calibration: {run_dir.name} ===")
        print(
            f"  {env_config.num_ships} ships, {paradigm}, "
            f"max_episode_steps={env_config.max_episode_steps}"
        )

        ladder_count = sum(1 for e in roster["entries"] if e["kind"] == "checkpoint")
        progress.stage(f"loading {ladder_count} ladder snapshots...")
        players = build_players(
            run_dir,
            roster,
            model_config,
            ship_config,
            env_config.num_ships,
            device,
            reference_probabilities=config.reference_probabilities,
        )
        progress.done(f"field ({len(players)}): {', '.join(p.label for p in players)}")
        anchor = next(i for i, p in enumerate(players) if p.label == "random")

        tournament = Tournament(
            players,
            ship_config,
            env_config,
            paradigm,
            num_envs,
            device,
            include_bullets=model_config.reads_bullets,
        )
        pairs = len(players) * (len(players) - 1) // 2
        progress.stage(
            f"{num_envs} games/batch over {pairs} pairs, target +/-{config.target_stderr:.0f} "
            f"Elo, max {config.max_batches} batches"
        )
        fit, stats, reference = run_tournament(tournament, anchor, config, progress)
        wins, ties = tournament.wins, tournament.ties
        directed_outcomes = tournament.directed_outcomes.tolist()
        target_stderr = config.target_stderr

    # Report on the scripted-anchored scale (scripted = 1000) while keeping the
    # errors from the gauge that is actually resolved. The shift is a constant:
    # it moves every rating together and cancels in any comparison between two
    # of them.
    scripted_index = next(i for i, p in enumerate(players) if p.label == "scripted")
    shifted = fit.ratings - fit.ratings[scripted_index] + SCRIPTED_ANCHOR_ELO
    ratings = {player.label: float(shifted[i]) for i, player in enumerate(players)}
    stderrs = {player.label: float(fit.stderr[i]) for i, player in enumerate(players)}

    # Refit the same match counts under the other convention. No extra play is
    # needed, and the comparison is the diagnostic for a policy that farms draws:
    # half-win scoring grants it parity it never had to win, decisive-only does
    # not, so a large disagreement for one agent is that signature.
    alt_mode = next(mode for mode in TIE_MODES if mode != config.tie_mode)
    alt_fit = fit_bradley_terry(
        effective_wins(wins, ties, alt_mode), anchor=reference, prior_games=prior_games
    )
    alt_shifted = alt_fit.ratings - alt_fit.ratings[scripted_index] + SCRIPTED_ANCHOR_ELO
    alt_ratings = {player.label: float(alt_shifted[i]) for i, player in enumerate(players)}

    history = []
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    progress.stage(f"refitting {len(history)} update records under both draw conventions...")
    curve = calibrate_live_curve(history, ratings, config.tie_mode)
    alt_curve = {
        point["update"]: point for point in calibrate_live_curve(history, alt_ratings, alt_mode)
    }
    for point in curve:
        other = alt_curve.get(point["update"])
        if other is not None:
            point["live_calibrated_alt"] = other["live_calibrated"]
            point["live_stderr_alt"] = other["live_stderr"]
    progress.done(f"calibrated {len(curve)} live-curve points")

    result = {
        "run": run_dir.name,
        "players": [
            {
                "label": player.label,
                "training_elo": player.training_elo,
                "calibrated_elo": ratings[player.label],
                "calibrated_elo_alt": alt_ratings[player.label],
                "stderr": stderrs[player.label],
                "stderr_alt": float(alt_fit.stderr[index]),
                "global_step": player.global_step,
                "games": float(fit.games[index]),
            }
            for index, player in enumerate(players)
        ],
        # The raw tally is the expensive artifact — everything else is a refit of
        # it. Persisting it means a new draw convention or estimator can be tried
        # without replaying a single match (see ``refit``).
        "player_labels": [player.label for player in players],
        "wins_matrix": np.asarray(wins).tolist(),
        "ties_matrix": np.asarray(ties).tolist(),
        "curve": curve,
        "batches": [vars(stat) for stat in stats],
        "tie_rates": tie_rate_table([p.label for p in players], wins, ties, shifted),
        "training_tie_rates": training_tie_rates(history, curve, ratings),
        "converged": bool(stats[-1].max_stderr <= target_stderr) if stats else False,
        "target_stderr": target_stderr,
        "reference": players[reference].label,
        "tie_mode": config.tie_mode,
        "tie_mode_alt": alt_mode,
        # Every reported rating is measured relative to the reference and then
        # shifted so scripted reads SCRIPTED_ANCHOR_ELO. That shift is itself
        # uncertain by this much, in common across all of them; it cancels
        # between any two ratings.
        "anchor": "scripted",
        "anchor_elo": SCRIPTED_ANCHOR_ELO,
        "anchor_offset_stderr": float(fit.stderr[scripted_index]),
    }
    if directed_outcomes is not None:  # absent from results stored before it existed
        result["directed_outcomes"] = directed_outcomes
    output = run_dir / "elo_calibrated.json"
    output.write_text(json.dumps(result, indent=2))
    progress.done(f"wrote {output}")

    # Also leave the calibrated ratings in the W&B export format, so the same
    # loader and chart system can read them alongside the run's in-training
    # history without reshaping either.
    from boost_and_broadside.modes.elo_calibrate_history import write_chart_data

    for path in write_chart_data(result, run_dir):
        progress.done(f"wrote {path}")

    _print_summary(result)
    if plot:
        from boost_and_broadside.modes.elo_calibrate_plots import write_plots

        progress.stage("rendering plots...")
        written = write_plots(result, run_dir, plot_decisive=config.plot_decisive)
        progress.done(f"wrote {len(written)} plots to {run_dir / 'elo_calibration'}")
        for path in written:
            print(f"    {path.name}")
    return result


def _print_summary(result: dict) -> None:
    """Print the calibrated-vs-training comparison for every ladder player."""
    print(f"\n  {'agent':<20} {'training':>10} {'calibrated':>12} {'+/-':>7} {'drift':>9}")
    print(f"  {'-' * 62}")
    for player in result["players"]:
        training = player["training_elo"]
        calibrated = player["calibrated_elo"]
        drift = f"{calibrated - training:+9.1f}" if training is not None else f"{'—':>9}"
        training_text = f"{training:10.1f}" if training is not None else f"{'—':>10}"
        print(
            f"  {player['label']:<20} {training_text} {calibrated:12.1f} "
            f"{player['stderr']:7.1f} {drift}"
        )
    primary, alt = result.get("tie_mode", "half_win"), result.get("tie_mode_alt", "decisive")
    print(f"\n  {'agent':<20} {primary:>14} {alt:>14} {'difference':>12}")
    print(f"  {'-' * 62}")
    for player in result["players"]:
        other = player.get("calibrated_elo_alt")
        if other is None:
            continue
        print(
            f"  {player['label']:<20} {player['calibrated_elo']:14.1f} {other:14.1f} "
            f"{player['calibrated_elo'] - other:+12.1f}"
        )
    random_rating = next(
        (p["calibrated_elo"] for p in result["players"] if p["label"] == "random"), None
    )
    random_text = (
        f" Random reads {random_rating:.0f} on this scale." if random_rating is not None else ""
    )
    print(
        f"\n  Errors are relative to '{result['reference']}'. Ratings are shifted so the "
        f"scripted\n  controller reads {result['anchor_elo']:.0f}; that shift is uncertain by "
        f"+/-{result['anchor_offset_stderr']:.0f} in common across\n  every rating above and "
        f"cancels whenever two are compared.{random_text}\n  Random's own link to the field is "
        "coarse because every trained agent beats it\n  decisively — nothing plays near its "
        "level, so those games say little however\n  they are scored. The live curve does not "
        "have this problem: early in training\n  it drew against random constantly, and draws "
        "are informative."
    )
    # Spacing is the part a shared offset cannot flatter, so report it directly.
    # Random is excluded: the step from it to the first rung is the coarse anchor
    # link, not a rung-to-rung gap, and listing it here would read as one.
    ladder = [
        p for p in result["players"] if p["training_elo"] is not None and p["label"] != "random"
    ]
    ladder.sort(key=lambda p: p["global_step"] or 0)
    if len(ladder) > 1:
        print(f"\n  {'rung-to-rung gap':<20} {'training':>10} {'calibrated':>12} {'delta':>9}")
        print(f"  {'-' * 54}")
        for previous, current in zip(ladder, ladder[1:]):
            training_gap = current["training_elo"] - previous["training_elo"]
            calibrated_gap = current["calibrated_elo"] - previous["calibrated_elo"]
            print(
                f"  {current['label']:<20} {training_gap:10.1f} {calibrated_gap:12.1f} "
                f"{calibrated_gap - training_gap:+9.1f}"
            )
