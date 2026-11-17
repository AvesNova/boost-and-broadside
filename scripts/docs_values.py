"""Build docs/values.json, the numbers the README and docs quote, from stored results.

Reads the reference run's artifacts under `checkpoints/<run>/artifacts/` (the newest
artifact of each type) plus the extra result files listed in `docs/values_sources.yml`,
and writes one flat JSON object {name: number, string or null}. A null is a result that
has not landed yet; the pages show it as pending (scripts/render_docs.py).

Offline and CPU only: it replays no game. Joining the fleet-scale tournament to the
reference ladder refits a Bradley-Terry model on the stored matrices, exactly as the
fleet-scale figure does.

    uv run --no-sync python scripts/docs_values.py            # write docs/values.json
    uv run --no-sync python scripts/docs_values.py --check    # fail if it is out of date

Names are full words: `<measurement>_<subject>_<quantity>`, e.g.
`crossover_4_ships_beats_up_to`, `calibration_final_live_elo`. Win rates are fractions;
the pages format them (`{{name:.1%}}`). "live" is the policy's own weights at a step;
"averaged" is the running average of the weights the run keeps beside them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "values_sources.yml"
VALUES = ROOT / "docs" / "values.json"

sys.path.insert(0, str(ROOT / "src"))


def newest_artifact(run_dir: Path, artifact_type: str) -> Path:
    candidates = sorted((run_dir / "artifacts" / artifact_type).glob("*/artifact.json"))
    if not candidates:
        raise FileNotFoundError(f"{run_dir.name} has no {artifact_type} artifact")
    return candidates[-1].parent


def read_json(path: Path):
    return json.loads(path.read_text())


def rounded(value, digits: int = 6):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    return value


# ---------------------------------------------------------------------- the run itself


def run_values(run_dir: Path, run_name: str) -> dict:
    run = read_json(run_dir / "run.json")
    elo_history = (run_dir / "elo_history.jsonl").read_text().splitlines()
    last_live = json.loads(elo_history[-1])
    export = newest_artifact(run_dir, "wandb-export")
    config = read_json(export / "config.json")
    meta = read_json(export / "run_meta.json")

    def setting(section: str, key: str):
        return config[f"{section}/{key}"]

    steps = run["global_step"]
    seconds = run["elapsed_seconds"]
    ships = setting("env", "num_ships")
    action_repeat = setting("env", "action_repeat")
    episode_ticks = setting("env", "max_episode_steps")
    rewards = setting("train", "rewards")
    return {
        "run_name": run_name,
        "run_wandb_id": meta["id"],
        "run_status": run["status"],
        "run_steps": steps,
        "run_steps_millions": steps / 1e6,
        "run_updates": run["update"],
        "run_target_steps": setting("train", "total_timesteps"),
        "run_target_steps_millions": setting("train", "total_timesteps") / 1e6,
        "run_hours": seconds / 3600,
        "run_days": seconds / 86400,
        "run_steps_per_second": steps / seconds,
        "run_ship_tokens_per_second": steps * ships / seconds,
        "run_git_commit": run["git_commit"][:8],
        "training_live_elo_final": last_live["live"],
        "training_averaged_live_elo_final": last_live["avg"],
        "config_parallel_environments": config["train/scales"][0]["num_envs"],
        "config_rollouts_per_update": setting("train", "rollouts_per_update"),
        "config_ships_per_environment": ships,
        "config_ships_per_team": ships // 2,
        "config_fields": setting("env", "num_fields"),
        "config_rollout_steps": setting("train", "num_steps"),
        "config_action_repeat": action_repeat,
        "config_decisions_per_second": 60 / action_repeat,
        "config_minibatches": setting("train", "num_minibatches"),
        "config_token_width": setting("model", "d_model"),
        "config_attention_heads": setting("model", "n_heads"),
        "config_blocks": setting("model", "n_yemong_blocks"),
        "config_episode_ticks": episode_ticks,
        "config_episode_seconds": episode_ticks / 60,
        "config_paradigm": setting("train", "paradigm"),
        "config_elo_games_per_matchup": setting("train", "elo_eval")["envs_per_matchup"],
        "config_bullet_min_damage_fraction": setting("ship", "bullet_min_damage_frac"),
        "config_reward_win": rewards["win_weight"],
        "config_reward_kill_payout_ratio": rewards["kill_payout_ratio"],
        "config_reward_kill_shot_fraction": rewards["kill_shot_fraction"],
        "config_reward_death": rewards["death_weight"],
        "config_reward_damage": rewards["damage_weight"],
        "config_reward_damage_payout_ratio": rewards["damage_payout_ratio"],
        "config_reward_facing": rewards["facing_weight"],
        "config_reward_closing_speed": rewards["closing_speed_weight"],
        "config_reward_shoot_quality": rewards["shoot_quality_weight"],
    }


# ---------------------------------------------------------------- post-hoc calibration


def calibration_values(run_dir: Path, final_step: int) -> dict:
    artifact = newest_artifact(run_dir, "elo-calibration")
    recipe = read_json(artifact / "artifact.json")["recipe"]["parameters"]
    result = read_json(artifact / "result.json")
    players = {player["label"]: player for player in result["players"]}

    def player(prefix: str, label: str) -> dict:
        entry = players[label]
        return {
            f"{prefix}_elo": entry["calibrated_elo"],
            f"{prefix}_stderr": entry["stderr"],
            f"{prefix}_games": entry["games"],
            f"{prefix}_steps_millions": (entry["global_step"] or 0) / 1e6,
        }

    ladder = sorted(
        (
            entry
            for label, entry in players.items()
            if label.startswith("ckpt_") and entry["global_step"] != final_step
        ),
        key=lambda entry: entry["global_step"],
    )
    last = result["curve"][-1]
    values = {
        **player("calibration_final_live", f"ckpt_{final_step}"),
        **player("calibration_best_live", "best_training"),
        **player("calibration_best_averaged", "best_avg"),
        **player("calibration_last_ladder", ladder[-1]["label"]),
        **player("calibration_random", "random"),
        "calibration_scripted_stderr": result["anchor_offset_stderr"],
        "calibration_scripted_games": players["scripted"]["games"],
        "calibration_reference_player": result["reference"],
        "calibration_target_stderr": result["target_stderr"],
        "calibration_max_batches": recipe["max_batches"],
        "calibration_batches": len(result["batches"]),
        "calibration_games": result["batches"][-1]["cumulative_games"],
        "calibration_converged": "yes" if result["converged"] else "no",
        "calibration_ladder_checkpoints": len(ladder),
        "calibration_refit_live_elo": last["live_calibrated"],
        "calibration_refit_live_stderr": last["live_stderr"],
        "calibration_refit_live_games": last["games"],
        "calibration_refit_live_steps_millions": last["global_step"] / 1e6,
        "calibration_refit_averaged_elo": last.get("avg_calibrated"),
        "calibration_refit_averaged_stderr": last.get("avg_stderr"),
    }
    final = values["calibration_final_live_elo"]
    values["calibration_final_live_lead_over_scripted"] = final - 1000.0
    values["calibration_best_live_gap_to_final"] = final - values["calibration_best_live_elo"]
    values["calibration_best_averaged_gap_to_final"] = (
        final - values["calibration_best_averaged_elo"]
    )
    values["calibration_random_to_scripted"] = 1000.0 - values["calibration_random_elo"]

    # Milestones of the refit live curve; one update's refit is noisy, so these are read
    # off a centred five-update mean.
    steps = [row["global_step"] for row in result["curve"]]
    ratings = [row["live_calibrated"] for row in result["curve"]]
    smooth = [
        sum(ratings[max(0, i - 2) : i + 3]) / len(ratings[max(0, i - 2) : i + 3])
        for i in range(len(ratings))
    ]

    def at(step: float) -> float:
        index = min(range(len(steps)), key=lambda i: abs(steps[i] - step))
        return smooth[index]

    for millions in (50, 100, 150, 250):
        values[f"calibration_curve_elo_at_{millions}m"] = at(millions * 1e6)
    for margin in (100, 50):
        below = [s for s, r in zip(steps, smooth) if r < final - margin]
        values[f"calibration_curve_within_{margin}_from_millions"] = (
            below[-1] / 1e6 if below else 0.0
        )
    return values


# ------------------------------------------------------------------ fleet-scale ratings


def scale_values(run_dir: Path) -> dict:
    from boost_and_broadside.charts.renderers.elo_scale import combine_reference_ladder

    scale = read_json(newest_artifact(run_dir, "elo-scale") / "result.json")
    reference = read_json(newest_artifact(run_dir, "semi-random-ladder") / "result.json")
    joined = combine_reference_ladder(scale, reference)
    values = {
        "scale_target_stderr": scale["target_stderr"],
        "scale_max_batches": scale["max_batches"],
    }
    final = scale["player_labels"].index("final")
    joined_final = joined["player_labels"].index("final")
    for key, size in scale["scales"].items():
        prefix = f"scale_{key}_ships"
        own = size["ratings"]["scripted_1000"]
        fit = joined["scales"][key]["ratings"]["scripted_1000"]
        values.update(
            {
                f"{prefix}_elo": fit["ratings"][joined_final],
                f"{prefix}_stderr": fit["stderr"][joined_final],
                f"{prefix}_tournament_only_elo": own["ratings"][final],
                f"{prefix}_tournament_only_stderr": own["stderr"][final],
                f"{prefix}_games": size["batches"][-1]["cumulative_games"],
                f"{prefix}_batch_games": size["parallel_envs"],
                f"{prefix}_batches": len(size["batches"]),
                f"{prefix}_max_stderr": size["batches"][-1]["max_stderr"],
                f"{prefix}_converged": "yes" if size["converged"] else "no",
            }
        )
    best = max(scale["scales"], key=lambda key: values[f"scale_{key}_ships_elo"])
    values["scale_peak_ships"] = int(best)
    return values


def ladder_values(run_dir: Path) -> dict:
    result = read_json(newest_artifact(run_dir, "semi-random-ladder") / "result.json")
    scores = [
        matchup["higher_expected_score"]
        for size in result["scales"].values()
        for matchup in size["adjacent_matchups"]
    ]
    values = {
        "ladder_games_per_pair": result["games_per_pair"],
        "ladder_scales": len(result["scales"]),
        "ladder_adjacent_comparisons": len(scores),
        "ladder_adjacent_score_min": min(scores),
        "ladder_adjacent_score_max": max(scores),
    }
    for rung in result["scales"]["4"]["live_gauge_error"]:
        name = f"{rung['p_scripted']:g}".replace(".", "p")
        values[f"ladder_4_ships_live_gauge_error_{name}"] = rung["live_elo_error"]
    return values


# --------------------------------------------------------------------------- crossover


def crossover_values(run_dir: Path) -> dict:
    artifact = newest_artifact(run_dir, "crossover")
    result = read_json(artifact / "result.json")
    checkpoint = read_json(artifact / "artifact.json")["recipe"]["subjects"]["trained"]
    values = {
        "crossover_checkpoint": checkpoint["checkpoint"],
        "crossover_games_per_matchup": result["num_envs"],
        "crossover_sizes": ", ".join(str(row["trained"]) for row in result["rows"]),
        "crossover_largest_size": max(row["trained"] for row in result["rows"]),
    }
    for row in result["rows"]:
        size = row["trained"]
        prefix = f"crossover_{size}_ships"
        beats, below = str(row["beats_up_to"]), str(row["crossover"])
        curve = row["curve"]
        values.update(
            {
                f"{prefix}_beats_up_to": row["beats_up_to"],
                f"{prefix}_beats_up_to_win_rate": row["win_rate_at_beats_up_to"],
                f"{prefix}_beats_up_to_games": curve[beats]["games"],
                f"{prefix}_beats_up_to_wins": curve[beats]["wins"],
                f"{prefix}_beats_up_to_losses": curve[beats]["losses"],
                f"{prefix}_crossover": row["crossover"],
                f"{prefix}_crossover_win_rate": row["win_rate_at_crossover"],
                f"{prefix}_crossover_games": curve[below]["games"],
                f"{prefix}_ratio": row["beats_up_to"] / size,
                f"{prefix}_equal_count_win_rate": (
                    curve[str(size)]["win_rate"] if str(size) in curve else None
                ),
            }
        )
        rate = row["win_rate_at_beats_up_to"]
        games = curve[beats]["games"]
        values[f"{prefix}_beats_up_to_binomial_stderr"] = math.sqrt(rate * (1 - rate) / games)
        for scripted, point in curve.items():
            for field in ("win_rate", "games", "wins", "losses", "ties"):
                values[f"{prefix}_vs_{scripted}_{field}"] = point[field]
    return values


# ------------------------------------------------------------------- training history


def history_values(run_dir: Path) -> dict:
    from boost_and_broadside.viz import history

    rows = history.load_history(newest_artifact(run_dir, "wandb-export") / "history.jsonl")
    values = {}

    steps, win = history.series(rows, "overview/win_rate_vs_scripted")
    values["scripted_win_rate_points"] = len(steps)
    for threshold in (95, 99):
        reached = [s for s, w in zip(steps, win) if w >= threshold / 100]
        values[f"scripted_win_rate_first_{threshold}_millions"] = (
            reached[0] / 1e6 if reached else None
        )
    values["scripted_win_rate_final"] = float(win[-1])
    late = [w for s, w in zip(steps, win) if s > 200e6]
    values["scripted_win_rate_min_after_200m"] = float(min(late)) if late else None

    _, learning_rate = history.series(rows, "schedule/learning_rate")
    values["learning_rate_peak"] = float(max(learning_rate))

    _, explained = history.series(rows, "overview/explained_variance")
    values["explained_variance_final"] = float(explained[-1])
    values["explained_variance_max"] = float(max(explained))

    for name, keys in {
        "position": ["next_state/position_x_0"],
        "velocity": ["next_state/velocity_0"],
        "power": ["next_state/power_0"],
        "health": ["next_state/health_0"],
    }.items():
        _, error = history.combine_series(rows, keys)
        values[f"next_state_{name}_error_first"] = float(error[0])
        values[f"next_state_{name}_error_final"] = float(error[-1])
    return values


# ----------------------------------------------------------------------- earlier runs


def earlier_run_values(key: str, run_dir: Path) -> dict:
    result = read_json(newest_artifact(run_dir, "elo-calibration") / "result.json")
    players = {player["label"]: player for player in result["players"]}
    run = run_dir / "run.json"
    step = (
        read_json(run)["global_step"]
        if run.exists()
        else max(
            int(m.group(1))
            for m in (
                re.search(r"step_(\d+)\.pt$", path)
                for path in glob.glob(str(run_dir / "step_*.pt"))
            )
            if m
        )
    )
    final = players[f"ckpt_{step}"]
    crossover = {}
    try:
        rows = read_json(newest_artifact(run_dir, "crossover") / "result.json")["rows"]
    except FileNotFoundError:
        rows = []
    for row in rows:
        size = row["trained"]
        crossover[f"history_{key}_crossover_{size}_ships_beats_up_to"] = row["beats_up_to"]
        crossover[f"history_{key}_crossover_{size}_ships_ratio"] = row["beats_up_to"] / size
    return {
        **crossover,
        f"history_{key}_steps_millions": step / 1e6,
        f"history_{key}_final_live_elo": final["calibrated_elo"],
        f"history_{key}_final_live_stderr": final["stderr"],
        f"history_{key}_random_to_scripted": 1000.0 - players["random"]["calibrated_elo"],
    }


# --------------------------------------------------------------- incoming result files


def fan_values(spec: dict) -> dict:
    """Per fan size N: the win rate of the final live and final averaged checkpoints
    against every scripted count S, and the largest S each still beats."""
    values = {}
    for size in spec["sizes"]:
        path = ROOT / spec["rows"].format(size=size)
        rows = []
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        final_step = max(
            (int(re.search(r"(\d+)\.pt$", row["ckpt"]).group(1)) for row in rows), default=None
        )
        for kind, prefix in (("live", "step_"), ("averaged", "avg_step_")):
            name = f"{prefix}{final_step:012d}.pt" if final_step is not None else None
            chosen = {row["scripted"]: row for row in rows if row["ckpt"].split("/")[-1] == name}
            for scripted in range(size, 2 * size + 1):
                row = chosen.get(scripted)
                values[f"fan_{size}_ships_vs_{scripted}_{kind}_win_rate"] = (
                    row["win_rate"] if row else None
                )
                values[f"fan_{size}_ships_vs_{scripted}_{kind}_games"] = (
                    row["games"] if row else None
                )
            beaten = [s for s, row in chosen.items() if row["win_rate"] >= 0.5]
            values[f"fan_{size}_ships_{kind}_beats_up_to"] = max(beaten) if beaten else None
            values[f"fan_{size}_ships_{kind}_ratio"] = max(beaten) / size if beaten else None
    return values


def clip_values(spec: dict) -> dict:
    """Every numeric field of each clip's survivors file, as clip_<file stem>_<field>."""
    values = {}
    for path in sorted(glob.glob(str(ROOT / spec["survivors"]))):
        stem = re.sub(r"[^a-z0-9]+", "_", Path(path).stem.lower()).strip("_")
        data = read_json(Path(path))
        for field, value in data.items() if isinstance(data, dict) else []:
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                key = re.sub(r"[^a-z0-9]+", "_", str(field).lower()).strip("_")
                values[f"clip_{stem}_{key}"] = value
    return values


def build() -> dict:
    sources = yaml.safe_load(SOURCES.read_text())
    run_name = sources["run"]
    run_dir = ROOT / "checkpoints" / run_name
    values = run_values(run_dir, run_name)
    values.update(sources.get("facts", {}))
    values.update(calibration_values(run_dir, values["run_steps"]))
    values.update(scale_values(run_dir))
    values.update(ladder_values(run_dir))
    values.update(crossover_values(run_dir))
    values.update(history_values(run_dir))
    for key, path in sources.get("history", {}).items():
        values.update(earlier_run_values(key, ROOT / path))
    values.update(fan_values(sources["fan"]))
    values.update(clip_values(sources["clips"]))
    return {name: rounded(value) for name, value in sorted(values.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if values.json is stale")
    args = parser.parse_args(argv)
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if not VALUES.exists() or VALUES.read_text() != text:
            print("docs/values.json is out of date: run scripts/docs_values.py", file=sys.stderr)
            return 1
        print("docs/values.json is up to date")
        return 0
    VALUES.write_text(text)
    print(f"wrote {VALUES.relative_to(ROOT)} ({len(json.loads(text))} values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
