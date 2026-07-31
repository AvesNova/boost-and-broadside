"""Render the README result figures for a landmark run.

Reads the two W&B-format sources a run leaves behind — the in-training archive in
``wandb_export/`` and the calibrated re-rating in ``elo_calibration/`` — through
the one ``viz.history`` reader, and paints each figure with ``viz.charts``.

Usage:
    uv run scripts/render_charts.py            # run 682 -> docs/results/
    uv run scripts/render_charts.py --run <name> --out <dir>
"""

import argparse
from pathlib import Path

from boost_and_broadside.viz import charts, history
from boost_and_broadside.viz.charts import Line, Panel

DEFAULT_RUN = "resilient-resonance-682"

# The next-state prediction error logs symmetric spatial channels separately;
# each pair is averaged into one honest curve. Order sets the palette slots.
_NEXT_STATE = [
    ("position", ["next_state/pos_x_dphase", "next_state/pos_y_dphase"]),
    ("velocity", ["next_state/vel_dvx_norm", "next_state/vel_dvy_norm"]),
    ("angular vel", ["next_state/ang_vel_abs"]),
    ("attitude", ["next_state/att_dphase"]),
    ("cooldown", ["next_state/cooldown_dphase"]),
    ("health", ["next_state/health_dphase"]),
    ("power", ["next_state/power_dphase"]),
]


def _line(rows: list[dict], key: str, label: str) -> Line:
    x, y = history.series(rows, key)
    return Line(x, y, label)


def render(run: str, out_dir: Path) -> list[Path]:
    run_dir = Path("checkpoints") / run
    wandb_dir = run_dir / "wandb_export"
    calib_dir = run_dir / "elo_calibration"
    for required in (wandb_dir / "history.jsonl", calib_dir / "history.jsonl"):
        if not required.exists():
            raise SystemExit(f"missing {required}; export W&B and run --mode elo_calibrate first")

    wandb_rows = history.load_history(wandb_dir / "history.jsonl")
    calib_rows = history.load_history(calib_dir / "history.jsonl")
    calib_summary = history.load_summary(calib_dir / "summary.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    # The scripted controller — the one opponent shared across runs — is a landmark for
    # reading where the policy actually got to.
    scripted = calib_summary.get("elo/scripted")
    elo_landmarks = [("scripted controller", float(scripted))] if history._finite(scripted) else None

    written = [
        charts.trend(
            [_line(calib_rows, "ladder/elo/live", "Elo")], out_dir / "elo_curve.png",
            title="Calibrated Elo over training", ylabel="Elo (scripted = 1000)",
            reference_lines=elo_landmarks,
        ),
        charts.trend(
            [_line(wandb_rows, "overview/win_rate_vs_scripted", "vs scripted")],
            out_dir / "win_rate_vs_scripted.png",
            title="Win rate vs the scripted controller", ylabel="win rate", percent=True,
        ),
        charts.grid(
            [
                Panel([_line(wandb_rows, "overview/explained_variance", "EV")],
                      title="Critic explained variance", ylabel="explained variance"),
                Panel([_line(wandb_rows, "overview/reward_mean", "reward")],
                      title="Mean episode reward", ylabel="reward"),
                Panel([_line(wandb_rows, "overview/kl", "KL")],
                      title="Policy update KL divergence", ylabel="KL divergence"),
                Panel([_line(wandb_rows, "overview/clip_fraction", "clip")],
                      title="PPO clip fraction", ylabel="clip fraction", percent=True),
            ],
            out_dir / "training_health.png",
        ),
        charts.trend(
            [Line(*history.combine_series(wandb_rows, keys), label=label)
             for label, keys in _NEXT_STATE],
            out_dir / "next_state_error.png",
            title="Next-state prediction error by dimension",
            subtitle="Symmetric spatial channels (x/y) averaged; normalised error, log scale",
            ylabel="prediction error (normalised)", log_y=True, size=(10.5, 6.2),
        ),
    ]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="landmark run name under checkpoints/")
    parser.add_argument("--out", type=Path, default=Path("docs/results"), help="output directory")
    args = parser.parse_args()
    for path in render(args.run, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
