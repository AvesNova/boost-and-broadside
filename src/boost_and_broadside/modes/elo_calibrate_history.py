"""Re-express a calibration result in the W&B export format.

The W&B archive (``scripts/export_wandb_run.py``) leaves two files per run: a
``history.jsonl`` of time-series rows keyed by ``_step`` (environment steps) with
sparse metric keys, and a flat ``summary.json`` of each metric's final value. The
calibration result (``elo_calibrated.json``) carries the same information for the
*calibrated* ratings, just in its own richer shape.

Reducing the calibration result to those two shapes — under the same metric key
names W&B itself uses (``ladder/elo/live``, ``ladder/elo/ckpt_<step>``,
``elo/scripted``) — means one loader and one chart system read the in-training
and calibrated ratings interchangeably. "In-training vs calibrated" becomes
simply which file is opened, exactly like reading two W&B runs.

The conversion is pure and takes the already-persisted ``result`` dict, so a
finished run can be back-filled from its ``elo_calibrated.json`` without replaying
a single tournament match.
"""

import json
import math
from pathlib import Path


def _finite(value) -> bool:
    """True for a real, plottable number. Clean sweeps fit to +/-inf and empty
    records fit to NaN; both are dropped so a row simply lacks the key, exactly
    as W&B omits a metric it did not log that step."""
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _put(row: dict, key: str, value) -> None:
    """Set a metric on a row only when it is a finite measurement."""
    if _finite(value):
        row[key] = float(value)


def to_history_rows(result: dict) -> list[dict]:
    """Calibrated ratings as W&B-style history rows keyed by ``_step``.

    The live/averaged curves become one row per update at that update's global
    step; each checkpoint's calibrated rating is added to the row at its own
    step (or a fresh row) under the same ``ladder/elo/ckpt_<step>`` key W&B logs
    the in-training rating under.
    """
    rows: dict[int, dict] = {}

    def row_at(step: int) -> dict:
        return rows.setdefault(int(step), {"_step": int(step)})

    for point in result.get("curve", []):
        step = point.get("global_step")
        if step is None:
            continue
        row = row_at(step)
        _put(row, "ladder/elo/live", point.get("live_calibrated"))
        _put(row, "ladder/elo/live_stderr", point.get("live_stderr"))
        _put(row, "ladder/elo/avg", point.get("avg_calibrated"))
        _put(row, "ladder/elo/avg_stderr", point.get("avg_stderr"))
        _put(row, "ladder/elo/live_decisive", point.get("live_calibrated_alt"))
        _put(row, "ladder/elo/live_decisive_stderr", point.get("live_stderr_alt"))

    for player in result.get("players", []):
        step = player.get("global_step")
        # Random is a fixed reference player, not a ladder rung; scripted and
        # other non-timeline agents have no step. None belongs on the step axis.
        if step is None or player.get("training_elo") is None or player["label"] == "random":
            continue
        row = row_at(step)
        _put(row, f"ladder/elo/ckpt_{int(step)}", player.get("calibrated_elo"))
        _put(row, f"ladder/elo/ckpt_{int(step)}_stderr", player.get("stderr"))

    # A row carrying only ``_step`` means every fit at that step was non-finite;
    # drop it rather than emit a metric-less row (W&B never logs an empty step).
    return [rows[step] for step in sorted(rows) if len(rows[step]) > 1]


def to_summary(result: dict) -> dict:
    """Calibrated finals + calibration metadata as a flat W&B-style summary.

    Per-checkpoint calibrated ratings land under the same ``ladder/elo/ckpt_<step>``
    keys W&B's summary uses, and the scripted agent — the one opponent shared
    across runs — under ``elo/scripted``, so both files are directly comparable.
    """
    summary: dict = {}
    final_step = 0
    for player in result.get("players", []):
        label = player["label"]
        calibrated = player.get("calibrated_elo")
        if label == "scripted":
            _put(summary, "elo/scripted", calibrated)
            continue
        step = player.get("global_step")
        if step is None or player.get("training_elo") is None or label == "random":
            continue
        _put(summary, f"ladder/elo/ckpt_{int(step)}", calibrated)
        _put(summary, f"ladder/elo/ckpt_{int(step)}_stderr", player.get("stderr"))
        final_step = max(final_step, int(step))

    curve = result.get("curve", [])
    if curve:
        last = curve[-1]
        _put(summary, "ladder/elo/live", last.get("live_calibrated"))
        _put(summary, "ladder/elo/avg", last.get("avg_calibrated"))
        final_step = max(final_step, int(last.get("global_step") or 0))

    summary["_step"] = final_step
    for key in ("reference", "tie_mode", "tie_mode_alt", "converged", "run", "anchor"):
        if key in result:
            summary[key] = result[key]
    for key in ("target_stderr", "anchor_offset_stderr", "anchor_elo"):
        if _finite(result.get(key)):
            summary[key] = float(result[key])
    return summary


def write_chart_data(result: dict, run_dir: Path) -> list[Path]:
    """Write ``elo_calibration/{history.jsonl,summary.json}`` in W&B export format.

    Grouped alongside the calibration plots so every chart-ready artifact for a
    run lives in one place.
    """
    output = run_dir / "elo_calibration"
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    summary_path = output / "summary.json"

    with history_path.open("w") as fh:
        for row in to_history_rows(result):
            fh.write(json.dumps(row) + "\n")
    summary_path.write_text(json.dumps(to_summary(result), indent=2))
    return [history_path, summary_path]
