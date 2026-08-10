"""Read the W&B export format — the one shape every source is reduced to.

A run leaves a ``history.jsonl`` (one JSON object per line, each a row keyed by
``_step`` with sparse metric keys) and a flat ``summary.json`` of final values.
The calibration mode writes the same two shapes under the same metric keys (see
``modes/elo_calibrate_history.py``), so the readers here treat an in-training run
and a calibrated re-rating identically — a chart's two curves are just two files.
"""

import json
import math
from pathlib import Path

import numpy as np


def load_history(path: str | Path) -> list[dict]:
    """Parse a ``history.jsonl`` into a list of row dicts."""
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_summary(path: str | Path) -> dict:
    """Parse a flat ``summary.json`` of final metric values."""
    return json.loads(Path(path).read_text())


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """The ``(x, y)`` of one metric: its ``_step`` against its value.

    Rows where the metric is absent, ``None``, or non-finite are dropped — the
    same sparsity W&B logs with — so the returned arrays contain only real
    measurements, sorted by step. ``x`` is in environment steps.
    """
    xs, ys = [], []
    for row in rows:
        step = row.get("_step")
        value = row.get(key)
        if _finite(step) and _finite(value):
            xs.append(float(step))
            ys.append(float(value))
    order = np.argsort(xs) if xs else np.array([], dtype=int)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def series_band(
    rows: list[dict], key: str, stderr_key: str, n_sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A metric with a ``±n_sigma`` band from its standard-error companion key.

    Only points that carry *both* a finite value and a finite standard error get
    a band; returns ``(x, y, lo, hi)`` aligned and step-sorted.
    """
    xs, ys, los, his = [], [], [], []
    for row in rows:
        step, value, err = row.get("_step"), row.get(key), row.get(stderr_key)
        if _finite(step) and _finite(value) and _finite(err):
            xs.append(float(step))
            ys.append(float(value))
            los.append(float(value) - n_sigma * float(err))
            his.append(float(value) + n_sigma * float(err))
    order = np.argsort(xs) if xs else np.array([], dtype=int)
    arrays = (np.asarray(a) for a in (xs, ys, los, his))
    return tuple(a[order] for a in arrays)  # type: ignore[return-value]


def combine_series(rows: list[dict], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean of several metrics per step — for folding symmetric dims into one.

    The next-state error logs ``pos_x``/``pos_y`` and ``vel_dvx``/``vel_dvy`` as
    separate, near-identical channels; averaging the pair is the honest single
    curve. A step is kept only where *every* member is a finite measurement, so a
    partially-logged step never averages fewer channels than the rest.
    """
    xs, ys = [], []
    for row in rows:
        step = row.get("_step")
        values = [row.get(k) for k in keys]
        if _finite(step) and all(_finite(v) for v in values):
            xs.append(float(step))
            ys.append(float(np.mean([float(v) for v in values])))
    order = np.argsort(xs) if xs else np.array([], dtype=int)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def checkpoint_points(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-checkpoint ratings from a summary: ``(steps, ratings, stderr)``.

    Reads the ``calibrated_elo/ckpt_<step>`` keys (and their ``_stderr``
    companions) a calibration summary exposes, so a run's frozen ladder can be
    scattered as points on the calibrated Elo curve. ``stderr`` is NaN where
    absent. The prefix is deliberately not the trainer's ``live_elo/``: these
    points are post-hoc fits and are never mixed with live readings on one axis.
    """
    steps, ratings, errs = [], [], []
    for key, value in summary.items():
        if not key.startswith("calibrated_elo/ckpt_") or key.endswith("_stderr"):
            continue
        if not _finite(value):
            continue
        step = key[len("calibrated_elo/ckpt_") :]
        if not step.isdigit():
            continue
        steps.append(float(step))
        ratings.append(float(value))
        err = summary.get(f"calibrated_elo/ckpt_{step}_stderr")
        errs.append(float(err) if _finite(err) else float("nan"))
    order = np.argsort(steps) if steps else np.array([], dtype=int)
    arrays = (np.asarray(a) for a in (steps, ratings, errs))
    return tuple(a[order] for a in arrays)  # type: ignore[return-value]
