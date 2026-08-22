"""The next-state noise report: aggregates in, one JSON and four figures out.

The measurement writes per-feature sigma, bias, lag-1 autocorrelation, team and
combat stratification, and the AR error growth curve. This turns exactly that
into the published report, so the figures can be re-rendered from stored
evidence without touching the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boost_and_broadside.charts.renderer_api import Renderer, RenderInputs, register

# Suppress the library version stamp so re-rendering is byte-identical.
_DETERMINISTIC = {"Software": None}


def write_outputs(
    data: dict,
    output_dir: str,
    feature_groups: dict[str, tuple[list[int], str]],
    dim_names: list[str],
) -> None:
    """Render the noise report from one stored measurement."""

    os.makedirs(output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(output_dir, "noise_params.json"), "w") as f:
        json.dump(data, f, indent=2)

    feats = data["features"]
    feat_names = list(feature_groups)

    # --- error_distributions.png: sigma bar chart per target dim with bias overlay ---
    num_targets = len(dim_names)
    num_columns = 5
    num_rows = (num_targets + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 3 * num_rows))
    fig.suptitle("Per-dim sigma (bar) and bias (line marker)", fontsize=11)

    # Rebuild per-dim sigma and bias from feature groups
    sigma_arr = np.zeros(num_targets)
    bias_arr = np.zeros(num_targets)
    for name, (dims, _) in feature_groups.items():
        sigma_arr[dims] = feats[name]["sigma"]
        bias_arr[dims] = feats[name]["bias"]

    for i, ax in enumerate(axes.flat):
        if i < num_targets:
            ax.bar([0], [sigma_arr[i]], color="steelblue", alpha=0.8, label="sigma")
            ax.axhline(bias_arr[i], color="red", linewidth=1.5, linestyle="--", label="bias")
            ax.set_title(dim_names[i], fontsize=8)
            ax.set_xticks([])
            ax.set_ylim(0, max(sigma_arr[i] * 1.5, 1e-6))
            if i == 0:
                ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "error_distributions.png"), dpi=120, metadata=_DETERMINISTIC
    )
    plt.close(fig)

    # --- autocorrelation.png: rho_lag1 per feature group ---
    rho_vals = [feats[n]["rho_lag1"] for n in feat_names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(feat_names, rho_vals, color="darkorange", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Lag-1 autocorrelation of prediction error (rho) per feature")
    ax.set_ylabel("rho")
    ax.set_ylim(-1.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "autocorrelation.png"), dpi=120, metadata=_DETERMINISTIC)
    plt.close(fig)

    # --- ar_growth.png: RMSE vs depth per feature group ---
    depths = data["ar_growth"]["depth"]
    rmse_by_feat = data["ar_growth"]["rmse_per_feature"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in feat_names:
        ax.plot(depths, rmse_by_feat[name], label=name, marker="o", markersize=3)
    ax.set_title("AR rollout RMSE vs depth (closed-loop, teacher-forced)")
    ax.set_xlabel("Rollout depth (steps)")
    ax.set_ylabel("RMSE (AUX space)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ar_growth.png"), dpi=120, metadata=_DETERMINISTIC)
    plt.close(fig)

    # --- team_symmetry.png: sigma_team0 vs sigma_team1 per feature group ---
    s0 = [feats[n]["sigma_team0"] for n in feat_names]
    s1 = [feats[n]["sigma_team1"] for n in feat_names]
    x = np.arange(len(feat_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, s0, w, label="team 0", color="royalblue", alpha=0.85)
    ax.bar(x + w / 2, s1, w, label="team 1", color="tomato", alpha=0.85)
    ax.set_title("Sigma per team (should be symmetric)")
    ax.set_ylabel("sigma")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "team_symmetry.png"), dpi=120, metadata=_DETERMINISTIC)
    plt.close(fig)


def _feature_groups(data: dict) -> dict[str, tuple[list[int], str]]:
    """Rebuild the report layout the measurement recorded."""

    return {
        name: (list(feature["aux_dims"]), name)
        for name, feature in data["features"].items()
    }


def _render(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    data = inputs.artifact("noise").read_json()
    write_outputs(data, str(out_dir), _feature_groups(data), list(data["dim_names"]))
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


register(
    Renderer(
        name="noise-calibration-v1",
        description="Next-state prediction error: sigma, autocorrelation, and AR growth.",
        render=_render,
        required_artifacts=("noise",),
        supported_schemas={"noise": (1,)},
        multi_file=True,
    )
)
