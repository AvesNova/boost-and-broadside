"""The next-state figure set: the head's one-step forecast of each ship's position against
where the ship is one tick later, for own ships, seen enemies and hidden ones.

The forecast at t (`bnb.models.yemong.decode.next_state`, world coordinates) is scored
against the privileged row at t + 1, on the rows the learner trains (`next_state_mask`).
Two baselines use only what the model was given at t: the ship stays where it is (the
row's position, or for a hidden enemy its belief's), and the ship keeps its velocity for
one tick.
"""

import math
import pathlib

import torch

from bnb.contracts.obs import SHIP

from . import output

POPULATIONS = ("own", "enemy_seen", "enemy_hidden")
ONE_SIGMA = 1.0 - math.exp(-0.5)  # P(|e| < sigma) for an isotropic 2D Gaussian


def forecasts(artifact: dict) -> dict:
    """Per row (L, R, N): the forecast's position error (m) and sigma (m), the two
    baselines' errors, the population masks and the learner's mask."""
    meta, rollout = artifact["meta"], artifact["rollout"]
    own, tick = meta["ships_per_team"], meta["tick_length"]
    rows, following = rollout["rows"], rollout["next_rows"]
    xy, velocity_columns = [SHIP["x"], SHIP["y"]], [SHIP["velocity_x"], SHIP["velocity_y"]]
    position, velocity = rows[..., xy].clone(), rows[..., velocity_columns].clone()
    belief = rollout["belief"]
    visible = rollout["ships"][..., SHIP["visible"]] > 0.5
    hidden = ~visible
    hidden[..., :own] = False
    # what the model was given for a hidden enemy: its belief
    position[..., own:, :] = torch.where(
        hidden[..., own:, None], belief["position"], position[..., own:, :]
    )
    velocity[..., own:, :] = torch.where(
        hidden[..., own:, None], belief["velocity"], velocity[..., own:, :]
    )
    truth = following[..., xy]
    forecast = rollout["next_state"]
    rows_count = rows.shape[-2]
    is_own = (torch.arange(rows_count) < own).expand(hidden.shape)
    return dict(
        error=(forecast["position"] - truth).norm(dim=-1),
        sigma=forecast["position_sigma"],
        stay=(position - truth).norm(dim=-1),
        constant_velocity=(position + velocity * tick - truth).norm(dim=-1),
        mask=rollout["next_state_mask"] & (following[..., SHIP["alive"]] > 0.5),
        own=is_own,
        enemy_seen=~is_own & visible,
        enemy_hidden=hidden,
        truth=truth,
        forecast=forecast["position"],
    )


def measures(artifact: dict) -> dict:
    """Per population: the rows scored, the mean position error (m) of the head and of the
    two baselines, the head's mean sigma, and the share within one sigma (ONE_SIGMA when
    the sigma is right)."""
    part = forecasts(artifact)
    out = {}
    for name in POPULATIONS:
        members = part["mask"] & part[name]
        if not members.any():
            continue
        out[name] = dict(
            rows=int(members.sum()),
            head_error=round(float(part["error"][members].mean()), 3),
            stay_error=round(float(part["stay"][members].mean()), 3),
            constant_velocity_error=round(float(part["constant_velocity"][members].mean()), 3),
            head_sigma=round(float(part["sigma"][members].mean()), 3),
            within_one_sigma=round(
                float((part["error"] <= part["sigma"])[members].float().mean()), 3
            ),
        )
    out["expected_within_one_sigma"] = round(ONE_SIGMA, 3)
    return out


def tracked_ship(artifact: dict, part: dict) -> tuple[int, int]:
    """(view, row): the enemy row with the most hidden scored ticks among those also seen
    for at least a quarter of the segment."""
    hidden = (part["mask"] & part["enemy_hidden"]).sum(0).float()  # (R, N)
    seen = (part["mask"] & part["enemy_seen"]).sum(0).float()
    length = part["mask"].shape[0]
    score = torch.where(seen >= length / 4, hidden, -1.0)
    view, row = divmod(int(score.argmax()), score.shape[1])
    return view, row


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The next-state figure (next_state_track): one enemy's path with the forecasts, its
    error over time with its hidden spans shaded, and the populations' errors."""
    from matplotlib.lines import Line2D

    meta, rollout = artifact["meta"], artifact["rollout"]
    numbers = measures(artifact)
    part = forecasts(artifact)
    view, row = tracked_ship(artifact, part)
    scored = part["mask"][:, view, row]
    hidden = part["enemy_hidden"][:, view, row]
    truth = part["truth"][:, view, row]
    forecast = part["forecast"][:, view, row]
    ticks = torch.arange(len(scored)) + 1  # a forecast at t is of t + 1
    figure, axes = output.new_figure(1, 3, size=(4.8, 4.4))
    axis = axes[0, 0]
    alive = rollout["next_rows"][:, view, row, SHIP["alive"]] > 0.5
    axis.plot(truth[alive, 0], truth[alive, 1], color=output.TRUE, linewidth=1.5)
    for members, colour in ((scored & ~hidden, output.PREDICTED), (scored & hidden, output.SECOND)):
        axis.plot(forecast[members, 0], forecast[members, 1], "o", color=colour, markersize=2)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(f"view {view}, ship {row}: path and one-step forecasts", fontsize=9)
    axis.legend(
        handles=[
            Line2D([], [], color=output.TRUE, label="true path"),
            Line2D(
                [],
                [],
                marker="o",
                color=output.PREDICTED,
                linestyle="none",
                label="forecast, ship seen at t",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color=output.SECOND,
                linestyle="none",
                label="forecast, ship hidden at t",
            ),
        ],
        frameon=False,
        fontsize=7,
    )
    axis = axes[0, 1]
    error = torch.where(scored, part["error"][:, view, row], torch.nan)
    stay = torch.where(scored, part["constant_velocity"][:, view, row], torch.nan)
    sigma = torch.where(scored, part["sigma"][:, view, row], torch.nan)
    seconds = ticks * meta["tick_length"]
    axis.plot(seconds, error, color=output.PREDICTED, linewidth=1.2, label="head error")
    axis.plot(
        seconds,
        stay,
        color=output.BASELINE,
        linewidth=1,
        linestyle="--",
        label="constant velocity error",
    )
    axis.plot(
        seconds, sigma, color=output.PREDICTED, linewidth=1, linestyle=":", label="head sigma"
    )
    for start, end in _spans(hidden & scored):
        axis.axvspan(
            float(seconds[start]), float(seconds[end]), color=output.GRID, alpha=0.9, linewidth=0
        )
    axis.set_yscale("log")
    axis.set_xlabel("time (s); shaded: the ship hidden at t")
    axis.set_ylabel("position error one tick ahead (m)")
    axis.legend(frameon=False, fontsize=7)
    axis = axes[0, 2]
    names = [name for name in POPULATIONS if name in numbers]
    places = torch.arange(len(names)).float()
    width = 0.27
    for offset, key, colour, label in (
        (-width, "head_error", output.PREDICTED, "head"),
        (0.0, "constant_velocity_error", output.BASELINE, "constant velocity"),
        (width, "stay_error", output.MUTED, "stays put"),
    ):
        axis.bar(
            places + offset,
            [numbers[name][key] for name in names],
            width * 0.92,
            color=colour,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(places.tolist(), [name.replace("_", " ") for name in names], fontsize=8)
    axis.set_ylabel("mean position error one tick ahead (m)")
    axis.legend(frameon=False, fontsize=7)
    axis.set_title(
        "by population (within one sigma: "
        + ", ".join(f"{numbers[n]['within_one_sigma']}" for n in names)
        + f"; right sigma {numbers['expected_within_one_sigma']})",
        fontsize=8,
    )
    figure.suptitle("Next state: one-step position forecasts", color=output.INK)
    figure.tight_layout()
    return [output.save(figure, directory, "next_state_track", meta, numbers)], numbers


def _spans(flags: torch.Tensor) -> list[tuple[int, int]]:
    """The runs of True in a (L,) bool tensor as (first, last) indices."""
    spans, start = [], None
    for index, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index - 1))
            start = None
    return spans
