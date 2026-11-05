"""The belief figure set: where the model believes each hidden enemy is (the ghost: the
belief's mean and sigma) against where it is, and whether the sigma is honest.

The belief is the replay record's (what the policy believed at t, after its sight), decoded
to world coordinates (`bnb.models.yemong.decode.ship_forecast`); the truth is the
privileged row at t. Coverage uses the training metric's definition
(`bnb.learn.head_statistics.belief_sums`): within k sigma when the squared error over the
quantity's dimensions, each divided by its variance and averaged, is at most k^2. For a
two-dimensional Gaussian with the stated sigma that holds with probability 1 - exp(-k^2), so
an honest sigma covers EXPECTED_COVERAGE.
"""

import math
import pathlib

import torch

from bnb.contracts.obs import SHIP
from bnb.learn.head_statistics import AGE_BUCKETS

from . import output

EXPECTED_COVERAGE = {1: 1.0 - math.exp(-1.0), 2: 1.0 - math.exp(-4.0)}
FRAMES = 4  # frames in the ghost figure
RECENTLY_HIDDEN = (0.2, 3.0)  # s: the hidden ages the ghost frames favour
TINY_VARIANCE = 1e-12


def hidden_beliefs(artifact: dict) -> dict:
    """Per enemy row (L, R, E): chosen (hidden, alive and holding a belief), age (s), the
    position error (m), sigma (m), and the normalised squared errors of position and
    velocity (the training metric's)."""
    meta, rollout = artifact["meta"], artifact["rollout"]
    own = meta["ships_per_team"]
    belief, rows = rollout["belief"], rollout["rows"][..., own:, :]
    visible = rollout["ships"][..., own:, SHIP["visible"]] > 0.5
    alive = rows[..., SHIP["alive"]] > 0.5
    chosen = ~visible & alive & belief["valid"]
    truth = rows[..., [SHIP["x"], SHIP["y"]]]
    error = belief["position"] - truth
    sigma = belief["position_sigma"]
    position_square = error.square().sum(-1) / sigma.square().clamp_min(TINY_VARIANCE) / 2.0
    velocity_error = belief["velocity"] - rows[..., [SHIP["velocity_x"], SHIP["velocity_y"]]]
    xx, xy, yy = belief["velocity_covariance"].unbind(-1)
    determinant = (xx * yy - xy * xy).clamp_min(TINY_VARIANCE**2)
    x, y = velocity_error.unbind(-1)
    velocity_square = (yy * x * x - 2.0 * xy * x * y + xx * y * y) / determinant / 2.0
    return dict(
        chosen=chosen,
        age=belief["age"].float() * meta["tick_length"],
        error=error.norm(dim=-1),
        sigma=sigma,
        position_square=position_square,
        velocity_square=velocity_square,
        visible=visible,
        alive=alive,
        truth=truth,
    )


def measures(artifact: dict) -> dict:
    """Per hidden-age bucket (`AGE_BUCKETS`): the count, the mean position error and sigma
    (m), and the share within one and two sigma of position and velocity; and the
    coverage an honest sigma gives."""
    part = hidden_beliefs(artifact)
    out = {}
    for name, low, high in AGE_BUCKETS:
        members = part["chosen"] & (part["age"] >= low) & (part["age"] < high)
        count = int(members.sum())
        if count == 0:
            continue

        def share(square: torch.Tensor, k: int) -> float:
            return round(float((square[members] <= k * k).float().mean()), 3)

        out[name] = dict(
            count=count,
            position_error=round(float(part["error"][members].mean()), 1),
            position_sigma=round(float(part["sigma"][members].mean()), 1),
            position_within_one_sigma=share(part["position_square"], 1),
            position_within_two_sigma=share(part["position_square"], 2),
            velocity_within_one_sigma=share(part["velocity_square"], 1),
            velocity_within_two_sigma=share(part["velocity_square"], 2),
        )
    out["expected"] = {f"within_{k}_sigma": round(v, 3) for k, v in EXPECTED_COVERAGE.items()}
    return out


def frames(part: dict, count: int = FRAMES) -> list[tuple[int, int]]:
    """(tick, view) of the frames with the most hidden believed enemies seen within the last
    RECENTLY_HIDDEN seconds (where the ghost still has a chance), at least 32 ticks apart."""
    low, high = RECENTLY_HIDDEN
    recent = part["chosen"] & (part["age"] >= low) & (part["age"] < high)
    score = recent.sum(-1).float() + 0.01 * part["chosen"].sum(-1)  # (L, R)
    chosen: list[tuple[int, int]] = []
    for flat in score.flatten().argsort(descending=True).tolist():
        tick, view = divmod(flat, score.shape[1])
        if any(abs(tick - t) < 32 for t, _ in chosen):
            continue
        chosen.append((tick, view))
        if len(chosen) == count:
            break
    return sorted(chosen)


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The belief figures: belief_ghosts (frames) and belief_coverage."""
    numbers = measures(artifact)
    part = hidden_beliefs(artifact)
    return [
        _ghosts(artifact, part, directory, numbers),
        _coverage(artifact, numbers, directory),
    ], numbers


def _ghosts(artifact, part, directory, numbers) -> pathlib.Path:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    meta, rollout = artifact["meta"], artifact["rollout"]
    own = meta["ships_per_team"]
    chosen = frames(part)
    figure, axes = output.new_figure(1, max(len(chosen), 1), size=(4.4, 4.4))
    for axis, (tick, view) in zip(axes.flat, chosen, strict=False):
        rows = rollout["rows"][tick, view]
        position = rows[:, [SHIP["x"], SHIP["y"]]]
        alive = rows[:, SHIP["alive"]] > 0.5
        hidden = part["chosen"][tick, view]
        ghosts = rollout["belief"]["position"][tick, view]
        sigma = part["sigma"][tick, view]
        points = [position[alive], ghosts[hidden]]
        every = torch.cat(points)
        centre = 0.5 * (every.amin(0) + every.amax(0))
        half = 0.5 * float((every.amax(0) - every.amin(0)).max()) + 250.0
        window = tuple(
            float(v)
            for v in (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
        )
        output.field_map(axis, rollout["fields"][tick, view], window)
        mine = alive.clone()
        mine[own:] = False
        seen = alive.clone()
        seen[:own] = False
        seen[own:] &= part["visible"][tick, view]
        axis.plot(*position[mine].T, "^", color=output.SECOND, markersize=6)
        axis.plot(*position[seen].T, "o", color=output.TRUE, markersize=6)
        for enemy in torch.nonzero(hidden)[:, 0].tolist():
            truth, ghost = position[own + enemy], ghosts[enemy]
            axis.plot(
                *truth, "o", markerfacecolor="none", markeredgecolor=output.TRUE, markersize=7
            )
            axis.plot([truth[0], ghost[0]], [truth[1], ghost[1]], color=output.MUTED, linewidth=0.8)
            axis.plot(*ghost, "x", color=output.PREDICTED, markersize=6)
            for k, style in ((1, "-"), (2, ":")):
                radius = k * math.sqrt(2.0) * float(sigma[enemy])
                axis.add_patch(
                    Circle(
                        ghost.tolist(),
                        radius,
                        fill=False,
                        linestyle=style,
                        edgecolor=output.PREDICTED,
                        linewidth=1,
                    )
                )
            age = float(part["age"][tick, view, enemy])
            axis.annotate(
                f"{age:.1f} s",
                ghost.tolist(),
                fontsize=7,
                color=output.INK,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axis.set_xlim(window[0], window[1])
        axis.set_ylim(window[2], window[3])
        axis.set_aspect("equal")
        axis.set_title(f"t={tick} view {view}: {int(hidden.sum())} hidden", fontsize=9)
    handles = [
        Line2D([], [], marker="^", color=output.SECOND, linestyle="none", label="own ship"),
        Line2D([], [], marker="o", color=output.TRUE, linestyle="none", label="enemy, seen"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=output.TRUE,
            linestyle="none",
            label="enemy, hidden (true)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            color=output.PREDICTED,
            linestyle="none",
            label="belief mean, 1 and 2 sigma (label: time since seen)",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(
        "Belief: the ghost of each hidden enemy against its true position", color=output.INK
    )
    return output.save(figure, directory, "belief_ghosts", meta, numbers)


def _coverage(artifact, numbers, directory) -> pathlib.Path:
    meta = artifact["meta"]
    buckets = [name for name, _, _ in AGE_BUCKETS if name in numbers]
    labels = [
        name.removeprefix("age_").removesuffix("_seconds").replace("_", "-") + " s"
        for name in buckets
    ]
    figure, axes = output.new_figure(1, 2, size=(5.4, 4.2))
    axis = axes[0, 0]
    places = range(len(buckets))
    for key, colour, style in (
        ("position_within_one_sigma", output.PREDICTED, "-o"),
        ("position_within_two_sigma", output.PREDICTED, ":o"),
        ("velocity_within_one_sigma", output.TRUE, "-s"),
        ("velocity_within_two_sigma", output.TRUE, ":s"),
    ):
        axis.plot(
            places,
            [numbers[b][key] for b in buckets],
            style,
            color=colour,
            markersize=4,
            label=key.replace("_", " "),
        )
    for k, value in EXPECTED_COVERAGE.items():
        axis.axhline(value, color=output.BASELINE, linestyle="--", linewidth=1)
        axis.annotate(
            f"honest, {k} sigma",
            (len(buckets) - 1, value),
            fontsize=7,
            color=output.MUTED,
            ha="right",
            va="bottom",
        )
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(list(places), labels, rotation=30, fontsize=7)
    axis.set_xlabel("time since seen")
    axis.set_ylabel("share of hidden enemies within k sigma")
    axis.legend(frameon=False, fontsize=7)
    axis = axes[0, 1]
    axis.plot(
        places,
        [numbers[b]["position_error"] for b in buckets],
        "-o",
        color=output.TRUE,
        label="mean position error",
    )
    axis.plot(
        places,
        [numbers[b]["position_sigma"] * math.sqrt(2.0) for b in buckets],
        "-o",
        color=output.PREDICTED,
        label="mean sigma x sqrt 2 (an honest RMS error)",
    )
    axis.set_yscale("log")
    axis.set_xticks(list(places), labels, rotation=30, fontsize=7)
    axis.set_xlabel("time since seen")
    axis.set_ylabel("m")
    axis.legend(frameon=False, fontsize=7)
    counts = ", ".join(str(numbers[b]["count"]) for b in buckets)
    figure.suptitle(
        f"Belief coverage by time since seen (rows per bucket: {counts})", color=output.INK
    )
    figure.tight_layout()
    return output.save(figure, directory, "belief_coverage", meta, numbers)
