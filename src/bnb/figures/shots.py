"""The shot figure set: does the shot head's predicted path bend through the fields as real
shots do?

For the labelled shots of a capture (the learner's mask: valid shots of own ships and of
enemies seen at launch), it compares three paths at each age: the true flight at zero
spread (the game's launch and transport through the fields), the head's decoded mean, and
the straight line, launch position plus launch velocity times age. A flight is bent when the
true path ends more than BEND_THRESHOLD to the side of that line; the error is measured on
bent and straight flights apart, and on bent ones the lateral part (across the launch
velocity), which only curvature makes.
"""

import pathlib

import torch

from bnb.models.yemong.decode import launch_frame_to_world

from . import output

BEND_THRESHOLD = 2.0  # m: a quarter of the smallest hit radius, to the side at the last age
GALLERY = 8  # shots in the gallery: the most bent, then straight ones
STRAIGHT_IN_GALLERY = 2
ONE_SIGMA_DISC = 1.0 - torch.exp(torch.tensor(-0.5)).item()  # P(|e| < sigma), isotropic 2D


def geometry(shots: dict, tick_length: float) -> dict:
    """The three paths of every shot, (L, R, N, A, 2) m in the launch frame, and what the
    measures read: age_valid (the shot is labelled: every age of it, past a hit too),
    last (the last valid age's index, -1 for none), unit (the launch velocity's direction),
    lateral and predicted_lateral (L, R, N, A): the true and the predicted path's offset
    to the left of the straight line, bend (L, R, N): the true lateral offset at the last
    age, and bent: |bend| > BEND_THRESHOLD."""
    true = shots["exact_snapshots"][..., :2]
    ages = torch.arange(1, true.shape[-2] + 1, dtype=true.dtype) * tick_length
    velocity = shots["launch_velocity"]
    straight = velocity[..., None, :] * ages[:, None]
    predicted = shots["predicted"]["displacement"]
    age_valid = shots["exact_age_valid"] & shots["labelled"][..., None]
    last = age_valid.sum(-1) - 1
    unit = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def left_of_line(path: torch.Tensor) -> torch.Tensor:
        away = path - straight
        return away[..., 1] * unit[..., None, 0] - away[..., 0] * unit[..., None, 1]

    lateral = left_of_line(true)
    bend = lateral.gather(-1, last.clamp_min(0)[..., None])[..., 0]
    return dict(
        true=true,
        straight=straight,
        predicted=predicted,
        age_valid=age_valid,
        last=last,
        unit=unit,
        lateral=lateral,
        predicted_lateral=left_of_line(predicted),
        bend=bend,
        bent=(bend.abs() > BEND_THRESHOLD) & (last >= 0),
    )


def _mean_by_age(values: torch.Tensor, mask: torch.Tensor) -> list[float]:
    count = mask.flatten(0, -2).sum(0)
    total = torch.where(mask, values, 0.0).flatten(0, -2).sum(0)
    return [round(float(value), 3) for value in total / count.clamp_min(1)]


def _lateral_fit(true: torch.Tensor, predicted: torch.Tensor) -> dict:
    """The predicted lateral offsets against the true ones (m, at the last age): the shot
    count, the mean lateral error of the head and of the straight line (the true offset),
    and the least-squares slope through the origin and the correlation."""
    if true.numel() < 2:
        return dict(shots=int(true.numel()))
    return dict(
        shots=int(true.numel()),
        head_lateral_error=round(float((predicted - true).abs().mean()), 3),
        straight_line_lateral_error=round(float(true.abs().mean()), 3),
        slope=round(float((true * predicted).sum() / true.square().sum()), 3),
        correlation=round(float(torch.corrcoef(torch.stack([true, predicted]))[0, 1]), 3),
    )


def last_age(values: torch.Tensor, paths: dict) -> torch.Tensor:
    """(L, R, N, A) values at each shot's last valid age, (L, R, N)."""
    return values.gather(-1, paths["last"].clamp_min(0)[..., None])[..., 0]


def measures(shots: dict, tick_length: float) -> dict:
    """The numbers: per class (straight, bent) the shot count, the mean displacement error
    (m) by age of the head and of the straight line, the head's mean sigma by age, and the
    share of shots whose last age lies within the head's one-sigma circle (ONE_SIGMA_DISC
    when its sigma is right). The lateral fit (`_lateral_fit`) of the bent flights: all of
    them, those launched inside a field's band (a nonzero ln n gradient at the launch
    point, which the shooter's row shows the model) and those that meet a band after
    launch. A slope of 1 is a head that bends as shots do; 0, straight lines only."""
    paths = geometry(shots, tick_length)
    head_error = (paths["predicted"] - paths["true"]).norm(dim=-1)
    straight_error = (paths["straight"] - paths["true"]).norm(dim=-1)
    sigma = shots["predicted"]["displacement_sigma"]
    covered = last_age((head_error <= sigma).float(), paths)
    flown = paths["last"] >= 0
    out = {}
    for name, members in (("straight", ~paths["bent"] & flown), ("bent", paths["bent"])):
        mask = paths["age_valid"] & members[..., None]
        out[name] = dict(
            shots=int(members.sum()),
            head_error_by_age=_mean_by_age(head_error, mask),
            straight_line_error_by_age=_mean_by_age(straight_error, mask),
            head_sigma_by_age=_mean_by_age(sigma, mask),
            within_one_sigma_last_age=round(float(covered[members].mean()), 3)
            if members.any()
            else None,
        )
    bent = paths["bent"]
    true, predicted = last_age(paths["lateral"], paths), last_age(paths["predicted_lateral"], paths)
    in_band = shots["launch_gradient"] > 0.0
    out["lateral"] = dict(
        bent=_lateral_fit(true[bent], predicted[bent]),
        launched_in_band=_lateral_fit(true[bent & in_band], predicted[bent & in_band]),
        met_band_after_launch=_lateral_fit(true[bent & ~in_band], predicted[bent & ~in_band]),
    )
    out["expected_within_one_sigma"] = round(ONE_SIGMA_DISC, 3)
    return out


def gallery_shots(paths: dict, count: int = GALLERY, straight: int = STRAIGHT_IN_GALLERY):
    """(tick, view, row) of the gallery: the most bent labelled shots, then straight ones,
    at most one per ship within the bullet's lifetime twice over (consecutive ticks'
    shots look alike)."""
    lifetime = paths["true"].shape[-2]
    whole = paths["last"] == lifetime - 1  # flew all its moves: no hit cuts it short
    score = torch.where(whole & paths["bent"], paths["bend"].abs(), -1.0)
    chosen: list[tuple[int, int, int]] = []

    def take(order: torch.Tensor, limit: int) -> None:
        taken = 0
        for flat in order.tolist():
            if taken == limit:
                return
            tick, view, row = (
                int(value) for value in torch.unravel_index(torch.tensor(flat), score.shape)
            )
            if any(v == view and r == row and abs(t - tick) < 2 * lifetime for t, v, r in chosen):
                continue
            chosen.append((tick, view, row))
            taken += 1

    bent_order = score.flatten().argsort(descending=True)
    bent_order = bent_order[score.flatten()[bent_order] > 0]
    take(bent_order, count - straight)
    plain = whole & ~paths["bent"]
    plain_order = torch.nonzero(plain.flatten())[:, 0]
    take(plain_order[:: max(1, plain_order.numel() // (8 * max(straight, 1)))], straight)
    return chosen


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The shot figures of a capture: the gallery (shots_gallery) and the error by age with
    the lateral bend (shots_error). Returns their paths and the numbers."""
    meta, rollout = artifact["meta"], artifact["rollout"]
    shots = rollout["shots"]
    numbers = measures(shots, meta["tick_length"])
    paths = geometry(shots, meta["tick_length"])
    written = [
        _gallery(meta, rollout, paths, directory, numbers),
        _errors(meta, paths, numbers, shots, directory),
    ]
    return written, numbers


def _world(shots: dict, index, displacement: torch.Tensor) -> torch.Tensor:
    tick, view, row = index
    return launch_frame_to_world(
        displacement,
        shots["launch_x"][tick, view, row],
        shots["launch_y"][tick, view, row],
        shots["heading_cos"][tick, view, row],
        shots["heading_sin"][tick, view, row],
    )


def _gallery(meta, rollout, paths, directory, numbers) -> pathlib.Path:
    from matplotlib.patches import Circle

    shots = rollout["shots"]
    chosen = gallery_shots(paths)
    columns = 4
    rows = max(1, -(-len(chosen) // columns))
    figure, axes = output.new_figure(rows, columns, size=(3.6, 3.6))
    image = None
    for axis, index in zip(axes.flat, chosen, strict=False):
        tick, view, row = index
        valid = paths["age_valid"][index]
        true = _world(shots, index, paths["true"][index])[valid]
        predicted = _world(shots, index, paths["predicted"][index])[valid]
        straight = _world(shots, index, paths["straight"][index])[valid]
        sigma = shots["predicted"]["displacement_sigma"][index][valid]
        fan = [
            _world(shots, index, shots["fan"][sample][index])[shots["fan_age_valid"][sample][index]]
            for sample in range(shots["fan"].shape[0])
        ]
        start = torch.stack([shots["launch_x"][index], shots["launch_y"][index]])
        every = torch.cat([true, predicted, straight, start[None]], 0)
        centre = 0.5 * (every.amin(0) + every.amax(0))
        half = 0.5 * float((every.amax(0) - every.amin(0)).max()) + 30.0
        window = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
        image = output.field_map(axis, rollout["fields"][tick, view], tuple(map(float, window)))
        for path in fan:
            line = torch.cat([start[None], path], 0)
            axis.plot(line[:, 0], line[:, 1], color=output.TRUE, linewidth=0.7, alpha=0.35)
        line = torch.cat([start[None], straight], 0)
        axis.plot(line[:, 0], line[:, 1], color=output.BASELINE, linewidth=1.2, linestyle="--")
        line = torch.cat([start[None], true], 0)
        axis.plot(line[:, 0], line[:, 1], color=output.TRUE, linewidth=2.0)
        for point, radius in zip(predicted, sigma, strict=True):
            axis.add_patch(
                Circle(
                    point.tolist(),
                    float(radius),
                    fill=False,
                    edgecolor=output.PREDICTED,
                    linewidth=0.8,
                    alpha=0.8,
                )
            )
        axis.plot(predicted[:, 0], predicted[:, 1], "o", color=output.PREDICTED, markersize=3)
        axis.set_xlim(window[0], window[1])
        axis.set_ylim(window[2], window[3])
        axis.set_aspect("equal")
        bend = float(paths["bend"][index])
        error = float((predicted[-1] - true[-1]).norm())
        axis.set_title(
            f"t={tick} view {view} ship {row}\nbend {bend:+.1f} m, head error {error:.1f} m",
            fontsize=8,
        )
    for axis in list(axes.flat)[len(chosen) :]:
        axis.set_visible(False)
    if image is not None:
        bar = figure.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
        bar.set_label("ln n (field)", color=output.INK)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=output.TRUE, linewidth=2.0, label="true flight, zero spread"),
        Line2D([], [], color=output.TRUE, linewidth=0.7, alpha=0.5, label="spread samples"),
        Line2D(
            [],
            [],
            color=output.PREDICTED,
            marker="o",
            linestyle="none",
            label="head mean, one-sigma circle",
        ),
        Line2D([], [], color=output.BASELINE, linestyle="--", label="straight line"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(
        "Shots: the true flight against the head's path, one dot per age", color=output.INK
    )
    return output.save(figure, directory, "shots_gallery", meta, numbers)


def _errors(meta, paths, numbers, shots, directory) -> pathlib.Path:
    figure, axes = output.new_figure(1, 3, size=(4.6, 4.0))
    ages = range(1, paths["true"].shape[-2] + 1)
    for axis, name in zip(axes[0, :2], ("straight", "bent"), strict=True):
        part = numbers[name]
        axis.plot(
            ages,
            part["head_error_by_age"],
            "-o",
            color=output.PREDICTED,
            markersize=4,
            linewidth=2,
            label="head mean",
        )
        axis.plot(
            ages,
            part["straight_line_error_by_age"],
            "--",
            color=output.BASELINE,
            linewidth=2,
            label="straight line",
        )
        axis.plot(
            ages,
            part["head_sigma_by_age"],
            ":",
            color=output.PREDICTED,
            linewidth=1.5,
            label="head sigma",
        )
        axis.set_title(
            f"{name} flights ({part['shots']} shots)\nlast age within one sigma: "
            f"{part['within_one_sigma_last_age']} (right sigma: "
            f"{numbers['expected_within_one_sigma']})",
            fontsize=9,
        )
        axis.set_xlabel("age (ticks)")
        axis.set_ylabel("mean displacement error (m)")
        axis.legend(frameon=False, fontsize=8)
    axis = axes[0, 2]
    bent = paths["bent"]
    true = last_age(paths["lateral"], paths)
    predicted = last_age(paths["predicted_lateral"], paths)
    in_band = shots["launch_gradient"] > 0.0
    limit = float(torch.cat([true[bent], predicted[bent]]).abs().max()) * 1.05 if bent.any() else 1
    axis.plot([-limit, limit], [-limit, limit], color=output.MUTED, linewidth=1)
    axis.axhline(0.0, color=output.BASELINE, linestyle="--", linewidth=1)
    lateral = numbers["lateral"]
    for members, colour, name in (
        (bent & ~in_band, output.PREDICTED, "met_band_after_launch"),
        (bent & in_band, output.SECOND, "launched_in_band"),
    ):
        fit = lateral[name]
        axis.plot(
            true[members],
            predicted[members],
            "o",
            color=colour,
            markersize=3,
            alpha=0.45,
            label=f"{name.replace('_', ' ')}: {fit['shots']} shots, slope {fit.get('slope')}",
        )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    axis.set_title("bent flights, at the last age", fontsize=9)
    axis.set_xlabel("true offset left of the straight line (m)")
    axis.set_ylabel("head's offset left of the straight line (m)")
    axis.set_aspect("equal")
    figure.suptitle(
        "Shot head: displacement error by age, bent against straight flights "
        f"(bent: more than {BEND_THRESHOLD:g} m to the side)",
        color=output.INK,
    )
    figure.tight_layout()
    return output.save(figure, directory, "shots_error", meta, numbers)
