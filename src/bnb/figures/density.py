"""The density figure set: the hex field of where each side's ships are, true against the
density head's expected counts, and how much of it the head knows beyond what it sees.

The true field is the learner's target (`bnb.learn.labels.density_targets`): each living
ship splits one unit over the three cells of its grid triangle. The head's field is its
expected count per cell (`bnb.models.yemong.decode.density`). Two cheap baselines: the
constant map (each cell's mean count over the capture) and the visible ships alone (the
same counting over the ships the view sees; for the own side, every ship). R^2 is
1 - (the squared error summed over cells and steps) / (the constant map's).
"""

import pathlib

import torch

from bnb.contracts.obs import GLOBAL, SHIP
from bnb.models.yemong.density import HexGrid, hex_cells

from . import output

TEAMS = ("ally", "enemy")
FRAMES = 2


def fields(artifact: dict) -> dict:
    """Per team, (L, R, C): true, head and visible (the visible-ships baseline) counts, and
    hidden (L, R) the living enemies the view does not see."""
    meta, rollout = artifact["meta"], artifact["rollout"]
    own, rings = meta["ships_per_team"], meta["hex_rings"]
    density, rows = rollout["density"], rollout["rows"]
    cells = density["ally"].shape[-1]
    visible = rollout["ships"][..., SHIP["visible"]] > 0.5
    alive = rows[..., SHIP["alive"]] > 0.5
    is_ally = (torch.arange(rows.shape[-2]) < own).expand(alive.shape)
    radius = rollout["globals"][..., GLOBAL["arena_radius"]][..., None]
    seen = HexGrid(rings).counts(
        rows[..., SHIP["x"]], rows[..., SHIP["y"]], is_ally, alive & visible, radius
    )
    out = {"hidden": (alive & ~visible & ~is_ally).sum(-1)}
    for index, team in enumerate(TEAMS):
        part = slice(index * cells, (index + 1) * cells)
        out[team] = dict(
            true=density["counts"][..., part], head=density[team], visible=seen[..., part]
        )
    return out


def rings_of(rings: int) -> torch.Tensor:
    """(C,) each cell's ring (0 the centre)."""
    axial, _ = hex_cells(rings)
    q, r = axial[:, 0], axial[:, 1]
    return torch.stack([q.abs(), r.abs(), (q + r).abs()]).amax(0)


def measures(artifact: dict) -> dict:
    """Per team: R^2 of the head and of the visible ships against the constant map; the
    head's mean squared error per cell by ring; and for the enemy side, the squared error
    per step (summed over cells) of the head and of the visible ships by the number of
    hidden living enemies."""
    part = fields(artifact)
    ring = rings_of(artifact["meta"]["hex_rings"])
    out = {}
    for team in TEAMS:
        true, head, seen = part[team]["true"], part[team]["head"], part[team]["visible"]
        constant = true.flatten(0, -2).mean(0)
        total = (true - constant).square().sum()

        def r_squared(prediction: torch.Tensor) -> float:
            return round(float(1.0 - (prediction - true).square().sum() / total), 4)

        square = (head - true).square().flatten(0, -2).mean(0)  # (C,) per cell
        out[team] = dict(
            head_r_squared=r_squared(head),
            visible_r_squared=r_squared(seen),
            head_square_error_by_ring=[
                round(float(square[ring == k].mean()), 5) for k in range(int(ring.max()) + 1)
            ],
        )
    true, head, seen = (part["enemy"][name] for name in ("true", "head", "visible"))
    by_hidden = {}
    for count in range(int(part["hidden"].max()) + 1):
        members = part["hidden"] == count
        if not members.any():
            continue
        by_hidden[str(count)] = dict(
            steps=int(members.sum()),
            head=round(float((head - true).square().sum(-1)[members].mean()), 4),
            visible=round(float((seen - true).square().sum(-1)[members].mean()), 4),
        )
    out["enemy"]["square_error_by_hidden_enemies"] = by_hidden
    return out


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The density figures: density_maps (true, head and signed error on the hex cells, for
    frames with hidden enemies) and density_error."""
    numbers = measures(artifact)
    part = fields(artifact)
    return [
        _maps(artifact, part, directory, numbers),
        _errors(artifact, numbers, directory),
    ], numbers


def _hexes(axis, centres, radius, values, colours, low, high):
    from matplotlib.collections import PolyCollection

    angles = torch.deg2rad(torch.arange(6) * 60.0 + 30.0)  # pointy-top cells
    corners = torch.stack([angles.cos(), angles.sin()], -1) * float(radius)
    polygons = (centres[:, None, :] + corners[None]).numpy()
    collection = PolyCollection(polygons, cmap=colours, edgecolors=output.SURFACE, linewidths=0.4)
    collection.set_array(values.numpy())
    collection.set_clim(low, high)
    axis.add_collection(collection)
    reach = float(centres.abs().max() + radius)
    axis.set_xlim(-reach, reach)
    axis.set_ylim(-reach, reach)
    axis.set_aspect("equal")
    return collection


def _maps(artifact, part, directory, numbers) -> pathlib.Path:
    meta, rollout = artifact["meta"], artifact["rollout"]
    own = meta["ships_per_team"]
    hidden = part["hidden"]
    order = (
        (
            hidden.float()
            + torch.rand(hidden.shape, generator=torch.Generator().manual_seed(0)) * 0.1
        )
        .flatten()
        .argsort(descending=True)
    )
    chosen = []
    for flat in order.tolist():
        tick, view = divmod(flat, hidden.shape[1])
        if all(abs(tick - t) >= 64 for t, _ in chosen):
            chosen.append((tick, view))
        if len(chosen) == FRAMES:
            break
    figure, axes = output.new_figure(len(chosen), 3, size=(4.2, 4.0))
    for line, (tick, view) in enumerate(chosen):
        centres = rollout["density"]["centres"][tick, view]
        radius = rollout["density"]["cell_radius"][tick, view]
        true = part["enemy"]["true"][tick, view]
        head = part["enemy"]["head"][tick, view]
        high = float(torch.maximum(true.max(), head.max()).clamp_min(1e-3))
        rows = rollout["rows"][tick, view]
        seen = rollout["ships"][tick, view, :, SHIP["visible"]] > 0.5
        alive = rows[:, SHIP["alive"]] > 0.5
        for column, (values, colours, low, top, title) in enumerate(
            (
                (true, "Blues", 0.0, high, "enemy density, true"),
                (head, "Blues", 0.0, high, "head's expected count"),
                (head - true, "RdBu_r", -high, high, "head minus true"),
            )
        ):
            axis = axes[line, column]
            image = _hexes(axis, centres, radius, values, colours, low, top)
            for row in range(own, rows.shape[0]):
                if not alive[row]:
                    continue
                point = rows[row, [SHIP["x"], SHIP["y"]]]
                if seen[row]:
                    axis.plot(*point, "o", color=output.TRUE, markersize=4)
                else:
                    axis.plot(
                        *point,
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=output.TRUE,
                        markersize=5,
                    )
            axis.set_title(f"t={tick} view {view}: {title}", fontsize=8)
            figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle(
        "Density: enemy ships per hex cell (dots: seen enemies; rings: hidden ones)",
        color=output.INK,
    )
    return output.save(figure, directory, "density_maps", meta, numbers)


def _errors(artifact, numbers, directory) -> pathlib.Path:
    meta = artifact["meta"]
    figure, axes = output.new_figure(1, 3, size=(4.4, 4.0))
    axis = axes[0, 0]
    places = torch.arange(len(TEAMS)).float()
    for offset, key, colour, label in (
        (-0.2, "head_r_squared", output.PREDICTED, "head"),
        (0.2, "visible_r_squared", output.BASELINE, "visible ships alone"),
    ):
        axis.bar(
            places + offset, [numbers[team][key] for team in TEAMS], 0.37, color=colour, label=label
        )
    axis.axhline(0.0, color=output.MUTED, linewidth=1)
    axis.set_xticks(places.tolist(), list(TEAMS))
    axis.set_ylabel("R^2 against the constant map")
    axis.legend(frameon=False, fontsize=8)
    axis = axes[0, 1]
    by_hidden = numbers["enemy"]["square_error_by_hidden_enemies"]
    counts = list(by_hidden)
    axis.plot(
        [int(c) for c in counts],
        [by_hidden[c]["head"] for c in counts],
        "-o",
        color=output.PREDICTED,
        label="head",
    )
    axis.plot(
        [int(c) for c in counts],
        [by_hidden[c]["visible"] for c in counts],
        "--o",
        color=output.BASELINE,
        label="visible ships alone",
    )
    axis.set_xlabel("hidden living enemies")
    axis.set_ylabel("enemy map: squared error per step")
    axis.legend(frameon=False, fontsize=8)
    axis = axes[0, 2]
    for team, colour in zip(TEAMS, (output.SECOND, output.PREDICTED), strict=True):
        values = numbers[team]["head_square_error_by_ring"]
        axis.plot(range(len(values)), values, "-o", color=colour, markersize=3, label=team)
    axis.set_xlabel("ring (0: the arena's centre)")
    axis.set_ylabel("head's squared error per cell")
    axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Density head: R^2, error against hidden enemies, error by ring", color=output.INK
    )
    figure.tight_layout()
    return output.save(figure, directory, "density_error", meta, numbers)
