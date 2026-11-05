"""The scenario lab: named, deterministic shots whose true path the physics gives, against
the shot head's prediction.

Every scenario starts from one template: a real moment of the capture's rollout, an own ship
of one view at the segment's last tick, alive throughout and far from every other ship. The
model sees that view's real history (its ships, its beliefs, its carry from the segment's
start); only the map changes. A scenario places one field relative to the shooter's launch
point and its bullet's launch direction, moves every other field that the flight could
reach to the far side of the arena, and shows the edited map in every observation of the
segment, with each visible ship's `log_refractive_index` and its gradient recomputed from
it. The history's motion is the real one (flown on the real map), so the edit is only
plausible where the shooter has not been: the fields are placed ahead of it.

The true flight is the learner's re-simulation (`bnb.learn.shots.resimulate`) of the one
shot, from the template's launch state with the medium at the launch point recomputed, at
zero spread and over a fan of spread draws, with no ship to hit.
"""

import dataclasses
import math
import pathlib

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import FIELD, GLOBAL, SHIP
from bnb.game import api
from bnb.game.physics.bullets import Launch, launch
from bnb.learn import shots
from bnb.learn.labels import privileged_rows
from bnb.models.yemong import decode

from . import output

HIGH, LOW = 2, -2  # field levels: n = 2 and n = 1/2
RADIUS = 400.0  # m, a scenario field's radius
BAND = 40.0  # m, its band width (the game's)
CLEARANCE = 250.0  # m: no other ship within this of the launch point at the template's tick
FAN = 6  # spread draws per scenario


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A field (radius m, level) placed by where the flight meets its boundary: the
    boundary point lies `along` flight lengths down the launch direction, and the
    boundary's normal there (outward, away from the field's centre) makes `incidence`
    radians with the launch direction, positive to the right. inside: the launch point is
    inside the field (the flight leaves it), else outside (the flight enters it). A
    scenario without a level places no field."""

    name: str
    description: str
    level: int | None = None
    radius: float = RADIUS
    along: float = 0.4
    incidence: float = 0.0
    inside: bool = False


SCENARIOS = (
    Scenario("straight_open", "no field in reach: n = 1 along the whole flight (the control)"),
    Scenario(
        "straight_core",
        "inside a large field's core: n = 2 along the whole flight",
        level=HIGH,
        radius=1200.0,
        along=-0.5,
        incidence=math.pi,
        inside=True,
    ),
    Scenario(
        "high_to_low",
        "leaves an n = 2 field at 20 degrees from the normal",
        level=HIGH,
        incidence=math.radians(20.0),
        inside=True,
    ),
    Scenario(
        "low_to_high",
        "enters an n = 2 field at 50 degrees from the normal",
        level=HIGH,
        incidence=math.radians(50.0),
    ),
    Scenario(
        "glancing",
        "passes through a small n = 2 field off its centre: enters at 55 "
        "degrees from the normal and leaves again, deflected",
        level=HIGH,
        radius=80.0,
        along=0.25,
        incidence=math.radians(55.0),
    ),
    Scenario(
        "internal_reflection",
        "meets the edge of an n = 2 field from inside at 60 "
        "degrees, past the critical angle (30 degrees)",
        level=HIGH,
        incidence=math.radians(60.0),
        inside=True,
    ),
    Scenario(
        "low_index_entry",
        "enters an n = 1/2 field at 20 degrees from the normal "
        "(bends away from the normal, and speeds up)",
        level=LOW,
        incidence=math.radians(20.0),
    ),
)


def _rotate(vector: torch.Tensor, angle: float) -> torch.Tensor:
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.stack([vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos])


def field_centre(scenario: Scenario, start: torch.Tensor, direction: torch.Tensor, flight):
    """The scenario field's centre (2,) m for a launch at start (2,) along direction (2,),
    a unit vector, over a flight of `flight` m."""
    boundary = start + direction * scenario.along * flight
    # the outward normal, `incidence` to the right of the flight (from inside, the flight
    # leaves along it; from outside, it enters against it)
    normal = _rotate(direction, -scenario.incidence)
    if not scenario.inside:
        normal = -normal
    return boundary - normal * scenario.radius


def template(trajectory, own: int) -> tuple[int, int, int]:
    """(tick, view, row): the segment's last tick and the own ship, alive at every tick of
    a view with no reset after the first, farthest from every other living ship (at least
    CLEARANCE)."""
    tick = trajectory.done.shape[0] - 1
    reset = trajectory.obs.globals[1:, :, GLOBAL["reset"]] > 0.5  # (L - 1, R)
    alive = trajectory.alive[..., :own].all(0) & ~reset.any(0)[:, None]  # (R, S)
    rows = privileged_rows(trajectory)[tick]  # (R, N, columns)
    position = rows[..., [SHIP["x"], SHIP["y"]]]
    living = rows[..., SHIP["alive"]] > 0.5
    distance = (position[:, :own, None, :] - position[:, None, :, :]).norm(dim=-1)
    distance = torch.where(living[:, None, :], distance, math.inf)
    distance = torch.where(torch.eye(position.shape[1], dtype=torch.bool)[:own], math.inf, distance)
    nearest = torch.where(alive, distance.amin(-1), -1.0)  # (R, S)
    best = int(nearest.argmax())
    view, row = divmod(best, own)
    if float(nearest[view, row]) < CLEARANCE:
        raise ValueError("no own ship of the capture is alive throughout and clear of the rest")
    return tick, view, row


def edited_fields(
    fields: torch.Tensor, scenario: Scenario, start, direction, flight, arena_radius: float
) -> torch.Tensor:
    """The map (M, len(FIELD_COLUMNS)) with every field the flight could reach moved to the
    far side of the arena, and the scenario's field in the first moved slot (or the
    nearest field's, when none had to move)."""
    fields = fields.clone()
    centres = fields[:, [FIELD["x"], FIELD["y"]]]
    reach = (
        (centres - start).norm(dim=-1) - fields[:, FIELD["radius"]] - fields[:, FIELD["band_width"]]
    )
    near = reach < flight + 2.0 * RADIUS
    away = -start / start.norm().clamp_min(1.0) * 0.9 * arena_radius
    fields[near, FIELD["x"]] = away[0]
    fields[near, FIELD["y"]] = away[1]
    if scenario.level is not None:
        slot = int(torch.nonzero(near)[0]) if near.any() else int(reach.argmin())
        centre = field_centre(scenario, start, direction, flight)
        fields[slot] = torch.tensor(
            [float(centre[0]), float(centre[1]), scenario.radius, BAND, float(scenario.level)]
        )
    return fields


def _medium(fields: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """(ln n, gradient x, gradient y) of fields (B, M, 5) at points (B, P)."""
    return api.log_index_gradient(
        fields[..., FIELD["x"]],
        fields[..., FIELD["y"]],
        fields[..., FIELD["radius"]],
        fields[..., FIELD["band_width"]],
        fields[..., FIELD["level"]].to(torch.int32),
        x,
        y,
    )


def with_map(ships: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
    """Ship rows (L, N, columns) with each visible row's medium columns recomputed for the
    fields (M, 5) of every tick."""
    ships = ships.clone()
    length = ships.shape[0]
    log_index, gradient_x, gradient_y = _medium(
        fields.expand(length, -1, -1), ships[..., SHIP["x"]], ships[..., SHIP["y"]]
    )
    shown = ships[..., SHIP["visible"]] > 0.5
    for column, value in (
        ("log_refractive_index", log_index),
        ("log_index_gradient_x", gradient_x),
        ("log_index_gradient_y", gradient_y),
    ):
        ships[..., SHIP[column]] = torch.where(shown, value, ships[..., SHIP[column]])
    return ships


def _one_shot(rules, types, start: Launch, fields: torch.Tensor, rows: int, spread):
    """The flight of one shot fired from `start` (one row's launch state, scalars), alone,
    through fields (M, 5): snapshots (draws, A, 5) and age_valid (draws, A) for spread
    (draws, 2) standard normals."""
    draws, ages = spread.shape[0], rules.bullet_moves
    shape = (ages, draws, rows)

    def column(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(()).expand(shape).clone()

    alive = torch.zeros(shape, dtype=torch.bool)
    alive[0, :, 0] = True  # the shooter fires once; no ship is there to be hit
    capture = shots.ShotCapture(
        launch=Launch(
            **{f.name: column(getattr(start, f.name)) for f in dataclasses.fields(Launch)}
        ),
        displacement_x=torch.zeros(shape),
        displacement_y=torch.zeros(shape),
        alive=alive,
    )
    noise = torch.zeros(shape + (2,))
    noise[0, :, 0] = spread
    result = shots.resimulate(
        rules,
        types,
        capture,
        fields.expand(ages, draws, -1, -1),
        torch.zeros((ages, draws), dtype=torch.bool),
        noise,
        rows,
    )
    return result.snapshots[0, :, 0], result.age_valid[0, :, 0]


@torch.no_grad()
def measure(model, rules, types, trajectory, config) -> dict:
    """The scenario lab of a capture: per scenario its fields, the true flight at zero
    spread and its spread fan (launch frame), and the head's decoded path, from one
    template. `bnb.figures.capture.capture` calls it with the rollout's segment."""
    own = trajectory.log_probability.shape[-1]
    tick, view, row = template(trajectory, own)
    at = (tick, view, row)
    start = Launch(
        **{
            f.name: getattr(trajectory.capture.launch, f.name)[at]
            for f in dataclasses.fields(Launch)
        }
    )
    zero = torch.zeros((1, 2))
    bullet = launch(rules, types, start, zero[0])
    direction = torch.stack([bullet.direction_x, bullet.direction_y]).reshape(2)
    speed = float(bullet.proper_speed / torch.exp(bullet.log_refractive_index))
    flight = speed * rules.bullet_moves * rules.tick_length
    origin = torch.stack([start.x, start.y]).reshape(2)
    arena_radius = float(trajectory.obs.globals[tick, view, GLOBAL["arena_radius"]])
    generator = torch.Generator().manual_seed(config.seed)
    spread = torch.cat([zero, torch.randn((FAN, 2), generator=generator)])
    rows = trajectory.obs.ships.shape[-2]
    maps, ship_rows, truths, valid = [], [], [], []
    for scenario in SCENARIOS:
        fields = edited_fields(
            trajectory.obs.fields[tick, view], scenario, origin, direction, flight, arena_radius
        )
        log_index, gradient_x, gradient_y = _medium(
            fields[None], start.x.reshape(1, 1), start.y.reshape(1, 1)
        )
        launched = dataclasses.replace(
            start,
            log_refractive_index=log_index.reshape(()),
            log_index_gradient_x=gradient_x.reshape(()),
            log_index_gradient_y=gradient_y.reshape(()),
        )
        snapshots, age_valid = _one_shot(rules, types, launched, fields, rows, spread)
        maps.append(fields)
        ship_rows.append(with_map(trajectory.obs.ships[: tick + 1, view], fields))
        truths.append(snapshots)
        valid.append(age_valid)
    count = len(SCENARIOS)
    length = tick + 1

    def repeated(tensor: torch.Tensor) -> torch.Tensor:  # (L, ...) of the view, per scenario
        return tensor[:length, view][:, None].expand((length, count) + tensor.shape[2:])

    obs = Obs(
        ships=torch.stack(ship_rows, 1),
        fields=torch.stack(maps)[None].expand(length, -1, -1, -1),
        globals=repeated(trajectory.obs.globals),
        zones=None if trajectory.obs.zones is None else repeated(trajectory.obs.zones),
    )
    replay = torch.utils._pytree.tree_map_only(torch.Tensor, repeated, trajectory.replay)
    carry = torch.utils._pytree.tree_map_only(
        torch.Tensor,
        lambda x: x[view][None].expand((count,) + x.shape[1:]),
        trajectory.initial_carry,
    )
    reset = obs.globals[..., GLOBAL["reset"]] > 0.5
    unrolled = model.unroll(carry, obs, replay, reset)
    ages = torch.arange(1, model.bullet_moves + 1, dtype=torch.float32)
    logits = model.shot_trajectory(unrolled.shot_latents[tick, :, row], ages)
    path = decode.shot_path(model.shot_code, logits)
    return dict(
        names=[scenario.name for scenario in SCENARIOS],
        template=dict(tick=tick, view=view, row=row),
        launch_x=start.x.reshape(()),
        launch_y=start.y.reshape(()),
        heading_cos=start.heading_cos.reshape(()),
        heading_sin=start.heading_sin.reshape(()),
        fields=torch.stack(maps),
        true=torch.stack(truths)[:, 0],  # (scenarios, A, 5) at zero spread
        true_age_valid=torch.stack(valid)[:, 0],
        fan=torch.stack(truths)[:, 1:, :, :2],  # (scenarios, FAN, A, 2)
        predicted={f.name: getattr(path, f.name) for f in dataclasses.fields(path)},
    )


def measures(lab: dict) -> dict:
    """Per scenario, at the last age the true flight reaches: the true path's offset from
    the zero-field line of the control (m, the lateral bend the field adds, left positive)
    and the head's, and the head's displacement error and its sigma (m)."""
    true = lab["true"][..., :2]
    predicted = lab["predicted"]["displacement"]
    sigma = lab["predicted"]["displacement_sigma"]
    control = lab["names"].index("straight_open")
    reference = true[control]  # (A, 2)
    unit = reference[-1] / reference[-1].norm().clamp_min(1e-6)
    out = {}
    for index, name in enumerate(lab["names"]):
        last = int(lab["true_age_valid"][index].sum()) - 1

        def left(point: torch.Tensor) -> float:
            away = point - reference[last]
            return round(float(away[1] * unit[0] - away[0] * unit[1]), 2)

        out[name] = dict(
            true_bend=left(true[index, last]),
            head_bend=left(predicted[index, last]),
            head_error=round(float((predicted[index, last] - true[index, last]).norm()), 2),
            head_sigma=round(float(sigma[index, last]), 2),
            true_reach=round(float(true[index, last].norm()), 1),
            head_reach=round(float(predicted[index, last].norm()), 1),
        )
    return out


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The lab's figure (shots_scenarios): one panel per scenario."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    meta, lab = artifact["meta"], artifact.get("scenarios")
    if lab is None:
        raise SystemExit("the capture holds no scenario lab: capture again with this code")
    numbers = measures(lab)
    names = lab["names"]
    columns = 4
    rows = -(-len(names) // columns)
    figure, axes = output.new_figure(rows, columns, size=(4.2, 4.0))

    def world(displacement: torch.Tensor) -> torch.Tensor:
        return decode.launch_frame_to_world(
            displacement, lab["launch_x"], lab["launch_y"], lab["heading_cos"], lab["heading_sin"]
        )

    start = torch.stack([lab["launch_x"], lab["launch_y"]])
    image = None
    for axis, (index, name) in zip(axes.flat, enumerate(names), strict=False):
        valid = lab["true_age_valid"][index]
        true = world(lab["true"][index, :, :2])[valid]
        predicted = world(lab["predicted"]["displacement"][index])
        sigma = lab["predicted"]["displacement_sigma"][index]
        every = torch.cat([true, predicted, start[None]])
        centre = 0.5 * (every.amin(0) + every.amax(0))
        half = 0.5 * float((every.amax(0) - every.amin(0)).max()) + 60.0
        window = tuple(
            float(v)
            for v in (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
        )
        image = output.field_map(axis, lab["fields"][index], window)
        for sample in lab["fan"][index]:
            line = torch.cat([start[None], world(sample)[valid]])
            axis.plot(line[:, 0], line[:, 1], color=output.TRUE, linewidth=0.7, alpha=0.35)
        line = torch.cat([start[None], true])
        axis.plot(line[:, 0], line[:, 1], color=output.TRUE, linewidth=2.0)
        for point, radius in zip(predicted, sigma, strict=True):
            axis.add_patch(
                Circle(
                    point.tolist(),
                    float(radius),
                    fill=False,
                    edgecolor=output.PREDICTED,
                    linewidth=0.8,
                )
            )
        axis.plot(predicted[:, 0], predicted[:, 1], "o", color=output.PREDICTED, markersize=3)
        axis.set_xlim(window[0], window[1])
        axis.set_ylim(window[2], window[3])
        axis.set_aspect("equal")
        part = numbers[name]
        axis.set_title(
            f"{name}\nbend: true {part['true_bend']:+.1f} m, head {part['head_bend']:+.1f} m",
            fontsize=8,
        )
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.6, pad=0.01).set_label("ln n (field)")
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
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    template_at = lab["template"]
    figure.suptitle(
        f"Scenario lab: one shot (view {template_at['view']}, ship "
        f"{template_at['row']}, tick {template_at['tick']}) on edited maps; bend "
        "is the offset from the open-field flight",
        color=output.INK,
    )
    return [output.save(figure, directory, "shots_scenarios", meta, numbers)], numbers
