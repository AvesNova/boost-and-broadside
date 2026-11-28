"""One sandbox session in its own process: the card is process-global (`tuning.flying`), so a
session that flies a tuned design at its own time scale cannot share a process with the
envelope computations. The child ticks lean.sandbox.world.World at 30 Hz and sends a frame per
tick; the parent forwards controls. No physics here: World runs the real game step."""

import time

from lean.contract import REAL
from lean.game import card, core, tuning
from lean.sandbox.world import HitRange, World, field_list, stick_index
from lean.tool import flight_config


def frame(world: World) -> dict:
    """What the browser draws: poses, bullets and the player's readouts; per ship its sight
    (the egg's reaches ahead, to the side and behind: core.sight_pairs) and whether the other ship
    knows it this tick: in its sight (or held by its tracking circle), or revealed by a shot or a
    hit (fog off: always)."""
    s = world.state
    seen = [True, True] if s.ticks_since_seen is None else (s.ticks_since_seen[0] == 0).tolist()
    ships = []
    for i, (x, y, heading) in enumerate(world.poses()):
        ships.append(
            dict(
                x=x,
                y=y,
                heading=heading,
                alive=bool(s.alive[0, i]),
                angle=float(s.angle[0, i]),
                speed=float(s.speed[0, i]),
                vision_ahead=world.constant("vision_ahead", i),
                vision_side=world.constant("vision_side", i),
                vision_behind=world.constant("vision_behind", i),
                tracking_radius=_tracking_radius(world, i),
                seen=bool(seen[i]),
            )
        )
    bullets = []
    slots = s.bullet_x.shape[2]
    # The tick each slot's bullet left the gun (the newest was written on the tick before).
    fired_tick = (s.tick - 1)[:, None] - core.bullet_ages(s.tick - 1, slots)
    for ship in range(2):
        for k in range(slots):
            # Real bullets in flight (a contacted one is no longer drawn).
            if int(s.bullet_state[0, ship, k]) == REAL:
                step = float(s.bullet_speed[0, ship, k]) * world.h
                bx, by = float(s.bullet_x[0, ship, k]), float(s.bullet_y[0, ship, k])
                bullets.append(
                    [
                        bx,
                        by,
                        float(s.bullet_dir_x[0, ship, k]) * step,
                        float(s.bullet_dir_y[0, ship, k]) * step,
                        ship,
                        # the bullet's identity: its slot and firing tick (the page joins a
                        # bullet's positions tick to tick, so a deflector's bend shows as a kink)
                        f"{ship * slots + k}:{int(fired_tick[0, k])}",
                    ]
                )
    extra = {}
    if isinstance(world, HitRange):
        extra = dict(
            range=world.range_readout(), deflector=world.deflector(), finished=world.finished
        )
    return dict(
        type="frame",
        tick=world.tick,
        ships=ships,
        bullets=bullets,
        readout=_finite(world.readout()),
        deaths=world.deaths,
        kills=world.opponent_kills,
        opponent_on=world.opponent_on,
        radii=dict(hit=world.constant("hit_radius", 0), core=world.constant("core_radius", 0)),
        **extra,
    )


def _tracking_radius(world: World, slot: int) -> float | None:
    """The tracking circle's radius (core.sight_pairs: a ship seen on the tick before is held
    within the observer's longest sight reach all round, the constants' tracking_range); None
    when the game has tracking off (GameConfig.fog_tracking) or fog off."""
    if not (world.config.fog and world.config.fog_tracking):
        return None
    return world.constant("tracking_range", slot)


def _finite(values: dict) -> dict:
    return {
        k: (v if v is not None and v == v and abs(v) != float("inf") else None)
        for k, v in values.items()
    }


def setup(world: World, config: core.GameConfig) -> dict:
    return dict(
        type="setup",
        arena_radius=config.arena_radius,
        ramp=config.boundary_ramp_width,
        tick=world.h,
        gravity=card.GRAVITY,
        fields=field_list(world.state),
        **hit_model_message(world),
    )


def hit_model_message(world: World) -> dict:
    """The hit model's inputs for the sliders (core.HIT_MODEL: name, label, unit, range, the
    world's value) and the deflector presets (card.DEFLECTOR_PRESETS): the page has no list of
    its own."""
    values = core.hit_model_of(world.state.constants)
    return dict(
        hit_model=[dict(entry._asdict(), value=values[entry.name]) for entry in core.HIT_MODEL],
        deflector_presets=card.DEFLECTOR_PRESETS,
    )


def run(
    connection,
    row: dict,
    factor: float,
    opponent: bool = True,
    realtime: bool = True,
    max_ticks: int | None = None,
    mode: str = "sandbox",
) -> None:
    """The child's loop: apply the controls the parent sent, tick, send a frame; stop on
    {"stop": true} (or after max_ticks, for tests). Messages: {"side", "level", "power",
    "fire"} sets the held controls; {"reset": true}; {"opponent": bool}; {"hit_model": {name:
    value}} changes hit-model inputs live (World.set_hit_model; answered with the whole model).
    mode "hit_range" runs lean.sandbox.world.HitRange: {"shooter": [x, y], "target_heading":
    radians} replace the controls and the frame carries the range readouts."""
    import torch

    torch.set_num_threads(1)
    with tuning.flying(row, factor):
        config = flight_config()
        hit_range = mode == "hit_range"
        world = (
            HitRange(config, player_type=0)
            if hit_range
            else World(config, player_type=0, opponent_on=opponent)
        )
        control = dict(side=0, level=3, power=0, fire=False)
        aim = dict(shooter=[-200.0, 0.0], target_heading=0.0)
        connection.send(setup(world, config))
        due = time.perf_counter()
        while max_ticks is None or world.tick < max_ticks:
            while connection.poll():
                message = connection.recv()
                if message.get("stop"):
                    return
                if message.get("reset"):
                    world.reset()
                    connection.send(setup(world, config))
                if "hit_model" in message:
                    try:
                        world.set_hit_model(**message["hit_model"])
                    except (KeyError, ValueError) as error:
                        connection.send(dict(type="error", message=str(error)))
                    connection.send(dict(type="hit_model", **hit_model_message(world)))
                aim.update({k: message[k] for k in aim if k in message})
                if (
                    "opponent" in message
                    and not hit_range
                    and bool(message["opponent"]) != world.opponent_on
                ):
                    world.toggle_opponent()
                control.update({k: message[k] for k in control if k in message})
            if hit_range:
                world.advance_range(
                    tuple(aim["shooter"]), float(aim["target_heading"]), bool(control["fire"])
                )
            else:
                world.advance(
                    stick_index(int(control["side"]), int(control["level"])),
                    int(control["power"]),
                    bool(control["fire"]),
                )
            connection.send(frame(world))
            if realtime:
                due += world.h
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:  # fell behind (a slow machine): do not run to catch up
                    due = time.perf_counter()
