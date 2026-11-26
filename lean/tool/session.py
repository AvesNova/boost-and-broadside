"""One sandbox session in its own process: the card is process-global (`tuning.flying`), so a
session that flies a tuned design at its own time scale cannot share a process with the
envelope computations. The child ticks lean.sandbox.world.World at 30 Hz and sends a frame per
tick; the parent forwards controls. No physics here: World runs the real game step."""

import os
import time

from lean.game import card, core, tuning
from lean.sandbox.world import World, field_list, stick_index

# No clock: the game without its time limit. BNB_POST_STALL=0 in the server's environment flies
# with the post-stall regime off (GameConfig.post_stall), for comparison.
CONFIG_SETTINGS = {
    "clock_ticks": 10**9,
    "post_stall": os.environ.get("BNB_POST_STALL", "1") != "0",
}


def frame(world: World) -> dict:
    """What the browser draws: poses, bullets and the player's readouts; per ship its sight
    (vision range p to the side, eccentricity e: core.sight_pairs) and whether the other ship
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
                vision_range=world.constant("vision_range", i),
                vision_eccentricity=world.constant("vision_eccentricity", i),
                seen=bool(seen[i]),
            )
        )
    bullets = []
    for ship in range(2):
        for k in range(s.bullet_x.shape[2]):
            if int(s.bullet_moves_left[0, ship, k]) > 0 and bool(s.bullet_real[0, ship, k]):
                step = float(s.bullet_speed[0, ship, k]) * world.h
                bx, by = float(s.bullet_x[0, ship, k]), float(s.bullet_y[0, ship, k])
                bullets.append(
                    [
                        bx,
                        by,
                        float(s.bullet_dir_x[0, ship, k]) * step,
                        float(s.bullet_dir_y[0, ship, k]) * step,
                        ship,
                    ]
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
    )


def _finite(values: dict) -> dict:
    return {k: (v if v == v and abs(v) != float("inf") else None) for k, v in values.items()}


def setup(world: World, config: core.GameConfig) -> dict:
    return dict(
        type="setup",
        arena_radius=config.arena_radius,
        ramp=config.boundary_ramp_width,
        tick=world.h,
        gravity=card.GRAVITY,
        fields=field_list(world.state),
    )


def run(
    connection,
    row: dict,
    factor: float,
    opponent: bool = True,
    realtime: bool = True,
    max_ticks: int | None = None,
) -> None:
    """The child's loop: apply the controls the parent sent, tick, send a frame; stop on
    {"stop": true} (or after max_ticks, for tests). Messages: {"side", "level", "power",
    "fire"} sets the held controls; {"reset": true}; {"opponent": bool}."""
    import torch

    torch.set_num_threads(1)
    with tuning.flying(row, factor):
        config = core.GameConfig(**CONFIG_SETTINGS)
        world = World(config, player_type=0, opponent_on=opponent)
        control = dict(side=0, level=3, power=0, fire=False)
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
                if "opponent" in message and bool(message["opponent"]) != world.opponent_on:
                    world.toggle_opponent()
                control.update({k: message[k] for k in control if k in message})
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
