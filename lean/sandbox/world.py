"""The sandbox world, without any drawing: the player (slot 0, team 0) and an optional opponent
(slot 1, team 1), stepped by lean.game.core.tick at batch size 1 on the CPU. The opponent is
the scripted opponent (the yardstick) or the chaser.

The player never dies: a lethal tick restores its shield and counts a death. A killed opponent
comes back two seconds later 1500 m from the player.
"""

import dataclasses
import math
from collections import deque

import torch

from lean import scripted
from lean.contract import CONTACTED
from lean.game import card, core

TRAIL_SECONDS = 5.0
HISTORY_SECONDS = 10.0
OPPONENT_RESPAWN_SECONDS = 2.0
OPPONENT_DISTANCE = 1500.0


def stick_index(side: int, level: int) -> int:
    """side +1 left (counter-clockwise), -1 right, 0 straight; level 1 Sustain, 2 Max,
    3 Snap. Even indices turn left (card.STICK_LEVELS: positive level = left)."""
    if side == 0 or level == 0:
        return 0
    return 2 * level if side > 0 else 2 * level - 1


class World:
    def __init__(
        self,
        config: core.GameConfig,
        player_type: int = 0,
        opponent_type: int = 0,
        opponent_on: bool = True,
        seed: int = 0,
        opponent: str = "scripted",
    ):
        if opponent not in ("scripted", "chaser"):
            raise ValueError(f"opponent {opponent!r}: 'scripted' or 'chaser'")
        self.config = config
        self.h = config.tick_length
        self.player_type, self.opponent_type, self.opponent_on = (
            player_type,
            opponent_type,
            opponent_on,
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.opponent = opponent
        self.scripted = scripted.ScriptedConfig()
        self.reset()

    # ------------------------------------------------------------------ setup

    def _types(self) -> torch.Tensor:
        return torch.tensor([[self.player_type, self.opponent_type]])

    def _constants(self):
        """Both ships' constants; any type may be flown, so the card is read for every type; the
        hit model's inputs set by `set_hit_model` on top."""
        every_type = dataclasses.replace(self.config, lineup=card.TYPE_NAMES)
        constants = core.ship_constants(every_type, self._types())
        hit_model = getattr(self, "hit_model", {})
        return core.with_hit_model(constants, **hit_model) if hit_model else constants

    def set_hit_model(self, **inputs: float) -> dict[str, float]:
        """Change hit-model inputs (core.HIT_MODEL names) live, for both ships: the next tick
        uses them (no restart, no recompile); kept across resets. Returns the whole model."""
        constants = core.with_hit_model(self.state.constants, **inputs)  # raises before keeping
        self.hit_model = {**getattr(self, "hit_model", {}), **inputs}
        self.state = self.state.replace(constants=constants)
        return core.hit_model_of(self.state.constants)

    def constant(self, name: str, slot: int) -> float:
        """One ship's constant (a number when equal for every type, else a per-slot row)."""
        return float(torch.as_tensor(self.state.constants[name]).expand(1, 2)[0, slot])

    def reset(self) -> None:
        state = core.reset(self.config, 1, torch.tensor([[1, 1]]), self.generator, "cpu")
        state = state.replace(
            type=self._types(), constants=self._constants(), alive=torch.tensor([[False, False]])
        )
        # The player at the centre, nose north, by the game's spawn rule.
        self.state = self._spawned(state, 0, 0.0, 0.0, math.pi / 2)
        self._place_opponent()
        self.state = self.state.replace(alive=torch.tensor([[True, self.opponent_on]]))
        self.tick = 0
        self.deaths = 0
        self.opponent_kills = 0
        self.opponent_respawn = -1
        self.trail = deque(maxlen=int(TRAIL_SECONDS / self.h))
        self.history = deque(maxlen=int(HISTORY_SECONDS / self.h))  # (speed, store fraction)
        self.turn_rate = 0.0
        self._last_track = self._track()
        self.previous = self.poses()

    def _spawned(self, state, slot: int, x: float, y: float, heading: float):
        """`state` with ship `slot` spawned at (x, y) heading `heading` (radians) through the
        game's spawn (lean.game.core.spawn: speed, store, shield and medium as at a deal)."""
        chosen = torch.zeros(1, 2, dtype=torch.bool)
        chosen[0, slot] = True
        x, y, heading_cos, heading_sin = (
            torch.full((1, 2), value) for value in (x, y, math.cos(heading), math.sin(heading))
        )
        return core.spawn(
            self.config, state, chosen, x, y, heading_cos, heading_sin, self.generator
        )

    def _place_opponent(self) -> None:
        """A new life for the opponent: placed 1500 m from the player, facing it."""
        s = self.state
        bearing = 2.0 * math.pi * float(torch.rand((), generator=self.generator))
        x = float(s.x[0, 0]) + OPPONENT_DISTANCE * math.cos(bearing)
        y = float(s.y[0, 0]) + OPPONENT_DISTANCE * math.sin(bearing)
        radius = math.hypot(x, y)
        if radius > 0.8 * self.config.arena_radius:  # keep it inside the arena
            x, y = (
                x * 0.8 * self.config.arena_radius / radius,
                y * 0.8 * self.config.arena_radius / radius,
            )
        toward = math.atan2(float(s.y[0, 0]) - y, float(s.x[0, 0]) - x)
        self.state = self._spawned(s, 1, x, y, toward)

    def set_player_type(self, ship_type: int) -> None:
        """Swap the player's ship card in flight; position, speed and store are kept."""
        self.player_type = ship_type
        self.state = self.state.replace(type=self._types(), constants=self._constants())
        s = self.state
        self.state = s.replace(
            energy=torch.minimum(s.energy, torch.as_tensor(s.constants["max_energy"])),
            shield=torch.minimum(s.shield, torch.as_tensor(s.constants["max_shield"])),
        )

    def toggle_opponent(self) -> None:
        self.opponent_on = not self.opponent_on
        if self.opponent_on:
            self._place_opponent()
        else:
            self.state.alive[0, 1] = False
            self.state.bullet_state[0, 1] = CONTACTED  # its bullets never touch a ship
        self.opponent_respawn = -1

    # ------------------------------------------------------------------ stepping

    def _track(self) -> float:
        """The player's flight-path direction: the nose turned back by the angle of attack."""
        s = self.state
        return math.atan2(float(s.heading_sin[0, 0]), float(s.heading_cos[0, 0])) - float(
            s.angle[0, 0]
        )

    def poses(self) -> list[tuple[float, float, float]]:
        s = self.state
        return [
            (
                float(s.x[0, i]),
                float(s.y[0, i]),
                math.atan2(float(s.heading_sin[0, i]), float(s.heading_cos[0, i])),
            )
            for i in range(2)
        ]

    @torch.no_grad()
    def advance(self, stick: int, power: int, fire: bool) -> None:
        """One 30 Hz game tick with the player's action."""
        self.previous = self.poses()
        if self.opponent == "scripted":
            uniforms = scripted.draw_uniforms(1, 2, self.generator, "cpu")
            action = scripted.act(self.scripted, self.state, 1, None, uniforms)
        else:
            action = scripted.chaser_act(self.state, team=1)
        action[0, 0] = torch.tensor([stick, power, int(fire)])
        self.state, out = core.tick(self.config, self.state, action, self.generator)
        s = self.state
        self.tick += 1
        if bool(out.died[0, 0]):
            self.deaths += 1
            s.alive[0, 0] = True
            s.shield[0, 0] = self.constant("max_shield", 0)
        if bool(out.died[0, 1]):
            self.opponent_kills += 1
            self.opponent_respawn = self.tick + int(OPPONENT_RESPAWN_SECONDS / self.h)
        if self.opponent_on and self.tick == self.opponent_respawn:
            self._place_opponent()
        track = self._track()
        turned = (track - self._last_track + math.pi) % (2.0 * math.pi) - math.pi
        self._last_track = track
        self.turn_rate = turned / self.h
        self.trail.append((float(s.x[0, 0]), float(s.y[0, 0])))
        self.history.append((self.ground_speed(), self.store_fraction()))

    # ------------------------------------------------------------------ readouts

    def ground_speed(self) -> float:
        """Metres per second over the map: the proper speed over the refractive index."""
        s = self.state
        return float(s.speed[0, 0] * torch.exp(-s.log_index[0, 0]))

    def store_fraction(self) -> float:
        return float(self.state.energy[0, 0]) / self.constant("max_energy", 0)

    def readout(self) -> dict[str, float]:
        speed = self.ground_speed()
        rate = self.turn_rate
        lateral = speed * abs(rate)
        s = self.state
        return {
            "speed": speed,
            "turn_rate_degrees": math.degrees(rate),
            "turn_radius": speed / abs(rate) if abs(rate) > 1e-4 else math.inf,
            "lateral_g": lateral / card.GRAVITY,
            "store": self.store_fraction(),
            "angle_of_attack_degrees": math.degrees(float(s.angle[0, 0])),
            "stall_degrees": math.degrees(self.constant("stall", 0)),
            "shield": float(s.shield[0, 0]) / self.constant("max_shield", 0),
        }


def field_list(state) -> list[tuple[float, float, float, float, float]]:
    """The first environment's fields as (x, y, radius, band, level in half-octaves)."""
    columns = state.fields[0]  # rows x, y, radius, band, target ln n
    return [
        (float(x), float(y), float(r), float(b), float(t) / card.LN_SQRT2)
        for x, y, r, b, t in zip(*columns)
    ]


class HitRange(World):
    """The hit-range mode: slot 1 is a target pinned at the origin (zero velocity, heading set by
    the page), slot 0 the shooter pinned to a point, facing due east (+x), always firing. Each
    tick both poses are
    overwritten (before and after the step, so bullets leave the pinned muzzle and the frame
    shows the pinned poses); everything else is the real step: the guns, bullet flight, the hit
    test, the egg, damage. The target is refilled when it dies. Statistics over a rolling window
    of ticks: damage, real shots fired and hits."""

    WINDOW_SECONDS = 10.0
    SIGMAS = 2.0  # the spreads are reported as 2 sigma (about 95 %)

    def __init__(
        self, config: core.GameConfig, player_type: int = 0, opponent_type: int = 0, seed: int = 0
    ):
        super().__init__(config, player_type, opponent_type, opponent_on=True, seed=seed)

    def reset(self) -> None:
        super().reset()
        self.window = deque(maxlen=max(1, round(self.WINDOW_SECONDS / self.h)))
        self.paths: dict = {}  # live real bullets by ring slot: dict(points, contact, a, b, range)
        self.finished: list = []  # this tick's ended flights: [points, contact index or None]
        self.finished_hits: list = []
        self.spreads = deque()  # (end tick, straight miss a, turned miss b, range at fire)
        self.kill_log = deque()  # (tick, time to kill)
        # no refractive-index fields: a clean range (they are moved far off the map)
        self.state.fields[:, 0] = 1e7
        self.state.fields[:, 1] = 1e7
        self.shooter_speed = float(self.state.speed[0, 0])
        self.shooter_ground = math.hypot(
            float(self.state.velocity_x[0, 0]), float(self.state.velocity_y[0, 0])
        )
        self.target_heading = 0.0
        self.shooter_xy = (-200.0, 0.0)
        self.target_kills = 0
        self.alive_ticks = 0
        self.kill_times: list[float] = []

    def _pin(self) -> None:
        s = self.state
        sx, sy = self.shooter_xy
        face = 0.0  # due east
        for slot, (x, y, heading, speed, ground) in enumerate(
            [
                (sx, sy, face, self.shooter_speed, self.shooter_ground),
                (0.0, 0.0, self.target_heading, 0.0, 0.0),
            ]
        ):
            s.x[0, slot], s.y[0, slot] = x, y
            s.heading_cos[0, slot], s.heading_sin[0, slot] = math.cos(heading), math.sin(heading)
            s.speed[0, slot] = speed
            s.velocity_x[0, slot], s.velocity_y[0, slot] = (
                ground * math.cos(heading),
                ground * math.sin(heading),
            )
            s.angle[0, slot] = 0.0
            s.angle_cos[0, slot], s.angle_sin[0, slot] = 1.0, 0.0
            s.nose_rate[0, slot] = 0.0
        s.alive[0, 0] = s.alive[0, 1] = True

    @torch.no_grad()
    def advance_range(
        self, shooter_xy: tuple[float, float], target_heading: float, fire: bool
    ) -> None:
        self.shooter_xy, self.target_heading = shooter_xy, target_heading
        self._pin()
        slot = int(self.state.tick[0]) % self.state.bullet_x.shape[2]  # this tick's ring slot
        self.previous = self.poses()
        fire = True  # always firing in this mode
        action = torch.zeros(1, 2, 3, dtype=torch.long)
        action[0, 0, 2] = int(fire)
        self.state, out = core.tick(self.config, self.state, action, self.generator)
        self.tick += 1
        s = self.state
        fired = bool(s.fired[0, 0])  # a real shot left the gun this tick (else the slot is a ghost)
        damage = float(out.hit_damage[0, 0, 1])
        self._follow_bullets(out, slot, fired)
        hits = len(self.finished_hits)
        self.window.append((damage, int(fired), hits))
        if damage > 0.0 or self.alive_ticks:
            self.alive_ticks += 1  # the kill clock starts at the first damage
        if bool(out.died[0, 1]):
            self.target_kills += 1
            self.kill_times.append(self.alive_ticks * self.h)
            self.kill_log.append((self.tick, self.alive_ticks * self.h))
            self.alive_ticks = 0
            s.shield[0, 1] = self.constant("max_shield", 1)
        s.shield[0, 0] = self.constant("max_shield", 0)
        self._pin()

    def _follow_bullets(self, out, slot: int, fired: bool) -> None:
        """Follow every real bullet of the shooter for its whole life (a ring of BULLET_SLOTS
        ticks; a contact makes it CONTACTED and it flies on): its path (tick-end points), `contact`
        the index of the point where it first touched the target (None while it has not), a: the
        signed miss distance of its straight launch line from the target's centre, b: the same
        for the post-turn line (`_after_egg`). The game's shot_outcome (1 hit an enemy, 0 missed
        at the end of the life) reports it once; that is when its numbers go to `spreads` (only
        real bullets: a ghost reports too). At the end of its life the flight is moved to
        `finished`: [points, contact index]. `finished_hits` lists this tick's real first hits."""
        s = self.state
        life = s.bullet_x.shape[2]
        self.finished, self.finished_hits = [], []
        sx, sy = self.shooter_xy
        if fired:
            dx, dy = float(s.bullet_dir_x[0, 0, slot]), float(s.bullet_dir_y[0, 0, slot])
            # the straight line from the muzzle: signed distance of the target (origin) from it
            a = dx * (-sy) - dy * (-sx)
            self.paths[slot] = dict(
                born=self.tick - 1,
                points=[[sx, sy]],
                contact=None,
                reported=False,
                a=a,
                b=self._after_egg(sx, sy, dx, dy, a),
                range=math.hypot(sx, sy),
            )
        for k, path in list(self.paths.items()):
            path["points"].append([float(s.bullet_x[0, 0, k]), float(s.bullet_y[0, 0, k])])
            outcome = int(out.shot_outcome[0, 0, k])
            if outcome == 1 and path["contact"] is None:
                path["contact"] = len(path["points"]) - 1
                self.finished_hits.append(k)
            if outcome >= 0 and not path["reported"]:
                path["reported"] = True
                self.spreads.append((self.tick, path["a"], path["b"], path["range"]))
            if self.tick - 1 - path["born"] >= life - 1:  # last tick of its life
                self.finished.append([path["points"], path["contact"]])
                del self.paths[k]
        horizon = self.tick - round(self.WINDOW_SECONDS / self.h)
        while self.spreads and self.spreads[0][0] <= horizon:
            self.spreads.popleft()
        while self.kill_log and self.kill_log[0][0] <= horizon:
            self.kill_log.popleft()

    def _after_egg(self, x: float, y: float, dx: float, dy: float, straight: float) -> float:
        """The signed miss distance b' of the bullet's post-turn straight line from the target's
        centre (same sign convention as the straight line's `straight`): the game's own egg on
        the bullet's straight path from (x, y) along (dx, dy): `core.egg_entry` finds the entry,
        `core.egg_turn` the turn (cos, sin) of the direction there; the line through the entry
        point along the turned direction, extended. A path that never enters the egg keeps its
        straight miss."""
        t = lambda v: torch.tensor(float(v), dtype=torch.float64)  # noqa: E731
        far = 2.0 * (math.hypot(x, y) + self.constant("egg_front", 1)) + 100.0
        mx, my = far * dx, far * dy
        heading = self.target_heading
        inverse_half = self.constant("inverse_egg_half_width", 1)
        entry = core.egg_entry(
            t(x), t(y), t(mx), t(my), t(math.cos(heading)), t(math.sin(heading)),
            self.constant("inverse_egg_front", 1), self.constant("inverse_egg_rear", 1),
            inverse_half,
        )  # fmt: skip
        if not bool(entry < torch.inf):
            return straight
        turn = core.egg_turn(
            t(x), t(y), t(mx), t(my), entry, inverse_half, t(self.constant("deflector_turn", 1))
        )
        cos, sin = float(turn.cos), float(turn.sin)
        px, py = x + float(entry) * mx, y + float(entry) * my
        ex, ey = dx * cos - dy * sin, dx * sin + dy * cos
        return ex * (-py) - ey * (-px)

    def _sigma(self, values: list[float]) -> float | None:
        if len(values) < 5:
            return None
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    def spread_readout(self) -> dict:
        """2 sigma (SIGMAS) of the signed miss distance at the target over the window, in metres
        and in milliradians (each miss over its own range at the shot): `straight` the line before
        any egg turn, `turned` the miss b' of the post-turn line (`_after_egg`). `width`
        the target's apparent half-width from the shooter: half the game's hit width
        (core.hit_width over the
        constants' hit_widths row, the agent's own) at the cosine of the bullets' path (due
        east) off the target's nose; `radius` the hit radius with no egg."""
        out = {}
        n = self.SIGMAS
        for name, index in (("straight", 1), ("turned", 2)):
            rows = [(r[index], r[3]) for r in self.spreads if r[index] is not None]
            sigma = self._sigma([m for m, _ in rows])
            angle = self._sigma([m / max(d, 1.0) for m, d in rows])
            out[name] = None if sigma is None else dict(metres=n * sigma, mrad=n * angle * 1000.0)
        widths = torch.as_tensor(self.state.constants["hit_widths"])[0, :, 1]
        hit_cos = torch.tensor(-math.cos(self.target_heading))
        half = 0.5 * float(core.hit_width(widths, hit_cos))
        distance = max(math.hypot(*self.shooter_xy), 1.0)
        out["width"] = dict(metres=half, mrad=1000.0 * half / distance)
        radius = self.constant("hit_radius", 1)
        out["radius"] = dict(metres=radius, mrad=1000.0 * radius / distance)
        out["samples"] = len(self.spreads)
        radius_now = self.constant("hit_radius", 1)
        out["inside_radius"] = sum(1 for r in self.spreads if abs(r[2]) < radius_now)
        return out

    def range_readout(self) -> dict:
        window = max(len(self.window), 1) * self.h
        damage = sum(w[0] for w in self.window) / window
        full = self.constant("max_shield", 1)
        sx, sy = self.shooter_xy
        # aspect: 0 = shooter dead astern of the target, 180 = head on
        bearing = (math.atan2(sy, sx) - self.target_heading + math.pi) % (2 * math.pi) - math.pi
        return dict(
            distance=math.hypot(sx, sy),
            aspect=180.0 - abs(math.degrees(bearing)),
            damage_per_second=damage,
            shots=sum(w[1] for w in self.window),
            hits=sum(w[2] for w in self.window),
            window_seconds=window,
            full_shield=full,
            shield=float(self.state.shield[0, 1]),
            implied_time_to_kill=full / damage if damage > 0 else None,
            kills=self.target_kills,
            last_kill_time=self.kill_times[-1] if self.kill_times else None,
            mean_kill_time=(
                sum(t for _, t in self.kill_log) / len(self.kill_log) if self.kill_log else None
            ),
            spreads=self.spread_readout(),
        )

    def egg(self) -> dict:
        """The target's egg and circle rows as the game holds them (core.HIT_MODEL names)."""
        names = ("egg_front", "egg_rear", "egg_half_width", "hit_radius", "core_radius")
        return {n: self.constant(n, 1) for n in names}
