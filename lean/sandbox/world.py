"""The sandbox world, without any drawing: the player (slot 0, team 0) and an optional opponent
(slot 1, team 1), stepped by lean.game.core.step at batch size 1 on the CPU. The opponent is
the scripted opponent (the yardstick, its memory carried from tick to tick as everywhere
else) or the chaser.

The player never dies: a lethal tick restores its shield and counts a death. A killed opponent
comes back two seconds later 1500 m from the player.
"""

import dataclasses
import math
from collections import deque

import torch

from lean import scripted
from lean.game import card, core

TRAIL_SECONDS = 5.0
HISTORY_SECONDS = 10.0
OPPONENT_RESPAWN_SECONDS = 2.0
OPPONENT_DISTANCE = 1500.0


def stick_index(side: int, level: int) -> int:
    """side +1 left (counter-clockwise), -1 right, 0 straight; level 1 gradual, 2 middle,
    3 full. Even indices turn left (card.STICK_LEVELS: positive level = left)."""
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
        """Both ships' constants; any type may be flown, so the card is read for every type."""
        every_type = dataclasses.replace(self.config, lineup=card.TYPE_NAMES)
        return core.ship_constants(every_type, self._types())

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
        """A new life for the opponent: placed 1500 m from the player, facing it, with a fresh
        scripted memory (no kept target, a new head-on side), as at a match's start."""
        self.memory = scripted.fresh_memory(1, 2, self.generator, "cpu")
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
            self.state.bullet_moves_left[0, 1] = 0
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
            action, self.memory = scripted.act(
                self.scripted, self.state, 1, None, uniforms, self.memory
            )
        else:
            action = scripted.chaser_act(self.scripted, self.state, team=1)
        action[0, 0] = torch.tensor([stick, power, int(fire)])
        noise = core.draw_noise(self.config, 1, 2, self.generator, "cpu")
        self.state, out = core.step(self.config, self.state, action, noise)
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
