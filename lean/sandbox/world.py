"""The sandbox world, without any drawing: the player (slot 0, team 0) and an optional chaser
(slot 1, team 1), stepped by lean.game.core.step at batch size 1 on the CPU.

The player never dies: a lethal tick restores its shield and counts a death. A killed chaser
comes back two seconds later 1500 m from the player.
"""

import dataclasses
import math
from collections import deque

import torch

from lean.game import card, core
from lean.scripted.chaser import chaser_act
from lean.scripted.opponent import ScriptedConfig

TRAIL_SECONDS = 5.0
HISTORY_SECONDS = 10.0
CHASER_RESPAWN_SECONDS = 2.0
CHASER_DISTANCE = 1500.0


def stick_index(side: int, level: int) -> int:
    """side +1 left (counter-clockwise), -1 right, 0 straight; level 1 fine, 2 sustained,
    3 full. Even indices turn left (card.STICK_LEVELS: positive level = left)."""
    if side == 0 or level == 0:
        return 0
    return 2 * level if side > 0 else 2 * level - 1


class World:
    def __init__(
        self,
        config: core.GameConfig,
        player_type: int = 1,
        chaser_type: int = 2,
        chaser_on: bool = True,
        seed: int = 0,
    ):
        self.config = config
        self.h = config.tick_length
        self.player_type, self.chaser_type, self.chaser_on = player_type, chaser_type, chaser_on
        self.generator = torch.Generator().manual_seed(seed)
        self.scripted = ScriptedConfig(arena_radius=config.arena_radius)
        self.reset()

    # ------------------------------------------------------------------ setup

    def _types(self) -> torch.Tensor:
        return torch.tensor([[self.player_type, self.chaser_type]])

    def reset(self) -> None:
        state = core.reset(self.config, 1, torch.tensor([[1, 1]]), self.generator, "cpu")
        constants = core.ship_constants(self.config, self._types())
        log_index, _, _ = core.medium(constants, torch.zeros(1, 2), torch.zeros(1, 2))
        self.state = dataclasses.replace(
            state,
            type=self._types(),
            constants=constants,
            x=torch.zeros(1, 2),
            y=torch.zeros(1, 2),
            heading_cos=torch.tensor([[0.0, 0.0]]),
            heading_sin=torch.tensor([[1.0, 1.0]]),
            speed=card.SPAWN_SPEED * torch.exp(log_index),
            energy=constants["spawn_energy"].clone(),
            shield=constants["max_shield"].clone(),
            alive=torch.tensor([[True, False]]),
        )
        self._place_chaser()
        self.state.alive[0, 1] = self.chaser_on
        self.tick = 0
        self.deaths = 0
        self.chaser_kills = 0
        self.chaser_respawn = -1
        self.trail = deque(maxlen=int(TRAIL_SECONDS / self.h))
        self.history = deque(maxlen=int(HISTORY_SECONDS / self.h))  # (speed, store fraction)
        self.turn_rate = 0.0
        self._last_track = self._track()
        self.previous = self.poses()

    def _place_chaser(self) -> None:
        s = self.state
        bearing = 2.0 * math.pi * float(torch.rand((), generator=self.generator))
        x = float(s.x[0, 0]) + CHASER_DISTANCE * math.cos(bearing)
        y = float(s.y[0, 0]) + CHASER_DISTANCE * math.sin(bearing)
        radius = math.hypot(x, y)
        if radius > 0.8 * self.config.arena_radius:  # keep it inside the arena
            x, y = (
                x * 0.8 * self.config.arena_radius / radius,
                y * 0.8 * self.config.arena_radius / radius,
            )
        toward = math.atan2(float(s.y[0, 0]) - y, float(s.x[0, 0]) - x)
        s.x[0, 1], s.y[0, 1] = x, y
        s.heading_cos[0, 1], s.heading_sin[0, 1] = math.cos(toward), math.sin(toward)
        s.speed[0, 1] = card.SPAWN_SPEED
        s.angle[0, 1], s.angle_cos[0, 1], s.angle_sin[0, 1], s.nose_rate[0, 1] = 0.0, 1.0, 0.0, 0.0
        s.energy[0, 1] = s.constants["spawn_energy"][0, 1]
        s.shield[0, 1] = s.constants["max_shield"][0, 1]
        s.alive[0, 1] = True

    def set_player_type(self, ship_type: int) -> None:
        """Swap the player's ship card in flight; position, speed and store are kept."""
        self.player_type = ship_type
        self.state.type = self._types()
        self.state.constants = core.ship_constants(self.config, self.state.type)
        s = self.state
        s.energy = torch.minimum(s.energy, s.constants["max_energy"])
        s.shield = torch.minimum(s.shield, s.constants["max_shield"])

    def toggle_chaser(self) -> None:
        self.chaser_on = not self.chaser_on
        if self.chaser_on:
            self._place_chaser()
        else:
            self.state.alive[0, 1] = False
            self.state.bullet_moves_left[0, 1] = 0
        self.chaser_respawn = -1

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
        action = chaser_act(self.scripted, self.state, team=1)
        action[0, 0] = torch.tensor([stick, power, int(fire)])
        noise = core.draw_noise(self.config, 1, 2, self.generator, "cpu")
        self.state, out = core.step(self.config, self.state, action, noise)
        s = self.state
        self.tick += 1
        if bool(out.died[0, 0]):
            self.deaths += 1
            s.alive[0, 0] = True
            s.shield[0, 0] = s.constants["max_shield"][0, 0]
        if bool(out.died[0, 1]):
            self.chaser_kills += 1
            self.chaser_respawn = self.tick + int(CHASER_RESPAWN_SECONDS / self.h)
        if self.chaser_on and self.tick == self.chaser_respawn:
            self._place_chaser()
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
        log_index, _, _ = core.medium(s.constants, s.x[:, :1], s.y[:, :1])
        return float(s.speed[0, 0] * torch.exp(-log_index[0, 0]))

    def store_fraction(self) -> float:
        return float(self.state.energy[0, 0] / self.state.constants["max_energy"][0, 0])

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
            "stall_degrees": math.degrees(float(s.constants["stall"][0, 0])),
            "shield": float(s.shield[0, 0] / s.constants["max_shield"][0, 0]),
        }
