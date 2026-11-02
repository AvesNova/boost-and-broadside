"""Seat symmetry (seat-symmetry-notes.md): the game treats its two teams alike.

The mirror test: build a state's mirror (the teams swapped, every position and direction
mapped by the map's symmetry, every per-team quantity swapped), step both with mirrored
actions and mirrored randomness, and compare the mirror's result with the mirror of the
original's. Team deathmatch maps are point-symmetric (a rotation by pi); Frontline's are
mirror-symmetric about the ring's neutral axis (a reflection, which also swaps the stick's
left and right and negates the angle of attack and the nose rate).
"""

import dataclasses
import math

import pytest
import torch

from bnb.contracts import Action, Obs
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.game.state import Bullets, Ships, State, replace

TYPES = api.ship_types()
STICK_REFLECTED = torch.tensor([0, 2, 1, 4, 3, 6, 5])  # left and right swap
ZONE_REFLECTED = torch.tensor([0, 4, 3, 2, 1])  # zone k <-> zone -k (mod 5)
VECTORS = {
    Ships: (
        ("x", "y"),
        ("heading_cos", "heading_sin"),
        ("log_index_gradient_x", "log_index_gradient_y"),
    ),
    Bullets: (
        ("x", "y"),
        ("direction_x", "direction_y"),
        ("log_index_gradient_x", "log_index_gradient_y"),
    ),
}
OBSERVED_VECTORS = (
    ("x", "y"),
    ("heading_cos", "heading_sin"),
    ("velocity_x", "velocity_y"),
    ("log_index_gradient_x", "log_index_gradient_y"),
)


class Mirror:
    """The map symmetry of one batch: a point reflection, or the reflection about each
    environment's Frontline axis (B,) rad. `seats` swaps the teams' sim indices."""

    def __init__(self, rules, axis: torch.Tensor | None = None):
        S, N = rules.ships_per_team, rules.num_ships
        self.rules, self.reflect = rules, axis is not None
        self.seats = torch.cat([torch.arange(S, N), torch.arange(S)])
        self.axis = axis

    def vector(self, x, y):
        if not self.reflect:
            return -x, -y
        shape = (-1,) + (1,) * (x.dim() - 1)
        c, s = torch.cos(2.0 * self.axis).reshape(shape), torch.sin(2.0 * self.axis).reshape(shape)
        return c * x + s * y, s * x - c * y

    def _per_ship(self, value, kind):
        leaves = {f.name: getattr(value, f.name)[:, self.seats] for f in dataclasses.fields(kind)}
        for a, b in VECTORS[kind]:
            leaves[a], leaves[b] = self.vector(leaves[a], leaves[b])
        if self.reflect and kind is Ships:
            for name in ("angle_of_attack", "angle_of_attack_sin", "nose_rate"):
                leaves[name] = -leaves[name]
        return kind(**leaves)

    def state(self, state: State) -> State:
        seats, fields, match, mode = self.seats, state.fields, state.match, state.mode
        field_x, field_y = self.vector(fields.x, fields.y)
        mode = replace(mode, score=mode.score.flip(-1))
        if self.reflect:
            zone = ZONE_REFLECTED
            zone_x, zone_y = self.vector(mode.zone_x[:, zone], mode.zone_y[:, zone])
            mode = replace(
                mode,
                front=-mode.front,
                zone_x=zone_x,
                zone_y=zone_y,
                zone_radius=mode.zone_radius[:, zone],
                role=4 - mode.role[:, zone],
                progress=mode.progress[:, zone],
                movement=mode.movement[:, zone],
                attack_ledger=mode.attack_ledger[:, zone][..., seats],
                defense_ledger=mode.defense_ledger[:, zone][..., seats],
            )
        return State(
            ships=self._per_ship(state.ships, Ships),
            bullets=self._per_ship(state.bullets, Bullets),
            fields=replace(fields, x=field_x, y=field_y),
            match=replace(
                match,
                ticks_since_seen=match.ticks_since_seen.flip(1)[..., seats],
                vision_range=match.vision_range[:, seats],
            ),
            mode=mode,
        )

    def action(self, action: Action) -> Action:
        packed = action.packed()[:, self.seats]
        if self.reflect:
            packed[..., 1] = STICK_REFLECTED[packed[..., 1].long()].int()
        return Action.unpack(packed)

    def spread(self, spread):
        x, y = self.vector(spread[:, self.seats, 0], spread[:, self.seats, 1])
        return torch.stack([x, y], -1)

    def uniforms(self, uniforms, done):
        """The tick uniforms that draw the mirrored map and spawns. An angle uniform u maps
        to u + 1/2 under the point reflection and to 2 axis / 2 pi - u under the
        reflection (the axis of a reset environment's new ring, drawn by the mode's first
        uniform)."""
        rules, (B, _) = self.rules, uniforms.shape
        M, N = rules.num_fields, rules.num_ships
        u = uniforms.clone()
        game = rules.game_uniform_count
        if self.reflect:
            axis = torch.where(done, 2.0 * math.pi * u[:, game], self.axis)

            def angle(value):
                return torch.remainder(
                    (axis / math.pi).reshape((-1,) + (1,) * (value.dim() - 1)) - value, 1.0
                )
        else:

            def angle(value):
                return torch.remainder(value + 0.5, 1.0)

        drawn = u[:, : 4 * M].reshape(B, M, 4)
        drawn[:, : M // 2, 1] = angle(drawn[:, : M // 2, 1])
        u[:, : 4 * M] = drawn.reshape(B, -1)
        start = rules.map_uniform_count
        rotation = u[:, start : start + 2].clone()
        u[:, start : start + 2] = torch.remainder(rotation.flip(-1) + 0.5, 1.0)
        u[:, start + 2 : start + 2 + N] = u[:, start + 2 : start + 2 + N][:, self.seats]
        mode = rules.modes[0]
        if mode.name == "frontline":
            per_ship = u[:, game + 2 : game + 2 + 2 * N].reshape(B, N, 2)[:, self.seats]
            per_ship[..., 1] = angle(per_ship[..., 1])
            u[:, game + 2 : game + 2 + 2 * N] = per_ship.reshape(B, -1)
        elif mode.name == "team_deathmatch":
            assert not mode.spawn_groups  # the anchor draw is by sim order: see the notes
            K = mode.spawn_candidates
            per_ship = u[:, game : game + N * (2 * K + 1)].reshape(B, N, 2 * K + 1)[:, self.seats]
            per_ship[..., K:] = angle(per_ship[..., K:])
            u[:, game : game + N * (2 * K + 1)] = per_ship.reshape(B, -1)
        return u

    def events(self, events):
        """Seat-swapped events; the killer as a sim index of the mirrored game."""
        seats = self.seats
        renumbered = seats.argsort()[events.killer.clamp_min(0).long()]
        killer = torch.where(events.killer >= 0, renumbered, -1)[:, seats].int()
        return dict(
            died=events.died[:, seats],
            fired=events.fired[:, seats],
            spawned=events.spawned[:, seats],
            killer=killer,
            done=events.done,
            result=-events.result,
            damage=events.damage[:, seats][:, :, seats],
            boundary_damage=events.boundary_damage[:, seats],
            objective=events.objective[:, seats],
            objective_progress=events.objective_progress[:, seats],
        )

    def view(self, ships: torch.Tensor) -> torch.Tensor:
        """A view's ship rows (B, N, C) mapped back by the symmetry."""
        ships = ships.clone()
        for a, b in OBSERVED_VECTORS:
            ships[..., SHIP[a]], ships[..., SHIP[b]] = self.vector(
                ships[..., SHIP[a]], ships[..., SHIP[b]]
            )
        if self.reflect:
            for name in ("angle_of_attack", "nose_rate"):
                ships[..., SHIP[name]] = -ships[..., SHIP[name]]
        return ships


def _leaves(value, prefix=""):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _leaves(getattr(value, field.name), f"{prefix}{field.name}.")
    else:
        yield prefix[:-1], value


def _assert_mirrored(rules, state, action, spread, uniforms, failures):
    """One tick of the state and of its mirror. Flags, counts and indices must agree
    exactly; floats to 1e-4 relative (the reflection rounds; the point reflection only
    reorders sums over ships). Appends the names of any leaves that differ."""

    def mirror_of(state):  # Frontline reflects about each environment's own ring axis
        frontline = rules.modes[0].name == "frontline"
        return Mirror(rules, state.mode.ring_rotation if frontline else None)

    after, events = api.step(rules, TYPES, state, action, spread, uniforms)
    mirror, mirror_after = mirror_of(state), mirror_of(after)  # a reset draws a new ring
    mirrored, mirrored_events = api.step(
        rules,
        TYPES,
        mirror.state(state),
        mirror.action(action),
        mirror.spread(spread),
        mirror.uniforms(uniforms, events.done),
    )
    want = dict(_leaves(mirror_after.state(after))) | mirror.events(events)
    got = dict(_leaves(mirrored)) | {
        name: getattr(mirrored_events, name) for name in mirror.events(events)
    }
    _, observed, _ = api.perceive(rules, after)
    _, observed_mirror, _ = api.perceive(rules, mirrored)
    want["view 0"] = observed.ships[:, 0]
    got["view 0"] = mirror_after.view(observed_mirror.ships[:, 1])
    want["zones"], got["zones"] = observed.zones[:, 0, :, 3:], observed_mirror.zones[:, 1, :, 3:]
    want["globals"], got["globals"] = observed.globals[:, 0], observed_mirror.globals[:, 1]
    for name, value in want.items():
        other = got[name]
        if value.is_floating_point():
            same = torch.isclose(value, other, rtol=1e-4, atol=1e-3)
        else:
            same = value == other
        if not bool(same.all()):
            failures.add(name)
    return after, events


def _melee(rules, batch, generator):
    """Every team's ships packed into one 30 m disc at a random point, shields low, and
    bullets in flight among them: contacts at the clamp (tau exactly 0 or 1) on several
    ships at once are common."""
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    B, N, K = state.bullets.moves_left.shape

    def uniform(*shape, low=-1.0, high=1.0):
        return low + (high - low) * torch.rand(*shape, generator=generator)

    centre_x, centre_y = uniform(B, 1) * 1500.0, uniform(B, 1) * 1500.0
    angle = uniform(B, N, low=0.0, high=2.0 * math.pi)
    shield = torch.where(uniform(B, N) < 0.0, 0.0, uniform(B, N, low=0.0, high=40.0))
    ships = replace(
        state.ships,
        x=centre_x + 30.0 * uniform(B, N),
        y=centre_y + 30.0 * uniform(B, N),
        heading_cos=angle.cos(),
        heading_sin=angle.sin(),
        shield=shield,
    )
    direction = uniform(B, N, K, low=0.0, high=2.0 * math.pi)
    bullets = replace(
        state.bullets,
        x=centre_x[..., None] + 40.0 * uniform(B, N, K),
        y=centre_y[..., None] + 40.0 * uniform(B, N, K),
        direction_x=direction.cos(),
        direction_y=direction.sin(),
        proper_speed=uniform(B, N, K, low=300.0, high=500.0),
        moves_left=torch.randint(0, 13, (B, N, K), generator=generator).int(),
    )
    return replace(state, ships=ships, bullets=bullets)


def _random_tick(rules, batch, generator):
    choice = torch.rand(batch, rules.num_ships, 3, generator=generator)
    action = Action.unpack((choice * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32))
    return (
        action,
        api.spread(rules, batch, generator=generator),
        api.uniforms(rules, batch, generator=generator),
    )


@pytest.mark.parametrize("preset", ["deathmatch", "frontline"])
def test_a_melee_tick_and_its_mirror_agree(preset):
    """Packed fights, where one bullet often touches an ally and an enemy at the same tau:
    hits, damage, deaths, kill credit, respawns, scores and the observation agree with the
    mirror. (An order by sim index failed here: team 0's bullets took the ally.)"""
    rules = api.rules_of(PRESETS[preset])
    generator = torch.Generator().manual_seed(3)
    failures = set()
    hits = 0
    for _ in range(4):
        state = _melee(rules, 64, generator)
        _, events = _assert_mirrored(rules, state, *_random_tick(rules, 64, generator), failures)
        hits += int((events.damage > 0).sum())
    assert hits > 200, hits
    assert not failures, sorted(failures)


def _view(observation: Obs, view: int) -> Obs:
    return Obs(
        ships=observation.ships[:, view],
        fields=observation.fields[:, view],
        globals=observation.globals[:, view],
        zones=observation.zones[:, view],
    )


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["deathmatch", "frontline"])
def test_chaser_games_and_their_mirror_agree_every_tick(preset):
    """The chaser against itself, fogged, compared with its mirror every tick after a
    warm-up: fire, flight, bullets, hits and damage, kill credit, death, respawns, captures,
    scores, episode ends with new maps and spawns, and the observation. The clocks are set
    so every environment's episode ends inside the compared window."""
    rules = api.rules_of(PRESETS[preset])
    generator = torch.Generator().manual_seed(5)
    batch, S, warm, compared = 24, rules.ships_per_team, 900, 300
    chaser = Chaser(S)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    ticks = rules.pools[0].episode_ticks
    state = api.start_clocks(
        state, torch.randint(ticks - warm - compared, ticks - warm, (batch,), generator=generator)
    )
    state, observation, _ = api.perceive(rules, state)
    failures, resets, deaths = set(), 0, 0
    done = torch.zeros(batch, dtype=torch.bool)
    for tick in range(warm + compared):
        action = Action.unpack(
            torch.cat(
                [chaser.step(None, _view(observation, view), done)[0].packed() for view in (0, 1)],
                1,
            )
        )
        inputs = (
            action,
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        if tick < warm:
            state, events = api.step(rules, TYPES, state, *inputs)
        else:
            state, events = _assert_mirrored(rules, state, *inputs, failures)
            resets += int(events.done.sum())
            deaths += int(events.died.sum())
        state, observation, _ = api.perceive(rules, state)
        done = events.done
    assert not failures, sorted(failures)
    assert resets >= batch and deaths > 10, (resets, deaths)
