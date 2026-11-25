"""The scripted opponent and the chaser: invariants, the power rule, the memory, the head-on
side, the persona population, and matches in the game."""

import dataclasses
import functools
import math

import pytest
import torch

from lean import game
from lean.contract import FIRES, POWERS, STICKS
from lean.game import core
from lean.scripted import (
    ScriptedConfig,
    act,
    chaser_act,
    draw_uniforms,
    fresh_memory,
    personas,
)
from lean.scripted.opponent import CLIMB, COAST, DIVE, explain
from lean.scripted.parameters import PARAMETERS, SAMPLED


def ramp_end(config: game.GameConfig) -> float:
    """The end of the boundary's damage ramp, in arena radii, read from the game's config."""
    return 1.0 + config.boundary_ramp_width / config.arena_radius


ARENA_RADIUS = game.GameConfig().arena_radius
BOUNDARY_RAMP_END = ramp_end(game.GameConfig())
# The left- and right-turning stick indices (lean.contract's STICKS; ScriptedConfig.left_is_even
# says which parity turns left; index 0 is straight).
LEFT = [i for i in range(1, STICKS) if (i % 2 == 0) == ScriptedConfig().left_is_even]
RIGHT = [i for i in range(1, STICKS) if (i % 2 == 0) != ScriptedConfig().left_is_even]

CPU = torch.device("cpu")


@functools.cache
def card() -> dict:
    """The default game's ship card (its speeds, radii and ranges; core.ship_card)."""
    return core.ship_card(core.GameConfig()).uniform


def random_state(envs: int, slots: int, seed: int):
    """A dealt core-game state with ships moved to random places, headings and speeds inside
    a 3 km square, two teams, and a few dead and absent rows; no fog, no fields."""
    generator = torch.Generator().manual_seed(seed)
    config = core.GameConfig(field_count=0, fog=False)
    sizes = torch.full((envs, 2), slots // 2, dtype=torch.long)
    state = core.reset(config, envs, sizes, generator, CPU)

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand((envs, slots), generator=generator)

    heading, angle = uniform(-math.pi, math.pi), uniform(-0.2, 0.2)
    exists = uniform(0.0, 1.0) > 0.15
    return core.with_velocity(
        state.replace(
            x=uniform(-1500.0, 1500.0),
            y=uniform(-1500.0, 1500.0),
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            # From just above the stall speed to a fifth above the top level speed (a dive).
            speed=uniform(1.1 * card()["stall_speed"], 1.2 * card()["cruise_speed"]),
            angle=angle,
            angle_cos=angle.cos(),
            angle_sin=angle.sin(),
            energy=uniform(0.0, 1.0) * state.constants["max_energy"],
            cooldown=uniform(0.0, 3.0).floor(),
            alive=exists & (uniform(0.0, 1.0) > 0.2),
            exists=exists,
        )
    )


def placed(
    own: dict, target: dict, energy_fraction: float = 0.5, arena_radius: float = ARENA_RADIUS
):
    """A 1v1 state with no fog or fields: the own ship (slot 0) and its target (slot 1),
    each given by x, y, heading (radians) and speed (m/s); straight flight, gun ready."""
    config = core.GameConfig(arena_radius=arena_radius, spawn="mixed", field_count=0, fog=False)
    state = core.reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    rows = (own, target)
    heading = torch.tensor([[row["heading"] for row in rows]])
    zeros = torch.zeros(1, 2)
    return core.with_velocity(
        state.replace(
            x=torch.tensor([[row["x"] for row in rows]]),
            y=torch.tensor([[row["y"] for row in rows]]),
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            speed=torch.tensor([[row["speed"] for row in rows]]),
            angle=zeros,
            angle_cos=torch.ones(1, 2),
            angle_sin=zeros,
            nose_rate=zeros,
            energy=torch.full((1, 2), energy_fraction) * state.constants["max_energy"],
            cooldown=torch.zeros_like(state.cooldown),
        )
    )


def memory_of(slots: int = 2, side: float = 1.0, target=None, lean: float = 0.0) -> dict:
    """A fresh memory for one environment: no kept target (or `target`), the drawn side, the
    plan lean (0: the plan by the energy advantage alone)."""
    memory = fresh_memory(1, slots, None, CPU)
    memory["side"] = torch.full((1, slots), side)
    memory["plan_lean"] = torch.full((1, slots), lean)
    if target is not None:
        memory["target"] = target
    return memory


def decide(state, memory=None):
    """The yardstick's marginals for team 0 with a fresh memory (side +1, no kept target)."""
    return explain(ScriptedConfig(), state, 0, None, memory or memory_of()).marginals


def test_actions_in_range_and_zero_for_uncontrolled_rows():
    """Catches an out-of-range index or an action leaking onto a dead, absent or enemy row."""
    config = ScriptedConfig()
    state = random_state(envs=32, slots=10, seed=1)
    generator = torch.Generator().manual_seed(2)
    limits = torch.tensor([STICKS, POWERS, FIRES])
    for spread in (0.0, 1.0):
        persona = personas(config, 32, 10, generator, CPU, spread)
        memory = fresh_memory(32, 10, generator, CPU)
        for team in (0, 1):
            controlled = state.exists & state.alive & (state.team == team)
            scripted, _ = act(
                config, state, team, persona, draw_uniforms(32, 10, generator, CPU), memory
            )
            for action in (scripted, chaser_act(config, state, team)):
                assert action.shape == (32, 10, 3) and action.dtype == torch.long
                assert (action >= 0).all() and (action < limits).all()
                assert (action[~controlled] == 0).all()
            assert scripted[controlled].any()


def test_marginals_are_distributions():
    """Catches a gate or floor that lets a head's probabilities leave [0, 1] or not sum to 1."""
    config = ScriptedConfig()
    state = random_state(envs=32, slots=10, seed=3)
    for spread in (0.0, 1.0):
        persona = personas(config, 32, 10, torch.Generator().manual_seed(4), CPU, spread)
        parts = explain(config, state, 0, persona, fresh_memory(32, 10, None, CPU)).marginals
        rows = parts.controlled
        for head in (parts.stick[rows], parts.power[rows]):
            assert torch.isfinite(head).all() and (head >= 0).all()
            assert torch.allclose(head.sum(-1), torch.ones(head.shape[0]), atol=1e-5)
        assert ((parts.fire[rows] >= 0) & (parts.fire[rows] <= 1)).all()


def test_spread_zero_is_the_yardstick_without_per_ship_tensors():
    """Review finding 2: the yardstick carries no persona at all (no dict of constants per
    ship) and uses no draw; equal inputs give equal outputs."""
    config = ScriptedConfig()
    generator = torch.Generator().manual_seed(6)
    before = generator.get_state()
    assert personas(config, 16, 10, generator, CPU, 0.0) is None
    assert torch.equal(generator.get_state(), before)
    state = random_state(envs=16, slots=10, seed=5)
    uniforms = draw_uniforms(16, 10, torch.Generator().manual_seed(8), CPU)
    memory = fresh_memory(16, 10, torch.Generator().manual_seed(9), CPU)
    first, second = (
        act(config, state, 0, None, uniforms, memory),
        act(config, state, 0, None, uniforms, memory),
    )
    assert torch.equal(first[0], second[0]) and torch.equal(first[1]["target"], second[1]["target"])
    with pytest.raises(ValueError):
        personas(config, 1, 2, generator, CPU, 1.5)


def test_randomness_only_from_sampling():
    """The action hold is gone (author 2026-11-22: randomness only from sampling): no hold
    fields, and the decision does not depend on the tick, only on the state, the memory and
    the uniforms."""
    config = ScriptedConfig()
    assert not [name for name in PARAMETERS if "hold" in name.split("_")]
    persona = personas(config, 8, 10, torch.Generator().manual_seed(1), CPU, 1.0)
    assert set(persona) == set(SAMPLED)
    state = random_state(envs=8, slots=10, seed=2)
    uniforms = draw_uniforms(8, 10, torch.Generator().manual_seed(3), CPU)
    memory = fresh_memory(8, 10, torch.Generator().manual_seed(4), CPU)
    first, _ = act(config, state, None, persona, uniforms, memory)
    for tick in (1, 2, 5, 1000):
        later = state.replace(tick=torch.full_like(state.tick, tick))
        assert torch.equal(act(config, later, None, persona, uniforms, memory)[0], first)


def test_turns_toward_a_target_on_its_left():
    """Catches a flipped side: a target at +90 degrees must get the left-turning stick."""
    config = ScriptedConfig()
    speed = card()["corner_speed"]
    state = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=speed), dict(x=0.0, y=400.0, heading=0.0, speed=speed)
    )
    assert decide(state).stick[0, 0, LEFT].sum() > 0.9
    assert chaser_act(config, state, 0)[0, 0, 0].item() == LEFT[-1]  # full stick left


# The power's cases (lean/scripted/NOTES.md "Power"): own ship at the origin heading +x; the
# target at `bearing` (radians, 0 ahead) and `distance` gun ranges, flying the same way at the
# same speed. Speeds in corner speeds (the sustained speed is below corner speed), the store as
# a fraction of full. Expected: the most likely power.
POWER_CASES = [
    # name, speed, bearing, distance, store, expected
    ("disengaged, slow: no dive (nothing to spend for)", 0.6, 0.0, 6.0, 0.5, COAST),
    ("disengaged at merge speed", 1.0, 0.0, 6.0, 0.5, COAST),
    ("disengaged and fast: bank the speed", 1.15, 0.0, 6.0, 0.5, CLIMB),
    ("disengaged and fast, store full", 1.15, 0.0, 6.0, 1.0, COAST),
    ("engaged, below the fight speed", 0.45, 0.0, 1.5, 0.5, DIVE),
    ("engaged, below the fight speed, store empty", 0.45, 0.0, 1.5, 0.0, COAST),
    # 0.92 is above the coast band of the fight speed (BFM-CMA2: 1.29 sustained speeds, 0.85
    # corner speeds), where the climb is what a straight target holds off. (0.7 sat in the band
    # of the earlier fight speed of 1.0; 0.8 dives under the 2,200 hp card's, 2026-11-24.)
    ("out of range behind a straight target: the engine closes", 0.92, 0.0, 1.5, 0.8, COAST),
]


def power_case(speed: float, bearing: float, distance: float, store: float):
    probe = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=100.0), dict(x=1.0, y=0.0, heading=0.0, speed=100.0)
    )
    corner, gun_range = probe.constants["corner_speed"], probe.constants["gun_range"]
    target = dict(
        x=distance * gun_range * math.cos(bearing),
        y=distance * gun_range * math.sin(bearing),
        heading=0.0,
        speed=speed * corner,
    )
    return decide(
        placed(dict(x=0.0, y=0.0, heading=0.0, speed=speed * corner), target, store)
    ).power[0, 0]


@pytest.mark.parametrize(
    "name, speed, bearing, distance, store, expected", POWER_CASES, ids=[c[0] for c in POWER_CASES]
)
def test_power_tracks_the_speed_target(name, speed, bearing, distance, store, expected):
    """Design B5: climb above the speed target v*, dive below it, coast in the band; v* the
    merge speed disengaged, the fight speed engaged (moved by the yo-yo: up to corner speed
    when falling behind out of range); the dive only engaged, each faded at its edge of the
    store. The tactics' own cases: test_tactics.py."""
    power = power_case(speed, bearing, distance, store)
    assert int(power.argmax()) == expected and float(power[expected]) > 0.6, power.tolist()


def test_kept_target_switches_only_for_a_much_closer_enemy():
    """Decision 5.2: keep the target unless another is closer than 0.8 of its distance."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=False)
    state = core.reset(config, 1, torch.tensor([[1, 2]]), torch.Generator().manual_seed(0), CPU)
    memory = memory_of(3, target=torch.tensor([[1, -1, -1]]))
    for other, expected in ((0.85, 1), (0.75, 2)):
        moved = state.replace(
            x=torch.tensor([[0.0, 1000.0, 1000.0 * other]]), y=torch.tensor([[0.0, 0.0, 1.0]])
        )
        parts = explain(ScriptedConfig(), moved, 0, None, memory).marginals
        assert int(parts.target[0, 0]) == expected, other
        fresh = explain(
            ScriptedConfig(), moved, 0, None, {**memory, "target": torch.full((1, 3), -1)}
        ).marginals
        assert int(fresh.target[0, 0]) == 2  # without memory the nearest


def test_head_on_side_is_drawn_once_per_ship_per_match_and_kept():
    """Decision 5.5: in the symmetric head-on pass (the target straight behind, flying
    straight) the plan's sense has no side to go by, so each ship turns by a side drawn once
    per ship per match (two-circle with more energy: to the side; one-circle: away from it);
    the side is never changed by `act` (a new match takes a fresh memory, `fresh_memory`)."""
    # Stores either side of the plan threshold (BFM-CMA2: -0.33, so 0.2 against 0.5 is two-circle).
    for own_store, plan in ((0.8, 1.0), (0.1, -1.0)):
        speed = card()["corner_speed"]
        state = placed(
            dict(x=0.0, y=0.0, heading=0.0, speed=speed),
            dict(x=-400.0, y=0.0, heading=math.pi, speed=speed),
        )
        store = torch.tensor([[own_store, 0.5]]) * state.constants["max_energy"]
        state = state.replace(energy=store)
        left = decide(state, memory_of(side=plan)).stick[0, 0]
        right = decide(state, memory_of(side=-plan)).stick[0, 0]
        assert left[LEFT].sum() > 0.9 and right[RIGHT].sum() > 0.9, own_store

    generator = torch.Generator().manual_seed(5)
    envs = 64
    memory = fresh_memory(envs, 2, generator, CPU)
    side = memory["side"].clone()
    assert set(side.unique().tolist()) == {-1.0, 1.0}
    play = game.GameConfig(arena_radius=1500.0)
    match = game.reset(play, envs, torch.ones(envs, 2, dtype=torch.long), generator, CPU)
    for _ in range(60):
        action, memory = act(
            ScriptedConfig(), match, None, None, draw_uniforms(envs, 2, generator, CPU), memory
        )
        match, _ = game.step(play, match, action, game.draw_noise(play, envs, 2, generator, CPU))
        assert torch.equal(memory["side"], side)


def test_the_lead_counts_the_medium():
    """Review finding 9: velocities over the map are proper velocities over the index n, and a
    bullet's time of flight is n (the shooter's) times longer. A crossing target inside a field
    (n = e^0.5) moves slower over the map, so the shooter outside leads it less: a smaller
    angle to the gun's lead point."""
    speed = card()["corner_speed"]
    base = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=speed),
        dict(x=500.0, y=0.0, heading=math.pi / 2, speed=speed),
    )
    lead = []
    for log_index in (0.0, 0.5):
        fields = torch.tensor(
            [[[500.0], [0.0], [150.0], [50.0], [log_index]]]
        )  # a field at the target
        state = base.replace(fields=fields)
        state.ships[0, core.LOG_INDEX] = torch.tensor([0.0, log_index])
        memory = memory_of()
        lead.append(float(explain(ScriptedConfig(), state, 0, None, memory).gun_error[0, 0]))
    print(f"angle to the lead point without and with the field: {lead}")
    assert 0.0 < lead[1] < lead[0] - 0.05


def test_reads_every_number_from_the_game():
    """Decision 5 (known weaknesses): the game provides every number. ScriptedConfig holds no
    ship characteristic or arena radius; the opponent steers by the game's arena: two gun
    ranges from the centre, flying outward at a target further out at corner speed, it turns
    back toward the centre in an arena whose edge is 0.6 sustained radii further out, and in
    the default arena (more than the boundary margin test's 4.48 sustained radii further out)
    flies at the target."""
    names = set(PARAMETERS) | {field.name for field in dataclasses.fields(ScriptedConfig)}
    for name in (
        "arena_radius",
        "gun_range",
        "corner_speed",
        "stall_angle",
        "max_energy",
        "hit_radius",
        "bullet_drag",
        "bullet_launch_speed",
        "max_acceleration",
    ):
        assert name not in names, name
    c = card()
    x = 2.0 * c["gun_range"]
    small, default = x + 0.6 * c["sustained_radius"], ARENA_RADIUS
    assert default - x > 4.48 * c["sustained_radius"]  # far from the default arena's edge
    heading = math.pi / 2 - 0.3  # north-east-ish on the +x axis: the centre is to the left
    bearing = heading - 0.4  # the target 600 m away, to the right
    own = dict(x=x, y=0.0, heading=heading, speed=c["corner_speed"])
    target = dict(
        x=x + 600.0 * math.cos(bearing),
        y=600.0 * math.sin(bearing),
        heading=heading,
        speed=c["corner_speed"],
    )
    left = {}
    for radius in (small, default):
        left[radius] = float(
            decide(placed(own, target, arena_radius=radius)).stick[0, 0, LEFT].sum()
        )
    assert left[small] > 0.9 and left[default] < 0.1


def test_boundary_margin_is_in_turn_radii_at_the_ship_speed():
    """Design B5: the edge band is a number of turn radii at the ship's speed, the sustained
    radius up to corner speed (as speed squared above), not speed^2 / max acceleration (half
    the real radius when slow). A slow ship 1.12 sustained radii inside the edge, flying
    outward at a target further out, turns back toward the centre; 4.48 radii inside it flies
    at the target; a fast ship (1.6 corner speeds)
    turns back from 4.48 radii. In sustained radii, so a flight change that moves the radius
    does not move the test."""
    heading = math.pi / 2 - 0.3  # the centre is to the left
    bearing = heading - 0.4  # the target 600 m away, to the right

    def left_share(inside_in_radii: float, speed_in_corner: float) -> float:
        probe = placed(
            dict(x=0.0, y=0.0, heading=0.0, speed=100.0),
            dict(x=1.0, y=0.0, heading=0.0, speed=100.0),
        )
        speed = speed_in_corner * probe.constants["corner_speed"]
        x = ARENA_RADIUS - inside_in_radii * probe.constants["sustained_radius"]
        own = dict(x=x, y=0.0, heading=heading, speed=speed)
        target = dict(
            x=x + 600.0 * math.cos(bearing),
            y=600.0 * math.sin(bearing),
            heading=heading,
            speed=speed,
        )
        return float(decide(placed(own, target)).stick[0, 0, LEFT].sum())

    assert left_share(1.12, 0.6) > 0.9
    assert left_share(4.48, 0.6) < 0.1
    assert left_share(4.48, 1.6) > 0.9


def test_never_shoots_through_a_team_mate():
    """Author 2026-11-23: no shot with a team-mate in the line of fire. A shooter aimed at an
    enemy 400 m ahead does not fire with a team-mate 200 m ahead on the line, and may fire
    with the team-mate 150 m off it."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=False)  # placed by hand
    state = core.reset(config, 1, torch.tensor([[2, 1]]), torch.Generator().manual_seed(0), CPU)
    zeros = torch.zeros_like(state.x)
    base = core.with_velocity(
        state.replace(
            heading_cos=torch.ones_like(state.x),
            heading_sin=zeros,
            speed=torch.full_like(state.x, card()["corner_speed"]),
            angle=zeros,
            angle_cos=torch.ones_like(state.x),
            angle_sin=zeros,
            cooldown=torch.zeros_like(state.cooldown),
        )
    )
    memory = fresh_memory(1, 3, None, CPU)
    fire = {}
    for name, ally_y in (("blocked", 0.0), ("clear", 150.0)):
        moved = base.replace(
            x=torch.tensor([[0.0, 200.0, 400.0]]), y=torch.tensor([[0.0, ally_y, 0.0]])
        )
        fire[name] = float(explain(ScriptedConfig(), moved, 0, None, memory).marginals.fire[0, 0])
    assert fire["blocked"] == 0.0
    assert fire["clear"] > 0.1


def test_one_call_for_both_teams_equals_the_two_single_team_calls():
    """Agreed fix 6: `team=None` decides for every ship in one call, each ship exactly as in
    its own team's call (under fog too); the memory of the two single calls, chained, is the
    memory of the one call."""
    generator = torch.Generator().manual_seed(3)
    config = ScriptedConfig()
    for game_config in (
        game.GameConfig(arena_radius=1500.0, spawn="mixed", field_count=0, fog=False),
        game.GameConfig(arena_radius=1500.0, spawn="mixed"),
    ):
        envs = 8
        sizes = torch.randint(3, 6, (envs, 2), generator=generator)
        state = game.reset(game_config, envs, sizes, generator, CPU)
        slots = state.x.shape[1]
        persona = personas(config, envs, slots, generator, CPU, 1.0)
        memory = fresh_memory(envs, slots, generator, CPU)
        deaths = 0
        for _ in range(120):
            uniforms = draw_uniforms(envs, slots, generator, CPU)
            both = explain(config, state, None, persona, memory).marginals
            for team in (0, 1):
                single = explain(config, state, team, persona, memory).marginals
                rows = single.controlled
                assert torch.equal(rows, both.controlled & (state.team == team))
                for name in ("stick", "power", "fire", "target"):
                    assert torch.equal(getattr(single, name)[rows], getattr(both, name)[rows]), name
            action, after = act(config, state, None, persona, uniforms, memory)
            first, middle = act(config, state, 0, persona, uniforms, memory)
            second, chained = act(config, state, 1, persona, uniforms, middle)
            assert torch.equal(action, first + second)
            assert torch.equal(after["target"], chained["target"])
            memory = after
            noise = game.draw_noise(game_config, envs, slots, generator, CPU)
            state, out = game.step(game_config, state, action, noise)
            deaths += int(out.died.sum())
        assert deaths > 0  # the ships fought


def play_scripted(
    envs: int,
    ticks: int,
    spread: float,
    seed: int,
    chaser_team: int | None = None,
    persona=None,
    config: game.GameConfig | None = None,
):
    """Scripted (or scripted against the chaser on `chaser_team`) 4v4 games in the core game
    (`config`, default the default game), finished games re-dealt with a fresh memory;
    `persona` (E, 8) per-slot rows instead of a draw at `spread`. Returns per-slot tallies over
    the run."""
    generator = torch.Generator().manual_seed(seed)
    config = config or game.GameConfig()
    sizes = torch.full((envs, 2), 4, dtype=torch.long)
    state = game.reset(config, envs, sizes, generator, CPU)
    slots = state.x.shape[1]
    scripted_config = ScriptedConfig()
    if persona is None:
        persona = personas(scripted_config, envs, slots, generator, CPU, spread)
    memory = fresh_memory(envs, slots, generator, CPU)
    radius = state.constants["arena_radius"]
    farthest = torch.zeros(envs, slots)
    fired = torch.zeros(envs, slots, dtype=torch.bool)
    lived = torch.zeros(envs, slots, dtype=torch.long)
    result = torch.full((envs,), -1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(ticks):
            action, memory = act(
                scripted_config,
                state,
                None,
                persona,
                draw_uniforms(envs, slots, generator, CPU),
                memory,
            )
            if chaser_team is not None:
                chaser = chaser_act(scripted_config, state, chaser_team)
                action = torch.where((state.team == chaser_team)[..., None], chaser, action)
            living = state.alive & state.exists
            distance = torch.sqrt(state.x * state.x + state.y * state.y)
            farthest = torch.maximum(farthest, torch.where(living, distance, 0.0))
            fired |= living & (action[..., 2] == 1)
            lived += living.long()
            state, out = game.step(
                config, state, action, game.draw_noise(config, envs, slots, generator, CPU)
            )
            result = torch.where((result < 0) & out.done, out.result, result)
            # A finished game starts afresh, as lean.learn re-deals: a fresh memory and a deal.
            fresh = fresh_memory(envs, slots, generator, CPU)
            memory = {
                name: torch.where(out.done[:, None], fresh[name], memory[name]) for name in memory
            }
            if chaser_team is None:
                state = game.take_deal(
                    state, out.done, game.deal(config, envs, slots, sizes, generator, CPU)
                )
            elif bool((result >= 0).all()):
                break
    return dict(farthest=farthest / radius, fired=fired, lived=lived, result=result)


def test_every_persona_is_competent_at_spread_one():
    """Sign-off 2026-11-23: every persona competent. At spread 1, 32 environments of 4v4 for
    1,800 ticks (finished games re-dealt; each slot keeps its persona): no ship goes past the
    end of the boundary's damage ramp, and of the ships that lived 20 s at least 97 in 100
    fired (not all: with BFM a defender survives longer without a firing solution)."""
    tallies = play_scripted(envs=32, ticks=1800, spread=1.0, seed=7)
    print(f"farthest {float(tallies['farthest'].max()):.3f} of the arena radius")
    assert float(tallies["farthest"].max()) < BOUNDARY_RAMP_END
    long_lived = tallies["lived"] >= round(20.0 / game.GameConfig().tick_length)  # 20 s
    assert long_lived.sum() > 100
    assert float(tallies["fired"][long_lived].float().mean()) >= 0.97


def test_opponent_beats_chaser():
    """Catches an opponent that does not fight: over a few dozen 4v4 games in the core game it
    must win clearly more than half against the chaser, seats alternated."""
    wins = losses = ended = 0
    for chaser_team in (0, 1):
        result = play_scripted(
            envs=24, ticks=3000, spread=0.0, seed=11 + chaser_team, chaser_team=chaser_team
        )["result"]
        wins += int((result == 1 - chaser_team).sum())
        losses += int((result == chaser_team).sum())
        ended += int((result >= 0).sum())
    print(f"opponent against chaser: {wins} wins, {losses} losses, {ended} ended of 48")
    assert ended >= 24
    assert wins > 0.65 * ended


def box_end(name: str, end: int) -> float:
    """A sampled row's value at one end of its box (spread 1)."""
    row = PARAMETERS[name]
    if row.law == "log":
        return row.default * math.exp(end * row.width)
    if row.law == "linear":
        return row.default + end * row.width
    return row.default + (row.width if end > 0 else 0.0)  # upward


def test_every_parameter_alone_at_both_ends_of_its_box_is_competent():
    """Each parameter is competent alone: every sampled row at each end of its box, the rest
    at the yardstick, in one batch (two environments of 4v4 per case, 900 ticks): no ship goes
    past the end of the boundary's damage ramp, and in every case at least half the ships
    fire. Fields are excluded on purpose (author decision 2026-11-25: scripted ships may be
    lost to fields; field-aware steering is not a competence asked of them): a field at the
    arena edge pushed a defending ship past the ramp."""
    cases = [
        (name, end)
        for name in SAMPLED
        for end in (-1, 1)
        if not (PARAMETERS[name].law == "upward" and end < 0)
    ]
    per_case, slots = 2, 8
    envs = per_case * len(cases)
    persona = {name: torch.full((envs, slots), PARAMETERS[name].default) for name in SAMPLED}
    for index, (name, end) in enumerate(cases):
        persona[name][index * per_case : (index + 1) * per_case] = box_end(name, end)
    config = game.GameConfig(field_count=0)
    tallies = play_scripted(
        envs=envs, ticks=900, spread=0.0, seed=3, persona=persona, config=config
    )
    farthest = tallies["farthest"].reshape(len(cases), -1).amax(-1)
    fired = tallies["fired"].reshape(len(cases), -1).float().mean(-1)
    worst = sorted(zip(fired.tolist(), cases))[:3]
    outermost = sorted(zip(farthest.tolist(), cases))[-3:]
    print(f"farthest {outermost}; lowest shares of ships that fired {worst}")
    assert float(farthest.max()) < ramp_end(config)
    assert float(fired.min()) >= 0.5


def ghost_state(distances: tuple[float, ...], seconds: tuple[float, ...]):
    """A fog-on state, no fields: the own ship (slot 0, team 0) at the centre heading +x, and
    one enemy per distance on the +x axis, each last seen `seconds` ago at that spot (its
    sighting recorded there, its ticks since seen set from the game's tick length)."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=True)
    sizes = torch.tensor([[1, len(distances)]])
    state = core.reset(config, 1, sizes, torch.Generator().manual_seed(0), CPU)
    zeros = torch.zeros_like(state.x)
    state = core.with_velocity(
        state.replace(
            x=torch.tensor([[0.0, *distances]]),
            y=zeros,
            heading_cos=torch.ones_like(state.x),
            heading_sin=zeros,
            speed=torch.full_like(state.x, state.constants["sustained_speed"]),
            angle=zeros,
            angle_cos=torch.ones_like(state.x),
            angle_sin=zeros,
        )
    )
    state = core.perceive(state, torch.ones_like(state.alive))  # every row seen where it is
    ticks = [round(value / state.constants["tick_length"]) for value in (0.0, *seconds)]
    return state.replace(ticks_since_seen=torch.tensor([ticks]))


def test_a_ghost_costs_its_roam_in_distance():
    """Author 2026-11-25: a remembered enemy's distance grows by the seconds since it was seen
    times its sustained speed. An enemy last seen 600 m away is chosen over one seen now at
    900 m while it is fresh, and not once its roam takes it past 900 m."""
    c = ghost_state((600.0, 900.0), (0.0, 0.0)).constants
    late = (900.0 - 600.0) / c["sustained_speed"]  # seconds at which the ghost is as far
    for seconds, expected in ((0.0, 1), (0.5 * late, 1), (1.5 * late, 2)):
        state = ghost_state((600.0, 900.0), (seconds, 0.0))
        parts = explain(ScriptedConfig(), state, 0, None, memory_of(3)).marginals
        assert int(parts.target[0, 0]) == expected, seconds


def test_a_ghost_that_could_be_anywhere_in_the_arena_is_dropped():
    """A remembered enemy stays a target while its roam is within the arena radius
    (ghost_cutoff 1), and is dropped beyond it: with no other enemy the ship then has no
    target (it flies toward the centre and does not fire)."""
    c = ghost_state((600.0,), (0.0,)).constants
    limit = ScriptedConfig().values["ghost_cutoff"] * c["arena_radius"] / c["sustained_speed"]
    for seconds, has_target in ((0.9 * limit, True), (1.1 * limit, False)):
        reasons = explain(ScriptedConfig(), ghost_state((600.0,), (seconds,)), 0, None, memory_of())
        assert bool(reasons.weights["disengaged"][0, 0] < 1.0) == has_target, seconds
        if not has_target:
            assert float(reasons.marginals.fire[0, 0]) == 0.0


def test_no_shot_inside_the_nose_cone_even_at_an_empty_shield():
    """Author 2026-11-25: inside the nose cone a hit does no damage, so the gun holds fire nose
    to nose, even at a target with no shield left (only damage kills); it fires at the same
    target from astern, shield or none."""
    own = dict(x=0.0, y=0.0, heading=0.0, speed=card()["corner_speed"])

    def fire(target_heading: float, shield: float | None = None) -> float:
        target = dict(x=450.0, y=0.0, heading=target_heading, speed=card()["corner_speed"])
        state = placed(own, target)
        if shield is not None:
            state = state.replace(shield=torch.tensor([[state.constants["max_shield"], shield]]))
        return float(decide(state).fire[0, 0])

    assert fire(math.pi) == 0.0  # nose to nose
    assert fire(0.0) > 0.5  # from astern
    assert fire(math.pi, shield=0.0) == 0.0  # nose to nose: no damage, so no kill
    assert fire(0.0, shield=0.0) > 0.5  # from astern the hit kills


def test_the_merge_miss_offset_moves_only_the_stick_and_only_inside_its_range():
    """Author 2026-11-25 (collisions): closing head-on on a collision course inside the miss
    range, the stick aims the miss offset beside the target's centre, away from the side he is
    on (him a little to my left: I turn right); the gun's aim is unchanged; beyond the range,
    off a collision course, or with the offset 0, nothing changes."""
    base = ScriptedConfig()
    off = base.replace(merge_miss_offset=0.0)
    hit_radius, radius = card()["hit_radius"], card()["sustained_radius"]
    offset = base.values["merge_miss_offset"] * hit_radius
    reach = base.values["merge_miss_range"] * radius
    speed = card()["corner_speed"]
    for distance, lateral, moved in (
        (0.8 * reach, 1.0, True),
        (0.8 * reach, -1.0, True),
        (1.2 * reach, 1.0, False),
        (0.8 * reach, 2.0 * offset, False),
    ):
        state = placed(
            dict(x=0.0, y=0.0, heading=0.0, speed=speed),
            dict(x=distance, y=lateral, heading=math.pi, speed=speed),
        )
        plain = explain(off, state, 0, None, memory_of())
        offsetting = explain(base, state, 0, None, memory_of())
        assert torch.equal(plain.gun_error, offsetting.gun_error)
        shift = float(offsetting.stick_error[0, 0] - plain.stick_error[0, 0])
        if moved:
            # He is on the side of `lateral`: I aim to his other side, by about offset / distance.
            assert math.copysign(1.0, shift) == -math.copysign(1.0, lateral)
            assert 0.5 * offset / distance < abs(shift) < 1.5 * offset / distance
        else:
            assert shift == 0.0


def test_the_merge_miss_offset_also_avoids_a_team_mate():
    """A team-mate closing head-on on a collision course (the enemy far off to the side): the
    stick turns away from him; with the offset 0 it does not."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=False)
    state = core.reset(config, 1, torch.tensor([[2, 1]]), torch.Generator().manual_seed(0), CPU)
    speed = card()["corner_speed"]
    heading = torch.tensor([[0.0, math.pi, 0.5 * math.pi]])
    state = core.with_velocity(
        state.replace(
            x=torch.tensor([[0.0, 150.0, 0.0]]),
            y=torch.tensor([[0.0, 1.0, 2500.0]]),
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            speed=torch.full((1, 3), speed),
            angle=torch.zeros(1, 3),
            angle_cos=torch.ones(1, 3),
            angle_sin=torch.zeros(1, 3),
            nose_rate=torch.zeros(1, 3),
        )
    )
    assert int(state.team[0, 0]) == int(state.team[0, 1]) != int(state.team[0, 2])
    memory = memory_of(slots=3)
    plain = explain(ScriptedConfig().replace(merge_miss_offset=0.0), state, 0, None, memory)
    offsetting = explain(ScriptedConfig(), state, 0, None, memory)
    assert float(offsetting.stick_error[0, 0]) < float(plain.stick_error[0, 0])  # turns right


def test_in_play_the_plan_is_held_across_passes_and_switches_only_when_losing():
    """Author 2026-11-26: in played 1v1 games (no fog, no fields), a ship's plan (memory
    "plan") is chosen once per engagement and outlives its passes (several commits of the turn
    sense under one plan); it turns to the other plan only when the angles off at that pass
    exceed those at its choice by the switch loss."""
    envs = 32
    generator = torch.Generator().manual_seed(11)
    config = game.GameConfig(field_count=0, fog=False)
    state = game.reset(config, envs, torch.ones(envs, 2, dtype=torch.long), generator, CPU)
    agent = ScriptedConfig()
    loss = agent.values["switch_loss"]
    memory = fresh_memory(envs, 2, generator, CPU)
    commits_under_one_plan = 0
    with torch.no_grad():
        for _ in range(900):
            reasons = explain(agent, state, None, None, memory)
            action, new = act(
                agent, state, None, None, draw_uniforms(envs, 2, generator, CPU), memory
            )
            living = state.alive & state.exists
            before, after = memory["plan"], new["plan"]
            flipped = living & (before * after < 0.0)
            grown = reasons.situation.angles_off - before.abs()
            assert bool((grown[flipped] > loss).all())
            commit = living & (memory["sense"] == 0.0) & (new["sense"] != 0.0)
            commits_under_one_plan += int((commit & (before != 0.0) & (before == after)).sum())
            memory = new
            state, out = game.step(
                config, state, action, game.draw_noise(config, envs, 2, generator, CPU)
            )
            if bool(out.done.all()):
                break
    assert commits_under_one_plan > 0
