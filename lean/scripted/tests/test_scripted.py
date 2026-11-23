"""The scripted opponent and the chaser: invariants, the power rule, the memory, the head-on
side, the persona population, and matches in the game."""

import dataclasses
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
    marginals,
    personas,
    renew_memory,
)
from lean.scripted.opponent import CLIMB, COAST, DIVE
from lean.scripted.parameters import PARAMETERS, SAMPLED

CPU = torch.device("cpu")


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
            speed=uniform(100.0, 440.0),
            angle=angle,
            angle_cos=angle.cos(),
            angle_sin=angle.sin(),
            energy=uniform(0.0, 1.0) * state.constants["max_energy"],
            cooldown=uniform(0.0, 3.0).floor(),
            alive=exists & (uniform(0.0, 1.0) > 0.2),
            exists=exists,
        )
    )


def placed(own: dict, target: dict, energy_fraction: float = 0.5, arena_radius: float = 4200.0):
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


def decide(state, memory=None):
    """The yardstick's marginals for team 0 with a fresh memory (side +1, no kept target)."""
    memory = memory or {"target": torch.full((1, 2), -1), "side": torch.ones(1, 2)}
    return marginals(ScriptedConfig(), state, 0, None, memory)


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
        parts = marginals(config, state, 0, persona, fresh_memory(32, 10, None, CPU))
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
    assert not [name for name in PARAMETERS if "hold" in name]
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
    state = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=260.0), dict(x=0.0, y=400.0, heading=0.0, speed=260.0)
    )
    left_indices = [2, 4, 6] if config.left_is_even else [1, 3, 5]
    assert decide(state).stick[0, 0, left_indices].sum() > 0.9
    assert chaser_act(config, state, 0)[0, 0, 0].item() == left_indices[-1]


# The power rule's cases (lean/scripted/NOTES.md "Power"): own ship at the origin heading +x;
# the target at `bearing` (radians, 0 ahead) and `distance` gun ranges, flying the same way.
# Speeds in corner speeds, the store as a fraction of full. Expected: the most likely power.
POWER_CASES = [
    # name, speed, bearing, distance, store, expected
    ("slow, straight, target ahead", 0.55, 0.0, 2.0, 0.5, DIVE),
    ("below corner in a hard turn", 0.85, math.pi / 2, 1.5, 0.5, DIVE),
    ("slow in a hard turn, store empty", 0.6, math.pi / 2, 1.5, 0.0, COAST),
    ("at corner speed, straight", 1.0, 0.0, 1.5, 0.5, COAST),
    ("just above corner in a hard turn", 1.08, math.pi / 2, 1.5, 0.5, COAST),
    ("well above corner speed", 1.3, 0.0, 1.5, 0.5, CLIMB),
    ("very close behind the target", 1.0, 0.0, 0.12, 0.5, CLIMB),
    ("far, store low", 1.0, 0.0, 6.0, 0.1, CLIMB),
    ("far, store high", 1.0, 0.0, 6.0, 0.8, COAST),
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
def test_power_rule(name, speed, bearing, distance, store, expected):
    """Author 2026-11-23 (5.1): dive when too slow (above all in a turn), climb when too fast or
    too close, climb a little when far with a low store, otherwise coast."""
    power = power_case(speed, bearing, distance, store)
    if name == "far, store low":  # "climb a little": about half the ticks, never a dive
        assert 0.3 < float(power[CLIMB]) < 0.7 and float(power[DIVE]) < 0.05, power.tolist()
    else:
        assert int(power.argmax()) == expected and float(power[expected]) > 0.6, power.tolist()


def test_kept_target_switches_only_for_a_much_closer_enemy():
    """Decision 5.2: keep the target unless another is closer than 0.8 of its distance."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=False)
    state = core.reset(config, 1, torch.tensor([[1, 2]]), torch.Generator().manual_seed(0), CPU)
    memory = {"target": torch.tensor([[1, -1, -1]]), "side": torch.ones(1, 3)}
    for other, expected in ((0.85, 1), (0.75, 2)):
        moved = state.replace(
            x=torch.tensor([[0.0, 1000.0, 1000.0 * other]]), y=torch.tensor([[0.0, 0.0, 1.0]])
        )
        parts = marginals(ScriptedConfig(), moved, 0, None, memory)
        assert int(parts.target[0, 0]) == expected, other
        fresh = marginals(
            ScriptedConfig(), moved, 0, None, {**memory, "target": torch.full((1, 3), -1)}
        )
        assert int(fresh.target[0, 0]) == 2  # without memory the nearest


def test_head_on_side_is_drawn_once_per_ship_per_match_and_kept():
    """Decision 5.5: in the symmetric head-on pass (the target straight behind) each ship turns
    to a side drawn once per ship per match; the side is never changed by `act` and is redrawn
    only for a new match (`renew_memory`)."""
    state = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=260.0),
        dict(x=-400.0, y=0.0, heading=math.pi, speed=260.0),
    )
    left = decide(state, {"target": torch.full((1, 2), -1), "side": torch.ones(1, 2)}).stick[0, 0]
    right = decide(state, {"target": torch.full((1, 2), -1), "side": -torch.ones(1, 2)}).stick[0, 0]
    assert left[[2, 4, 6]].sum() > 0.9 and right[[1, 3, 5]].sum() > 0.9

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
    done = torch.arange(envs) < envs // 2
    renewed = renew_memory(memory, done, generator)
    assert torch.equal(renewed["side"][~done], side[~done])
    assert (renewed["target"][done] == -1).all()
    assert not torch.equal(renewed["side"][done], side[done])  # 64 fresh coins differ


def test_the_lead_counts_the_medium():
    """Review finding 9: velocities over the map are proper velocities over the index n, and a
    bullet's time of flight is n (the shooter's) times longer. A crossing target inside a field
    (n = e^0.5) moves slower over the map, so the shooter outside leads it less: a smaller aim
    error, so less of the full stick level."""
    base = placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=250.0),
        dict(x=500.0, y=0.0, heading=math.pi / 2, speed=250.0),
    )
    full_left = []
    for log_index in (0.0, 0.5):
        fields = torch.tensor(
            [[[500.0], [0.0], [150.0], [50.0], [log_index]]]
        )  # a field at the target
        state = base.replace(fields=fields)
        state.ships[0, core.LOG_INDEX] = torch.tensor([0.0, log_index])
        full_left.append(float(decide(state).stick[0, 0, 6]))
    print(f"P(full left) without and with the field: {full_left}")
    assert full_left[1] < full_left[0] - 0.05


def test_reads_every_number_from_the_game():
    """Decision 5 (known weaknesses): the game provides every number. ScriptedConfig holds no
    ship characteristic or arena radius; the opponent steers by the game's arena: near the edge
    of a 1,500 m arena, flying outward at a target further out, it turns back toward the
    centre, where in the 4,200 m arena it flies at the target."""
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
    heading = math.pi / 2 - 0.3  # north-east-ish at x = 1,350 m: the centre is to the left
    bearing = heading - 0.4  # the target 600 m away, to the right
    own = dict(x=1350.0, y=0.0, heading=heading, speed=260.0)
    target = dict(
        x=1350.0 + 600.0 * math.cos(bearing),
        y=600.0 * math.sin(bearing),
        heading=heading,
        speed=260.0,
    )
    left = {}
    for radius in (1500.0, 4200.0):
        left[radius] = float(
            decide(placed(own, target, arena_radius=radius)).stick[0, 0, [2, 4, 6]].sum()
        )
    assert left[1500.0] > 0.9 and left[4200.0] < 0.1


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
            speed=torch.full_like(state.x, 250.0),
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
        fire[name] = float(marginals(ScriptedConfig(), moved, 0, None, memory).fire[0, 0])
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
            both = marginals(config, state, None, persona, memory)
            for team in (0, 1):
                single = marginals(config, state, team, persona, memory)
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


def play_scripted(envs: int, ticks: int, spread: float, seed: int, chaser_team: int | None = None):
    """Scripted (or scripted against the chaser on `chaser_team`) 4v4 games in the core game,
    finished games re-dealt with a fresh memory. Returns per-slot tallies over the run."""
    generator = torch.Generator().manual_seed(seed)
    config = game.GameConfig()
    sizes = torch.full((envs, 2), 4, dtype=torch.long)
    state = game.reset(config, envs, sizes, generator, CPU)
    slots = state.x.shape[1]
    scripted_config = ScriptedConfig()
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
            memory = renew_memory(memory, out.done, generator)
            if chaser_team is None:
                state = game.reset_done(config, state, out.done, sizes, generator)
            elif bool((result >= 0).all()):
                break
    return dict(farthest=farthest / radius, fired=fired, lived=lived, result=result)


def test_every_persona_is_competent_at_spread_one():
    """Sign-off 2026-11-23: every persona competent. At spread 1, 32 environments of 4v4 for
    1,800 ticks (finished games re-dealt; each slot keeps its persona): no ship goes past the
    end of the boundary's damage ramp, and of the ships that lived 20 s at least 49 in 50
    fired (the yardstick itself has a rare survivor that never gets a firing solution: 1 of
    236 at this seed)."""
    tallies = play_scripted(envs=32, ticks=1800, spread=1.0, seed=7)
    print(f"farthest {float(tallies['farthest'].max()):.3f} of the arena radius")
    assert float(tallies["farthest"].max()) < 1.0 + 300.0 / 4200.0
    long_lived = tallies["lived"] >= 600
    assert long_lived.sum() > 100
    assert float(tallies["fired"][long_lived].float().mean()) >= 0.98


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
