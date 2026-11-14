"""Field-count sampling (`bnb.contracts.ActiveFields`): an episode plays a drawn number of
its field slots and the rest are absent. An absent field has no effect on the game (the
medium, flight, bullets, sight), its observation row shows it absent, the draw is keyed by
(seed, environment, episode) in the rollout, and a Yemong with `mask_absent_fields` reads
nothing of an absent field's row (the nearest model skips it too)."""

import contextlib
import dataclasses

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, ActiveFields, GameConfig, Obs, Pool
from bnb.contracts.obs import FIELD, GLOBAL
from bnb.controllers.scripted import Chaser, Random
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.game.layout import environment
from bnb.game.physics.spawning import draw_active_fields, present_fields
from bnb.game.state import replace
from bnb.models.nearest import NearestConfig, features
from bnb.models.yemong.model import Ablations, Yemong, YemongConfig, yemong_build
from bnb.runs.serialise import from_json, to_json
from bnb.runtime import lanes
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import TrainConfig, samples_field_counts, training_pools

CPU = torch.device("cpu")
TYPES = api.ship_types()
SEATS = ("a", "b")


@pytest.fixture(autouse=True)
def _default_generator_untouched():
    with torch.random.fork_rng(devices=[]):
        yield


# ------------------------------------------------------------------ the setting and the draw


def test_the_counts_follow_the_maps_symmetry():
    assert ActiveFields(2, 7).counts(10) == (2, 4, 6)  # pairs only
    assert ActiveFields(2, 7).counts(9) == (2, 3, 4, 5, 6, 7)  # a centre field
    assert ActiveFields(0, 64).counts(4) == (0, 2, 4)
    with pytest.raises(ValueError):
        ActiveFields(3, 2)
    with pytest.raises(ValueError):
        GameConfig(2, 4, 2000.0, active_fields=ActiveFields(1, 6))  # more than the slots
    with pytest.raises(ValueError):
        GameConfig(2, 4, 2000.0, active_fields=ActiveFields(3, 3))  # no even count


def test_the_present_fields_are_the_first_pairs_and_the_centre():
    count = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)
    present = present_fields(5, count)  # pairs (0, 2), (1, 3); the centre is 4
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(present, expected)
    assert torch.equal(present.sum(-1), count.long())


def test_the_draw_is_uniform_over_the_allowed_counts():
    game = GameConfig(2, 10, 2000.0, active_fields=ActiveFields(2, 6))
    batch = 30_000
    rules = api.rules_of(game)
    arena = environment(rules, batch)
    uniform = torch.rand(batch, generator=torch.Generator().manual_seed(0))
    count = draw_active_fields(10, arena.field_count_first, arena.field_count_options, uniform)
    count = count.sum(-1)
    shares = torch.stack([(count == c).float().mean() for c in (2, 4, 6)])
    assert int(count.min()) == 2 and int(count.max()) == 6 and bool((count % 2 == 0).all())
    torch.testing.assert_close(shares, torch.full((3,), 1.0 / 3.0), atol=0.015, rtol=0)


def test_without_sampling_the_rules_and_the_draws_are_unchanged():
    plain = api.rules_of(GameConfig(2, 4, 2000.0))
    sampled = api.rules_of(GameConfig(2, 4, 2000.0, active_fields=ActiveFields(0, 4)))
    assert not plain.samples_fields and sampled.samples_fields
    assert sampled.tick_uniform_count == plain.tick_uniform_count + 1  # one draw, the last
    uniforms = api.uniforms(plain, 8, generator=torch.Generator().manual_seed(1))
    state = api.reset(plain, TYPES, uniforms)
    assert bool(state.fields.present.all())
    _, obs, _ = api.perceive(plain, state)
    assert bool((obs.fields[..., FIELD["present"]] == 1.0).all())


# ------------------------------------------------------------------ no effect on the game


def _sandbox(fields: int) -> api.Rules:
    return api.rules_of(GameConfig(2, fields, 2000.0, episode_seconds=20.0, mode="sandbox"))


def _flown(rules, state, ticks: int, seed: int):
    """Every tick's ship state, bullets and perception over `ticks` ticks of random flying
    and shooting (the same actions and spread for any field count)."""
    generator = torch.Generator().manual_seed(seed)
    batch = state.ships.x.shape[0]
    record = []
    for _ in range(ticks):
        packed = torch.stack(
            [
                torch.randint(0, 3, (batch, rules.num_ships), generator=generator),
                torch.randint(0, 7, (batch, rules.num_ships), generator=generator),
                torch.randint(0, 2, (batch, rules.num_ships), generator=generator),
            ],
            -1,
        ).to(torch.int32)
        spread = torch.randn((batch, rules.num_ships, 2), generator=generator)
        state, _ = api.step(
            rules,
            TYPES,
            state,
            Action.unpack(packed),
            spread,
            torch.zeros(batch, rules.tick_uniform_count),
        )
        state, obs, _ = api.perceive(rules, state)
        record.append((state.ships, state.bullets, obs.ships))
    return record


def test_an_absent_field_has_no_effect_on_the_game():
    """Four field slots, the second pair absent and laid right over the ships (big, strong
    and in their way), against the same game with only the first pair: the flight, the
    bullets and every observed ship row agree within rounding. Present, the same pair
    changes the game (the test can tell)."""
    four, two = _sandbox(4), _sandbox(2)
    batch = 6
    uniforms = api.uniforms(four, batch, generator=torch.Generator().manual_seed(5))
    state = api.reset(four, TYPES, uniforms, torch.ones(batch, dtype=torch.bool))
    ships = state.ships
    fields = state.fields
    centre_x, centre_y = ships.x[:, [0, 2]], ships.y[:, [0, 2]]  # on a ship of each team
    fields = replace(
        fields,
        x=torch.cat([fields.x[:, :1], centre_x[:, :1], fields.x[:, 1:2], centre_x[:, 1:]], 1),
        y=torch.cat([fields.y[:, :1], centre_y[:, :1], fields.y[:, 1:2], centre_y[:, 1:]], 1),
        radius=torch.cat([fields.radius[:, :1], torch.full((batch, 1), 900.0)] * 2, 1),
        level=torch.cat(
            [fields.level[:, :1], torch.full((batch, 1), -2, dtype=torch.int32)] * 2, 1
        ),
    )
    absent = replace(fields, present=torch.tensor([True, False, True, False]).expand(batch, 4))
    # the ships' cached medium as a game without the pair would have it
    pair = [0, 2]
    without = replace(
        state,
        fields=api.observed_fields(
            torch.stack(
                [
                    fields.x[:, pair],
                    fields.y[:, pair],
                    fields.radius[:, pair],
                    fields.band_width[:, pair],
                    fields.level[:, pair].float(),
                    torch.ones(batch, 2),
                ],
                -1,
            )
        ),
    )
    log_index, gradient_x, gradient_y = api.log_index_gradient(
        without.fields.x,
        without.fields.y,
        without.fields.radius,
        without.fields.band_width,
        without.fields.level,
        ships.x,
        ships.y,
    )
    ships = replace(
        ships,
        log_refractive_index=log_index,
        log_index_gradient_x=gradient_x,
        log_index_gradient_y=gradient_y,
    )
    without = replace(without, ships=ships)
    with_absent = replace(state, ships=ships, fields=absent)
    with_present = replace(state, ships=ships, fields=fields)

    reference = _flown(two, without, 45, seed=11)
    for other in (_flown(four, with_absent, 45, seed=11),):
        for (ship, bullet, row), (ship_2, bullet_2, row_2) in zip(reference, other, strict=True):
            for leaf, leaf_2 in zip(
                _pytree.tree_leaves((ship, bullet, row)),
                _pytree.tree_leaves((ship_2, bullet_2, row_2)),
                strict=True,
            ):
                torch.testing.assert_close(leaf, leaf_2, rtol=1e-5, atol=1e-3)
    present = _flown(four, with_present, 45, seed=11)
    assert not torch.allclose(reference[-1][0].x, present[-1][0].x, atol=1.0)


def test_an_absent_field_blocks_no_sight_line():
    """A field core between two ships hides them from each other under fog; absent, it
    hides nothing."""
    rules = _sandbox(1)
    state = api.reset(rules, TYPES, torch.zeros(1, rules.tick_uniform_count))
    ships = replace(
        state.ships,
        x=torch.tensor([[-500.0, -500.0, 500.0, 500.0]]),
        y=torch.tensor([[0.0, 10.0, 0.0, 10.0]]),
        spawned=torch.zeros(1, 4, dtype=torch.bool),  # a spawning ship is always seen
    )
    wall = replace(
        state.fields,
        x=torch.zeros(1, 1),
        y=torch.zeros(1, 1),
        radius=torch.full((1, 1), 200.0),
        level=torch.full((1, 1), 2, dtype=torch.int32),
    )
    for present, seen in ((True, False), (False, True)):
        fields = replace(wall, present=torch.full((1, 1), present))
        _, obs, _ = api.perceive(rules, replace(state, ships=ships, fields=fields))
        visible = api.visible(rules, obs)  # (1, 2, N)
        assert bool(visible[0, 0, 2:].all()) is seen and bool(visible[0, 1, :2].all()) is seen


def test_an_absent_fields_row_is_zero_but_its_flag():
    game = GameConfig(2, 6, 2000.0, active_fields=ActiveFields(2, 2))
    rules = api.rules_of(game)
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 3, generator=torch.Generator().manual_seed(2))
    )
    _, obs, _ = api.perceive(rules, state)
    rows = obs.fields  # (B, 2, M, F)
    present = torch.tensor([True, False, False, True, False, False])
    assert torch.equal(rows[..., FIELD["present"]] > 0.5, present.expand(3, 2, 6))
    assert bool((rows[..., ~present, :] == 0.0).all())
    assert bool((rows[..., present, FIELD["radius"]] > 0.0).all())


def test_a_reset_draws_a_new_count_and_the_shapes_stay():
    """Short episodes: each episode end draws its environment's field count anew."""
    game = GameConfig(2, 8, 2000.0, episode_seconds=0.2, active_fields=ActiveFields(0, 8))
    batch = 32
    rules = api.rules_of(game, (Pool("p", 1.0, SEATS),), batch)
    generator = torch.Generator().manual_seed(3)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    shapes = [leaf.shape for leaf in _pytree.tree_leaves(state)]
    changed = 0
    for _ in range(14):
        before = state.fields.present
        state, events = api.step(
            rules,
            TYPES,
            state,
            Action.unpack(torch.zeros(batch, rules.num_ships, 3, dtype=torch.int32)),
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        after = state.fields.present
        assert torch.equal(after[~events.done], before[~events.done])
        changed += int((after[events.done] != before[events.done]).any(-1).sum())
        assert [leaf.shape for leaf in _pytree.tree_leaves(state)] == shapes
        assert bool((after.sum(-1) % 2 == 0).all())
    assert changed > batch // 2


def test_frontline_keeps_the_absent_fields_absent():
    game = dataclasses.replace(api.PRESETS["frontline"], active_fields=ActiveFields(4, 4))
    rules = api.rules_of(game)
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 4, generator=torch.Generator().manual_seed(6))
    )
    assert torch.equal(state.fields.present.sum(-1), torch.full((4,), 4))


# ------------------------------------------------------------------ keyed by the episode

S = 2
SHORT = GameConfig(S, 6, 1500.0, episode_seconds=0.5)
SHORTER = GameConfig(S, 6, 1500.0, episode_seconds=0.4)
ENVIRONMENTS, LENGTH, SEGMENTS = 8, 30, 3
RANGE = ActiveFields(0, 6)


def _rollout(learner: str = "chaser", game: GameConfig = SHORT, active=RANGE, seed: int = 0):
    chaser = Chaser(characteristics(TYPES, api.rules_of(game)), S)
    players = {"chaser": chaser, "random": Random(S)}
    pools = (Pool("p", 1.0, ("learner", "other"), 0.5, active_fields=active),)
    return Rollout(
        game,
        api.ship_types(CPU),
        pools,
        {"learner": players[learner], "other": Random(S)},
        "learner",
        ENVIRONMENTS,
        LENGTH,
        CPU,
        compiled=False,
        seed=seed,
    )


def _starts(rollout: Rollout) -> dict:
    """Each (environment, episode)'s present fields and field centres, from its first tick."""
    act, advance, resolve, perceive = rollout.stages
    starts = {}

    def recorded(carry):
        perceive(carry)
        fields = carry["state"].fields
        for index in range(ENVIRONMENTS):
            key = (index, int(carry["episode"][index]))
            starts.setdefault(key, (fields.present[index].clone(), fields.x[index].clone()))

    rollout.stages = (act, advance, resolve, recorded)
    for _ in range(SEGMENTS):
        rollout.run()
    return starts


def test_the_count_is_keyed_by_seed_environment_and_episode():
    """Another player and shorter episodes: the k-th episode of each environment has the
    same active fields; the counts vary over episodes; the same rollout reproduces."""
    chaser, random = _starts(_rollout("chaser")), _starts(_rollout("random", SHORTER))
    common = set(chaser) & set(random)
    assert len(common) >= 2 * ENVIRONMENTS
    for key in common:
        assert torch.equal(chaser[key][0], random[key][0]), key
    assert len({int(present.sum()) for present, _ in chaser.values()}) > 1
    again = _starts(_rollout("chaser"))
    assert all(torch.equal(again[key][0], chaser[key][0]) for key in chaser)
    other_seed = _starts(_rollout("chaser", seed=1))
    assert any(not torch.equal(other_seed[key][0], chaser[key][0]) for key in chaser)


def test_turning_field_count_sampling_on_moves_no_other_draw():
    sampled, plain = _rollout(active=RANGE), _rollout(active=None)
    episode = torch.arange(ENVIRONMENTS, dtype=torch.int64) % 3
    width = plain.rules.tick_uniform_count
    assert torch.equal(
        sampled._episode_uniforms(episode)[:, :width], plain._episode_uniforms(episode)
    )
    assert torch.equal(sampled._seats(episode), plain._seats(episode))
    first, second = sampled.carry["state"].fields, plain.carry["state"].fields
    assert torch.equal(first.x, second.x) and torch.equal(first.radius, second.radius)
    assert bool(second.present.all()) and not bool(first.present.all())


# ------------------------------------------------------------------ the plumbing


def test_training_and_evaluation_settings():
    from bnb import cli

    config = TrainConfig(api.PRESETS["deathmatch"])
    assert config.active_fields is None
    assert all(pool.active_fields is None for pool in training_pools(config))
    assert not samples_field_counts(config.game, training_pools(config))
    varied = TrainConfig(api.PRESETS["deathmatch"], active_fields=ActiveFields(2, 10))
    pools = training_pools(varied)
    assert all(pool.active_fields == ActiveFields(2, 10) for pool in pools)
    assert samples_field_counts(varied.game, pools)
    # the rating lanes play their games' own fields
    modes = lanes.mode_games(varied.pools, varied.game, lambda pool, game: pool.mode or game.mode)
    every, _, _ = lanes.lane_pools(pools, modes, 64, lanes.LaneConfig(environments=64))
    assert [pool.active_fields for pool in every[len(pools) :]] == [None] * (
        len(every) - len(pools)
    )
    # the setting round-trips through the run's config.json
    assert from_json(TrainConfig, to_json(varied))[0].active_fields == ActiveFields(2, 10)
    with pytest.raises(ValueError):
        TrainConfig(api.PRESETS["deathmatch"], active_fields=ActiveFields(0, 12))
    # evaluation: the ratings' range on the rated game
    game = cli.rated_game("deathmatch", None, (4, 8))
    assert game.active_fields == ActiveFields(4, 8) and samples_field_counts(game)
    assert cli.rated_game("deathmatch", None).active_fields is None
    with pytest.raises(SystemExit):
        cli.rated_game("deathmatch", None, (4, 12))


# ------------------------------------------------------------------ the models

GAME = GameConfig(
    2, 4, 2000.0, episode_seconds=0.5, mode="sandbox", active_fields=ActiveFields(2, 2)
)
N = GAME.num_ships


@contextlib.contextmanager
def _default_generator(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.set_state(torch.Generator().manual_seed(seed).get_state())
        yield


def _model(seed: int = 0) -> Yemong:
    config = YemongConfig(
        width=8,
        blocks=1,
        spatial_per_block=1,
        heads=1,
        map_width=4,
        value_bins=11,
        hex_rings=1,
        rotary_octaves=1,
        heading_harmonics=(1,),
    )
    with _default_generator(seed):
        net = Yemong(yemong_build(config, api.rules_of(GAME), TYPES))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    return net.eval()


def _views(ticks: int = 20) -> list[Obs]:
    """Team 0's views of a chaser game with the second pair of fields absent."""
    rules = api.rules_of(GAME)
    generator = torch.Generator().manual_seed(4)
    state = api.reset(rules, TYPES, api.uniforms(rules, 2, generator=generator))
    state, observation, _ = api.perceive(rules, state)
    chaser = Chaser(characteristics(TYPES, rules), S)
    views = []
    for _ in range(ticks):
        team0, team1 = (
            Obs(
                **{
                    part: getattr(observation, part)[:, team]
                    for part in ("ships", "fields", "globals", "zones")
                }
            )
            for team in (0, 1)
        )
        views.append(team0)
        actions = [chaser.step(None, view, None, torch.arange(2))[0] for view in (team0, team1)]
        state, _ = api.step(
            rules,
            TYPES,
            state,
            Action.unpack(torch.cat([action.packed() for action in actions], 1)),
            api.spread(rules, 2, generator=generator),
            api.uniforms(rules, 2, generator=generator),
        )
        state, observation, _ = api.perceive(rules, state)
    return views


def _dirty(view: Obs, generator: torch.Generator) -> Obs:
    """The view with every absent field's row (but its flag) overwritten by noise."""
    fields = view.fields.clone()
    absent = fields[..., FIELD["present"]] < 0.5
    noise = 500.0 * torch.randn(fields.shape, generator=generator)
    noise[..., FIELD["present"]] = 0.0
    fields = torch.where(absent[..., None], noise, fields)
    return dataclasses.replace(view, fields=fields)


def _run(net, views, ablations=None):
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(2, S, N)
    decisions = []
    with torch.no_grad():
        for t, view in enumerate(views):
            reset = torch.full((2,), t == 0)
            decision = net.step(carry, view, reset, torch.arange(2), ablations)
            carry = dataclasses.replace(
                decision.carry, previous_action=decision.policy_logits.argmax(-1).to(torch.int32)
            )
            decisions.append(decision)
    return decisions


def _outputs(decision):
    return (
        decision.policy_logits,
        decision.values,
        decision.enemy_action_logits,
        decision.next_state_logits,
        decision.outcome_probabilities,
        decision.carry.belief.means,
        decision.carry.state,
    )


def test_with_the_mask_on_no_output_reads_an_absent_fields_row():
    """Noise in the absent fields' rows (step by step, the trunk's memory carried) leaves
    every output as it was, alone and with the absent-zone ablation; with the mask off it
    does not."""
    views = _views()
    assert bool((views[0].fields[..., FIELD["present"]] < 0.5).any())
    assert bool((views[0].globals[..., GLOBAL["reset"]] > 0.5).all())
    generator = torch.Generator().manual_seed(9)
    dirty = [_dirty(view, generator) for view in views]
    net = _model()
    net.mask_absent_fields = True
    zones = Ablations(
        rotary=torch.zeros(2, dtype=torch.bool),
        forecast=torch.zeros(2, dtype=torch.bool),
        absent_zones=torch.ones(2, dtype=torch.bool),
    )
    for ablations in (None, zones):
        clean_run, dirty_run = _run(net, views, ablations), _run(net, dirty, ablations)
        for one, two in zip(clean_run, dirty_run, strict=True):
            for a, b in zip(_outputs(one), _outputs(two), strict=True):
                torch.testing.assert_close(a, b)
    net.mask_absent_fields = False
    clean_run, dirty_run = _run(net, views), _run(net, dirty)
    assert not torch.allclose(clean_run[-1].policy_logits, dirty_run[-1].policy_logits)


def test_unroll_reads_no_absent_fields_row_with_the_mask_on():
    views = _views()
    dirty = [_dirty(view, torch.Generator().manual_seed(9)) for view in views]
    net = _model()
    net.mask_absent_fields = True
    decisions = _run(net, views)
    replay = _pytree.tree_map(lambda *x: torch.stack(x), *[d.replay for d in decisions])
    resets = torch.stack([torch.full((2,), t == 0) for t in range(len(views))])
    with torch.no_grad(), _default_generator(7):
        carry = net.initial_carry(2, S, N)

    def unrolled(segment):
        stacked = _pytree.tree_map(lambda *x: torch.stack(x), *segment)
        with torch.no_grad():
            return net.unroll(carry, stacked, replay, resets)

    clean, noisy = unrolled(views), unrolled(dirty)
    for name in ("policy_logits", "values", "enemy_action_logits", "next_state_logits"):
        torch.testing.assert_close(getattr(clean, name), getattr(noisy, name))
    torch.testing.assert_close(
        clean.policy_logits, torch.stack([d.policy_logits for d in decisions])
    )


def test_the_mask_on_a_full_map_computes_what_it_does_off():
    game = dataclasses.replace(GAME, active_fields=None)
    rules = api.rules_of(game)
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 2, generator=torch.Generator().manual_seed(1))
    )
    _, observation, _ = api.perceive(rules, state)
    view = Obs(
        **{
            part: getattr(observation, part)[:, 0]
            for part in ("ships", "fields", "globals", "zones")
        }
    )
    net = _model()
    off = _run(net, [view])[0]
    net.mask_absent_fields = True
    on = _run(net, [view])[0]
    for a, b in zip(_outputs(off), _outputs(on), strict=True):
        torch.testing.assert_close(a, b)


def test_the_nearest_model_skips_an_absent_field():
    views = _views(2)
    dirty = _dirty(views[0], torch.Generator().manual_seed(3))
    config = NearestConfig(fields=4)
    torch.testing.assert_close(features(views[0], config), features(dirty, config))
