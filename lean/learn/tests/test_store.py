"""The rollout store (lean.learn.store): streamed segments equal the unstreamed rollouts,
micro-batches give the one-pass gradient, and compact storage barely moves the policy."""

import copy
import dataclasses

import pytest
import torch

from lean import game, model, scripted
from lean.learn import LearnConfig, Opponent, RolloutStore, collect, start, update
from lean.learn.store import EPISODE_CONSTANT, stored, widened
from lean.learn.tests.setting import StepRecorder, small

ENVIRONMENTS = 6  # 3 self-play: 9 seats


def setting(fields: bool = True):
    torch.manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=3, equal_teams=False, minibatches=1, epochs=1)
    game_config = game.GameConfig() if fields else game.GameConfig(field_count=0, fog=False)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    return config, game_config, policy, opponent


def rollouts(config, game_config, policy, opponent, segments: int, ticks: int, store=None):
    generator = torch.Generator().manual_seed(1)
    state, carry = start(config, game_config, policy, opponent, ENVIRONMENTS, generator)
    collected = []
    for segment in range(segments):
        rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator, store=store, segment=segment)
        state, carry = rollout.state, rollout.carry
        collected.append(rollout)
    return collected


def whole_batch(store: RolloutStore, names) -> dict:
    index, _ = store.batch_index(torch.arange(store.views), store.views)
    (batch,) = list(store.batches(names, index))
    return batch


def test_streamed_segments_equal_the_unstreamed_rollouts():
    """Two segments written tick by tick into a store hold what the stacked rollouts hold,
    in the storage types: the episode-constant fields and the position offset through the
    deal table, the carries, the bootstrap values."""
    config, game_config, policy, opponent = setting()
    store = RolloutStore(2)
    rollouts(config, game_config, policy, opponent, 2, 12, store)
    reference = rollouts(config, game_config, policy, opponent, 2, 12)
    seats = reference[0].done.shape[1]
    names = store.observation_names + tuple(name for name in reference[0].TICK_FIELDS if name != "position_offset") + ("hit_label",)
    batch = whole_batch(store, names)
    observation, offset = store.observation(batch)

    def joined(get):
        return torch.cat([get(rollout) for rollout in reference], 1)

    def compact(name, value):
        return widened(stored(name, value))

    for field in dataclasses.fields(observation):
        expected = joined(lambda rollout, name=field.name: getattr(rollout.observation, name))
        if field.name not in EPISODE_CONSTANT:  # the deal table keeps the game's types
            expected = compact(field.name, expected)
        assert torch.equal(getattr(observation, field.name), expected), field.name
    assert any(getattr(observation, name).numel() for name in EPISODE_CONSTANT)  # fields drawn
    assert torch.equal(offset, joined(lambda rollout: rollout.position_offset))
    for name in names:
        if name in store.observation_names:
            continue
        expected = joined(lambda rollout, name=name: getattr(rollout, name))
        if name == "deal_row":
            continue  # rows of each segment's own table; checked through the lookups above
        if name == "hit_label":
            # The store also labels segment 0's tail from segment 1 (checked in
            # test_hit_labels_near_a_segment_boundary_come_from_the_next_segment).
            labelled = expected >= 0
            assert torch.equal(batch[name][labelled], expected[labelled])
            assert torch.equal(batch[name][:, seats:], expected[:, seats:])
            continue
        assert torch.equal(batch[name], compact(name, expected)), name
    carry = torch.cat([rollout.initial_carry for rollout in reference], 1)
    assert torch.equal(batch["carry"], carry)
    bootstrap = torch.cat([rollout.bootstrap_value for rollout in reference])
    assert torch.equal(store.bootstrap[: 2 * seats], bootstrap)


@pytest.mark.parametrize("targets_every_epoch", [False, True])
def test_micro_batches_give_the_one_pass_gradient(targets_every_epoch):
    """One minibatch run as micro-batches of 2 views (the last padded with the empty spare
    view) and as one pass of every view: the same gradient within float32 rounding. Plain
    SGD at rate 1 with no clipping, so the step is minus the gradient."""
    config, game_config, policy, opponent = setting()
    store = RolloutStore(1)
    rollouts(config, game_config, policy, opponent, 1, 16, store)
    assert store.views % 2 == 1  # so the last micro-batch is padded
    steps = []
    for size in (2, store.views):
        variant = dataclasses.replace(
            config, micro_batch_views=size, learning_rate=1.0, warmup_steps=1, gradient_clip=1e9,
            targets_every_epoch=targets_every_epoch,
        )
        twin = copy.deepcopy(policy)
        before = [parameter.detach().clone() for parameter in twin.parameters()]
        optimiser = torch.optim.SGD(twin.parameters(), lr=1.0, fused=True)
        metrics = update(variant, twin, optimiser, store, teacher_weight=1.0, generator=torch.Generator().manual_seed(2))
        steps.append(([after.detach() - start for after, start in zip(twin.parameters(), before, strict=True)], metrics))
    (small, small_metrics), (whole, whole_metrics) = steps
    scale = max(step.abs().max() for step in whole)
    for name, (a, b) in zip(dict(policy.named_parameters()), zip(small, whole, strict=True), strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * scale), name
    for name in ("loss", "surrogate", "value", "entropy", "next_state", "hit", "bc", "bc_kl", "approximate_kl"):
        assert small_metrics[name] == pytest.approx(whole_metrics[name], rel=1e-4, abs=1e-6), name


def policy_outputs(policy, observation, carry, reset, offset):
    logits, value_logits, *_ = policy.sequence(observation, carry, reset, offset)
    return torch.log_softmax(logits.float(), -1), value_logits.float()


@pytest.mark.parametrize("fields", [True, False])
def test_compact_storage_moves_the_policy_far_less_than_the_clip(fields):
    """The policy on the compact stored observation against the full-precision one, float32:
    the largest change of a sampled action's log-probability and the mean KL over the
    trained tokens, far below PPO's clip (0.2) and a typical update's KL (about 0.01)."""
    config, game_config, policy, opponent = setting(fields)
    (rollout,) = rollouts(config, game_config, policy, opponent, 1, 24)
    store = RolloutStore.of(rollout)
    batch = whole_batch(store, store.observation_names + ("deal_row", "reset", "action"))
    observation, offset = store.observation(batch)
    with torch.no_grad():
        full, full_value = policy_outputs(policy, rollout.observation, rollout.initial_carry, rollout.reset, rollout.position_offset)
        compact, compact_value = policy_outputs(policy, observation, batch["carry"], batch["reset"], offset)
    own = rollout.observation.own
    action = rollout.action[..., None]
    change = (full.gather(-1, action) - compact.gather(-1, action)).squeeze(-1)[own].abs().max()
    kl = (full.exp() * (full - compact)).sum(-1)[own].mean()
    print(f"fields={fields}: largest log-probability change {change:.2e}, mean KL {kl:.2e}, "
          f"largest value-logit change {(full_value - compact_value)[own].abs().max():.2e}")
    assert change < 1e-2 and kl < 1e-5


def test_every_stored_float16_field_is_within_range():
    """Every float16 record of a scripted rollout (both games) is finite after narrowing."""
    for fields in (True, False):
        config, game_config, policy, opponent = setting(fields)
        (rollout,) = rollouts(config, game_config, policy, opponent, 1, 24)
        records = {field.name: getattr(rollout.observation, field.name) for field in dataclasses.fields(rollout.observation)}
        records["next_state"] = rollout.next_state
        for name, value in records.items():
            narrow = stored(name, value)
            if narrow.dtype == torch.float16 and value.numel():
                assert torch.isfinite(narrow).all(), name
                assert value.abs().max() < 1e4, name


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory and the copy stream need a GPU")
def test_pinned_streaming_equals_the_device_store():
    """On a GPU: a store in pinned host memory (asynchronous copies, prefetched micro-batches)
    gives the same batches, targets and update as the same store kept on the device."""
    config, game_config, policy, opponent = setting()
    (rollout,) = rollouts(config, game_config, policy, opponent, 1, 16)
    policy, rollout = policy.cuda(), _to(rollout, "cuda")
    stores = [RolloutStore.of(rollout, stream) for stream in (True, False)]
    assert stores[0].pinned and not stores[1].pinned
    names = stores[0].observation_names + ("deal_row", "reset", "action", "log_probability", "value", "reward", "next_state")
    results = []
    for store in stores:
        store.ready()
        index, _ = store.batch_index(torch.randperm(store.views, generator=torch.Generator().manual_seed(3)), 4)
        results.append([{name: value.clone() for name, value in batch.items()} for batch in store.batches(names, index)])
    for first, second in zip(*results, strict=True):
        for name in first:
            assert torch.equal(first[name], second[name]), name
    variant = dataclasses.replace(config, micro_batch_views=4, minibatches=2, epochs=2)
    metrics = []
    for store in stores:
        twin = copy.deepcopy(policy)
        optimiser = torch.optim.SGD(twin.parameters(), lr=1e-3, fused=True)
        metrics.append(update(variant, twin, optimiser, store, generator=torch.Generator().manual_seed(4)))
    for name in metrics[0]:
        assert metrics[0][name] == pytest.approx(metrics[1][name], rel=1e-4, abs=1e-6), name


def _to(rollout, device):
    """A rollout with every tensor on `device` (the collect above ran on the CPU)."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {name: move(item) for name, item in value.items()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.replace(value, **{field.name: move(getattr(value, field.name)) for field in dataclasses.fields(value)})
        return value

    return dataclasses.replace(rollout, **{
        name: move(getattr(rollout, name))
        for name in ("observation", *rollout.TICK_FIELDS, "hit_label", "deal_table", "bootstrap_value", "value_centres", "initial_carry")
    })


def real_store(segments: int, ticks: int, envs: int = 4, clock: int = 10):
    """A store of `segments` consecutive segments of the real game (games end at `clock`)."""
    config, game_config, policy, opponent, state, carry, generator = small(
        envs, game_config=game.GameConfig(training_spawn=True, clock_ticks=clock)
    )
    store = RolloutStore(segments, stream=False)
    for segment in range(segments):
        rollout = collect(config, game_config, policy, opponent, state, carry, ticks, generator, store=store, segment=segment)
        state, carry = rollout.state, rollout.carry
    return config, game_config, policy, store


def seat_major(records: torch.Tensor, segments: int, seats: int) -> torch.Tensor:
    """Store records (T, V + 1, ...) as each seat's segments in time order (U x T, S, ...)."""
    from lean.learn.returns import join_segments

    return join_segments(records[:, : segments * seats], segments)


@pytest.mark.parametrize("fresh_values", [False, True])
def test_chained_targets_equal_one_gae_pass_over_the_concatenated_ticks(fresh_values):
    """Four 12-tick segments: the target pass (micro-batches of whole seats) gives what one
    GAE pass over each seat's 48 consecutive ticks gives, bootstrapped from the last
    segment only, within float32 rounding; with an episode end inside a segment and one on
    a segment's last tick (which must cut the chain there). Both the stored values and the
    critic's recomputed ones (targets_every_epoch)."""
    from lean.learn import generalised_advantages, normalise_advantages
    from lean.learn.training import minibatch_values, target_pass

    segments, ticks = 4, 12
    config, _, policy, store = real_store(segments, ticks)
    seats, views = store.seats, store.views
    generator = torch.Generator().manual_seed(5)
    # Rewards on every level and stored values, so every tick's delta matters.
    own = store.resident["own"][..., None]
    store.data["reward"].copy_(torch.randn(store.data["reward"].shape, generator=generator) * own)
    store.data["value"].copy_(torch.randn(store.data["value"].shape, generator=generator) * own)
    store.data["reward"][:, views] = 0
    store.data["value"][:, views] = 0
    done = torch.zeros_like(store.resident["done"])
    done[5, 2 * seats + 1] = True  # inside segment 2 of seat 1
    done[ticks - 1, 1 * seats + 0] = True  # the last tick of segment 1 of seat 0
    store.resident["done"].copy_(done)
    store.bootstrap.copy_(torch.randn(store.bootstrap.shape, generator=generator))
    store.bootstrap[views] = 0

    if fresh_values:
        index, _ = store.batch_index(torch.arange(views), views)
        (batch,) = list(store.batches(store.observation_names + ("deal_row", "reset"), index))
        observation, offset = store.observation(batch)
        value, _ = minibatch_values(config, policy, observation, batch["carry"], batch["reset"], offset)
        value = torch.cat([value, torch.zeros_like(value[:, :1])], 1)
    else:
        value = store.data["value"]
    discounts, lambdas = torch.tensor(config.discounts), torch.tensor(config.gae_lambdas)
    advantage, target = generalised_advantages(
        seat_major(store.data["reward"], segments, seats), seat_major(value, segments, seats),
        store.bootstrap[(segments - 1) * seats: views], seat_major(store.resident["done"], segments, seats),
        discounts, lambdas,
    )

    def store_major(records):  # (U x T, S, ...) back to (T, V, ...), by hand
        rest = records.shape[2:]
        return records.reshape((segments, ticks, seats) + rest).transpose(0, 1).reshape((ticks, views) + rest)

    expected_target = store_major(target)
    expected_advantage = normalise_advantages(store_major(advantage.sum(-1)), store.resident["own"][:, :views])
    index = store.chain_index(4 * segments)  # 4 seats a row; 6 seats, so the last row is padded
    assert index.shape == (2, 4 * segments) and seats == 6
    target_pass(config, policy, store, index, minibatch_values if fresh_values else None)
    assert torch.allclose(store.target[:, :views], expected_target, rtol=1e-5, atol=1e-5)
    assert torch.allclose(store.advantage[:, :views], expected_advantage, rtol=1e-5, atol=1e-5)
    # The chain really reaches across a boundary: seat 2's segment 0 sees segment 1's rewards.
    cut, _ = generalised_advantages(
        store.data["reward"][:, 2:3], value[:, 2:3], store.bootstrap[2:3], store.resident["done"][:, 2:3],
        discounts, lambdas,
    )
    assert not torch.allclose(cut[-1], target[ticks - 1, 2:3] - value[ticks - 1, 2:3], atol=1e-3)


def test_hit_labels_near_a_segment_boundary_come_from_the_next_segment(monkeypatch):
    """Three consecutive 12-tick segments (bullets live up to 11 ticks, games end every 20):
    every row of the store is labelled with the outcome of the bullet the ship emitted there,
    found by searching the recorded game steps over all 36 ticks, so a segment's tail is
    labelled by bullets that end in the next segment; only the last segment keeps an
    unlabelled tail."""
    recorder = StepRecorder().install(monkeypatch)
    segments, ticks = 3, 12
    _, _, _, store = real_store(segments, ticks, clock=20)
    seats = store.seats
    labels = seat_major(store.resident["hit_label"], segments, seats).long()  # (U x T, S, N)
    environment = torch.cat([torch.arange(4), torch.arange(2)])  # 4 environments, 2 self-play: the seats
    expected = recorder.hit_labels().index_select(1, environment)
    assert torch.equal(labels, expected)
    assert (labels[ticks - 3: ticks] >= 0).any()  # segment 0's tail is labelled from segment 1
    assert (store.resident["hit_label"][:, store.views] == -1).all()  # the spare view labels nothing
