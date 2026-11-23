"""Invariants of the policy model. Each test names the assumption it guards."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lean.model import ModelConfig, Policy, decode_value, hl_gauss_targets, value_bin_centres
from lean.model.position import ColourCode, FourierCode, GridCellCode, make_position_code
from lean.model.temporal import doubling_scan

SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES = 27, 3, 3
ARENA_RADIUS = 4200.0  # lean.game.GameConfig's


def make_policy(**overrides) -> Policy:
    torch.manual_seed(0)
    policy = Policy(ModelConfig(**overrides), SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES)
    # The heads' last layers start near zero; enlarge them so the tests see the trunk's output.
    with torch.no_grad():
        for last in (*policy.heads.lasts, *policy.auxiliary_heads.lasts):
            last.weight.normal_(std=0.1)
    return policy.eval()


def make_obs(
    lead: tuple[int, ...], slots: int, fields: int = 4, seed: int = 1, live_fraction: float = 0.8
):
    generator = torch.Generator().manual_seed(seed)
    key_mask = torch.rand(*lead, slots, generator=generator) < live_fraction
    return SimpleNamespace(
        ship=torch.randn(*lead, slots, SHIP_FEATURES, generator=generator),
        pos=(torch.rand(*lead, slots, 2, generator=generator) - 0.5) * 6000.0,
        heading=(torch.rand(*lead, slots, generator=generator) - 0.5) * 6.28,
        key_mask=key_mask,
        own=key_mask.clone(),
        globals=torch.randn(*lead, GLOBAL_FEATURES, generator=generator),
        field=torch.randn(*lead, fields, FIELD_FEATURES, generator=generator),
        field_pos=(torch.rand(*lead, fields, 2, generator=generator) - 0.5) * 6000.0,
        field_mask=torch.rand(*lead, fields, generator=generator) < 0.75,
        visible=torch.rand(*lead, slots, generator=generator) < 0.7,
        since_seen=torch.rand(*lead, slots, generator=generator) * 2.0,
    )


SHIP_NAMES = ("ship", "pos", "heading", "key_mask", "own", "visible", "since_seen")
FIELD_NAMES = ("field", "field_pos", "field_mask")


def index_obs(obs, index):
    return SimpleNamespace(**{name: value[index] for name, value in vars(obs).items()})


def select(obs, names, order):
    """Reorder (or subset) the ship or field slots of a single-tick observation."""
    return SimpleNamespace(
        **{name: value[:, order] if name in names else value for name, value in vars(obs).items()}
    )


def step_all(policy, obs, carry, reset, offset=None):
    outputs = []
    for tick in range(reset.shape[0]):
        logits, values, carry = policy.step(index_obs(obs, tick), carry, offset, reset[tick])
        outputs.append((logits, values))
    return outputs, carry


@pytest.mark.parametrize("rotate_values", [False, True])
def test_sequence_equals_repeated_steps(rotate_values):
    """Catches a learner forward that differs from the rollout forward: the carry with its
    convolution buffer, resets inside the segment (also at tick 0), masks, fields."""
    ticks, envs, slots = 9, 3, 5
    policy = make_policy(rotate_values=rotate_values)
    obs = make_obs((ticks, envs), slots)
    reset = torch.zeros(ticks, envs, dtype=torch.bool)
    reset[0, 2] = reset[1, 1] = reset[3, 0] = reset[4, 2] = reset[5, 1] = True
    offset = torch.randn(envs, 2) * 1000.0
    carry = torch.randn(policy.initial_carry(envs, slots, "cpu").shape)
    with torch.no_grad():
        logits, values, next_state, hit, final = policy.sequence(obs, carry, reset, offset)
        outputs, stepped = step_all(policy, obs, carry, reset, offset)
    for tick, (step_logits, step_values) in enumerate(outputs):
        live = obs.key_mask[tick]
        # float32 rounding: the scan sums in another order (measured 1.4e-5 on logits of 4.5)
        assert torch.allclose(step_logits[live], logits[tick][live], atol=1e-4)
        assert torch.allclose(step_values[live], values[tick][live], atol=1e-4)
    assert torch.allclose(stepped, final, atol=1e-5)
    assert next_state.shape == (ticks, envs, slots, policy.config.next_state_size)
    assert hit.shape == (ticks, envs, slots, policy.config.hit_classes)


def test_a_reset_forgets_state_and_convolution_buffer():
    """Catches a reset that zeroes the RG-LRU state but lets the convolution reach back over
    the episode start (or the reverse): after a reset at tick k the outputs equal a fresh
    run from a zero carry starting at tick k."""
    ticks, envs, slots, start = 7, 2, 4, 3
    policy = make_policy()
    obs = make_obs((ticks, envs), slots, live_fraction=2.0)
    reset = torch.zeros(ticks, envs, dtype=torch.bool)
    reset[start] = True
    carry = torch.randn(policy.initial_carry(envs, slots, "cpu").shape)
    with torch.no_grad():
        logits, _, _, _, final = policy.sequence(obs, carry, reset)
        later = index_obs(obs, slice(start, None))
        fresh_logits, _, _, _, fresh_final = policy.sequence(
            later,
            policy.initial_carry(envs, slots, "cpu"),
            torch.zeros(ticks - start, envs, dtype=torch.bool),
        )
    assert torch.allclose(logits[start:], fresh_logits, atol=1e-5)
    assert torch.allclose(final, fresh_final, atol=1e-5)
    assert not torch.allclose(logits[start - 1], logits[start], atol=1e-3)


def test_masked_rows_keep_a_zero_carry():
    """Catches state leaking through a dead or absent slot, in the RG-LRU state or the
    convolution buffer."""
    policy = make_policy()
    obs = make_obs((4,), 6)
    with torch.no_grad():
        _, _, carry = policy.step(obs, torch.randn(policy.initial_carry(4, 6, "cpu").shape))
    ships = carry[:, :, 1:]  # (blocks, E, N, 4, width)
    assert (ships[:, ~obs.key_mask] == 0).all()
    for row in range(4):  # the state and each buffer row
        assert (ships[:, obs.key_mask][:, :, row] != 0).any()


@pytest.mark.parametrize("positionless_rotary", ["centre", "zero"])
def test_padding_and_permutation_of_ships_leave_real_ships_unchanged(positionless_rotary):
    """Catches any dependence on slot index, slot count or the content of masked rows."""
    envs, real, padded = 3, 7, 12
    policy = make_policy(positionless_rotary=positionless_rotary)
    obs = make_obs((envs,), padded, live_fraction=2.0)
    obs.key_mask[:, real:] = False
    small = select(obs, SHIP_NAMES, torch.arange(real))
    order = torch.randperm(padded, generator=torch.Generator().manual_seed(3))
    shuffled = select(obs, SHIP_NAMES, order)
    with torch.no_grad():
        base_logits, base_values, base_carry = policy.step(
            small, policy.initial_carry(envs, real, "cpu")
        )
        pad_logits, pad_values, pad_carry = policy.step(
            obs, policy.initial_carry(envs, padded, "cpu")
        )
        mix_logits, mix_values, mix_carry = policy.step(
            shuffled, policy.initial_carry(envs, padded, "cpu")
        )
    # float32 rounding of softmaxes over different key counts: 0.4e-5 to 2.3e-5 on outputs
    # of magnitude 4 over eight weight seeds (2026-11-23), so 5e-5.
    assert torch.allclose(pad_logits[:, :real], base_logits, atol=5e-5)
    assert torch.allclose(pad_values[:, :real], base_values, atol=5e-5)
    assert torch.allclose(pad_carry[:, :, : 1 + real], base_carry, atol=1e-5)
    live = shuffled.key_mask
    assert torch.allclose(mix_logits[live], pad_logits[:, order][live], atol=1e-5)
    assert torch.allclose(mix_values[live], pad_values[:, order][live], atol=1e-5)
    assert torch.allclose(
        mix_carry[:, :, 1:][:, live], pad_carry[:, :, 1:][:, :, order][:, live], atol=1e-5
    )
    assert torch.allclose(mix_carry[:, :, 0], pad_carry[:, :, 0], atol=1e-5)


def test_fields_are_an_unordered_memory_with_no_outputs():
    """Catches field slots read by index, absent fields that still count, and fields that
    get outputs or carry: permuting fields or padding them with absent ones changes no ship
    output; F = 0 runs; the carry has no field rows; present fields are read."""
    envs, slots, real, padded = 3, 6, 5, 9
    policy = make_policy()
    obs = make_obs((envs,), slots, fields=padded)
    obs.field_mask[:, :real] = True
    obs.field_mask[:, real:] = False
    small = select(obs, FIELD_NAMES, torch.arange(real))
    order = torch.randperm(padded, generator=torch.Generator().manual_seed(5))
    shuffled = select(obs, FIELD_NAMES, order)
    none = select(obs, FIELD_NAMES, torch.arange(0))
    carry = torch.randn(policy.initial_carry(envs, slots, "cpu").shape)
    with torch.no_grad():
        base = policy.step(small, carry)
        pad = policy.step(obs, carry)
        mix = policy.step(shuffled, carry)
        empty = policy.step(none, carry)
    live = obs.key_mask
    for first, second in ((base, pad), (base, mix)):
        assert torch.allclose(first[0][live], second[0][live], atol=1e-5)
        assert torch.allclose(first[1][live], second[1][live], atol=1e-5)
        assert torch.allclose(first[2], second[2], atol=1e-5)
    assert base[2].shape == empty[2].shape == carry.shape
    assert torch.isfinite(empty[0]).all() and torch.isfinite(empty[1]).all()
    assert not torch.allclose(base[0][live], empty[0][live], atol=1e-4)


class ConstantCode(nn.Module):
    """A position code that reads nothing: holds every token's inputs fixed."""

    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, positions):
        return positions.new_zeros(positions.shape[:-1] + (self.features,))


def test_attention_reads_only_relative_positions():
    """Catches absolute position entering attention through the rotary terms: with the token
    position code held fixed and the game token positionless ("zero"), translating every
    ship and field together changes no ship output. (With "centre", v2's choice, the game
    token is the arena centre, an absolute place, on purpose.)"""
    envs, slots = 2, 6
    policy = make_policy(positionless_rotary="zero")
    policy.position_code = ConstantCode(policy.position_code.features)
    obs = make_obs((envs,), slots)
    shift = torch.tensor([1234.5, -876.5])
    moved = SimpleNamespace(
        **{**vars(obs), "pos": obs.pos + shift, "field_pos": obs.field_pos + shift}
    )
    carry = policy.initial_carry(envs, slots, "cpu")
    with torch.no_grad():
        base_logits, _, _ = policy.step(obs, carry)
        moved_logits, _, _ = policy.step(moved, carry)
    live = obs.key_mask
    assert torch.allclose(base_logits[live], moved_logits[live], atol=2e-3)


def test_same_weights_run_at_any_ship_and_field_count():
    """Catches a parameter or buffer sized by the number of slots or fields."""
    policy = make_policy()
    config = policy.config
    for slots, fields in ((10, 0), (40, 3), (100, 20)):
        obs = make_obs((2,), slots, fields=fields)
        with torch.no_grad():
            logits, values, carry = policy.step(obs, policy.initial_carry(2, slots, "cpu"))
        assert logits.shape == (2, slots, 42)
        assert values.shape == (2, slots, config.levels, 64)
        assert carry.shape == (config.blocks, 2, 1 + slots, 4, config.width)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_doubling_scan_equals_the_loop():
    """Catches an off-by-one in the doubling rounds at any segment length (powers of two
    and not, and T = 1, the step)."""
    generator = torch.Generator().manual_seed(0)
    for ticks in (1, 2, 3, 5, 8, 13, 64):
        decay = torch.rand(ticks, 3, 7, generator=generator, dtype=torch.float64)
        update = torch.randn(ticks, 3, 7, generator=generator, dtype=torch.float64)
        initial = torch.randn(3, 7, generator=generator, dtype=torch.float64)
        state, expected = initial, []
        for tick in range(ticks):
            state = decay[tick] * state + update[tick]
            expected.append(state)
        states, last = doubling_scan(decay, update, initial)
        assert torch.allclose(states, torch.stack(expected), atol=1e-12)
        assert torch.equal(last, states[-1])


def test_temporal_initialisation_is_griffins():
    """Catches v2's initialisation coming back: a^c within the paper's [0.9, 0.999] (v2's
    reached 0.004); the convolution at std 0.05 with a zero bias (v2: uniform +-1/2); the
    gates' biases zero (v2: random)."""
    policy = make_policy()
    for block in policy.blocks:
        temporal = block.temporal
        retention = temporal.retention_at_open_gate()
        assert retention.min() >= 0.9 - 1e-4 and retention.max() <= 0.999 + 1e-4
        assert retention.max() - retention.min() > 0.05
        assert 0.04 < temporal.convolution_weight.std() < 0.06
        assert (temporal.convolution_bias == 0).all() and (temporal.gates.bias == 0).all()


def test_attention_gate_starts_near_one_half():
    """Catches a gate initialised open (v2's sigmoid(2) = 0.88) or closed: released code
    starts it like the query projection, centred on 0.5."""
    policy = make_policy()
    spatial = policy.blocks[0].spatial[0]
    tokens = torch.randn(64, 11, policy.config.width)
    width = policy.config.width
    gate = torch.sigmoid(
        spatial.query_key_value_gate(spatial.attention_norm(tokens))[..., 3 * width :]
    )
    assert abs(gate.mean().item() - 0.5) < 0.03
    assert 0.05 < gate.std().item() < 0.25


def test_rotary_bands_stop_a_little_above_the_arena():
    """Catches rotary bands far larger than the map (decided 2026-11-22: drop them) and a
    ladder too short to tell apart positions across the arena."""
    config = ModelConfig()
    periods = config.rotary_periods
    assert abs(periods[0] - 64.0) < 1e-6
    assert 2.0 * ARENA_RADIUS < periods[-1] < 1.5 * 2.0 * ARENA_RADIUS
    assert 2 * config.rotary_pairs <= config.width // config.heads


@pytest.mark.parametrize("name", ["fourier", "grid_cells", "v2_codes"])
def test_each_position_code_runs_in_the_policy(name):
    """Catches a position code whose width the encoder does not match, or that breaks the
    forward (bf16 included)."""
    policy = make_policy(position_code=name)
    obs = make_obs((3,), 5)
    features = policy.position_code(obs.pos)
    assert features.shape == (3, 5, policy.position_code.features)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits, _, _ = policy.step(obs, policy.initial_carry(3, 5, "cpu"), torch.zeros(3, 2))
    assert torch.isfinite(logits.float()).all()
    assert make_position_code(policy.config).span == policy.config.offset_span


def test_position_codes_have_their_defining_symmetries():
    """Catches a mis-built code: a grid cell repeats over its hexagonal lattice (a shift of
    one spacing at 30 degrees) and not over half of it; v2's colour code is a distribution
    over 9 colour pairs per level; a Fourier code of one period P along 0, 60 and 120
    degrees repeats after 2 P along x (cos 60 = 1/2) and not after P."""
    points = (torch.rand(50, 2, generator=torch.Generator().manual_seed(0)) - 0.5) * 3000.0
    spacing = 300.0
    grid = GridCellCode((spacing,), phases=3)
    lattice = spacing * torch.tensor([math.cos(math.pi / 6), math.sin(math.pi / 6)])
    assert torch.allclose(grid(points), grid(points + lattice), atol=1e-3)
    assert not torch.allclose(grid(points), grid(points + 0.5 * lattice), atol=1e-2)
    assert grid(points).min() >= -1e-6
    colour = ColourCode(36.0 * 3.0**8, 9)(points).view(50, 9, 9)
    assert torch.allclose(colour.sum(-1), torch.ones(50, 9), atol=1e-5) and (colour >= 0).all()
    fourier = FourierCode((500.0,))
    assert torch.allclose(fourier(points), fourier(points + torch.tensor([1000.0, 0.0])), atol=1e-3)
    assert not torch.allclose(
        fourier(points), fourier(points + torch.tensor([500.0, 0.0])), atol=1e-2
    )


def test_grid_cells_tell_a_position_from_its_reflection():
    """With 2 phases per axis every grid feature is even (f(x) = f(-x)), so a ship at p and one
    at -p got the same code; the default must separate them."""
    from lean.model.policy import ModelConfig

    points = (torch.rand(200, 2, generator=torch.Generator().manual_seed(1)) - 0.5) * 6000.0
    config = ModelConfig()
    grid = GridCellCode(config.position_periods, config.grid_phases)
    difference = (grid(points) - grid(-points)).abs().amax(-1)
    assert difference.min() > 0.1


def test_hl_gauss_targets_sum_to_one_and_decode_to_the_value():
    """Catches a mis-normalised or shifted value target."""
    centres = value_bin_centres(torch.tensor([-2.0, -10.0]), torch.tensor([2.0, 30.0]), 64)
    width = centres[:, 1] - centres[:, 0]
    inside = torch.rand(500, 2, generator=torch.Generator().manual_seed(0))
    values = centres[:, 0] + 3.0 * width + inside * (centres[:, -1] - centres[:, 0] - 6.0 * width)
    targets = hl_gauss_targets(values, centres)
    assert targets.shape == (500, 2, 64)
    assert torch.allclose(targets.sum(-1), torch.ones(500, 2), atol=1e-5)
    assert (targets >= 0).all()
    decoded = decode_value(targets.clamp_min(1e-30).log(), centres)
    assert ((decoded - values).abs() < 1e-2 * width).all()
    outside = hl_gauss_targets(torch.tensor([100.0, -100.0]), centres)
    assert torch.allclose(outside.sum(-1), torch.ones(2), atol=1e-5)


def training_batch(ticks=8, envs=4, slots=10, fields=5):
    obs = make_obs((ticks, envs), slots, fields=fields)
    obs.key_mask[:, :, -1] = False  # a slot that is never live
    obs.key_mask[:, :, 0] = True
    obs.field_mask[:, :, 0] = True
    obs.field_mask[:, :, -1] = False  # a field that is never present
    reset = torch.zeros(ticks, envs, dtype=torch.bool)
    reset[3, 1] = reset[5, envs - 1] = True
    generator = torch.Generator().manual_seed(7)
    config = ModelConfig()
    targets = SimpleNamespace(
        actions=torch.randint(0, 42, (ticks, envs, slots), generator=generator),
        bins=torch.randint(
            0, config.bins, (ticks, envs, slots, config.levels), generator=generator
        ),
        next_state=torch.randn(ticks, envs, slots, config.next_state_size, generator=generator),
        hit=torch.randint(0, config.hit_classes, (ticks, envs, slots), generator=generator),
    )
    return obs, reset, targets


def training_loss(policy, obs, reset, targets, carry=None):
    ticks, envs, slots = obs.key_mask.shape
    if carry is None:
        carry = policy.initial_carry(envs, slots, "cpu")
    logits, values, next_state, hit, final = policy.sequence(obs, carry, reset)
    live = obs.key_mask
    policy_loss = torch.nn.functional.cross_entropy(logits[live], targets.actions[live])
    value_loss = torch.nn.functional.cross_entropy(
        values[live].flatten(0, 1), targets.bins[live].flatten()
    )
    next_loss = (next_state[live] - targets.next_state[live]).square().mean()
    hit_loss = torch.nn.functional.cross_entropy(hit[live], targets.hit[live])
    return policy_loss + value_loss + next_loss + hit_loss, policy_loss, final


def test_backward_reaches_every_parameter_and_no_masked_row():
    """Catches dead parameters, non-finite gradients, and masked ship rows or absent fields
    leaking into the loss."""
    policy = make_policy().train()
    obs, reset, targets = training_batch()
    for name in ("ship", "pos", "heading", "globals", "field", "field_pos"):
        getattr(obs, name).requires_grad_(True)
    carry = torch.randn(policy.initial_carry(4, 10, "cpu").shape).requires_grad_(True)
    loss, _, final = training_loss(policy, obs, reset, targets, carry)
    (loss + final.square().mean()).backward()
    assert torch.isfinite(loss)
    dead = [
        name
        for name, p in policy.named_parameters()
        if p.grad is None or not p.grad.abs().sum() > 0
    ]
    bad = [
        name
        for name, p in policy.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not dead, f"parameters with no gradient: {dead}"
    assert not bad, f"parameters with a non-finite gradient: {bad}"
    for names, mask in (
        (("ship", "pos", "heading"), obs.key_mask),
        (("field", "field_pos"), obs.field_mask),
    ):
        for name in names:
            grad = getattr(obs, name).grad
            assert torch.isfinite(grad).all()
            assert grad[~mask].abs().max() == 0, f"a masked row's {name} has a gradient"
            assert grad[mask].abs().sum() > 0
    # The carry (state and buffer) of a row masked at tick 0 is cut by keep = 0.
    assert carry.grad[:, :, 1:][:, ~obs.key_mask[0]].abs().max() == 0
    assert carry.grad[:, :, 1:][:, obs.key_mask[0]].abs().sum() > 0

    # Changing masked rows' and absent fields' inputs leaves the loss unchanged.
    with torch.no_grad():
        changed = SimpleNamespace(**{k: v.detach().clone() for k, v in vars(obs).items()})
        masked, absent = ~obs.key_mask, ~obs.field_mask
        changed.ship[masked] = torch.randn_like(changed.ship[masked]) * 50.0
        changed.pos[masked] = torch.randn_like(changed.pos[masked]) * 5000.0
        changed.heading[masked] = 2.5
        changed.field[absent] = torch.randn_like(changed.field[absent]) * 50.0
        changed.field_pos[absent] = torch.randn_like(changed.field_pos[absent]) * 5000.0
        again, _, again_final = training_loss(policy, changed, reset, targets, carry)
    assert torch.allclose(again, loss, atol=1e-5)
    assert torch.allclose(again_final, final, atol=1e-5)


def test_gradient_without_perturbed_heads_or_loss_on_carry():
    """The same with the real initialisation and the loss alone: lists what a fresh model's
    first step actually trains."""
    torch.manual_seed(0)
    policy = Policy(ModelConfig(), SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES)
    obs, reset, targets = training_batch()
    loss, _, _ = training_loss(policy, obs, reset, targets)
    loss.backward()
    dead = [
        name
        for name, p in policy.named_parameters()
        if p.grad is None or not p.grad.abs().sum() > 0
    ]
    assert not dead, f"parameters with no gradient: {dead}"
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters())


def test_adam_fits_a_tiny_fixed_batch():
    """Catches a model that cannot fit: policy cross-entropy to fixed random actions."""
    torch.manual_seed(0)
    policy = Policy(ModelConfig(), SHIP_FEATURES, GLOBAL_FEATURES, FIELD_FEATURES)
    obs, reset, targets = training_batch(ticks=8, envs=2, slots=6)
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        optimiser.zero_grad()
        _, policy_loss, _ = training_loss(policy, obs, reset, targets)
        policy_loss.backward()
        optimiser.step()
        losses.append(policy_loss.item())
    print("policy cross-entropy first, last:", losses[0], losses[-1])
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


def test_bf16_autocast_step_stays_close_to_float32():
    """Catches position phases computed in bf16 (measured before they were pinned to
    float32: 0.18 to 0.26 logit drift) and a carry that leaves float32."""
    policy = make_policy()
    obs = make_obs((4,), 10)
    carry = torch.randn(policy.initial_carry(4, 10, "cpu").shape)
    with torch.no_grad():
        logits, values, new_carry = policy.step(obs, carry)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            half_logits, half_values, half_carry = policy.step(obs, carry)
    live = obs.key_mask
    assert half_carry.dtype == torch.float32
    assert (logits - half_logits.float())[live].abs().max() < 0.08
    assert (values - half_values.float())[live].abs().max() < 0.08
    # bf16 rounding: 0.025 to 0.036 over eight weight seeds (2026-11-23), so 0.05.
    assert (new_carry - half_carry).abs().max() < 0.05


def test_compiled_step_matches_eager():
    """Catches a forward that torch.compile cannot trace or traces differently (the doubling
    scan's Python loop runs over a constant T)."""
    policy = make_policy()
    obs = make_obs((2,), 4, fields=3)
    carry = torch.randn(policy.initial_carry(2, 4, "cpu").shape)
    compiled = torch.compile(policy.step)
    with torch.no_grad():
        eager_logits, _, eager_carry = policy.step(obs, carry)
        compiled_logits, _, compiled_carry = compiled(obs, carry)
    live = obs.key_mask
    assert torch.allclose(eager_logits[live], compiled_logits[live], atol=1e-4)
    assert torch.allclose(eager_carry, compiled_carry, atol=1e-4)


def test_the_model_reads_fog():
    """C10: under fog a last-seen row must not look current: the outputs of a ship depend on its
    own visible flag and time since seen."""
    policy = make_policy().eval()
    obs = make_obs((1,), 8, live_fraction=1.0)
    carry = policy.initial_carry(1, 8, torch.device("cpu"))
    with torch.no_grad():
        base, _, _ = policy.step(obs, carry)
        stale = SimpleNamespace(
            **{
                **vars(obs),
                "visible": torch.zeros_like(obs.visible),
                "since_seen": obs.since_seen + 3.0,
            }
        )
        changed, _, _ = policy.step(stale, carry)
    assert (base - changed).abs().max() > 1e-4
