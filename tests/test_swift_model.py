"""The whole model: unroll == loop(step), shapes, dtypes, the belief rules, symmetry, codes."""

import pytest
import torch

from bnb.contracts.obs import SHIP, SHIP_HIDDEN_ZERO, Obs
from bnb.models.swift.model import Replay, Yemong, YemongConfig
from bnb.models.swift.state_code import MEAN_CHANNELS, SPREAD_CHANNELS
from bnb.models.swift.synthetic import random_segment, time_slice

STEP_KEYS = ("action_logits", "enemy_action_logits", "next_state_logits", "values", "value",
             "outcome")
CUDA = torch.cuda.is_available()


def make_model(dtype=torch.float64, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = Yemong(YemongConfig()).to(device=device, dtype=dtype)
    with torch.no_grad():  # the next-state head starts at zero; make the forecast move
        model.next_state_head.last.weight.normal_(0, 0.02)
    return model


def nonzero_carry(model, views, ships, dtype, device="cpu"):
    carry = model.initial_carry(views, ships, device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(1)
    return carry._replace(
        state=torch.randn(carry.state.shape, generator=generator, device=device, dtype=dtype),
        convolution_buffer=torch.randn(carry.convolution_buffer.shape, generator=generator,
                                       device=device, dtype=dtype))


def loop_of_steps(model, carry, obs, reset, sample=True):
    outputs = []
    for t in range(reset.shape[0]):
        carry, out = model.step(carry, time_slice(obs, t), reset[t], sample=sample)
        outputs.append(out)
    replay = Replay(*(torch.stack([o["replay"][i] for o in outputs]) for i in range(6)))
    stacked = {key: torch.stack([o[key] for o in outputs]) for key in STEP_KEYS}
    return carry, stacked, replay


def relative_error(actual, expected):
    return ((actual - expected).abs().max() / max(1.0, expected.abs().max().item())).item()


@pytest.mark.parametrize("reset_steps", [(0, 6), (5,), ()], ids=["first-and-mid", "mid", "none"])
def test_unroll_equals_loop_of_steps_float64(reset_steps):
    model = make_model()
    steps, views, ships = 11, 4, 10
    obs, reset = random_segment(steps, views, ships, dtype=torch.float64, reset_steps=reset_steps)
    if reset_steps:
        reset[:, 1] = False
        reset[8, 1] = True  # a reset only one view sees
    start = nonzero_carry(model, views, ships, torch.float64)
    final, stepped, replay = loop_of_steps(model, start, obs, reset)
    unrolled = model.unroll(start, obs, replay, reset)
    for key in STEP_KEYS:
        assert relative_error(unrolled[key], stepped[key]) < 1e-9, key
    assert relative_error(unrolled["final_state"], final.state) < 1e-9
    assert relative_error(unrolled["final_convolution_buffer"], final.convolution_buffer) < 1e-9


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_unroll_equals_loop_of_steps_float32_cuda(compiled):
    model = make_model(torch.float32, "cuda")
    steps, views, ships = 9, 8, 10
    obs, reset = random_segment(steps, views, ships, device="cuda", reset_steps=(0, 4))
    reset[:, 1] = False
    start = nonzero_carry(model, views, ships, torch.float32, "cuda")
    step = torch.compile(model.step, fullgraph=True) if compiled else model.step
    unroll = torch.compile(model.unroll, fullgraph=True) if compiled else model.unroll
    carry, outputs = start, []
    for t in range(steps):
        carry, out = step(carry, time_slice(obs, t), reset[t])
        outputs.append({key: value.clone() for key, value in out.items() if key != "replay"}
                       | {"replay": Replay(*(leaf.clone() for leaf in out["replay"]))})
    replay = Replay(*(torch.stack([o["replay"][i] for o in outputs]) for i in range(6)))
    unrolled = unroll(start, obs, replay, reset)
    for key in STEP_KEYS:
        expected = torch.stack([o[key] for o in outputs])
        assert relative_error(unrolled[key], expected) < 1e-4, key


def test_shapes_dtypes_and_finite():
    model = make_model(torch.float32)
    steps, views, ships, fields = 5, 3, 10, 10
    obs, reset = random_segment(steps, views, ships, fields, reset_steps=(2,))
    carry = model.initial_carry(views, ships)
    carry, out = model.step(carry, time_slice(obs, 0), reset[0])
    own, levels = ships // 2, len(model.config.reward_levels)
    expected = {
        "action_logits": (views, own, 42), "enemy_action_logits": (views, ships, 42),
        "next_state_logits": (views, ships, 490), "values": (views, own, levels),
        "value": (views, own), "outcome": (views, 4), "action": (views, own),
        "logp": (views, own), "entropy": (views, own),
    }
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, key
        assert torch.isfinite(out[key].float()).all(), key
    assert out["action"].dtype == torch.int64
    assert out["values"].dtype == torch.float32
    assert tuple(carry.means.shape) == (views, ships, MEAN_CHANNELS)
    assert tuple(carry.spreads.shape) == (views, ships, SPREAD_CHANNELS)
    assert torch.isfinite(carry.means).all() and torch.isfinite(carry.spreads).all()
    _, _, replay = loop_of_steps(model, model.initial_carry(views, ships), obs, reset)
    ages = torch.arange(1, 5).expand(steps, views, ships, 4)
    unrolled = model.unroll(model.initial_carry(views, ships), obs, replay, reset, ages)
    more = {"density_log_rates": (steps, views, 662), "shot_outcome_logits":
            (steps, views, ships, 19), "shot_latent": (steps, views, ships, 128),
            "shot_trajectory_logits": (steps, views, ships, 4, 154)}
    for key, shape in more.items():
        assert tuple(unrolled[key].shape) == shape, key
        assert torch.isfinite(unrolled[key]).all(), key


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_bfloat16_autocast_runs(device):
    model = make_model(torch.float32, device)
    steps, views, ships = 4, 2, 10
    obs, reset = random_segment(steps, views, ships, device=device, reset_steps=(2,))
    with torch.autocast(device, dtype=torch.bfloat16):
        carry, out = model.step(model.initial_carry(views, ships, device=device),
                                time_slice(obs, 0), reset[0])
        assert out["values"].dtype == torch.float32
        assert carry.means.dtype == torch.float32
        _, _, replay = loop_of_steps(model, model.initial_carry(views, ships, device=device),
                                     obs, reset)
        unrolled = model.unroll(model.initial_carry(views, ships, device=device), obs, replay,
                                reset)
        loss = sum(value.float().mean() for key, value in unrolled.items()
                   if key not in ("final_state", "final_convolution_buffer"))
    loss.backward()
    for key in STEP_KEYS:
        assert torch.isfinite(unrolled[key]).all(), key
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


# --------------------------------------------------------------------------- belief rules


def first_step(model, obs, views, ships):
    reset = torch.ones(views, dtype=torch.bool)
    return model.step(model.initial_carry(views, ships, dtype=torch.float64), obs, reset)


def test_visible_ship_code_is_sharp_code_of_observation():
    model = make_model()
    obs, _ = random_segment(1, 3, 10, dtype=torch.float64, fog_share=1.0)
    single = time_slice(obs, 0)
    _, out = first_step(model, single, 3, 10)
    replay = out["replay"]
    inputs = model.compose(single.ships, single.fields, single.globals, replay)
    ship_type = model.ship_type(single.ships)
    sharp = model.codec.sharp(model.truth(single.ships), ship_type)
    visible = single.ships[..., SHIP["visible"]] > 0.5
    assert visible.any() and (~visible).any()
    assert torch.allclose(inputs.code[visible], sharp[visible], atol=1e-12)


def test_spawning_voids_a_belief():
    model = make_model()
    obs, reset = random_segment(3, 2, 10, dtype=torch.float64, fog_share=1.0)
    ships = obs.ships.clone()
    # ship 7 visible at step 0, hidden and spawned (without sight) at step 1
    ships[0, :, 7, SHIP["visible"]] = 1
    ships[1, :, 7, :SHIP_HIDDEN_ZERO] = 0
    ships[1, :, 7, SHIP["visible"]] = 0
    ships[1, :, 7, SHIP["spawned"]] = 1
    obs = Obs(ships, obs.fields, obs.globals, obs.zones)
    carry = model.initial_carry(2, 10, dtype=torch.float64)
    carry, out = model.step(carry, time_slice(obs, 0), torch.ones(2, dtype=torch.bool))
    assert out["replay"].valid[:, 7].all()
    carry, out = model.step(carry, time_slice(obs, 1), torch.zeros(2, dtype=torch.bool))
    assert not out["replay"].valid[:, 7].any()
    assert (out["replay"].means[:, 7] == 0).all()


def test_fog_off_spreads_are_zero():
    model = make_model()
    obs, reset = random_segment(4, 2, 10, dtype=torch.float64, fog_share=0.0)
    _, _, replay = loop_of_steps(model, model.initial_carry(2, 10, dtype=torch.float64), obs,
                                 reset)
    assert (replay.spreads == 0).all()


def test_hidden_zero_columns_of_hidden_ships_are_not_read():
    model = make_model()
    obs, reset = random_segment(5, 3, 10, dtype=torch.float64, fog_share=1.0, reset_steps=(0,))
    hidden = (obs.ships[..., SHIP["visible"]] < 0.5).unsqueeze(-1)
    assert hidden.any()
    noise = torch.randn(obs.ships.shape, dtype=torch.float64) * 100
    columns = torch.arange(obs.ships.shape[-1]) < SHIP_HIDDEN_ZERO
    perturbed = Obs(torch.where(hidden & columns, obs.ships + noise, obs.ships), obs.fields,
                    obs.globals, obs.zones)
    start = model.initial_carry(3, 10, dtype=torch.float64)
    _, clean, _ = loop_of_steps(model, start, obs, reset, sample=False)
    _, dirty, _ = loop_of_steps(model, start, perturbed, reset, sample=False)
    for key in STEP_KEYS:
        assert torch.equal(clean[key], dirty[key]), key


def test_permuting_ships_within_a_team_permutes_outputs():
    model = make_model()
    views, ships = 2, 10
    obs, reset = random_segment(4, views, ships, dtype=torch.float64, reset_steps=(0,))
    order = torch.tensor([2, 0, 1, 4, 3, 9, 5, 8, 6, 7])
    permuted = Obs(obs.ships[..., order, :], obs.fields, obs.globals, obs.zones)
    start = model.initial_carry(views, ships, dtype=torch.float64)
    _, plain, _ = loop_of_steps(model, start, obs, reset, sample=False)
    _, moved, _ = loop_of_steps(model, start, permuted, reset, sample=False)
    own = order[:5]
    assert relative_error(moved["action_logits"], plain["action_logits"][..., own, :]) < 1e-9
    assert relative_error(moved["values"], plain["values"][..., own, :]) < 1e-9
    for key in ("enemy_action_logits", "next_state_logits"):
        assert relative_error(moved[key], plain[key][..., order, :]) < 1e-9, key
    assert relative_error(moved["outcome"], plain["outcome"]) < 1e-9


@pytest.mark.parametrize("ships", [2, 10, 20])
def test_one_set_of_weights_runs_at_every_size(ships):
    model = make_model(torch.float32)
    obs, reset = random_segment(2, 2, ships, fields=7, reset_steps=(0,))
    carry, out = model.step(model.initial_carry(2, ships), time_slice(obs, 0), reset[0])
    assert tuple(out["action_logits"].shape) == (2, ships // 2, 42)


# --------------------------------------------------------------------------- codes


def test_state_code_round_trip():
    model = make_model()
    obs, _ = random_segment(1, 4, 10, dtype=torch.float64, fog_share=0.0)
    ships = obs.ships[0]
    truth = model.truth(ships)
    ship_type = model.ship_type(ships)
    code = model.codec.sharp(truth, ship_type)
    assert code.shape[-1] == 490
    means, spreads = model.codec.decode(code, ship_type)
    tolerance = torch.tensor([1e-6] * 4 + [1e-9] * 8, dtype=torch.float64)
    assert ((means - truth).abs() <= tolerance * truth.abs().clamp_min(1) + 1e-9).all()
    assert spreads[..., :11].abs().max() < 1e-6
    assert spreads[..., 11:].abs().max() < 1e-9
