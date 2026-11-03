"""Yemong in a mixed batch (mixed spec 4 and 6): absent zones are one learned map-memory
token at the global token's rotation, they change nothing, the offset stays hidden, and
unroll reproduces a stored mixed rollout."""

import dataclasses

import pytest
import torch
import yemong_cases as cases

from bnb.contracts import Obs, Pool
from bnb.contracts.obs import ZONE, ZONE_COLUMNS
from bnb.controllers.scripted import Chaser
from bnb.models.yemong import policy
from bnb.models.yemong.inputs import POSITION_CODE
from bnb.models.yemong.state_code import POSITION_PERIOD

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


OWN = 5
POOLS = (
    Pool("frontline", 0.5, ("learner", "chaser"), 0.5, mode="frontline"),
    Pool("deathmatch", 0.5, ("learner", "learner"), 0.5),
)


def _mixed_rollout(net, environments: int = 8, length: int = 32, runs: int = 1):
    """Yemong learning views of a mixed batch: Frontline against the chaser (first), team
    deathmatch in self-play (absent zones), short clocks so the segment has resets."""
    from bnb.runtime.rollout import Rollout

    config = cases.SHORT_SQUAD
    controllers = {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(
        config, POOLS, controllers, "learner", environments, length, cases.CPU, compiled=False
    )
    for _ in range(runs):
        trajectory = runner.run()
    return trajectory


def _views(ticks: int = 24):
    """float64 views (L, R, ...) of a mixed batch and the float64 model; rows of absent
    zones (team deathmatch) and present ones (Frontline)."""
    net = cases.model()
    obs = cases.cast(_mixed_rollout(net, length=ticks).obs, torch.float64)
    present = obs.zones[..., ZONE["present"]] > 0.5
    assert present.any() and (~present).any()
    return cases.double(net), obs


def _last(obs: Obs) -> Obs:
    return Obs(
        ships=obs.ships[-1], fields=obs.fields[-1], globals=obs.globals[-1], zones=obs.zones[-1]
    )


def _garbage(view: Obs, seed: int) -> Obs:
    """The view with every column of its absent zones but `present` replaced by noise."""
    zones = view.zones.clone()
    absent = zones[..., ZONE["present"]] < 0.5
    noise = (
        torch.randn(zones.shape, generator=torch.Generator().manual_seed(seed), dtype=zones.dtype)
        * 500.0
    )
    columns = torch.tensor([name != "present" for name in ZONE_COLUMNS])
    zones = torch.where(absent[..., None] & columns, noise, zones)
    return dataclasses.replace(view, zones=zones)


def test_absent_zone_columns_change_no_output():
    net, obs = _views()
    view = _last(obs)
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    reset = torch.zeros(views, dtype=torch.bool)
    with torch.no_grad():
        plain = net.step(carry, view, reset)
        noisy = net.step(carry, _garbage(view, 1), reset)
        for name in (
            "policy_logits",
            "enemy_action_logits",
            "next_state_logits",
            "values",
            "outcome_probabilities",
        ):
            assert torch.equal(getattr(plain, name), getattr(noisy, name)), name
        replay = cases.step_loop(
            net,
            net.step,
            net.initial_carry(views, OWN, 10),
            obs,
            torch.zeros(obs.ships.shape[:2] + (OWN,), dtype=torch.int32),
        )[0].replay
        noisy_obs = dataclasses.replace(obs, zones=_garbage(obs, 2).zones)
        first = net.unroll(carry, obs, replay, cases.resets(obs))
        second = net.unroll(carry, noisy_obs, replay, cases.resets(obs))
    for name in ("policy_logits", "values", "density_log_rates", "shot_latents"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name


def test_an_absent_zone_is_the_learned_token_at_the_global_tokens_rotation():
    net, obs = _views()
    view = _last(obs)
    views = view.ships.shape[0]
    carry = net.initial_carry(views, OWN, 10)
    with torch.no_grad():
        replay = net.step(carry, view, torch.zeros(views, dtype=torch.bool)).replay
        tokens = net.tokens(view, replay)
        _, memory, token_tables, map_tables = net._encode(tokens)
    M, N = view.fields.shape[-2], view.ships.shape[-2]
    absent = ~tokens.zone_present
    assert absent.any() and tokens.zone_present.any()
    zone_memory = memory[..., M:, :]
    learned = net.encoder.absent_zone.expand_as(zone_memory)
    assert torch.equal(zone_memory[absent], learned[absent])
    assert not torch.equal(zone_memory[~absent], learned[~absent])
    # the rotation: the absent zones' rotary tables are the global token's
    for token_table, map_table in zip(token_tables, map_tables, strict=True):
        global_table = token_table[..., N : N + 1, :, :].expand_as(map_table[..., M:, :, :])
        assert torch.equal(map_table[..., M:, :, :][absent], global_table[absent])
    centre = tokens.query_position[..., N, :]
    assert torch.equal(
        tokens.memory_position[..., M:, :][absent], centre[:, None, :].expand(-1, 5, -1)[absent]
    )


def test_the_offset_stays_hidden_with_absent_zones():
    """With the encoder's position-code columns zeroed, a decision does not depend on the
    per-team offset, in a batch with absent zones: the same history in a frame shifted by
    delta gives the same outputs."""
    net, obs = _views(ticks=24)
    with torch.no_grad():
        for kind in (
            net.encoder.ship,
            net.encoder.field,
            net.encoder.global_token,
            net.encoder.zone,
        ):
            kind.first.weight[:, :POSITION_CODE] = 0.0
    views = obs.ships.shape[1]
    tail = Obs(
        ships=obs.ships[:-1], fields=obs.fields[:-1], globals=obs.globals[:-1], zones=obs.zones[:-1]
    )
    joint = torch.zeros(obs.ships.shape[:2] + (OWN,), dtype=torch.int32)
    with torch.no_grad():
        _, carry = cases.step_loop(net, net.step, net.initial_carry(views, OWN, 10), tail, joint)
        delta = torch.rand(views, 2, dtype=torch.float64) * POSITION_PERIOD
        means = carry.belief.means.clone()
        means[..., 0:2] += delta[:, None, :]
        shifted = dataclasses.replace(
            carry,
            offset=carry.offset + delta,
            belief=dataclasses.replace(carry.belief, means=means),
        )
        reset = torch.zeros(views, dtype=torch.bool)
        view = _last(obs)
        plain, moved = net.step(carry, view, reset), net.step(shifted, view, reset)
    assert (view.zones[..., ZONE["present"]] < 0.5).any()
    for name in ("policy_logits", "enemy_action_logits", "values", "outcome_probabilities"):
        torch.testing.assert_close(getattr(moved, name), getattr(plain, name), rtol=1e-8, atol=1e-8)


def test_unroll_reproduces_a_stored_mixed_rollout():
    """The replay holds what the step read, absent zones included (they come from the
    stored observation), so unroll reproduces the stored decisions."""
    net = cases.model()
    trajectory = _mixed_rollout(net, length=48, runs=2)
    assert cases.resets(trajectory.obs)[1:].any()
    with torch.no_grad():
        unrolled = net.unroll(
            trajectory.initial_carry,
            trajectory.obs,
            trajectory.replay,
            cases.resets(trajectory.obs),
        )
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-4
    log_probability, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(trajectory))
    assert cases.scale_error(log_probability, trajectory.log_probability) <= 1e-4


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_compiled_captured_mixed_rollout_stores_what_unroll_needs(fresh_compiler):
    """The mixed batch compiles into the runtime's stages and captures as one CUDA graph,
    and the stored decisions still agree with unroll to float32 rounding."""
    from bnb.runtime.rollout import Rollout

    device = torch.device("cuda")
    net = cases.model().to(device)
    config = cases.SHORT_SQUAD
    controllers = {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(config, POOLS, controllers, "learner", 16, 64, device, compiled=True)
    trajectory = runner.run()
    assert runner.graph is not None and cases.resets(trajectory.obs)[1:].any()
    present = trajectory.obs.zones[..., ZONE["present"]]
    assert (present == 1).any() and (present == 0).any()
    with torch.no_grad():
        unrolled = net.unroll(
            trajectory.initial_carry,
            trajectory.obs,
            trajectory.replay,
            cases.resets(trajectory.obs),
        )
    assert cases.scale_error(unrolled.values, trajectory.values) <= 1e-5
    log_probability, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(trajectory))
    assert cases.scale_error(log_probability, trajectory.log_probability) <= 1e-5
