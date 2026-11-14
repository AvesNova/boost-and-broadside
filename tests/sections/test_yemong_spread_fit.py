"""`YemongConfig.spread_fit`: an old build record (no field, so "v1") decodes its belief's
spreads with v1's fit; a training build records "coarsest"; a policy file keeps
the setting."""

import json
import pathlib

import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.contracts.obs import SHIP
from bnb.runs import checkpoint
from bnb.runs.checkpoint import FilledDefaults
from bnb.runtime.train import TrainConfig, model_build

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
# The step outputs of the case model on the first 40 ticks of `cases.rollout`, its step loop
# with `cases.step_loop`'s decision keys (first recorded by the pre-fix tree, commit
# 67be0bca; re-recorded by `arm_goldens.py --record`, last for the keyed initialisation and
# offsets).
GOLDEN = pathlib.Path(__file__).parent / "golden" / "yemong_v1_spread_steps.pt"


def _steps(net, golden):
    """net's step loop over the golden's inputs: its outputs in the golden's names, and what
    the forecast check reads."""
    obs = cases.golden_obs(golden)
    dtype = next(net.parameters()).dtype
    obs = cases.cast(obs, dtype)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(net, net.step, carry, obs, golden["joint"])
    outputs = dict(
        policy_logits=decisions.policy_logits,
        enemy_action_logits=decisions.enemy_action_logits,
        values=decisions.values,
        outcome_probabilities=decisions.outcome_probabilities,
        **{f"replay.{k}": v for k, v in vars(decisions.replay).items()},
        **{f"carry.belief.{k}": v for k, v in vars(final.belief).items()},
        **{"carry.state": final.state, "carry.convolution_buffer": final.convolution_buffer},
    )
    extra = dict(
        next_state_logits=decisions.next_state_logits,
        ship_type=obs.ships[..., OWN:, SHIP["ship_type"]].long(),
        visible=obs.ships[..., OWN:, SHIP["visible"]] > 0.5,
    )
    return outputs, extra


def _forecast_check(net, outputs, extra):
    """The spreads held at t + 1 (valid, hidden, held at t) equal the decode of the forecast
    at t by net's state code."""
    stored = outputs["replay.spreads"]
    _, spreads = net.state_code.decode_logits(
        extra["next_state_logits"][:-1].to(stored.dtype), extra["ship_type"][:-1]
    )
    valid = outputs["replay.valid"]
    held = valid[1:] & ~extra["visible"][1:] & valid[:-1]
    assert held.any()
    tolerance = 100 * torch.finfo(stored.dtype).eps  # the same decode: 100 ulps is generous
    torch.testing.assert_close(stored[1:][held], spreads[held], rtol=tolerance, atol=tolerance)


def test_the_default_v1_fit_steps_as_recorded():
    """The v1 fit's step outputs as recorded (`arm_goldens`), to rounding."""
    golden = torch.load(GOLDEN, weights_only=True)
    net = cases.double(cases.model())  # float64, so that the threads' rounding is small
    assert net.config.spread_fit == "v1"
    outputs, extra = _steps(net, golden)
    cases.assert_matches_golden(outputs, golden["outputs"])
    _forecast_check(net, outputs, extra)


def test_the_coarsest_fit_decodes_the_belief_with_the_fixed_spread():
    golden = torch.load(GOLDEN, weights_only=True)
    net = cases.model(spread_fit="coarsest")
    outputs, extra = _steps(net, golden)
    _forecast_check(net, outputs, extra)
    assert not torch.equal(outputs["replay.spreads"], golden["outputs"]["replay.spreads"])


def test_a_training_build_records_the_coarsest_fit():
    config = TrainConfig(game=GameConfig(5, 10, 2600.0))
    assert config.yemong.spread_fit == "coarsest"
    assert model_build(config).config.spread_fit == "coarsest"


@pytest.mark.parametrize("spread_fit", ["v1", "coarsest"])
def test_a_policy_file_keeps_the_fit(tmp_path, spread_fit):
    checkpoint.save_policy(tmp_path / "policy", cases.model(spread_fit=spread_fit))
    loaded = checkpoint.load_policy(tmp_path / "policy", cases.CPU)
    assert loaded.config.spread_fit == spread_fit
    assert loaded.state_code.spread_fit == spread_fit


def test_a_build_record_from_before_the_option_loads_with_v1(tmp_path):
    checkpoint.save_policy(tmp_path / "policy", cases.model(spread_fit="coarsest"))
    path = tmp_path / "policy" / checkpoint.RECORD
    record = json.loads(path.read_text())
    record["model"]["build"]["config"].pop("spread_fit")
    path.write_text(json.dumps(record))
    with pytest.warns(FilledDefaults, match="config.spread_fit"):
        loaded = checkpoint.load_policy(tmp_path / "policy", cases.CPU)
    assert loaded.state_code.spread_fit == "v1"


def test_an_unknown_fit_is_refused():
    with pytest.raises(ValueError, match="spread_fit"):
        cases.model(spread_fit="finest")
