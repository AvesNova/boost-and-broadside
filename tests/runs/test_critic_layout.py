"""A policy file keeps its critic's layout: the split critic's ten values (each reward
level's offence and defence, the outcome head's P(win) and -P(loss) for the outcome's), and
a build from before the split, whose five still load and play."""

import dataclasses

import pytest
import runs_cases as cases
import torch

from bnb.contracts import REWARD_LEVELS, VALUE_LEVELS
from bnb.game import api
from bnb.models.yemong.model import Yemong, yemong_build
from bnb.runs.checkpoint import load_policy, save_policy

# loading a policy file initialises the model from torch's default generator
pytestmark = pytest.mark.usefixtures("global_generator")


def _model(levels: tuple[str, ...]) -> Yemong:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = dataclasses.replace(cases.tiny_yemong_config(), levels=levels)
        model = Yemong(yemong_build(config, api.rules_of(cases.GAME), api.ship_types()))
    return model.eval()


@pytest.mark.parametrize("levels", [VALUE_LEVELS, REWARD_LEVELS])
def test_a_policy_file_keeps_its_critic_layout(tmp_path, levels):
    model = _model(levels)
    save_policy(tmp_path / "policy", model)
    loaded = load_policy(tmp_path / "policy", cases.CPU).eval()
    assert loaded.config.levels == levels
    obs = cases.observation()
    carry = cases.yemong_carry(model, 1, 2)
    want = cases.single_step(model, obs, carry)
    got = cases.single_step(loaded, obs, carry)
    cases.assert_equal(got, want)
    values = got[".values"]
    assert values.shape[-1] == len(levels)
    outcome = got[".outcome_probabilities"][:, None, :].expand(*values.shape[:-1], 4)
    if levels == VALUE_LEVELS:
        index = VALUE_LEVELS.index
        torch.testing.assert_close(values[..., index("outcome_offence")], outcome[..., 0])
        torch.testing.assert_close(values[..., index("outcome_defence")], -outcome[..., 2])
    else:
        torch.testing.assert_close(values[..., -1], outcome[..., 0] - outcome[..., 2])
