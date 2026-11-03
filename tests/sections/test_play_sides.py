"""The sides (`bnb.play.sides`): trained agents from policy files play a match, a policy
for another game is refused with its reason, and flying one of an agent's ships replaces
that ship's action only."""

import copy
import dataclasses

import pytest
import torch
from play_cases import SMALL, tiny_policy

from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL
from bnb.play.loop import MatchLoop, MatchSetup, SideSetup, team_view
from bnb.play.sides import AgentError, describe, label_of, load_agent, policy_choices


def _agents(policy, temperature: float = 1.0) -> MatchSetup:
    side = SideSetup("agent", str(policy), temperature)
    return MatchSetup(SMALL, seed=4, sides=(side, side))


@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_an_agent_side_plays_a_match(tmp_path, model):
    policy = tiny_policy(tmp_path / "run" / "policy", SMALL, model)
    loop = MatchLoop(_agents(policy), compiled=False)
    actions = []
    for _ in range(40):
        loop.step()
        actions.append(loop.last_actions.clone())
    actions = torch.stack(actions)
    assert actions.shape == (40, SMALL.num_ships, 3)
    assert (actions >= 0).all() and len(torch.unique(actions[..., 1])) > 1  # it steers


def test_a_policy_for_another_game_is_refused_with_the_reason(tmp_path):
    policy = tiny_policy(tmp_path / "policy", SMALL)
    other = dataclasses.replace(SMALL, num_fields=6)
    setup = MatchSetup(other, sides=(SideSetup("agent", str(policy)), SideSetup()))
    with pytest.raises(AgentError, match="num_fields 4 .this match: 6."):
        MatchLoop(setup, compiled=False)
    lineup = MatchSetup(SMALL, lineup=(1, 1, 1), sides=setup.sides)
    with pytest.raises(AgentError, match="default lineup"):
        MatchLoop(lineup, compiled=False)


def test_a_policy_without_its_game_or_a_broken_file_is_refused(tmp_path):
    from play_cases import tiny_model

    from bnb.runs.checkpoint import save_policy

    bare = save_policy(tmp_path / "bare", tiny_model("relational", SMALL))
    loop = MatchLoop(MatchSetup(SMALL), compiled=False)
    with pytest.raises(AgentError, match="does not record the game"):
        load_agent(bare, SMALL, loop.rules, loop.types)
    with pytest.raises(AgentError, match="not a checkpoint"):
        load_agent(tmp_path / "nothing", SMALL, loop.rules, loop.types)


@pytest.mark.parametrize("model", ["relational", "yemong"])
def test_flying_an_agents_ship_leaves_its_other_ships_and_its_carry(tmp_path, model):
    """From one state, the agent's actions with a ship flown equal its actions without,
    but for the flown ship; its carry and generator move on alike."""
    policy = tiny_policy(tmp_path / "policy", SMALL, model)
    loop = MatchLoop(_agents(policy), compiled=False)
    for _ in range(12):
        loop.step()
    flown = 1
    alone = copy.deepcopy(loop.sides[0])
    loop.pilot = flown
    loop.inputs[:] = torch.tensor([True, False, True, False, True, False, True])
    with_pilot = loop.actions()
    observation = loop.realtime.obs
    reset = observation.globals[:, 0, GLOBAL["reset"]] > 0.5
    without = alone.act(team_view(observation, 0), reset)
    others = [ship for ship in range(SMALL.ships_per_team) if ship != flown]
    assert torch.equal(with_pilot[others], without[others])
    assert not torch.equal(with_pilot[flown], without[flown])
    if model == "yemong":
        for got, want in zip(
            torch.utils._pytree.tree_leaves(loop.sides[0].carry),
            torch.utils._pytree.tree_leaves(alone.carry),
            strict=True,
        ):
            assert torch.equal(got, want)
    assert torch.equal(loop.sides[0].generator.get_state(), alone.generator.get_state())
    assert torch.equal(loop.sides[0].model_draws.get_state(), alone.model_draws.get_state())


def test_temperature_zero_takes_the_most_likely_action(tmp_path):
    policy = tiny_policy(tmp_path / "policy", SMALL)
    loop = MatchLoop(_agents(policy, temperature=0.0), compiled=False)
    first = [loop.actions().clone() for _ in range(3)]
    assert all(torch.equal(first[0], actions) for actions in first)


def test_the_policy_list_finds_the_runs_policies(tmp_path):
    runs = tmp_path / "runs"
    tiny_policy(runs / "2026-11-01-a" / "policy", SMALL)
    tiny_policy(runs / "2026-11-02-b" / "checkpoints" / "step_100", SMALL)
    (runs / "2026-11-02-b" / "checkpoints" / "step_200.tmp").mkdir(parents=True)
    found = policy_choices(runs)
    assert [label_of(path) for path in found] == ["2026-11-02-b/step_100", "2026-11-01-a/policy"]
    assert describe(found[0]).startswith("relational")
    assert policy_choices(tmp_path / "missing") == []


def test_an_unsupported_model_is_refused(tmp_path):
    from bnb.game import api
    from bnb.models.mlp import MLP, MLPBuild
    from bnb.runs.checkpoint import save_policy

    with torch.random.fork_rng(devices=[]):
        model = MLP(MLPBuild(SMALL.ships_per_team, SMALL.num_ships, SMALL.num_fields, 8))
    path = save_policy(tmp_path / "mlp", model, game=SMALL, types=api.ship_types())
    loop = MatchLoop(MatchSetup(SMALL), compiled=False)
    with pytest.raises(AgentError, match="mlp models do not play"):
        load_agent(path, GameConfig(3, 4, 900.0, episode_seconds=20.0), loop.rules, loop.types)


def test_an_agent_match_is_reproducible_from_its_seed(tmp_path):
    """The agents draw only from their own generators (Yemong's offsets included)."""
    policy = tiny_policy(tmp_path / "policy", SMALL, "yemong")
    runs = []
    for _ in range(2):
        loop = MatchLoop(_agents(policy), compiled=False)
        actions = []
        for _ in range(30):
            loop.step()
            actions.append(loop.last_actions.clone())
        runs.append(torch.stack(actions))
    assert torch.equal(runs[0], runs[1])
