"""Shared cases for the Yemong section tests: a model, real rollouts, step loops."""

import copy
import dataclasses

import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs, Pool
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL
from bnb.controllers.scripted import Chaser
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.models.yemong.model import Yemong, YemongConfig

CPU = torch.device("cpu")
# squad's sizes in a smaller arena with short episodes, so a segment has fights and resets
SQUAD = GameConfig(5, 10, 2600.0)
SHORT_SQUAD = GameConfig(5, 10, 1500.0, episode_s=2.0, respawn_s=1.0)


def model(config: GameConfig = SQUAD, seed: int = 0, dtype=torch.float32) -> Yemong:
    """A Yemong with its zero-initialised and near-zero last layers randomised, so every
    head's output depends on its inputs."""
    torch.manual_seed(seed)
    net = Yemong(YemongConfig(), api.rules_of(config), api.ship_types())
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name.endswith("last.weight"):
                parameter.normal_(0.0, 0.05)
    return net.to(dtype).eval()


def controller(net: Yemong, config: GameConfig) -> YemongController:
    return YemongController(net, config.ships_per_team, config.num_ships)


def rollout(net: Yemong, config: GameConfig = SHORT_SQUAD, envs: int = 8, length: int = 128,
            fog_share: float = 0.5, device=CPU, compile: bool = False):
    """A real rollout with Yemong as the learner, in self-play and against the chaser."""
    from bnb.runtime.rollout import Rollout  # tests may import the runtime

    pools = (Pool("self_play", 0.5, ("learner", "learner"), fog_share),
             Pool("chaser", 0.5, ("learner", "chaser"), fog_share))
    controllers = {"learner": controller(net, config), "chaser": Chaser(config.ships_per_team)}
    runner = Rollout(config, pools, controllers, "learner", envs, length, device, compile=compile)
    return runner, runner.run()


def resets(obs: Obs) -> torch.Tensor:
    return obs.globals[..., GLOBAL["reset"]] > 0.5


def cast(tree, dtype):
    return _pytree.tree_map_only(torch.Tensor,
                                 lambda x: x.to(dtype) if x.is_floating_point() else x, tree)


def step_loop(net: Yemong, step, carry, obs: Obs, joint: torch.Tensor):
    """Run `step` over a stored segment, feeding back the stored joint actions.
    Returns the decisions stacked over time and the final carry."""
    decisions = []
    reset = resets(obs)
    for t in range(obs.ships.shape[0]):
        view = Obs(ships=obs.ships[t], fields=obs.fields[t], globals=obs.globals[t])
        decision = step(carry, view, reset[t])
        carry = dataclasses.replace(decision.carry, previous_action=joint[t])
        decisions.append(decision)
    stacked = _pytree.tree_map(lambda *x: torch.stack(x), *decisions)
    return stacked, carry


def joint_actions(trajectory) -> torch.Tensor:
    return to_joint(trajectory.action).to(torch.int32)


def double(net: Yemong) -> Yemong:
    return copy.deepcopy(net).double()


def scale_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """max |actual - expected| relative to max(1, max |expected|)."""
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1.0)).item()
