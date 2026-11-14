"""Shared cases for the Yemong section tests: a model, real rollouts, step loops."""

import copy
import dataclasses

import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig, Obs, Pool
from bnb.contracts.action import to_joint
from bnb.contracts.obs import COLUMNS_BEFORE_RECORDS, GLOBAL, GLOBAL_COLUMNS, SHIP_COLUMNS
from bnb.controllers.scripted import Chaser
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build

CPU = torch.device("cpu")
# squad's sizes in a smaller arena with short episodes, so a segment has fights and resets
SQUAD = GameConfig(5, 10, 2600.0)
SHORT_SQUAD = GameConfig(5, 10, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
SHORT_FRONTLINE = GameConfig(
    5, 10, 1500.0, episode_seconds=2.0, respawn_seconds=1.0, mode="frontline"
)


def model(config: GameConfig = SQUAD, seed: int = 0, dtype=torch.float32, **settings) -> Yemong:
    """A Yemong with its zero-initialised and near-zero last layers randomised, so every
    head's output depends on its inputs; settings: YemongConfig fields."""
    # The last layers' draws are the default generator's: seeded here, then restored.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        build = yemong_build(YemongConfig(**settings), api.rules_of(config), api.ship_types())
        net = Yemong(build, seed)
        for name, parameter in net.named_parameters():
            if name.endswith("last.weight") or ".lasts." in name and name.endswith(".weight"):
                parameter.normal_(0.0, 0.05)
    assert net.ship_heads.lasts[1].weight.abs().min() > 0, "the next-state head must be live"
    return net.to(dtype).eval()


def controller(net: Yemong, config: GameConfig) -> YemongController:
    return YemongController(net, config.ships_per_team, config.num_ships)


def rollout(
    net: Yemong,
    config: GameConfig = SHORT_SQUAD,
    environments: int = 8,
    length: int = 128,
    fog_share: float = 0.5,
    device=CPU,
    compiled: bool = False,
    runs: int = 1,
):
    """A real rollout with Yemong as the learner, in self-play and against the chaser. With
    runs > 1, the trajectory is the last run's, so its initial carry has history."""
    from bnb.runtime.rollout import Rollout  # tests may import the runtime

    pools = (
        Pool("self_play", 0.5, ("learner", "learner"), fog_share),
        Pool("chaser", 0.5, ("learner", "chaser"), fog_share),
    )
    controllers = {
        "learner": controller(net, config),
        "chaser": Chaser(
            characteristics(api.ship_types(), api.rules_of(config)), config.ships_per_team
        ),
    }
    runner = Rollout(
        config,
        api.ship_types(device),
        pools,
        controllers,
        "learner",
        environments,
        length,
        device,
        compiled=compiled,
    )
    for _ in range(runs - 1):
        runner.run()
    return runner, runner.run()


def resets(obs: Obs) -> torch.Tensor:
    return obs.globals[..., GLOBAL["reset"]] > 0.5


def cast(tree, dtype):
    return _pytree.tree_map_only(
        torch.Tensor, lambda x: x.to(dtype) if x.is_floating_point() else x, tree
    )


def step_loop(net: Yemong, step, carry, obs: Obs, joint: torch.Tensor, start: int = 0):
    """Run `step` over a stored segment from step `start`, feeding back the stored joint
    actions. Returns the decisions stacked over time and the final carry."""
    decisions = []
    reset = resets(obs)
    for t in range(start, obs.ships.shape[0]):
        view = Obs(
            ships=obs.ships[t],
            fields=obs.fields[t],
            globals=obs.globals[t],
            zones=obs.zones[t],
        )
        decision = step(carry, view, reset[t])
        carry = dataclasses.replace(decision.carry, previous_action=joint[t])
        decisions.append(decision)
    stacked = _pytree.tree_map(lambda *x: torch.stack(x), *decisions)
    return stacked, carry


def joint_actions(trajectory) -> torch.Tensor:
    """The learner's own joint actions (L, R, S); trajectory.action holds every row."""
    return to_joint(trajectory.action).to(torch.int32)[..., : trajectory.log_probability.shape[-1]]


def double(net: Yemong) -> Yemong:
    return copy.deepcopy(net).double()


def scale_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """max |actual - expected| relative to max(1, max |expected|)."""
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1.0)).item()


def assert_matches_golden(actual: dict, expected: dict, tolerance: float = 2e-4) -> None:
    """Every named output of a recorded run: floats to `tolerance` of the output's own scale
    (`scale_error`), flags and counts exactly. The goldens run a float64 network for 40
    ticks; the replay stores the enemy actions as int16 fixed point (`store_action`), so a
    rounding flip (one unit in ACTION_SCALE) feeds back into the belief: one thread against
    all threads, the outputs differ by 1e-6 of their scale and the worst (arm 1's carried
    state) by 3e-5. A wrong formula or a missing term moves an output by a good fraction (the
    stale goldens of this tree differed by 1). The stored actions themselves match to one
    unit."""
    assert actual.keys() == expected.keys()
    for name, want in expected.items():
        got = actual[name]
        if name == "replay.enemy_action":  # round(p ACTION_SCALE): a rounding flip is one unit
            assert (got.int() - want.int()).abs().max() <= 1, name
        elif want.is_floating_point():
            error = scale_error(got.double(), want.double())
            assert error <= tolerance, (name, error)
        else:
            assert torch.equal(got, want), name


def _in_todays_layout(recorded: torch.Tensor, part: str, columns: tuple[str, ...]) -> torch.Tensor:
    """recorded's last axis, the columns of COLUMNS_BEFORE_RECORDS[part], in `columns`: each
    column moved by name, the columns added since zero."""
    before = {name: i for i, name in enumerate(COLUMNS_BEFORE_RECORDS[part])}
    assert recorded.shape[-1] == len(before)
    zero = torch.zeros_like(recorded[..., 0])
    return torch.stack(
        [recorded[..., before[name]] if name in before else zero for name in columns], -1
    )


def golden_obs(golden: dict) -> Obs:
    """A golden file's observation, recorded with the columns of COLUMNS_BEFORE_RECORDS, in
    today's layout: each column moved by name, the columns added since zero (the ships'
    accelerations, as for a ship the team did not see move; the globals' mode_elimination, as
    in the golden's team deathmatch)."""
    ships = _in_todays_layout(golden["ships"], "ships", SHIP_COLUMNS)
    globals_ = _in_todays_layout(golden["globals"], "globals", GLOBAL_COLUMNS)
    return Obs(ships=ships, fields=golden["fields"], globals=globals_, zones=golden["zones"])
