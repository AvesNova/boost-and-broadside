"""Shared cases for the runs tests: the smallest model of each registered kind, a fixed
observation and carry, and a model's single step as named tensors."""

import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs
from bnb.game import api
from bnb.models.mlp import MLP, MLPBuild
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build

CPU = torch.device("cpu")
GAME = GameConfig(2, 2, 1500.0)
PARTS = ("ships", "fields", "globals", "zones")


def tiny_yemong_config(width: int = 8) -> YemongConfig:
    """Yemong's smallest configuration: one block, one head, one rotary octave."""
    return YemongConfig(
        width=width,
        blocks=1,
        spatial_per_block=1,
        heads=1,
        map_width=4,
        value_bins=11,
        hex_rings=1,
        rotary_octaves=1,
        heading_harmonics=(1,),
    )


def _randomised(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Every parameter redrawn, so every output depends on its inputs and on every weight
    (zero-initialised heads would hide a weight that failed to load)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def tiny_model(name: str, seed: int = 0, width: int = 8, game: GameConfig = GAME):
    """The smallest model registered as `name`, weights drawn from `seed`."""
    # The constructors initialise from the default generator: seeded here, then restored.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "yemong":
            model = Yemong(
                yemong_build(tiny_yemong_config(width), api.rules_of(game), api.ship_types())
            )
        elif name == "relational":
            build = RelationalBuild(
                RelationalConfig(hidden=width, value_bins=11), game.ships_per_team, game.num_ships
            )
            model = Relational(build)
        else:
            model = MLP(MLPBuild(game.ships_per_team, game.num_ships, game.num_fields, width))
    return _randomised(model, seed)


def observation(seed: int = 0, views: int = 2, game: GameConfig = GAME, device=CPU) -> Obs:
    """Team 0's views of `views` fresh games a few ticks in."""
    generator = torch.Generator().manual_seed(seed)
    rules, types = api.rules_of(game), api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, views, generator=generator))
    for _ in range(3):
        action = torch.randint(
            0, 3, (views, game.num_ships, 3), generator=generator, dtype=torch.int32
        )
        state, _ = api.step(
            rules,
            types,
            state,
            Action.unpack(action),
            api.spread(rules, views, generator=generator),
            api.uniforms(rules, views, generator=generator),
        )
    _, obs, _ = api.perceive(rules, state)
    return Obs(**{part: getattr(obs, part)[:, 0].to(device) for part in PARTS})


def yemong_carry(model: Yemong, seed: int, views: int, game: GameConfig = GAME, device=CPU):
    """A carry with history: a fresh one with every float leaf drawn from `seed`."""
    generator = torch.Generator().manual_seed(seed)
    carry = model.initial_carry(views, game.ships_per_team, game.num_ships, device)
    return _pytree.tree_map_only(
        torch.Tensor,
        lambda x: (
            torch.rand(x.shape, generator=generator).to(device, x.dtype)
            if x.is_floating_point()
            else x
        ),
        carry,
    )


def single_step(model: torch.nn.Module, obs: Obs, carry=None) -> dict[str, torch.Tensor]:
    """The model's single-step forward on `obs` (and Yemong's `carry`), no reset, as named
    tensors."""
    with torch.no_grad():
        if isinstance(model, Yemong):
            reset = torch.zeros(obs.ships.shape[0], dtype=torch.bool, device=obs.ships.device)
            out = model.step(carry, obs, reset)
        elif isinstance(model, MLP):
            logits, value = model(obs)
            return dict(logits=logits, value=value)
        else:
            out = model(obs)
    leaves, _ = _pytree.tree_flatten_with_path(out)
    return {_pytree.keystr(path): leaf for path, leaf in leaves if isinstance(leaf, torch.Tensor)}


def assert_equal(got: dict, want: dict) -> None:
    """Every named tensor bit for bit."""
    assert got.keys() == want.keys()
    for name in want:
        assert torch.equal(got[name], want[name]), name


def tiny_train_config(model: str = "relational", updates: int = 2, **changes):
    """A TrainConfig small enough to train on the CPU in seconds, uncompiled."""
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import TrainConfig

    learner = LearnerConfig(epochs=1, minibatches=2, warmup_steps=1000, micro_batch_rows=64)
    values = dict(
        game=GameConfig(2, 2, 1500.0, episode_seconds=4.0, respawn_seconds=1.0),
        environments=16,
        updates=updates,
        length=16,
        model=model,
        relational_hidden=16,
        league_every=1,
        learner=learner,
        compiled=False,
    )
    values.update(changes)
    return TrainConfig(**values)


TRAIN_OPTIONS = (
    "--device",
    "cpu",
    "--model",
    "relational",
    "--ships-per-team",
    "2",
    "--num-fields",
    "2",
    "--arena-radius",
    "1500",
    "--environments",
    "16",
    "--set",
    "length=16",
    "--set",
    "compiled=false",
    "--set",
    "learner.minibatches=2",
    "--set",
    "learner.epochs=1",
    "--set",
    "relational_hidden=16",
    "--set",
    "game.episode_seconds=4.0",
)
