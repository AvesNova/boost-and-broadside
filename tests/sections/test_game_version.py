"""`bnb.game.VERSION` rises with what the game computes: a fingerprint of a short seeded
match in each game preset (the states and observations, on the CPU, uncompiled) is kept
beside it, so a change to the game's numbers fails here until the version is raised and the
fingerprint updated (a refactor that computes the same numbers passes unchanged)."""

import hashlib

import torch
from torch.utils import _pytree

from bnb import game
from bnb.game.api import PRESETS
from bnb.runtime.realtime import Realtime

TICKS = 48
# the game presets the fingerprint plays: the training and rating games (the fleet-size
# presets are these games at other sizes, and a 40v40 match would make the test slow)
FINGERPRINTED = (
    "deathmatch",
    "deathmatch_wide",
    "frontline",
    "real_scale_deathmatch",
    "real_scale_frontline",
    "real_scale_elimination",
    "clover",
)
SIGNIFICANT = 5  # digits kept, so the fingerprint reads the game, not the last bit


def _rounded(tensor: torch.Tensor) -> bytes:
    if tensor.is_floating_point():
        tensor = tensor.double()
        scale = 10.0 ** (SIGNIFICANT - 1 - torch.floor(torch.log10(tensor.abs().clamp_min(1e-30))))
        tensor = torch.where(tensor == 0, 0.0, torch.round(tensor * scale) / scale)
    return tensor.contiguous().numpy().tobytes()


def fingerprint() -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINTED:
        config = PRESETS[name]
        sim = Realtime(config, compiled=False, generator=torch.Generator().manual_seed(1))
        draws = torch.Generator().manual_seed(2)
        for _ in range(TICKS):
            actions = torch.stack(
                [
                    torch.randint(0, 3, (config.num_ships,), generator=draws),
                    torch.randint(0, 7, (config.num_ships,), generator=draws),
                    torch.randint(0, 2, (config.num_ships,), generator=draws),
                ],
                -1,
            ).to(torch.int32)
            obs = sim.step(actions)
        digest.update(name.encode())
        for leaf in _pytree.tree_leaves((sim.state, obs)):
            if isinstance(leaf, torch.Tensor):
                digest.update(_rounded(leaf))
    return digest.hexdigest()[:16]


def test_the_game_version_names_what_the_game_computes():
    assert fingerprint() == game.FINGERPRINT, (
        "the game computes other numbers than bnb.game.FINGERPRINT records: if the change "
        "is meant, raise bnb.game.VERSION and set FINGERPRINT to the new value; if it is a "
        "refactor, it should not change them"
    )
