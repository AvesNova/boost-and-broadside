"""Turn-performance helpers around lean.game.core.flight_tick (no fields), for the charts in
this folder: the game's flight model with every card type in the lineup, so a tool can fly any
type by its card index."""
import os
from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
CONFIG = core.GameConfig(lineup=tuple(card.TYPE_NAMES))
TYPES = range(len(card.TYPE_NAMES))
THEME = os.environ.get("THEME", "light")  # "dark": the dark-mode palette and surfaces
SUFFIX = "-core" + ("-dark" if THEME == "dark" else "")  # output file names
H = CONFIG.tick_length
_SIZES = core.per_type_constants(CONFIG, torch.arange(len(card.TYPE_NAMES)))
MAX_ENERGY = [float(e) for e in _SIZES.max_energy]  # J/kg per ship type
SPAWN = [float(e) for e in _SIZES.spawn_energy]
FADE_BAND = CONFIG.store_edge_band  # share of the store


def make(ship_type, speeds, energy):
    b = speeds.shape[0]
    types = torch.full((b, 1), ship_type, dtype=torch.long)
    constants = core.ship_constants(CONFIG, types)
    z = torch.zeros(b, 1)
    s = SimpleNamespace(
        x=z.clone(), y=z.clone(), heading_cos=torch.ones(b, 1), heading_sin=z.clone(),
        speed=speeds.reshape(b, 1).float().clone(), angle=z.clone(), angle_cos=torch.ones(b, 1),
        angle_sin=z.clone(), nose_rate=z.clone(),
        energy=(energy if torch.is_tensor(energy) else torch.full((b, 1), float(energy))).reshape(b, 1).float().clone(),
        log_index=z.clone(), gradient_x=z.clone(), gradient_y=z.clone(),
    )
    return constants, s


def track(s):
    # velocity direction = heading rotated by -angle of attack
    return torch.atan2(s.heading_sin, s.heading_cos) - s.angle


def no_fields(envs: int) -> torch.Tensor:
    """The fields argument of `core.flight_tick` for no fields: (E, 5, 0)."""
    return torch.zeros(envs, 5, 0)

