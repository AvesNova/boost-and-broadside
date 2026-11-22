"""Turn-performance helpers around lean.game.core.flight_tick (no fields).

The flight model comes from the environment variable FLIGHT_MODEL: "v2" (the default) or
"core" (GameConfig.core()). Every script in this folder reads it from here.
"""
import math
import os
from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
MODEL = os.environ.get("FLIGHT_MODEL", "v2")
if MODEL not in ("v2", "core"):
    raise ValueError(f"FLIGHT_MODEL={MODEL!r}: need 'v2' or 'core'")
CONFIG = core.GameConfig.core() if MODEL == "core" else core.GameConfig()
THEME = os.environ.get("THEME", "light")  # "dark": the dark-mode palette and surfaces
SUFFIX = ("-core" if MODEL == "core" else "") + ("-dark" if THEME == "dark" else "")  # output file names
H = CONFIG.tick_length
_SIZES = core.ship_constants(CONFIG, torch.arange(3))
MAX_ENERGY = [float(e) for e in _SIZES["max_energy"]]  # J/kg per ship type, this model's store
SPAWN = [float(e) for e in _SIZES["spawn_energy"]]
FADE_BAND = CONFIG.store_edge_band if MODEL == "core" else card.STORE_FADE_BAND  # share of the store


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


def run(ship_type, speeds, energy, stick, power, ticks, every=1):
    """Return per-tick tensors (T, B): speed, energy, turn rate (rad/s, velocity), angle."""
    constants, s = make(ship_type, speeds, energy)
    b = speeds.shape[0]
    stick_t = torch.full((b, 1), stick, dtype=torch.long)
    power_t = torch.full((b, 1), power, dtype=torch.long)
    out = {"speed": [], "energy": [], "rate": [], "angle": [], "nose": []}
    previous = track(s)
    with torch.no_grad():
        for _ in range(ticks):
            s = core.flight_tick(constants, H, s, stick_t, power_t, model=MODEL)
            now = track(s)
            d = torch.remainder(now - previous + math.pi, 2 * math.pi) - math.pi
            previous = now
            out["speed"].append(s.speed[:, 0].clone())
            out["energy"].append(s.energy[:, 0].clone())
            out["rate"].append(d[:, 0] / H)
            out["angle"].append(s.angle[:, 0].clone())
            out["nose"].append(s.nose_rate[:, 0].clone())
    return {k: torch.stack(v) for k, v in out.items()}


def run_batch(types, speeds, energies, sticks, powers, ticks, keep_every=1, magnitude=None):
    """Every argument is a (B,) tensor; rows may mix ship types. Records every keep_every ticks."""
    b = speeds.shape[0]
    types = types.reshape(b, 1).long()
    constants = core.ship_constants(CONFIG, types)
    if magnitude is not None:  # scratch override: stick index 6 gets a continuous magnitude
        constants['stick_table'] = constants['stick_table'].clone()
        constants['stick_table'][:, 0, 6] = magnitude
    z = torch.zeros(b, 1)
    s = SimpleNamespace(
        x=z.clone(), y=z.clone(), heading_cos=torch.ones(b, 1), heading_sin=z.clone(),
        speed=speeds.reshape(b, 1).float().clone(), angle=z.clone(), angle_cos=torch.ones(b, 1),
        angle_sin=z.clone(), nose_rate=z.clone(), energy=energies.reshape(b, 1).float().clone(),
        log_index=z.clone(), gradient_x=z.clone(), gradient_y=z.clone(),
    )
    stick_t, power_t = sticks.reshape(b, 1).long(), powers.reshape(b, 1).long()
    out = {"speed": [], "energy": [], "rate": [], "angle": []}
    previous = track(s)
    with torch.no_grad():
        for k in range(ticks):
            s = core.flight_tick(constants, H, s, stick_t, power_t, model=MODEL)
            now = track(s)
            d = torch.remainder(now - previous + math.pi, 2 * math.pi) - math.pi
            previous = now
            if (k + 1) % keep_every == 0:
                out["speed"].append(s.speed[:, 0].clone())
                out["energy"].append(s.energy[:, 0].clone())
                out["rate"].append(d[:, 0] / H)
                out["angle"].append(s.angle[:, 0].clone())
    return {k: torch.stack(v) for k, v in out.items()}
