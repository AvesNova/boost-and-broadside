"""probe1: one card field at a time around E (Jujak turn x0.95, Yongma turn x1.12, Yongma
shield x0.55), to measure each field's kill-share profile across the modes and sizes (which
fields move Frontline 10v10 differently from deathmatch). Raw card factors, not handles: a
probe, not a candidate. Writes candidates/probe1.json."""

import copy
import json
import pathlib

HERE = pathlib.Path(__file__).parent
E = next(
    e
    for e in json.loads((HERE / "options.json").read_text())
    if e["name"] == "jt0.95-yt1.12-ys0.55"
)

PROBES = [  # (name, side, {field: factor})
    ("E", None, {}),
    ("E-again", None, {}),
    ("Y-shield0.9", "energy", {"max_shield": 0.9}),
    ("Y-recharge2", "energy", {"shield_recharge_rate_scale": 2.0}),
    ("Y-delay0.5", "energy", {"shield_recharge_delay_scale": 0.5}),
    ("Y-vision1.3", "energy", {"vision_range_scale": 1.3}),
    ("Y-power1.1", "energy", {"engine_power": 1.1, "engine_power_full_store": 1.1}),
    ("Y-drag0.85", "energy", {"zero_lift_drag": 0.85}),
    ("Y-damage1.1", "energy", {"hit_damage": 1.1}),
    ("Y-range1.2", "energy", {"bullet_life": 1.2}),
    ("Y-muzzle1.15", "energy", {"muzzle_speed": 1.15}),
    ("Y-hitradius0.85", "energy", {"hit_radius": 0.85, "core_radius": 0.85}),
    ("Y-explosive1", "energy", {"explosive_share": 2.0}),
    ("Y-store1.3", "energy", {"max_energy": 1.3}),
    ("Y-spawnenergy1.5", "energy", {"spawn_energy": 1.5}),
    ("Y-shotcost0.7", "energy", {"bullet_energy_cost": 0.7}),
    ("Y-maxlift1.1", "energy", {"max_lift_coefficient": 1.1}),
    ("Y-induced0.85", "energy", {"induced_drag_scale": 0.85}),
    ("J-shield1.15", "turn", {"max_shield": 1.15}),
    ("J-damage0.9", "turn", {"hit_damage": 0.9}),
    ("J-vision0.8", "turn", {"vision_range_scale": 0.8}),
    ("J-power0.9", "turn", {"engine_power": 0.9, "engine_power_full_store": 0.9}),
    ("J-recharge0.5", "turn", {"shield_recharge_rate_scale": 0.5}),
    ("J-range0.85", "turn", {"bullet_life": 0.85}),
]


def main() -> None:
    out = []
    for name, side, factors in PROBES:
        entry = copy.deepcopy(E)
        entry["name"] = f"p1-{name}"
        entry["probe"] = {"side": side, "factors": factors}
        for field, factor in factors.items():
            card = entry["card"][side]
            if isinstance(card.get(field), dict):
                card[field] = {"set": card[field]["set"] * factor}
            else:
                card[field] = factor
        out.append(entry)
    (HERE / "probe1.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main()
