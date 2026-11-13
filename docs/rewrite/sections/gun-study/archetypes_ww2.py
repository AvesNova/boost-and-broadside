"""Turn fighters against energy fighters, from the set's research (aircraft.json): each type's
quantities, the derived turn, dive and zoom figures, and each archetype's geometric mean
over the 18 researched types' (the fleet) and over the P-51D's. Writes archetypes_ww2.json.

Derived (sea-level density 1.225, clean, no thrust where it says so):
  corner speed       Vc = Vs sqrt(n)           (n the limit load factor)
  instant turn       g sqrt(n^2 - 1) / Vc      at the corner speed
  instant turn 100   g sqrt(m^2 - 1) / 100     at 100 m/s, m = min(n, (100 / Vs)^2)
  sustained radius   V / omega                 at the research model's best-rate speed
  drag per g at 200  rho V^2 C_D0 / (2 W/S g)  at 200 m/s (the dive's deceleration in g)
  ballistic coeff.   (W/S) / C_D0              kg/m^2
  zoom height        ln((g + k v0^2) / (g + k v1^2)) / (2 k), k = rho C_D0 / (2 W/S):
                     unpowered vertical zoom 200 -> 60 m/s
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
RESEARCH = HERE.parents[2] / "explanation/game/real-scale-research/aircraft.json"
G, RHO = 9.81, 1.225

GROUPS = {
    "turn": ["spitfire_lf9", "spitfire_14", "a6m5", "yak3", "ki84"],
    "energy": ["p47d", "fw190a8", "fw190d9", "f4u1d", "tempest5"],
    "extreme energy": ["me262a1a"],
    "mixed": ["p51d"],
}
MIXED_IN_GROUP = {
    "spitfire_14": "turn, leaning mixed",
    "yak3": "turn, light and fast low down",
    "ki84": "mixed",
    "tempest5": "energy, low altitude",
}


def value(x, key):
    entry = x.get(key)
    return None if not isinstance(entry, dict) else entry.get("value")


def row(x):
    from bnb.reference.real_planes import AIRCRAFT

    by_name = {a.key: a for a in AIRCRAFT}
    key = {
        "spitfire_lf9": "spitfire-ix",
        "spitfire_14": "spitfire-xiv",
        "tempest5": "tempest-v",
        "mosquito_fb6": "mosquito",
        "fw190a8": "fw190-a8",
        "fw190d9": "fw190-d9",
        "bf109k4": "bf109-k4",
        "me262a1a": "me262",
        "me163b": "me163",
        "f4u1d": "f4u-1d",
        "f6f5": "f6f-5",
    }.get(x["id"], x["id"])
    sustained = by_name[key].data["sustained_turn_rate"].value  # the set's (literature or model)
    stall, n = value(x, "stall_speed_clean_power_off"), value(x, "g_limit")
    wing_loading, cd0 = value(x, "wing_loading"), value(x, "cd0")
    corner = stall * math.sqrt(n)
    m100 = min(n, (100 / stall) ** 2)
    k = RHO * cd0 / (2 * wing_loading)
    model_rate, model_speed = (
        value(x, "sustained_turn_rate_sea_level"),
        value(x, "sustained_turn_speed"),
    )
    return dict(
        name=x["name"],
        mass=value(x, "combat_mass"),
        wing_loading=wing_loading,
        power_loading=value(x, "power_loading"),
        top_speed_sea_level=value(x, "max_speed_sea_level_military"),
        top_speed_altitude=value(x, "max_speed_altitude"),
        climb_rate=value(x, "climb_rate_sea_level"),
        service_ceiling=value(x, "service_ceiling"),
        low_speed_acceleration=by_name[key].data["low_speed_acceleration"].value,
        stall_speed=stall,
        g_limit=n,
        sustained_turn=sustained,
        sustained_radius=model_speed / math.radians(model_rate),
        corner_speed=corner,
        instant_turn_corner=math.degrees(G * math.sqrt(n * n - 1) / corner),
        instant_radius_corner=corner**2 / (G * math.sqrt(n * n - 1)),
        instant_turn_100=math.degrees(G * math.sqrt(max(m100 * m100 - 1, 0)) / 100),
        roll_rate=value(x, "roll_rate_peak"),
        dive_limit=value(x, "dive_limit_speed"),
        ballistic_coefficient=wing_loading / cd0,
        drag_per_g_200=k * 200**2 / G,
        zoom_height=math.log((G + k * 200**2) / (G + k * 60**2)) / (2 * k),
        weight_of_fire=value(x, "weight_of_fire"),
        survivability=x["survivability"]["rating"],
        half_span=value(x, "wing_span") / 2,
    )


def geometric_mean(values):
    values = [v for v in values if v]
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    aircraft = json.load(open(RESEARCH))["aircraft"]
    rows = {x["id"]: row(x) for x in aircraft}
    quantities = list(next(iter(rows.values())))[1:]
    fleet = {q: geometric_mean(r[q] for r in rows.values()) for q in quantities}
    groups = {
        g: {q: geometric_mean(rows[i][q] for i in ids) for q in quantities}
        for g, ids in GROUPS.items()
    }
    ratios = {g: {q: groups[g][q] / fleet[q] for q in quantities} for g in groups}
    turn_over_energy = {q: groups["turn"][q] / groups["energy"][q] for q in quantities}
    over_mixed = {g: {q: groups[g][q] / rows["p51d"][q] for q in quantities} for g in groups}
    out = dict(
        groups=GROUPS,
        notes=MIXED_IN_GROUP,
        rows=rows,
        fleet=fleet,
        group_means=groups,
        over_fleet=ratios,
        over_p51d=over_mixed,
        turn_over_energy=turn_over_energy,
    )
    json.dump(out, open(HERE / "archetypes_ww2.json", "w"), indent=1)
    show = [i for ids in GROUPS.values() for i in ids]
    print(
        "| quantity | "
        + " | ".join(rows[i]["name"].split("(")[0] for i in show)
        + " | fleet | turn / fleet | energy / fleet | turn / P-51D | energy / P-51D"
        + " | turn / energy |"
    )
    for q in quantities:
        cells = [f"{rows[i][q]:.3g}" for i in show]
        print(
            f"| {q} | "
            + " | ".join(cells)
            + f" | {fleet[q]:.3g} | {ratios['turn'][q]:.2f} | {ratios['energy'][q]:.2f}"
            + f" | {over_mixed['turn'][q]:.2f} | {over_mixed['energy'][q]:.2f}"
            + f" | {turn_over_energy[q]:.2f} |"
        )


if __name__ == "__main__":
    main()
