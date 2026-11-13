"""Game v13: the real-scale scripted agent's time-unit parameters at the set's x2 native
speed (times / 2, rates x 2), so it plays the x2 game as it played the x1 one. The
logistics are in log x, so only their centres move. Run once, after native_x2.py:

    uv run --no-sync python docs/rewrite/sections/lethality/scripted_x2.py

Parameters (`ScriptedParameters`): memory_fade_in_seconds' centre / 2; every switch's rate
x 2. Doctrines (`Doctrine`): lead_turn_in_seconds' centre / 2, the yo-yo's seconds / 2
(their weights are 0 in every stored doctrine today). Files: the real-scale parameter sets
(scripted_real_scale.json), the real-scale doctrines (scripted_doctrines.json, the play
shell's doctrines), the game-loop league and the maneuvers champions.
"""

import dataclasses
import json
import pathlib

from bnb.controllers.scripted_parameters import Switch, from_row, pack

ROOT = pathlib.Path(__file__).resolve().parents[4]
FACTOR = 2.0
SWITCHES = (
    "target_switch_in_log_distance",
    "side_switch_in_radians",
    "goal_switch_in_strength",
    "recovery_switch_in_log_ratio",
)
DOCTRINE_SECONDS = ("yoyo_after_seconds", "yoyo_climb_seconds", "yoyo_dive_seconds")


def parameters_named(p: dict) -> dict:
    p = dict(p)
    centre, scale = p["memory_fade_in_seconds"]
    p["memory_fade_in_seconds"] = [centre / FACTOR, scale]
    for name in SWITCHES:
        rate, centre, scale = p[name]
        p[name] = [rate * FACTOR, centre, scale]
    return p


def parameters_row(row: list[float]) -> list[float]:
    values = from_row(row)
    fade = values.memory_fade_in_seconds
    changed = dict(memory_fade_in_seconds=fade._replace(centre=fade.centre / FACTOR))
    for name in SWITCHES:
        switch = getattr(values, name)
        changed[name] = Switch(switch.rate * FACTOR, switch.centre, switch.scale)
    return pack(dataclasses.replace(values, **changed))


def doctrine_named(d: dict) -> dict:
    d = dict(d)
    if "lead_turn_in_seconds" in d:
        centre, scale = d["lead_turn_in_seconds"]
        d["lead_turn_in_seconds"] = [centre / FACTOR, scale]
    for name in DOCTRINE_SECONDS:
        if name in d:
            d[name] = d[name] / FACTOR
    return d


def rewrite(path: pathlib.Path, change) -> None:
    data = json.loads(path.read_text())
    path.write_text(json.dumps(change(data), indent=1) + "\n")
    print("scaled", path.relative_to(ROOT))


def main() -> None:
    controllers = ROOT / "src/bnb/controllers"
    rewrite(
        controllers / "scripted_real_scale.json",
        lambda d: {m: dict(v, parameters=parameters_named(v["parameters"])) for m, v in d.items()},
    )

    def real_scale_doctrines(d):
        for kind, entry in d.get("real-scale", {}).items():
            entry["doctrine"] = doctrine_named(entry["doctrine"])
        return d

    rewrite(controllers / "scripted_doctrines.json", real_scale_doctrines)
    for path in sorted((ROOT / "src/bnb/play/doctrines").glob("*.json")):
        rewrite(path, real_scale_doctrines)

    def player(entry):
        entry = dict(entry)
        for kind in ("yongma", "jujak"):
            if kind in entry:
                entry[kind] = doctrine_named(entry[kind])
        if "parameters" in entry:
            entry["parameters"] = parameters_row(entry["parameters"])
        return entry

    rewrite(ROOT / "docs/rewrite/sections/game-loop/league.json", lambda d: [player(e) for e in d])
    for path in sorted(ROOT.glob("docs/rewrite/sections/maneuvers*/champion*.json")):
        rewrite(path, player)


if __name__ == "__main__":
    main()
