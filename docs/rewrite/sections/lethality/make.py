"""Game v12's doctrine round: r9m2's extension gates on the new lethality rules, on the
ship files as they are (the "frozen" card: no factors). Fields over r9m (maneuvers-2's
make.py), a tuple for the Yongma's and the Jujak's.

    uv run --no-sync python docs/rewrite/sections/lethality/make.py round1
"""

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("m2", HERE.parent / "maneuvers-2" / "make.py")
m2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m2)
R9M, TEAM, de = m2.R9M, m2.TEAM, m2.de

NO_EXTEND = dict(extend=0.0)  # r9's own extension element off
NO_CROWD = dict(crowd_sense=0.0)  # r9m's crowd extension off


def gates(yongma: float, jujak: float, *extra: dict) -> tuple[dict, dict]:
    """The team-gated duel extension at these shield ratios (r9m2: 1.0 and 0.4)."""
    more = {k: v for e in extra for k, v in e.items()}
    return dict(de(yongma), **TEAM, **more), dict(de(jujak), **TEAM, **more)


ROUNDS = {
    "round1": [
        ("r9m", {}),
        ("r9m2", gates(1.0, 0.4)),
        ("DE Y 0.7 / J 0.25", gates(0.7, 0.25)),
        ("DE Y 0.5 / J 0.1", gates(0.5, 0.1)),
        ("r9m2 no crowd ext", gates(1.0, 0.4, NO_CROWD)),
        ("r9m2 no r9 extend", gates(1.0, 0.4, NO_EXTEND)),
        ("press (no extension)", dict(NO_CROWD, **NO_EXTEND)),
    ],
}


# round 10: the Frontline lock and wave options over r9m2, on the card the balance search
# chose (Yongma lift x1.5 + damage x1.5)
CARD = {"max_lift_coefficient": 1.5, "hit_damage": 1.5}
LOCK = dict(lock_press=1.0)
WAVE = dict(wave_hold=1.0)
ROUNDS["round10"] = [
    ("r9m2", gates(1.0, 0.4)),
    ("r9m2 + lock 1", gates(1.0, 0.4, LOCK)),
    ("r9m2 + lock 0.5", gates(1.0, 0.4, dict(lock_press=0.5))),
    ("r9m2 + wave 30 s", gates(1.0, 0.4, WAVE)),
    ("r9m2 + wave 10 s", gates(1.0, 0.4, dict(WAVE, wave_hold_seconds=10.0))),
    ("r9m2 + lock 1 + wave 30 s", gates(1.0, 0.4, LOCK, WAVE)),
]
CARDS = {"round10": CARD}


def main() -> None:
    name = sys.argv[1]
    out = []
    card = CARDS.get(name, {})
    for label, fields in ROUNDS[name]:
        pair = fields if isinstance(fields, tuple) else (fields, fields)
        out.append(
            dict(
                name=label,
                energy=card,
                turn={},
                doctrine=dict(energy=dict(R9M, **pair[0]), turn=dict(R9M, **pair[1])),
            )
        )
    path = HERE / name / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
