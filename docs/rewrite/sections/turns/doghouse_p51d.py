"""Doghouse charts of lean's own types, the P-51D and Vizia with v2's climb modifications
reverted (docs/rewrite/p51d-model.md), in game units and in real units, with doghouse.py's
layout and functions.

    for theme in light dark; do FLIGHT_MODEL=core THEME=$theme OMP_NUM_THREADS=1 PYTHONPATH=.:src \\
        python docs/rewrite/sections/turns/doghouse_p51d.py; done
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from doghouse import doghouse  # noqa: E402

from lean.game import card  # noqa: E402

if __name__ == "__main__":
    for name in ("p51d", "vizia_plain"):
        for real in (False, True):
            print(doghouse(card.TYPE_NAMES.index(name), real=real))
