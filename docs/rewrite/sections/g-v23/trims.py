"""Shield trims of option G on game v23 (G was Yongma-strong at 5v5 and 10v10 by 3-5 kill-share
points): Yongma's and Jujak's max_shield factors, re-solved like solve.py; writes trims.json.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/g-v23/trims.py
"""

import json
import pathlib
import sys

sys.argv = sys.argv[:1]
HERE = pathlib.Path(__file__).parent
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("solve", HERE / "solve.py")
solve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve)  # also rewrites candidates.json (unchanged)

TRIMS = {"Y0.50": (0.50, 1.1586), "Y0.46": (0.46, 1.1586),
         "J1.25": (0.55, 1.25), "Y0.50-J1.22": (0.50, 1.22)}
out = []
for name, (yongma, jujak) in TRIMS.items():
    energy = dict(solve.G["energy"][1], max_shield=yongma)
    turn = dict(solve.G["turn"][1], max_shield=jujak)
    out.append({"name": name, "card": {"energy": solve.solve("yongma", energy),
                                        "turn": solve.solve("jujak", turn)}})
(HERE / "trims.json").write_text(json.dumps(out, indent=1) + "\n")
print("wrote", HERE / "trims.json")
