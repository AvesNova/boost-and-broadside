"""Round 3: the revised sampler's 24 personas (group sampler2) with the flight temperature
reflected to the cold side of the round-3 centre (a draw above the centre's temperature is
mirrored below it; every other row unchanged): group sampler2c, a paired test of the one-sided
temperature (step 1: flat to 0.001 below, a cliff above).

    reflect3.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3  # noqa: E402

HERE = Path(__file__).resolve().parent

centre = json.loads((HERE / "candidates" / "robust-centre-3.json").read_text())["temperature"]
group = {}
for label, rows in round3.read_group("sampler2").items():
    rows = dict(rows)
    rows["temperature"] = centre * math.exp(-abs(math.log(rows["temperature"] / centre)))
    group[label.replace("gauss50b", "gauss50c")] = rows
round3.write_group("sampler2c", group)
