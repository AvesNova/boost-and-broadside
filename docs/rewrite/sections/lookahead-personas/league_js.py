"""Round 2 (revised brief): how differently the league candidates behave, on the state sets
around the robust centre (blur2.py states), per size: the mean per-state JS divergence of each
pair's joints, and each one's against the centre and the old yardstick.

    league_js.py DATA OUT.json NAME ...      # names of candidates/NAME.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blur2  # noqa: E402
import divergence  # noqa: E402

from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import vector  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    data, out, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    configs = {
        n: ScriptedConfig().replace(**json.loads((HERE / "candidates" / f"{n}.json").read_text()))
        for n in names
    }
    result = {}
    for size in blur2.SIZES:
        states = blur2.States(data, size)
        joints = {n: states.joint(vector(c)) for n, c in configs.items()}
        matrix = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j > i:
                    matrix[i, j] = matrix[j, i] = float(divergence.js(joints[a], joints[b]).mean())
        entropy = {n: float(divergence.entropy(p).mean()) for n, p in joints.items()}
        result[f"{size}v{size}"] = {"names": names, "js": matrix.tolist(), "entropy": entropy}
        blur2.log(
            f"{size}v{size}: JS mean off-diagonal {matrix[np.triu_indices(len(names), 1)].mean():.4f}"
        )
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
