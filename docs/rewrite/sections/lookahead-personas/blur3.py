"""Round 3: label blur and label spread (JS) of the sampler's personas (GPU or CPU, no games).

    blur3.py OUT.json [--group sampler] [--extra NAME=JSON ...]

On round 2's state sets around its robust centre (DATA2/behaviour-centre/states_*.pt: games
with sampled actions, the centre and spread-1 personas, fields, fog, study clocks). Per set of
personas (the group's labels by prefix: gauss25, gauss50, gauss100, box50) and size: the blur
(mixture entropy minus the members' mean entropy, nats; also as a share of the centre's
entropy), the members' mean JS from the centre, the mean pairwise JS (first 4096 ship-states).
`--extra` personas (league picks): their pairwise JS matrix per size.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blur2  # noqa: E402
import divergence  # noqa: E402
import round3  # noqa: E402

from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import vector  # noqa: E402

STATES = Path("/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2/behaviour-centre")
PAIR_STATES = 4096
PREFIXES = ("gauss25", "gauss50", "gauss100", "box50", "gauss50b", "gauss50c")


def joint(states, rows):
    return states.joint(vector(ScriptedConfig().replace(**rows)))


def pairwise(ps):
    sub = torch.stack([p[:PAIR_STATES] for p in ps])
    n = len(ps)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        values = divergence.js(sub[i][None], sub).mean(-1).tolist()
        out[i] = values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--group", default="sampler")
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--centre", default=str(round3.CENTRE))
    a = parser.parse_args()
    group = round3.read_group(a.group)
    centre = round3.rows_of(a.centre)
    result = {}
    for size in blur2.SIZES:
        states = blur2.States(STATES, size)
        p0 = joint(states, centre)
        h0 = float(divergence.entropy(p0).mean())
        entry = {"centre_entropy": h0, "ship_states": int(p0.shape[0])}
        for prefix in PREFIXES:
            labels = [k for k in group if k.startswith(prefix + "-")]
            if not labels:
                continue
            ps = [joint(states, group[k]) for k in labels]
            mixture = torch.stack(ps).mean(0)
            mean_h = float(torch.stack([divergence.entropy(p).mean() for p in ps]).mean())
            blur = float(divergence.entropy(mixture).mean()) - mean_h
            pw = pairwise(ps)
            n = len(ps)
            entry[prefix] = {
                "personas": n,
                "blur": blur,
                "blur_share": blur / h0,
                "from_centre": float(torch.stack([divergence.js(p, p0).mean() for p in ps]).mean()),
                "pairwise_mean": sum(sum(r) for r in pw) / (n * (n - 1)),
            }
            e = entry[prefix]
            print(
                f"{size}v{size} {prefix}: blur {blur:.4f} ({blur / h0:.1%}), from centre "
                f"{e['from_centre']:.4f}, pairwise {e['pairwise_mean']:.4f}",
                file=sys.stderr,
                flush=True,
            )
        if a.extra:
            names = [e.split("=", 1)[0] for e in a.extra]
            ps = [joint(states, round3.rows_of(e.split("=", 1)[1])) for e in a.extra]
            entry["extra"] = {"names": names, "pairwise": pairwise(ps)}
        result[f"{size}v{size}"] = entry
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
