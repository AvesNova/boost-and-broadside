"""Tables of the gradient diagnostic from a lean.train log's `gradients_json` lines.

    uv run --no-sync python docs/rewrite/sections/gradients-11/tables.py LOG
"""

import json
import sys

TOP = ("surrogate", "bc", "teacher", "entropy", "value", "next_state", "hit", "pair", "field", "edge")


def cosine(record: dict, a: str, b: str) -> float:
    for key in (f"gradient/cosine/trunk/{a}/{b}", f"gradient/cosine/trunk/{b}/{a}"):
        if key in record:
            return record[key]
    return float("nan")


def main(path: str) -> None:
    records = []
    for line in open(path):
        if " gradients_json " in line:
            stage, _, index = line.split()[:3]
            records.append((stage, int(index), json.loads(line.split(" gradients_json ", 1)[1])))
    for stage in ("bc", "rl"):
        rows = [(index, record) for s, index, record in records if s == stage]
        if not rows:
            continue
        policy = "bc" if stage == "bc" else "surrogate"
        names = [
            name.removeprefix("gradient/share/trunk/")
            for name in rows[0][1]
            if name.startswith("gradient/share/trunk/")
        ]
        print(f"\n## {stage}: per term, per update (trunk share / trunk norm / cosine with "
              f"{policy} / coherent fraction; model share)\n")
        updates = [index for index, _ in rows]
        print("| term | " + " | ".join(f"update {u}" for u in updates) + " |")
        print("|---|" + "---|" * len(updates))
        for name in names:
            cells = []
            for _, r in rows:
                cells.append(
                    f"{r[f'gradient/share/trunk/{name}']:.3f} / {r[f'gradient/norm/trunk/{name}']:.2e}"
                    f" / {cosine(r, policy, name) if name != policy else 1.0:.3f}"
                    f" / {r[f'gradient/coherent_fraction/{name}']:.3f}; {r[f'gradient/share/model/{name}']:.3f}"
                )
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("\n| measure | " + " | ".join(f"update {u}" for u in updates) + " |")
        print("|---|" + "---|" * len(updates))
        extra = [k for k in rows[0][1] if k.startswith(("gradient/agreement", "gradient/actor", "gradient/critic", "gradient/coherent_share"))]
        for key in extra:
            print(f"| {key.removeprefix('gradient/')} | " + " | ".join(f"{r[key]:.3f}" for _, r in rows) + " |")
        # mean over updates, top-level terms
        print(f"\n### {stage}: means over the {len(rows)} updates (top level)\n")
        print(f"| term | trunk share | trunk norm | cosine with {policy} | coherent | model share |")
        print("|---|---|---|---|---|---|")
        for name in names:
            if name not in TOP:
                continue
            def mean(f):
                values = [f(r) for _, r in rows]
                return sum(values) / len(values)
            print(
                f"| {name} | {mean(lambda r: r[f'gradient/share/trunk/{name}']):.3f} | "
                f"{mean(lambda r: r[f'gradient/norm/trunk/{name}']):.2e} | "
                f"{1.0 if name == policy else mean(lambda r: cosine(r, policy, name)):.3f} | "
                f"{mean(lambda r: r[f'gradient/coherent_fraction/{name}']):.3f} | "
                f"{mean(lambda r: r[f'gradient/share/model/{name}']):.3f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1])
