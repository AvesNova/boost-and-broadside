"""Loose Elo of the scripted agent after the pending-tick fix (lean.scripted) against the agent
before it (old_opponent.py, core 3f9769db), both with the round-3 robust centre rows, at 1v1,
4v4 and 16v16 with the study's game (fields and fog on, the study clocks: skill2.config_of).

    elo.py [--games 2048 4096 2048] [--minutes 25] [--envs 2048] [--out elo.json]

The runner plays every ship through `lean.scripted.decide`; here it is wrapped so a ship whose
stickiness row carries the old agent's marker (the centre's stickiness x (1 + 1e-4), a change
of nothing measurable, only so the two agents have different ids) takes the old agent's action
and every other ship the new one's (both computed for every ship, same uniforms). Games stored
in DATA/elo.sqlite per batch (resumable; `--minutes` bounds the call), Elo by lean.eval's
Bradley-Terry fit with the old agent at 1000.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lookahead-personas"))
import old_opponent  # noqa: E402
import skill2  # noqa: E402

from lean import scripted  # noqa: E402
from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402

DATA = Path("/home/vizia/avesnova/boost-and-broadside/data/scripted-delay")
CENTRE = HERE.parent / "lookahead-personas" / "candidates" / "robust-centre-3.json"
BATCH = 1024


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, nargs=3, default=[2048, 4096, 2048])
    parser.add_argument("--sizes", type=int, nargs=3, default=[1, 4, 16])
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--out", default=str(HERE / "elo.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    rows = json.loads(CENTRE.read_text())
    new = ScriptedAgent.of(ScriptedConfig().replace(**rows))
    marker = float(torch.tensor(rows["stickiness"] * (1.0 + 1e-4), dtype=torch.float32))
    old = ScriptedAgent.of(ScriptedConfig().replace(**{**rows, "stickiness": marker}))
    assert new.id != old.id
    new_decide = scripted.decide

    def both(config, state, team, persona, uniforms):
        action, choice = new_decide(config, state, team, persona, uniforms)
        old_action, _ = old_opponent.decide(config, state, team, persona, uniforms)
        is_old = (persona["stickiness"] == marker)[..., None]
        return torch.where(is_old, old_action, action), choice

    scripted.decide = both
    DATA.mkdir(parents=True, exist_ok=True)
    store = Store(DATA / "elo.sqlite")
    result = {}
    for size, target in zip(args.sizes, args.games, strict=True):
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=args.envs, minimum_envs=args.envs, seed=8800 + size
        )
        sizes = (size, size)
        while True:
            have = evaluator.counts(new, old, sizes)["games"]
            if have >= target:
                break
            if (time.perf_counter() - started) / 60.0 > args.minutes:
                log("time bound")
                break
            evaluator.ensure([(new, old, sizes, min(target, have + BATCH))])
            counts = evaluator.counts(new, old, sizes)
            log(f"{size}v{size}: {counts}, {time.perf_counter() - started:.0f} s")
        counts = evaluator.counts(new, old, sizes)
        elo = {i: (r, e) for i, r, e in evaluator.elo(sizes, anchor=old, agents=[new, old])}
        rating, error = elo.get(new.id, (None, None))
        result[f"{size}v{size}"] = {"elo_new": rating, "error": error, **counts}
        log(f"{size}v{size}: new {rating} +- {error} against old 1000, {counts}")
        Path(args.out).write_text(json.dumps(result, indent=1))
    log("done")


if __name__ == "__main__":
    main()
