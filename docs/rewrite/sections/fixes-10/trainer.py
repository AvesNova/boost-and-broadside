"""The real trainer (lean.train.main) with LearnConfig overrides, its update timed alone
(queue item 10's smoke runs; from item 6; also the store's bytes after each update).

    PYTHONPATH=. python docs/rewrite/sections/fixes-10/trainer.py NAME [field=value ...] -- [lean.train arguments]

A cold inductor cache of its own (DATA/cache/NAME, deleted at the end). Prints the machine's
state, then the trainer's lines, plus `timed update K: S s` (the update alone, synchronised)
and `timed collect K: S s`.
"""

import os
import shutil
import subprocess
import sys

DATA = "/home/vizia/avesnova/boost-and-broadside/data/fixes10"
NAME = sys.argv[1]
split = sys.argv.index("--")
OVERRIDES = dict(item.split("=", 1) for item in sys.argv[2:split])
CACHE = os.path.join(DATA, "cache", NAME)
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(CACHE)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
os.environ["TRITON_CACHE_DIR"] = os.path.join(CACHE, "triton")

import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import lean.train as train  # noqa: E402
from lean.learn import LearnConfig  # noqa: E402

units = subprocess.run(
    ["systemctl", "--user", "list-units", "bnb-*", "--state=active", "--no-legend", "--plain"],
    capture_output=True,
    text=True,
).stdout
print(
    json.dumps(
        {
            "job": NAME,
            "overrides": OVERRIDES,
            "power": subprocess.run(
                ["powerprofilesctl", "get"], capture_output=True, text=True
            ).stdout.strip(),
            "load": os.getloadavg(),
            "bnb_units": [line.split()[0] for line in units.splitlines() if line.strip()],
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
        }
    ),
    flush=True,
)
fields = {field.name: field for field in dataclasses.fields(LearnConfig)}


def configured(**keywords):
    values = {name: type(fields[name].default)(value) for name, value in OVERRIDES.items()}
    return LearnConfig(**{**keywords, **values})


train.LearnConfig = configured
timed = {"update": 0, "collect": 0}


def timing(name, function):
    def wrapped(*arguments, **keywords):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = function(*arguments, **keywords)
        torch.cuda.synchronize()
        print(f"timed {name} {timed[name]}: {time.perf_counter() - began:.2f} s", flush=True)
        timed[name] += 1
        return result

    return wrapped


train.update = timing("update", train.update)
timed_update = train.update


def hit_shares(store):
    """Measurement only: the enemy-hit share of the labelled observed rows in the store, all
    of them and only those at least a bullet's life - 1 rows before an episode end or the
    update's end (the rows the item-10 mask keeps; computed here from the store's own records,
    seat-major over the update's segments), and the labelled row counts."""
    from lean.contract import BULLET_SLOTS

    label = store.resident["hit_label"][:, : store.views].long()
    observed = store.resident["observed"][:, : store.views]
    done = store.resident["done"][:, : store.views]
    ticks, segments, seats = store.ticks, store.segments, store.seats
    chained = done.view(ticks, segments, seats).transpose(0, 1).reshape(segments * ticks, seats)
    reach = BULLET_SLOTS - 1
    padded = torch.cat([chained, chained.new_ones((reach, seats))])
    near = torch.zeros_like(chained)
    for offset in range(reach):
        near |= padded[offset : offset + segments * ticks]
    near |= torch.cat(
        [chained.new_zeros((segments * ticks - reach, seats)), chained.new_ones((reach, seats))]
    )
    near = near.view(segments, ticks, seats).transpose(0, 1).reshape(ticks, segments * seats)
    labelled = observed & (label >= 0)
    far = labelled & ~near[..., None]
    enemy = label == 1
    return (
        float((enemy & labelled).sum() / labelled.sum().clamp(min=1)),
        float((enemy & far).sum() / far.sum().clamp(min=1)),
        int(labelled.sum()),
        int(far.sum()),
    )


def update_and_store(config, policy, optimiser, store, **keywords):
    shares = hit_shares(store)
    print(
        f"hit share labelled {shares[0]:.5f} ({shares[2]} rows), away from ends "
        f"{shares[1]:.5f} ({shares[3]} rows)",
        flush=True,
    )
    result = timed_update(config, policy, optimiser, store, **keywords)
    sizes = store.nbytes()
    print(
        f"store bytes total {sum(sizes.values()) / 2**20:.1f} MiB, teacher "
        f"{sizes.get('teacher', 0) / 2**20:.1f} MiB, teacher_weight "
        f"{keywords.get('teacher_weight')}",
        flush=True,
    )
    return result


train.update = update_and_store
train.collect = timing("collect", train.collect)
sys.argv = ["lean.train"] + sys.argv[split + 1 :]
try:
    train.main()
finally:
    shutil.rmtree(CACHE, ignore_errors=True)
