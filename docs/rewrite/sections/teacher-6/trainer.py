"""The real trainer (lean.train.main) with LearnConfig overrides, its update timed alone
(queue item 6's smoke runs; also the store's bytes after each update).

    PYTHONPATH=. python docs/rewrite/sections/teacher-6/trainer.py NAME [field=value ...] -- [lean.train arguments]

A cold inductor cache of its own (DATA/cache/NAME, deleted at the end). Prints the machine's
state, then the trainer's lines, plus `timed update K: S s` (the update alone, synchronised)
and `timed collect K: S s`.
"""

import os
import shutil
import subprocess
import sys

DATA = "/home/vizia/avesnova/boost-and-broadside/data/teacher6"
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
            "power": subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout.strip(),
            "load": os.getloadavg(),
            "bnb_units": [line.split()[0] for line in units.splitlines() if line.strip()],
            "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
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


def update_and_store(config, policy, optimiser, store, **keywords):
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
