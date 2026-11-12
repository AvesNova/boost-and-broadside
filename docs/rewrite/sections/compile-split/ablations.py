"""ablations.py IN_DIR OUT_DIR: the act-stage ablations' configs from a resolved bc.json and
ppo.json (configs.sh): the rating lanes off, a single league slot (PPO, with the lanes off:
their matchups name every slot), and a single mode (the mix's frontline pools only, 0.5
each)."""

import copy
import json
import pathlib
import sys

source, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def no_lanes(config):
    config["lanes"]["environments"] = 0


def one_slot(config):
    config["league"]["slots"] = 1


def frontline(config):
    pools = [pool for pool in config["pools"] if pool["mode"] == "frontline"]
    for pool in pools:
        pool["share"] = 1.0 / len(pools)
    config["pools"] = pools


CASES = {
    "bc": {"full": [], "nolanes": [no_lanes], "frontline": [frontline],
           "frontline-nolanes": [frontline, no_lanes]},
    "ppo": {"full": [], "nolanes": [no_lanes], "nolanes-slot1": [no_lanes, one_slot],
            "frontline": [frontline], "frontline-nolanes-slot1": [frontline, no_lanes, one_slot]},
}
for stage, cases in CASES.items():
    base = json.loads((source / f"{stage}.json").read_text())
    for name, changes in cases.items():
        config = copy.deepcopy(base)
        for change in changes:
            change(config)
        (out / f"{stage}-{name}.json").write_text(json.dumps(config))
        print(stage, name)
