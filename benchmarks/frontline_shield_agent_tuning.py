"""Side-balanced recovery-threshold sweep against the pre-shield controller.

Both controllers play the CURRENT environment, with identical seeds and production
ship settings. The prior controller source is pinned to the overhaul base commit.
Four games per threshold is a tuning screen, not an Elo estimate.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontline_agent_head_to_head import play_pair


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Prior(StochasticScriptedAgent):
    def get_actions(self, state, team_visibility=None):
        return self.old.get_actions(state, team_visibility)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_root = Path(tempfile.mkdtemp(prefix="frontline-prior-agent-"))
    for filename in ("frontline_strategy.py", "stochastic_scripted.py"):
        source = subprocess.check_output(
            ["git", "show", f"85f2bf3:src/boost_and_broadside/agents/{filename}"]
        )
        (old_root / filename).write_bytes(source)
    oldstrategy = module("oldstrategy", old_root / "frontline_strategy.py")
    oldagent = module("oldagent", old_root / "stochastic_scripted.py")
    oldagent.frontline_strategy = oldstrategy.frontline_strategy

    torch.set_num_threads(1)
    ship = frontline_ship_config(SHIP_CONFIG)
    prior = Prior(ship, StochasticAgentConfig())
    prior.old = oldagent.StochasticScriptedAgent(ship, StochasticAgentConfig())
    results = []
    for recovery in (0.3, 0.5, 0.7):
        torch.manual_seed(9751)
        candidate = StochasticScriptedAgent(
            ship, StochasticAgentConfig(frontline_recovery_health=recovery)
        )
        result = play_pair(
            ResolvedAgent("scripted", candidate),
            ResolvedAgent("scripted", prior),
            games=4,
            team_size=5,
            seed=9751,
            seconds=120,
            ship_config=ship,
            env_config=PLAY_ENV_CONFIG,
        )
        result["recovery"] = recovery
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2))
        print(recovery, result, flush=True)


if __name__ == "__main__":
    main()
