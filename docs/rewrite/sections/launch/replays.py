"""replays.py: whole matches of a trained policy recorded as play-shell replays, with no
window (`bnb.play.replay`): open one with `bnb play --replay DIR`.

    replays.py --policy RUN/policies/step_N/uniform_average [--policy ...] --out DIR
               [--presets ...] [--matches 2] [--opponent scripted_agent|self]
               [--lineup NAME ...] [--seed 0]

Each match is the policy (team 0) against the opponent (team 1: the scripted agent, or the
policy itself), in every preset, from its first tick to its end. Besides the replays, the
directory gets matches.jsonl: per match its policy, preset, seed, winner, ticks and each
team's kills and boundary deaths (the deaths with no killer), for the reward check.
Runs on the GPU when there is one (the play shell's agents do).
"""

import argparse
import json
import pathlib
import sys

REPOSITORY = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY / "src"))


def match_loop(setup):
    """The match of `setup` as `bnb.play.loop.MatchLoop` builds it, but its agents load with
    `any_mode`: a policy of a mixed batch records the batch's first game only and plays
    every mode of the mix (`bnb.play.sides`)."""
    import torch

    from bnb.play.loop import MatchLoop
    from bnb.play.sides import ScriptedAgent, load_agent
    from bnb.play.tuning import base_rules
    from bnb.runtime.realtime import Realtime

    settings, types = setup.tuning.played(
        base_rules(setup.config, setup.lineup), setup.config.ship_set
    )
    realtime = Realtime(
        setup.config,
        compiled=False,
        fogged=setup.fogged,
        generator=torch.Generator().manual_seed(setup.seed),
        lineup=setup.lineup,
        settings=settings,
        types=types,
    )
    sides = [
        load_agent(
            side.policy,
            setup.config,
            realtime.rules,
            realtime.types,
            side.temperature,
            setup.seed + team,
            any_mode=True,
        )
        if side.controller == "agent"
        else ScriptedAgent(realtime.rules, realtime.types, setup.seed + team)
        for team, side in enumerate(setup.sides)
    ]
    return MatchLoop(setup, sides=sides, realtime=realtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["real_scale_frontline", "real_scale_deathmatch", "real_scale_elimination"],
    )
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--opponent", default="scripted_agent", choices=["scripted_agent", "self"])
    parser.add_argument("--lineup", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    import torch

    from bnb.game.api import PRESETS
    from bnb.play import replay
    from bnb.play.loop import MatchSetup, SideSetup
    from bnb.runtime.train import lineup_indices

    torch.set_num_threads(4)
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = open(out / "matches.jsonl", "a")
    for policy in arguments.policy:
        for preset in arguments.presets:
            config = PRESETS[preset]
            lineup = lineup_indices(tuple(arguments.lineup), config) if arguments.lineup else None
            for match in range(arguments.matches):
                seed = arguments.seed + match
                agent = SideSetup("agent", policy)
                other = agent if arguments.opponent == "self" else SideSetup("scripted_agent")
                setup = MatchSetup(config, lineup=lineup, seed=seed, sides=(agent, other))
                loop = match_loop(setup)
                name = f"{pathlib.Path(policy).parent.parent.parent.name}-{arguments.opponent}"
                recorder = replay.Recorder(setup, loop.realtime, name)
                loop.on_tick = recorder.record
                limit = int(config.episode_seconds * 30) + 300  # the episode, and a margin
                while not recorder.ended and recorder.ticks < limit:
                    loop.step()
                directory = replay.save(recorder, out, keep=10_000)
                ends = [e for e in recorder.events if e.kind == "end"]
                kills = [e for e in recorder.events if e.kind == "kill"]
                per_team = loop.rules.ships_per_team
                row = dict(
                    policy=policy,
                    preset=preset,
                    seed=seed,
                    opponent=arguments.opponent,
                    replay=str(directory),
                    ticks=recorder.ticks,
                    ended=recorder.ended,
                    winner=ends[0].team if ends else None,
                    kills=[
                        sum(1 for e in kills if e.other >= 0 and e.other // per_team == t)
                        for t in (0, 1)
                    ],
                    boundary_deaths=[
                        sum(1 for e in kills if e.other < 0 and e.ship // per_team == t)
                        for t in (0, 1)
                    ],
                )
                summary.write(json.dumps(row) + "\n")
                summary.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
