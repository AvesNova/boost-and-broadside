"""The unified evaluator's spec for one or more dry-run chains (box/unified_check.sh).

    python box/unified_spec.py RUNS CHAIN[,CHAIN...] [--parts "suite curves breakeven teamsize"]
        [--old-imugi] [--modes elimination] [--preset elimination=PRESET]
        [--lineup-types "Jujak"] [--curve-design static|adaptive] [--reach 2] [--pilot 1]
        [--windows 16,8[,64]]
        > spec.toml

Players: every chain's league-schedule snapshots of both stages (`scheduled_policies`, the
exponential averages; a run that keeps every update's policy gives the same snapshots) and
its champion (`RUNS/CHAIN-eval/champion.json`, else the last RL step's average); with
`--old-imugi`, "imugi_v24" (game v24's Imugi, `run.IMUGI_VARIANTS`) beside the anchor
("scripted", this tree's Imugi, 1000). One chain keeps the single-run names (bc_N, rl_N,
champion); several are prefixed with the chain ("dry-v24-s1/rl_N").

Targets, per mode (no map-scale targets: evaluation plays the trained map sizes):
- curves: the 5v5 Elo field of every player, +-60, with the rating pools' preset keys for
  --prior; by default the static design (`--curve-design static`: the anchor and the
  checkpoints only, each run resolved against the anchor in two fixed stages,
  `bnb.league.unified.static_target`; `--reach`, `--pilot`), else the adaptive planner
  (stopping on precision, a safety budget of `CURVE_BUDGET`);
- suite: each champion against the anchor at 5v5, 10v10 (and 1v1, 2v2 outside Frontline),
  +-0.02; against imugi_v24 at 5v5 only;
- breakeven: each champion at 5 ships against the anchor's 1 to 20 (+-0.1);
- teamsize: each champion at n v n, n v 1.5n, n v 2n, n = 1..10 (+-0.05).
Deathmatch's curves carry dry run 1's ticket note when a v24 chain is in the field.
"""

import argparse
import json
import pathlib
import sys

CURVE_BUDGET = 100_000  # games a mode: a safety bound
MODES = {
    "frontline": "real_scale_frontline",
    "deathmatch": "real_scale_deathmatch",
    "elimination": "real_scale_elimination",
}
TICKET_NOTE = (
    "dry run 1 (dry-v24-s1) trained deathmatch on 5N tickets; every evaluation plays the "
    "real rule 4N^(4/3) rounded to 5 (35 at 5v5 against 25): little effect expected, "
    "mainly in close games decided by a late kill"
)


def champion_of(runs: pathlib.Path, chain: str) -> str:
    try:
        return json.loads((runs / f"{chain}-eval" / "champion.json").read_text())["path"]
    except OSError, ValueError, KeyError:
        root = runs / f"{chain}-rl" / "policies"
        last = max(int(p.name[5:]) for p in root.glob("step_*") if p.name[5:].isdigit())
        return str(root / f"step_{last}" / "exponential_average")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=pathlib.Path)
    parser.add_argument("chains")
    parser.add_argument("--parts", default="suite curves breakeven teamsize")
    parser.add_argument("--old-imugi", action="store_true")
    parser.add_argument("--modes", default=" ".join(MODES), help="the modes to evaluate")
    parser.add_argument(
        "--curve-design",
        choices=("static", "adaptive"),
        default="static",
        help="the curves' design: static (fixed counts in two stages, `static_target`) or the "
        "adaptive per-game planner",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="MODE=PRESET: a mode's preset (e.g. elimination=real_scale_elimination_hemisphere)",
    )
    parser.add_argument("--lineup-types", default="Yongma Jujak", help="the lineups' ship types")
    parser.add_argument("--reach", type=int, default=2, help="static: chain neighbours a side")
    parser.add_argument("--pilot", type=float, default=1.0, help="static: stage 1's goal factor")
    parser.add_argument(
        "--windows",
        default="",
        help="static: SIZE,OVERLAP[,CAPACITY] fixed windows of the training order (16,8,64), "
        "a window's cells started, the next seated beside it; CAPACITY rows a lane; empty: sliding",
    )
    args = parser.parse_args()
    from bnb.league.rating_pool import scheduled_policies

    runs, chains, parts = args.runs, args.chains.split(","), set(args.parts.split())
    presets = dict(MODES) | dict(one.split("=", 1) for one in args.preset)
    several = len(chains) > 1
    players: dict[str, str] = {}
    champions = []
    for chain in chains:
        prefix = f"{chain}/" if several else ""
        for stage in ("bc", "rl"):
            if not (runs / f"{chain}-{stage}" / "policies").is_dir():
                print(f"note: {chain}-{stage} has no policies; left out", file=sys.stderr)
                continue
            for path in scheduled_policies(runs / f"{chain}-{stage}", ("exponential_average",)):
                players[f"{prefix}{stage}_{path.parent.name[5:]}"] = str(path.relative_to(runs))
        players[f"{prefix}champion"] = champion_of(runs, chain)
        champions.append(f"{prefix}champion")
    lineup = json.dumps(args.lineup_types.split())
    out = [f"lineup_types = {lineup}", "buckets = [5, 10, 20]", "seed = 0"]
    if args.windows:
        size, overlap, *capacity = (int(x) for x in args.windows.split(","))
        extra = f", capacity = {capacity[0]}" if capacity else ""
        out.append(f"windows = {{size = {size}, overlap = {overlap}{extra}}}")
    out.append("")
    out.append("[players]")
    if args.old_imugi:
        out.append('imugi_v24 = "imugi_v24"')
    out += [f'"{name}" = "{path}"' for name, path in players.items()]
    scripted = ["scripted"] + (["imugi_v24"] if args.old_imugi else [])
    field = json.dumps(scripted + list(players))
    v24 = any("v24" in chain for chain in chains)
    modes = args.modes.split()
    for mode, preset in ((m, p) for m, p in presets.items() if m in modes):
        sizes = [[5, 5], [10, 10]] + ([[1, 1], [2, 2]] if mode != "frontline" else [])
        team = []
        for n in range(1, 11):
            team.append([n, n])
            if n % 2 == 0:
                team.append([n, 3 * n // 2])
            team.append([n, 2 * n])
        if mode == "frontline":
            team = [lineup for lineup in team if min(lineup) >= 3]
        if "curves" in parts:
            out += ["", "[[elo]]", f'name = "curves/{mode}"', f'preset = "{preset}"']
            if args.curve_design == "static":  # the anchor and the checkpoints only
                # a champion outside the policies/step_N snapshots (an averaging sweep's pick) is
                # on no run's chain: the static design cannot place it, so it is left out here
                chained = [n for n, path in players.items() if "/policies/step_" in f"/{path}"]
                out += [f"players = {json.dumps(['scripted', *chained])}", 'design = "static"']
                out += [f"reach = {args.reach}", f"pilot = {args.pilot}"]
            else:
                out.append(f"players = {field}")
            out += ["stderr = 60.0", f"budget = {CURVE_BUDGET}"]
            out.append(f'prior_keys = ["{preset}", "{mode}_5v5"]')
            if mode == "deathmatch" and v24:
                out.append(f"note = {json.dumps(TICKET_NOTE)}")
        for champion in champions:
            if "suite" in parts:
                out += ["", "[[win_rate]]", 'name = "suite"', f'preset = "{preset}"']
                out += [f'first = "{champion}"', 'second = "scripted"', f"lineups = {sizes}"]
                out += ["stderr = 0.02", "budget = 512"]
                if args.old_imugi:
                    out += ["", "[[win_rate]]", 'name = "suite_v24"', f'preset = "{preset}"']
                    out += [f'first = "{champion}"', 'second = "imugi_v24"', "lineups = [[5, 5]]"]
                    out += ["stderr = 0.02", "budget = 512"]
            if "breakeven" in parts:  # the fan at the trained 5 ships, Imugi 1 to 20
                out += ["", "[[break_even]]", f'preset = "{preset}"', f'policy = "{champion}"']
                out += ['opponent = "scripted"', "stderr = 0.1", "budget = 2048"]
                out += ["policy_count = 5", "fixed_count = 20"]
            if "teamsize" in parts:
                out += ["", "[[win_rate]]", 'name = "teamsize"', f'preset = "{preset}"']
                out += [f'first = "{champion}"', 'second = "scripted"', f"lineups = {team}"]
                out += ["stderr = 0.05", "budget = 128"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
