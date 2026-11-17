"""Charts and values from the unified evaluator's report (`report.json`, with its `spec.toml`
beside it): one field of players, several chains' snapshots and champions, rated together
against the anchor "scripted" (the current Imugi, 1000), other fixed players (an earlier
game's Imugi, for example `imugi_v24`) rated as players.

A player is named `<chain>/bc_<steps>` or `<chain>/rl_<steps>` (a snapshot of the chain's
cloning or reinforcement learning, its exponential average), `<chain>/champion` (its steps
read from the spec's path, `.../step_<steps>...`), or a fixed player. Each `curves/<mode>`
target holds per player a rating, a status ("rated", a bound "above" or "below", or
"unplayed" / "undetermined", which have no rating) and a standard error.

- `elo`: the rating over each chain's steps, one panel per mode in `modes`' order, one line
  per chain (the `chain` given in the mode's colour, the others neutral), the band one
  standard error, a bound an open triangle, the champion a star, the hand-off from cloning
  dashed, the anchor's 1000 and each other fixed player's rating as reference lines.
- `champion-break-even`: per mode, each champion's score against the anchor at
  `policy_count` of its ships against 1 to N of Imugi's, the logistic fit of those lineups
  alone (`bnb.league.crossover.fit`; the report's own fit pools every lineup the field
  played, the team-size games' included) with its 95% band, the break-even marked.
"""

import json
import math
import pathlib
import re
import tomllib

from bnb.league import crossover

from .. import theme
from . import common
from .fan import Z95

PLAYER = re.compile(r"^(?P<chain>.+)/(?P<stage>bc|rl)_(?P<steps>\d+)$")
STEP = re.compile(r"step_(\d+)")
RATED, ABOVE, BELOW = "rated", "above", "below"
MODE_ORDER = ("elimination", "frontline", "deathmatch")  # the release mode first


def read(path: str | pathlib.Path) -> tuple[dict, dict]:
    """(the report, the spec's players {name: path}; empty without a spec beside it)."""
    path = pathlib.Path(path)
    report = json.loads(path.read_text())
    spec = path.with_name("spec.toml")
    players = tomllib.loads(spec.read_text()).get("players", {}) if spec.exists() else {}
    return report, players


def curves(report: dict) -> dict[str, dict]:
    """{mode: its `curves/<mode>` target}."""
    return {
        name.removeprefix("curves/"): target
        for name, target in report["targets"].items()
        if name.startswith("curves/") and target.get("kind") == "elo"
    }


def chains_of(target: dict) -> list[str]:
    """The chains with snapshots in a curves target, in the order they first appear."""
    found: list[str] = []
    for name in target["ratings"]:
        matched = PLAYER.match(name)
        chain = matched.group("chain") if matched else name.removesuffix("/champion")
        if (matched or name.endswith("/champion")) and chain not in found:
            found.append(chain)
    return found


def fixed_players(target: dict, anchor: str = "scripted") -> list[str]:
    """The target's players that are neither snapshots nor champions nor the anchor."""
    return [
        name
        for name in target["ratings"]
        if name != anchor and not PLAYER.match(name) and not name.endswith("/champion")
    ]


def handoff(target: dict, chain: str) -> int:
    """A chain's cloning steps: its last BC snapshot's (the RL steps start there)."""
    steps = [
        int(m.group("steps"))
        for name in target["ratings"]
        if (m := PLAYER.match(name)) and m.group("chain") == chain and m.group("stage") == "bc"
    ]
    return max(steps, default=0)


def champion_steps(chain: str, players: dict) -> int | None:
    """The champion's RL steps from its spec path (`.../step_<steps>/...`)."""
    path = players.get(f"{chain}/champion")
    found = STEP.findall(path or "")
    return int(found[-1]) if found else None


def describe(target: dict, name: str) -> str | None:
    """A player's rating as the pages quote it: "1234 ± 45", a bound "> 1721 (a bound)",
    or None (unplayed, undetermined or absent)."""
    status = target["status"].get(name)
    rating = target["ratings"].get(name)
    if rating is None or status not in (RATED, ABOVE, BELOW):
        return None
    if status == RATED:
        return f"{rating:.0f} ± {target['errors'].get(name) or 0.0:.0f}"
    return f"{'>' if status == ABOVE else '<'} {rating:.0f} (a bound)"


def chain_points(target: dict, chain: str) -> list[tuple[float, str, str]]:
    """[(chain steps, player, stage)] of a chain's snapshots with a rating or a bound, in
    order of steps."""
    offset = handoff(target, chain)
    points = []
    for name in target["ratings"]:
        matched = PLAYER.match(name)
        if not matched or matched.group("chain") != chain:
            continue
        if target["status"].get(name) not in (RATED, ABOVE, BELOW):
            continue
        steps = int(matched.group("steps"))
        stage = matched.group("stage")
        points.append((steps + (offset if stage == "rl" else 0), name, stage))
    return sorted(points)


def rated_segments(target: dict, chain: str) -> list[list[tuple[int, str]]]:
    """A chain's rated snapshots [(chain steps, player)] in runs of consecutive scheduled
    snapshots: a snapshot without a rating (a bound, unplayed or undetermined) ends a run,
    so no line is drawn across it."""
    offset = handoff(target, chain)
    scheduled = []
    for name in target["status"]:
        matched = PLAYER.match(name)
        if matched and matched.group("chain") == chain:
            steps = int(matched.group("steps"))
            scheduled.append((steps + (offset if matched.group("stage") == "rl" else 0), name))
    segments: list[list[tuple[int, str]]] = [[]]
    for x, name in sorted(scheduled):
        if target["status"][name] == RATED and target["ratings"].get(name) is not None:
            segments[-1].append((x, name))
        elif segments[-1]:
            segments.append([])
    return [segment for segment in segments if segment]


def ordered_modes(available, modes: tuple[str, ...]) -> list[str]:
    """`modes` when given (each must be available), else the available ones, the release
    mode first."""
    if modes:
        missing = [mode for mode in modes if mode not in available]
        if missing:
            raise SystemExit(f"no curves for {missing} in the report; it has {sorted(available)}")
        return list(modes)
    return [mode for mode in MODE_ORDER if mode in available]


def _colours(chains: list[str], main: str, mode: str) -> dict[str, str]:
    colours = {main: common.mode_colour(mode)}
    others = [chain for chain in chains if chain != main]
    for index, chain in enumerate(others):
        colours[chain] = theme.BASELINE if index == 0 else theme.slot(3 + index)
    return colours


def draw_elo(
    path,
    out,
    modes: tuple[str, ...] = (),
    chain: str | None = None,
    formats=common.FORMATS,
    name: str = "elo_over_training",
    labels: dict[str, str] | None = None,
):
    """The Elo chart (see the module); (the files written, the values: per chain and mode
    its last BC and last RL snapshot's rating as a string, `describe`, with their chain
    steps, its champion's, and each fixed player's rating; the `chain` given also under the
    plain names `elo_<mode>_bc_final` and `elo_<mode>_final`, pending for the modes not
    drawn)."""
    report, players = read(path)
    found = curves(report)
    drawn = ordered_modes(found, modes)
    figure, axes = common.new_figure(1, len(drawn), size=(7.2 if len(drawn) == 1 else 4.2, 3.8))
    values: dict[str, float | str | None] = {}
    for axis, mode in zip(axes.flat, drawn, strict=True):
        target = found[mode]
        chains = chains_of(target)
        main = chain if chain in chains else chains[0]
        colours = _colours(chains, main, mode)
        lines = []
        for each in chains:
            colour = colours[each]
            points = chain_points(target, each)
            segments = rated_segments(target, each)
            width = 1.6 if each == main else 1.2
            for segment in segments:
                xs = [x / 1e6 for x, _ in segment]
                ys = [target["ratings"][n] for _, n in segment]
                errors = [target["errors"].get(n) or 0.0 for _, n in segment]
                axis.plot(xs, ys, "-", color=colour, linewidth=width, marker="o", markersize=2.3)
                axis.fill_between(
                    xs,
                    [y - e for y, e in zip(ys, errors, strict=True)],
                    [y + e for y, e in zip(ys, errors, strict=True)],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
            last_segment = segments[-1] if segments else []
            xs = [x / 1e6 for x, _ in last_segment]
            ys = [target["ratings"][n] for _, n in last_segment]
            for x, n, _ in points:
                status = target["status"][n]
                if status != RATED:
                    axis.plot(
                        [x / 1e6],
                        [target["ratings"][n]],
                        marker="^" if status == ABOVE else "v",
                        markerfacecolor="none",
                        markeredgecolor=colour,
                        markersize=4.5,
                        linestyle="none",
                    )
            offset = handoff(target, each)
            champion = f"{each}/champion"
            steps = champion_steps(each, players)
            status = target["status"].get(champion)
            if xs and steps is not None and (offset + steps) / 1e6 > xs[-1]:
                xs, ys = [*xs, (offset + steps) / 1e6], [*ys, ys[-1]]  # the label past the star
            lines.append(((labels or {}).get(each, each), xs, ys, colour))
            if steps is not None and status in (RATED, ABOVE, BELOW):
                filled = status == RATED
                error = target["errors"].get(champion) or 0.0
                axis.errorbar(
                    [(offset + steps) / 1e6],
                    [target["ratings"][champion]],
                    yerr=[[error], [error]] if filled else None,
                    marker="*",
                    markersize=9,
                    color=colour,
                    markerfacecolor=colour if filled else "none",
                    markeredgecolor=theme.SURFACE if filled else colour,
                    linestyle="none",
                    capsize=0,
                    zorder=5,
                )
            stem = f"elo_{common.slug(each)}_{mode}"
            for stage, key in (("bc", "bc_final"), ("rl", "final")):
                last = [(x, n) for x, n, s in points if s == stage]
                values[f"{stem}_{key}"] = describe(target, last[-1][1]) if last else None
                values[f"{stem}_{key}_steps"] = float(last[-1][0]) if last else None
            values[f"{stem}_champion"] = describe(target, champion)
            values[f"{stem}_champion_steps"] = float(offset + steps) if steps is not None else None
            values[f"{stem}_unrated"] = float(
                sum(
                    1
                    for n in target["ratings"]
                    if (PLAYER.match(n) and PLAYER.match(n).group("chain") == each)
                    and target["status"].get(n) not in (RATED, ABOVE, BELOW)
                )
            )
            if each == main:
                for key in ("bc_final", "final"):
                    values[f"elo_{mode}_{key}"] = values[f"{stem}_{key}"]
                    values[f"elo_{mode}_{key}_steps"] = values[f"{stem}_{key}_steps"]
        for x in sorted({handoff(target, each) / 1e6 for each in chains} - {0.0}):
            axis.axvline(x, color=theme.MUTED, linestyle="--", linewidth=0.9)
            axis.annotate(
                "cloning | RL",
                (x, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=theme.MUTED,
            )
        axis.axhline(1000, color=theme.MUTED, linewidth=0.8, linestyle=":")
        _reference(axis, 1000, "Imugi = 1000", above=True)
        for fixed in fixed_players(target):
            rating = target["ratings"].get(fixed)
            values[f"elo_{common.slug(fixed)}_{mode}"] = describe(target, fixed)
            if rating is None:
                continue
            axis.axhline(rating, color=theme.MUTED, linewidth=0.8, linestyle=(0, (1, 3)))
            _reference(axis, rating, f"{_fixed_label(fixed)} ({rating:.0f})", above=False)
        if len(drawn) > 1:
            axis.set_title(mode, fontsize=9)
        axis.set_xlabel("environment steps (M)", fontsize=8)
        common.label_ends(axis, lines)
    axes[0, 0].set_ylabel("Elo (Imugi = 1000)", fontsize=8)
    every_chain = {each for target in found.values() for each in chains_of(target)}
    for mode in common.MODES:
        if mode in drawn:
            continue
        stems = [f"elo_{mode}"] + [f"elo_{common.slug(each)}_{mode}" for each in every_chain]
        for stem in stems:  # a mode not drawn is pending on the pages
            for key in ("bc_final", "final", "champion"):
                values[f"{stem}_{key}"] = None
                values[f"{stem}_{key}_steps"] = None
        for fixed in {f for target in found.values() for f in fixed_players(target)}:
            values[f"elo_{common.slug(fixed)}_{mode}"] = None
    values["elo_report_games"] = float(report.get("games") or 0)
    values["elo_modes_drawn"] = ", ".join(drawn)
    values.update(win_rate_values(report))
    title = common.captions.title(name)
    if len(drawn) == 1:
        title = f"{title}: {drawn[0]}"
    common.title(figure, name, title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    written = common.save(figure, out, name, [str(path)], _numbers(values), formats)
    return written, values


def _fixed_label(name: str) -> str:
    matched = re.match(r"^imugi_v(\d+)$", name)
    return f"Imugi of game v{matched.group(1)}" if matched else name.replace("_", " ")


def _reference(axis, y: float, text: str, above: bool) -> None:
    axis.annotate(
        text,
        (0.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(3, 2 if above else -2),
        textcoords="offset points",
        va="bottom" if above else "top",
        fontsize=7,
        color=theme.MUTED,
    )


def _numbers(values: dict) -> dict:
    """The values that are numbers (a PNG's provenance holds those)."""
    return {key: value for key, value in values.items() if isinstance(value, float)}


def win_rate_values(report: dict) -> dict[str, float | None]:
    """Each played head-to-head cell of the report's win-rate targets (`suite`,
    `suite_v24`, `teamsize`): `h2h_<chain>_<mode>_<lineup>_vs_<opponent>_score`, `_stderr`
    and `_games` (the lineup "5v5" when the preset names none)."""
    values: dict[str, float | None] = {}
    for target in report["targets"].values():
        if target.get("kind") != "win_rate" or not target.get("games"):
            continue
        parts = target["cell"].split("/")
        if len(parts) < 3:
            continue
        preset, chain, pairing = parts[0], "/".join(parts[1:-1]), parts[-1]
        sized = re.match(r"^real_scale_(?P<mode>[a-z]+)(?:_(?P<lineup>\d+v\d+))?$", preset)
        if not sized:
            continue
        opponent = pairing.split("_vs_", 1)[1]
        stem = (
            f"h2h_{common.slug(chain)}_{sized.group('mode')}_{sized.group('lineup') or '5v5'}"
            f"_vs_{common.slug(opponent)}"
        )
        values[f"{stem}_score"] = target["score"]
        values[f"{stem}_stderr"] = target["stderr"]
        values[f"{stem}_games"] = float(target["games"])
    return values


def break_even_lineups(target: dict, policy_count: int) -> dict[tuple[int, int], list[float]]:
    """{lineup: [score, games]} of a break-even target's lineups with the policy at
    `policy_count` ships."""
    found = {}
    for text, (score, games) in target.get("lineups", {}).items():
        ours, theirs = (int(part) for part in text.split("v"))
        if ours == policy_count and games:
            found[(ours, theirs)] = [float(score), float(games)]
    return found


def break_even_targets(report: dict, policy_count: int, modes: tuple[str, ...] = ()):
    """{mode: [(chain, target, its lineups at `policy_count`)]} for the modes where some
    champion has played `policy_count` against more than `policy_count` (the lineups that
    measure a break-even; 5v5 alone does not)."""
    found: dict[str, list] = {}
    for name, target in report["targets"].items():
        if target.get("kind") != "break_even":
            continue
        mode = common.mode_of(target["preset"])
        if modes and mode not in modes:
            continue
        lineups = break_even_lineups(target, policy_count)
        chain = target["policy"].removesuffix("/champion")
        found.setdefault(mode, []).append((chain, target, lineups))
    return {
        mode: rows
        for mode, rows in found.items()
        if any(theirs > policy_count for _, _, lineups in rows for _, theirs in lineups)
    }


def draw_break_even(
    path,
    out,
    policy_count: int = 5,
    modes: tuple[str, ...] = (),
    chain: str | None = None,
    formats=common.FORMATS,
    name: str = "champion_break_even",
    labels: dict[str, str] | None = None,
):
    """The champions' break-even chart (see the module); (the files written, the values:
    per chain and mode the break-even ratio, its standard error, the fit's status and the
    games). Nothing is drawn (([], {})) while no champion has played `policy_count`
    against more."""
    report, _ = read(path)
    found = break_even_targets(report, policy_count, modes)
    drawn = [mode for mode in MODE_ORDER if mode in found]
    values: dict[str, float | str | None] = {"break_even_policy_count": float(policy_count)}
    for target in report["targets"].values():  # pending until drawn
        mode = common.mode_of(target.get("preset", ""))
        if target.get("kind") == "break_even" and mode not in drawn:
            stem = f"break_even_{common.slug(target['policy'].removesuffix('/champion'))}_{mode}"
            for key in ("ratio", "ratio_stderr", "status", "games"):
                values[f"{stem}_{key}"] = None
    if not drawn:
        print(f"no lineups of {policy_count} against more yet in {path}: not drawn")
        return [], values
    figure, axes = common.new_figure(1, len(drawn), size=(5.6 if len(drawn) == 1 else 3.8, 3.4))
    for axis, mode in zip(axes.flat, drawn, strict=True):
        rows = found[mode]
        chains = [each for each, _, _ in rows]
        main = chain if chain in chains else chains[0]
        colours = _colours(chains, main, mode)
        ratios = [theirs / ours for _, _, lineups in rows for ours, theirs in lineups]
        low_x, high_x = min(ratios + [1.0]), max(ratios)
        lines = []
        for each, _, lineups in rows:
            colour = colours[each]
            fitted = crossover.fit(lineups)
            ratio, ratio_error = fitted.ratio()
            stem = f"break_even_{common.slug(each)}_{mode}"
            values[f"{stem}_ratio"] = ratio
            values[f"{stem}_ratio_stderr"] = ratio_error
            values[f"{stem}_status"] = fitted.status
            values[f"{stem}_games"] = float(sum(games for _, games in lineups.values()))
            for (ours, theirs), (score, games) in sorted(lineups.items()):
                rate, low, high = _wilson(score, games)
                x = theirs / ours
                axis.plot([x, x], [low, high], color=colour, linewidth=0.8, alpha=0.6)
                axis.plot([x], [rate], "o", color=colour, markerfacecolor="none", markersize=4)
            if fitted.status != "crossing":
                continue
            xs = [low_x * (high_x / low_x) ** (i / 120) for i in range(121)]
            curve = [_fit_at(fitted, -math.log(x)) for x in xs]
            axis.plot(xs, [c[0] for c in curve], color=colour, linewidth=1.5)
            axis.fill_between(
                xs,
                [c[1] for c in curve],
                [c[2] for c in curve],
                color=colour,
                alpha=0.15,
                linewidth=0,
            )
            axis.axvline(ratio, color=colour, linewidth=0.9, linestyle="--")
            lines.append(
                (
                    f"{(labels or {}).get(each, each)}: {ratio:.2f}",
                    list(xs),
                    [c[0] for c in curve],
                    colour,
                )
            )
        axis.axhline(0.5, color=theme.MUTED, linewidth=0.8, linestyle=":")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xscale("log")
        _ratio_ticks(axis, low_x, high_x)
        axis.set_xlabel(f"Imugi's ships per one of ours ({policy_count} of ours)", fontsize=8)
        if len(drawn) > 1:
            axis.set_title(mode, fontsize=9)
        common.label_ends(axis, lines)
    axes[0, 0].set_ylabel("score against Imugi", fontsize=8)
    title = common.captions.title(name)
    if len(drawn) == 1:
        title = f"{title}: {drawn[0]}"
    common.title(figure, name, title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    written = common.save(figure, out, name, [str(path)], _numbers(values), formats)
    return written, values


def _ratio_ticks(axis, low: float, high: float) -> None:
    """Plain ratio ticks (0.5, 1, 1.5, 2, 3, 4) on a log axis, within [low, high]."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ticks = [t for t in (0.2, 0.25, 0.5, 1, 1.5, 2, 3, 4) if low * 0.95 <= t <= high * 1.05]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels([f"{t:g}" for t in ticks])


def _fit_at(fitted, log_ratio: float) -> tuple[float, float, float]:
    """(the fit's score, its 95% low and high) at `log_ratio` (r = log ours / theirs)."""
    eta = fitted.intercept + fitted.slope * log_ratio
    (aa, as_), (_, ss) = fitted.covariance
    spread = math.sqrt(max(aa + 2 * log_ratio * as_ + log_ratio * log_ratio * ss, 0.0))

    def logistic(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    return logistic(eta), logistic(eta - Z95 * spread), logistic(eta + Z95 * spread)


def _wilson(score: float, games: float) -> tuple[float, float, float]:
    from .evaluation import wilson

    return wilson(score, games)
