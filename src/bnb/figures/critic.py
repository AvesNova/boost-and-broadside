"""The critic figure set: the value and the outcome head over whole games, with kills and
deaths marked, and whether the outcome head's win probability comes true.

It reads the capture's `games` record (the critic over the first segment and the segments
played on after it). The outcome head is discounted: its targets put (1 - gamma) per tick
on "unresolved", so its probability of a win is the discounted chance of winning. The
figures read the resolved probabilities renormalised, P(win) / (P(win) + P(tie) + P(loss)),
as its forecast of the match result, scored on the ticks of games that ended inside the
capture: a reliability diagram, and the Brier skill against the capture's result
frequencies by the time left to the game's end. A critic without an outcome head (the
single value arm, `bnb.models.yemong.model.YemongConfig`) draws the values alone.
"""

import pathlib

import torch

from bnb.models.yemong.critic import LOSS, TIE, WIN

from . import output

HORIZONS = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, 300.0))  # s
RELIABILITY_BINS = 10


def ended_games(artifact: dict) -> dict:
    """Per tick and view (L, R): ended (a game end follows in the capture), the class of
    that end's result (WIN, TIE or LOSS, for the view's team) and the seconds left to it,
    and the resolved probabilities (L, R, 3) in WIN, TIE, LOSS order."""
    games, tick = artifact["games"], artifact["meta"]["tick_length"]
    done, result = games["done"], games["result"]
    length, views = done.shape
    ended = torch.zeros((length, views), dtype=torch.bool)
    outcome = torch.zeros((length, views), dtype=torch.long)
    left = torch.zeros((length, views))
    next_end = torch.full((views,), -1)
    next_class = torch.zeros(views, dtype=torch.long)
    for time in reversed(range(length)):
        finished = done[time]
        next_end = torch.where(finished, time, next_end)
        resolved = torch.where(result[time] > 0, WIN, torch.where(result[time] < 0, LOSS, TIE))
        next_class = torch.where(finished, resolved, next_class)
        ended[time] = next_end >= 0
        outcome[time] = next_class
        left[time] = (next_end - time + 1).float() * tick
    probabilities = torch.stack(
        [games["outcome"][name] for name in ("win", "tie", "loss")], -1
    )  # in WIN, TIE, LOSS order (WIN, TIE, LOSS = 0, 1, 2)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-9)
    return dict(ended=ended, outcome=outcome, left=left, probabilities=probabilities)


def _brier(probabilities: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    actual = torch.nn.functional.one_hot(outcome, 3).float()
    return (probabilities - actual).square().sum(-1)


def measures(artifact: dict) -> dict:
    """The games that ended, the ticks scored, the result frequencies, the Brier score of
    the renormalised outcome head and of the frequencies, overall and per HORIZONS bucket
    (seconds left), and the Brier skill 1 - head / frequencies. Without an outcome head,
    the games that ended alone."""
    games = int(artifact["games"]["done"].sum())
    if "outcome" not in artifact["games"]:
        return dict(games_ended=games)
    part = ended_games(artifact)
    ended = part["ended"]
    out = dict(games_ended=games, ticks_scored=int(ended.sum()))
    if not ended.any():
        return out
    outcome = part["outcome"][ended]
    frequency = torch.bincount(outcome, minlength=3).float() / len(outcome)
    head = _brier(part["probabilities"][ended], outcome)
    climate = _brier(frequency.expand(len(outcome), 3), outcome)
    out["result_frequency"] = dict(
        win=round(float(frequency[WIN]), 3),
        tie=round(float(frequency[TIE]), 3),
        loss=round(float(frequency[LOSS]), 3),
    )
    out["brier_skill"] = round(float(1.0 - head.mean() / climate.mean().clamp_min(1e-9)), 3)
    by_horizon = {}
    left = part["left"][ended]
    for low, high in HORIZONS:
        members = (left >= low) & (left < high)
        if not members.any():
            continue
        by_horizon[f"{low:g}-{high:g} s"] = dict(
            ticks=int(members.sum()),
            brier_skill=round(
                float(1.0 - head[members].mean() / climate[members].mean().clamp_min(1e-9)), 3
            ),
        )
    out["brier_skill_by_seconds_left"] = by_horizon
    return out


def render(artifact: dict, directory: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """The critic figures: critic_trace (one view's game: the values and the outcome head,
    kills and deaths marked) and critic_outcome (reliability and Brier skill); without an
    outcome head, critic_trace of the values alone."""
    numbers = measures(artifact)
    if "outcome" not in artifact["games"]:
        return [_trace(artifact, None, directory, numbers)], numbers
    part = ended_games(artifact)
    return [
        _trace(artifact, part, directory, numbers),
        _outcome(artifact, part, directory, numbers),
    ], numbers


def _trace(artifact, part, directory, numbers) -> pathlib.Path:
    meta, games = artifact["meta"], artifact["games"]
    own, tick = meta["ships_per_team"], meta["tick_length"]
    done = games["done"]
    # the view whose first game is longest and ended; else view 0
    ends = [
        int(torch.nonzero(done[:, view])[0]) if done[:, view].any() else -1
        for view in range(done.shape[1])
    ]
    view = max(range(len(ends)), key=lambda v: ends[v])
    stop = ends[view] + 1 if ends[view] >= 0 else done.shape[0]
    seconds = torch.arange(stop) * tick
    alive = games["alive"][:stop, view]
    died = alive[:-1] & ~alive[1:]  # (stop - 1, N) a ship died on that tick
    result = int(games["result"][stop - 1, view]) if ends[view] >= 0 else None
    title = (
        f"view {view}: game result {result} (+1 win); vertical lines: an enemy "
        "killed (orange), an own ship lost (black)"
    )
    rows = 1 if part is None else 2
    figure, axes = output.new_figure(rows, 1, size=(11.0, 1.6 * rows), sharex=True)
    for time, row in torch.nonzero(died).tolist():
        colour = output.TRUE if row >= own else output.INK
        for each in axes[:, 0]:
            each.axvline(float((time + 1) * tick), color=colour, linewidth=0.8, alpha=0.6)
    if part is not None:
        _outcome_trace(axes[0, 0], games["outcome"], part, view, stop, seconds)
    axes[0, 0].set_title(title, fontsize=9)
    axis = axes[-1, 0]
    values = games["values"][:stop, view].mean(1)  # (stop, levels): the own ships' mean
    for level, name in enumerate(meta["levels"]):
        axis.plot(seconds, values[:, level], linewidth=1, label=name)
    axis.set_ylabel("critic value per level\n(own ships' mean)")
    axis.set_xlabel("seconds")
    axis.legend(frameon=False, fontsize=7, ncol=values.shape[-1])
    figure.tight_layout()
    return output.save(figure, directory, "critic_trace", meta, numbers)


def _outcome_trace(axis, outcome: dict, part: dict, view: int, stop: int, seconds) -> None:
    """The outcome head over one view's game, on `axis`."""
    axis.plot(
        seconds,
        outcome["value"][:stop, view],
        color=output.PREDICTED,
        label="P(win) - P(loss), as the head gives it",
    )
    axis.plot(
        seconds,
        part["probabilities"][:stop, view, WIN],
        color=output.SECOND,
        label="P(win), resolved and renormalised",
    )
    axis.plot(
        seconds,
        outcome["unresolved"][:stop, view],
        color=output.BASELINE,
        linestyle="--",
        label="P(unresolved)",
    )
    axis.set_ylabel("outcome head")
    axis.legend(frameon=False, fontsize=7, loc="upper left")


def _outcome(artifact, part, directory, numbers) -> pathlib.Path:
    meta = artifact["meta"]
    figure, axes = output.new_figure(1, 2, size=(5.0, 4.2))
    axis = axes[0, 0]
    axis.plot([0, 1], [0, 1], color=output.MUTED, linewidth=1)
    ended = part["ended"]
    if ended.any():
        predicted = part["probabilities"][..., WIN][ended]
        happened = (part["outcome"][ended] == WIN).float()
        index = (predicted * RELIABILITY_BINS).long().clamp(0, RELIABILITY_BINS - 1)
        count = torch.bincount(index, minlength=RELIABILITY_BINS).float()
        mean = torch.bincount(index, predicted, minlength=RELIABILITY_BINS) / count.clamp_min(1)
        share = torch.bincount(index, happened, minlength=RELIABILITY_BINS) / count.clamp_min(1)
        kept = count > 0
        axis.plot(mean[kept], share[kept], "-o", color=output.PREDICTED, markersize=4)
        for x, y, n in zip(
            mean[kept].tolist(), share[kept].tolist(), count[kept].tolist(), strict=True
        ):
            axis.annotate(
                f"{int(n)}",
                (x, y),
                fontsize=6,
                color=output.MUTED,
                xytext=(3, -8),
                textcoords="offset points",
            )
    axis.set_xlabel("P(win), resolved and renormalised")
    axis.set_ylabel("share of those ticks whose game was won")
    axis.set_aspect("equal")
    axis.set_title(
        f"reliability over {numbers.get('ticks_scored', 0)} ticks of "
        f"{numbers.get('games_ended', 0)} ended games (labels: ticks)",
        fontsize=8,
    )
    axis = axes[0, 1]
    by_horizon = numbers.get("brier_skill_by_seconds_left", {})
    names = list(by_horizon)
    axis.bar(
        range(len(names)),
        [by_horizon[n]["brier_skill"] for n in names],
        0.6,
        color=output.PREDICTED,
    )
    axis.axhline(0.0, color=output.MUTED, linewidth=1)
    axis.set_xticks(range(len(names)), names, fontsize=7)
    axis.set_xlabel("time left to the game's end")
    axis.set_ylabel("Brier skill against the result frequencies")
    axis.set_title(f"overall {numbers.get('brier_skill')}", fontsize=9)
    figure.suptitle("Outcome head: does its win probability come true?", color=output.INK)
    figure.tight_layout()
    return output.save(figure, directory, "critic_outcome", meta, numbers)
