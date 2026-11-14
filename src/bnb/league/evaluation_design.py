"""Choosing `bnb evaluate`'s sequential design from a pilot (`bnb evaluate design`): each
condition's per-game variance and engine seconds per game, read from pilot directories
(`bnb.league.evaluation`'s records), give the engine seconds a unit of the overall score's
information costs with the cost-optimal allocation (`evaluation.cost_per_information`);
then for each candidate number of looks K the planned boundaries
(`bnb.league.sequential.plan`), I_max and its cost, and the outcomes' chances and the
expected cost at chosen true differences (`operating_characteristics`).

**The procedure simulated** (`simulate`): the whole suite, look by look, with games drawn
instead of played. A head-to-head condition's games are wins, draws and losses at its
pilot draw rate, A's mean score shifted to one half plus the true difference; break-even's
estimate is drawn normal about the true difference with its pilot per-game variance over
its games. Each look's targets come from `evaluation.targets` on the drawn games'
estimates, and the test and the gate are `evaluation.look`'s, so the simulation checks the
error rates the plan promises with estimated variances, batch-sized steps and the
cost-aware allocation; and the estimate's mean at the stop.
"""

import dataclasses
import math
import pathlib

import torch

from . import evaluation, sequential
from .evaluation import BREAK_EVEN, Condition, Estimate, EvaluationConfig, Tally


@dataclasses.dataclass(frozen=True)
class ConditionModel:
    """One condition's pilot numbers: per-game variance, engine seconds per game, the draw
    rate (head-to-head) and the games per batch."""

    condition: Condition
    per_game: float
    seconds: float
    draws: float
    batch_games: int


def pilot_models(directory: pathlib.Path) -> tuple[EvaluationConfig, list[ConditionModel], dict]:
    """The settings, each scored condition's model from a pilot directory's records, and its
    summary (the observed difference)."""
    _, config = evaluation.read_settings(directory)
    batches, _, _ = evaluation.read_batches(directory)
    tallies = evaluation.tallies_of(batches)
    models = []
    for condition in config.scored():
        tally = tallies.get(condition.name, Tally())
        found = evaluation.estimate(condition, tally)
        if found is None or tally.seconds_per_game() is None:
            raise ValueError(f"{directory}: {condition.name} has no estimate yet")
        won, drawn, lost = tally.outcomes
        models.append(
            ConditionModel(
                condition,
                found.per_game,
                tally.seconds_per_game(),
                drawn / max(won + drawn + lost, 1.0),
                config.layout(condition).games,
            )
        )
    return config, models, evaluation.summarise(config.scored(), tallies)


def pooled_models(groups: list[list[ConditionModel]]) -> list[ConditionModel]:
    """Each condition's model averaged over pilots (per-game variance, seconds and draws)."""
    pooled = []
    for index, first in enumerate(groups[0]):
        members = [group[index] for group in groups]
        pooled.append(
            dataclasses.replace(
                first,
                per_game=sum(m.per_game for m in members) / len(members),
                seconds=sum(m.seconds for m in members) / len(members),
                draws=sum(m.draws for m in members) / len(members),
            )
        )
    return pooled


def seconds_per_information(models: list[ConditionModel]) -> float:
    """Engine seconds per unit of information, cost-optimal allocation."""
    conditions = [m.condition for m in models]
    return evaluation.cost_per_information(
        conditions,
        {m.condition.name: m.per_game for m in models},
        {m.condition.name: m.seconds for m in models},
    )


@dataclasses.dataclass(frozen=True)
class DesignRow:
    """One candidate design at one true difference: the outcomes' chances and the expected
    engine hours."""

    looks: int
    difference: float
    a_wins: float
    b_wins: float
    tie: float
    expected_hours: float
    maximum_hours: float


def design_table(
    config: EvaluationConfig,
    models: list[ConditionModel],
    looks: tuple[int, ...],
    differences: tuple[float, ...],
) -> tuple[list[DesignRow], dict[int, sequential.Boundaries]]:
    """Each K's planned boundaries and, at each true difference, its outcomes' chances and
    expected hours (the planned looks at their planned information; `plan`)."""
    per_information = seconds_per_information(models)
    rows, plans = [], {}
    for count in looks:
        design = dataclasses.replace(config, looks=count).design()
        drift, found = sequential.plan(design)
        plans[count] = found
        maximum = (drift / config.delta) ** 2
        for difference in differences:
            result = sequential.operating_characteristics(found, difference * math.sqrt(maximum))
            rows.append(
                DesignRow(
                    count,
                    difference,
                    result.a_wins,
                    result.b_wins,
                    result.tie,
                    result.expected_fraction * maximum * per_information / 3600.0,
                    maximum * per_information / 3600.0,
                )
            )
    return rows, plans


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """The procedure simulated at one true difference: the decisions' shares, the mean
    engine hours and looks, and the mean and standard error of the estimate at the stop."""

    difference: float
    a_wins: float
    b_wins: float
    tie: float
    hours: float
    looks: float
    estimate_mean: float
    estimate_stderr: float


def _draw_head_to_head(
    model: ConditionModel, difference: float, games: int, generator: torch.Generator
) -> list[float]:
    """(wins, draws, losses) of `games` games with A's mean score 1/2 + difference."""
    draws = min(model.draws, 1.0 - 2.0 * abs(difference))
    win = 0.5 + difference - draws / 2.0
    probabilities = torch.tensor([win, draws, 1.0 - win - draws], dtype=torch.float64)
    counts = torch.multinomial(probabilities.clamp(min=0.0), games, True, generator=generator)
    return [float((counts == outcome).sum()) for outcome in range(3)]


def simulate(
    config: EvaluationConfig,
    models: list[ConditionModel],
    difference: float,
    repeats: int,
    generator: torch.Generator,
    initial_batches: int | None = None,
) -> SimulationResult:
    """The procedure, `repeats` times, at a true difference (every condition's; see the
    module docstring)."""
    conditions = [m.condition for m in models]
    by_name = {m.condition.name: m for m in models}
    maximum = evaluation.maximum_information(config)
    batch_games = {m.condition.name: m.batch_games for m in models}
    initial = config.initial_batches if initial_batches is None else initial_batches
    decisions = dict(A=0, B=0, tie=0)
    hours, looks_taken, estimates = 0.0, 0, []
    for _ in range(repeats):
        tallies = {c.name: Tally() for c in conditions}
        break_even_sums = {c.name: 0.0 for c in conditions if c.kind == BREAK_EVEN}

        def add(name: str, games: int) -> None:
            model, tally = by_name[name], tallies[name]
            if model.condition.kind == BREAK_EVEN:
                spread = math.sqrt(model.per_game * games)
                break_even_sums[name] += games * difference + spread * float(
                    torch.randn((), generator=generator, dtype=torch.float64)
                )
            else:
                drawn = _draw_head_to_head(model, difference, games, generator)
                tally.outcomes = [a + b for a, b in zip(tally.outcomes, drawn, strict=True)]
            tally.games += games
            tally.batches += games // batch_games[name]
            tally.steady_games += games
            tally.steady_seconds += games * model.seconds

        def estimates_now() -> dict[str, Estimate]:
            found = {}
            for c in conditions:
                tally = tallies[c.name]
                if c.kind == BREAK_EVEN:
                    variance = by_name[c.name].per_game / tally.games
                    found[c.name] = Estimate(
                        break_even_sums[c.name] / tally.games, variance, by_name[c.name].per_game
                    )
                else:
                    found[c.name] = evaluation.head_to_head_estimate(tally.outcomes)
            return found

        for c in conditions:
            add(c.name, initial * batch_games[c.name])
        earlier: list[float] = []
        while True:
            summary = evaluation.combine(conditions, estimates_now())
            spent = sum(t.steady_seconds for t in tallies.values())
            wanted = evaluation.targets(
                conditions,
                tallies,
                summary,
                evaluation.next_variance(config, len(earlier), maximum),
                batch_games,
            )
            short = {c.name: wanted[c.name] - tallies[c.name].games for c in conditions}
            if any(value > 0 for value in short.values()) and spent < config.budget_hours * 3600:
                for name, value in short.items():
                    if value > 0:
                        add(name, value)
                continue
            if summary.information() / maximum <= (earlier[-1] if earlier else 0.0):
                # no new information (a variance estimate grew): one more batch everywhere
                for c in conditions:
                    add(c.name, batch_games[c.name])
                continue
            result = evaluation.look(config, summary, earlier, spent, maximum)
            earlier.append(result.fraction)
            if result.decision != "continue":
                decisions[result.decision] += 1
                hours += spent / 3600.0
                looks_taken += len(earlier)
                estimates.append(summary.overall[0])
                break
    values = torch.tensor(estimates, dtype=torch.float64)
    return SimulationResult(
        difference,
        decisions["A"] / repeats,
        decisions["B"] / repeats,
        decisions["tie"] / repeats,
        hours / repeats,
        looks_taken / repeats,
        float(values.mean()),
        float(values.std() / math.sqrt(repeats)) if repeats > 1 else math.inf,
    )
