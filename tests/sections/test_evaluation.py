"""The decision suite (`bnb.league.evaluation`, `bnb.league.sequential`,
`bnb.league.evaluation_design`, `bnb evaluate`): the score's fixed weights and its
unbiasedness under the cost-aware allocation, the break-even score, the sequential
boundaries' error rates, the gate, the records (merged per writer, resumed), and a small
evaluation on the CPU."""

import json
import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.league import crossover, evaluation, evaluation_design, sequential
from bnb.league.evaluation import Estimate, Tally
from bnb.league.evaluation_settings import (
    BREAK_EVEN,
    CONDITIONS,
    HEAD_TO_HEAD,
    Condition,
    EvaluationConfig,
)
from bnb.league.matches import player
from bnb.runs import checkpoint
from tests.runs.runs_cases import tiny_model

CPU = torch.device("cpu")


# ---- the score


def test_the_weights_are_a_third_per_mode_shared_equally_within_it():
    weights = evaluation.weights(list(CONDITIONS))
    assert math.isclose(sum(weights.values()), 1.0)
    by_mode = {}
    for condition in CONDITIONS:
        by_mode.setdefault(condition.mode, []).append(weights[condition.name])
    for shares in by_mode.values():
        assert math.isclose(sum(shares), 1 / 3)
        assert all(math.isclose(share, shares[0]) for share in shares)
    assert math.isclose(weights["frontline_5v5"], 1 / 9)
    assert math.isclose(weights["deathmatch_1v1"], 1 / 15)


def test_the_suite_has_the_decided_conditions():
    names = [condition.name for condition in CONDITIONS]
    assert len(names) == 13
    assert sum(condition.kind == BREAK_EVEN for condition in CONDITIONS) == 3
    assert "frontline_1v1" not in names and "frontline_10v10" in names


def test_a_head_to_head_score_is_the_mean_less_one_half_with_its_sample_variance():
    found = evaluation.head_to_head_estimate([6.0, 2.0, 2.0])  # scores 1 x6, 1/2 x2, 0 x2
    scores = torch.tensor([1.0] * 6 + [0.5] * 2 + [0.0] * 2, dtype=torch.float64)
    assert math.isclose(found.score, float(scores.mean()) - 0.5)
    assert math.isclose(found.variance, float(scores.var(unbiased=False)) / 10)
    assert evaluation.head_to_head_estimate([0.0, 0.0, 0.0]) is None
    # the allocation's per-game variance is shrunk toward 1/4, so all-won games keep a cost
    assert evaluation.head_to_head_estimate([10.0, 0.0, 0.0]).per_game > 0.0


def test_the_overall_score_is_the_fixed_weight_mean_whatever_the_games():
    conditions = [c for c in CONDITIONS if c.kind == HEAD_TO_HEAD]
    weights = evaluation.weights(conditions)
    tallies = {}
    for index, condition in enumerate(conditions):
        games = 10 * (index + 1)  # very unequal games
        wins = games * (index % 3) // 2
        tallies[condition.name] = Tally(outcomes=[float(wins), 0.0, float(games - wins)])
    summary = evaluation.summarise(conditions, tallies)
    expected = sum(weights[name] * summary.conditions[name].score for name in weights)
    assert math.isclose(summary.overall[0], expected)
    variance = sum(weights[name] ** 2 * summary.conditions[name].variance for name in weights)
    assert math.isclose(summary.overall[1], math.sqrt(variance))
    for mode, (score, _) in summary.modes.items():
        members = [c.name for c in conditions if c.mode == mode]
        assert math.isclose(score, sum(summary.conditions[n].score for n in members) / len(members))


def _curve_tallies(ratio: float, slope: float, lineups, games: int) -> dict:
    """Exact expected tallies of a logistic curve in the log ratio crossing at `ratio`."""
    return {
        lineup: [
            games / (1 + math.exp(-slope * math.log(lineup[0] / lineup[1] / ratio))),
            float(games),
        ]
        for lineup in lineups
    }


STRONGER_SIDE = [(4, 20), (6, 20), (8, 20), (10, 20), (12, 20), (14, 20), (16, 20)]


def test_the_break_even_score_favours_the_player_needing_fewer_ships():
    tally = Tally()
    tally.sides = dict(
        A=_curve_tallies(0.4, 4.0, STRONGER_SIDE, 200),
        B=_curve_tallies(0.6, 4.0, STRONGER_SIDE, 200),
    )
    tally.games = 2 * 200 * len(STRONGER_SIDE)
    found = evaluation.break_even_estimate(tally)
    # A in B's 12v20 lineup: sigmoid(4 log 1.5); B in A's 8v20: sigmoid(4 log(2/3))
    p = 1 / (1 + math.exp(-4 * math.log(1.5)))
    assert math.isclose(found.score, 0.5 * (p - 0.5) - 0.5 * ((1 - p) - 0.5), rel_tol=1e-4)
    assert found.score > 0 and found.variance > 0
    assert math.isclose(found.per_game, found.variance * tally.games)
    tally.sides["B"] = tally.sides["A"]
    assert abs(evaluation.break_even_estimate(tally).score) < 1e-6
    tally.sides["B"] = {(4, 20): [0.0, 10.0], (16, 20): [10.0, 10.0]}  # separated: no crossing
    assert evaluation.break_even_estimate(tally) is None


def test_the_break_even_score_holds_across_parity():
    """A stronger than the opponent (its break-even 15v20) and B weaker (20v16): each one's
    score is taken in the other's lineup, on opposite sides of parity, and both fits play
    lineups of their own side only."""
    weaker_side = [(20, 20), (20, 18), (20, 16), (20, 14), (20, 12), (20, 10)]
    tally = Tally()
    tally.sides = dict(
        A=_curve_tallies(0.75, 5.0, STRONGER_SIDE, 200),
        B=_curve_tallies(1.25, 3.0, weaker_side, 200),
    )
    tally.games = 200 * (len(STRONGER_SIDE) + len(weaker_side))
    found = evaluation.break_even_estimate(tally)
    # A in B's 20v16 lineup: sigmoid(5 log(1.25 / 0.75)); B in A's 15v20: sigmoid(3 log 0.6)
    a_at_b = 1 / (1 + math.exp(-5 * math.log(1.25 / 0.75)))
    b_at_a = 1 / (1 + math.exp(-3 * math.log(0.75 / 1.25)))
    assert math.isclose(found.score, 0.5 * (a_at_b - 0.5) - 0.5 * (b_at_a - 0.5), rel_tol=1e-4)
    assert found.score > 0.2 and found.variance > 0
    swapped = Tally(sides=dict(A=tally.sides["B"], B=tally.sides["A"]), games=tally.games)
    assert math.isclose(evaluation.break_even_estimate(swapped).score, -found.score, rel_tol=1e-6)


def test_the_break_even_variance_is_the_delta_methods():
    a = torch.tensor([-8.0, 4.0], dtype=torch.float64)
    b = torch.tensor([-10.0, 4.0], dtype=torch.float64)
    step = 1e-6
    for which in (a, b):
        for index in range(2):
            moved = which.clone()
            moved[index] += step
            pair = (moved, b) if which is a else (a, moved)
            numeric = (
                evaluation.break_even_score(*pair) - evaluation.break_even_score(a, b)
            ) / step
            exact = torch.func.grad(
                lambda x, which=which: evaluation.break_even_score(
                    *((x, b) if which is a else (a, x))
                )
            )(which)[index]
            assert math.isclose(float(numeric), float(exact), rel_tol=1e-4, abs_tol=1e-8)


def _models(costs, variances, draws=0.2, batch=64):
    conditions = [c for c in CONDITIONS if c.kind == HEAD_TO_HEAD][: len(costs)]
    return [
        evaluation_design.ConditionModel(c, v, s, draws, batch)
        for c, s, v in zip(conditions, costs, variances, strict=True)
    ]


def test_the_cost_aware_allocation_leaves_the_score_unbiased():
    """Games added by estimated variance per second, from noisy first batches, over
    conditions of different true means: the fixed-weight estimate's mean is the true
    weighted mean (within its Monte Carlo error)."""
    generator = torch.Generator().manual_seed(0)
    models = _models([0.1, 1.0, 4.0, 0.3, 2.0], [0.2] * 5, draws=0.1)
    conditions = [m.condition for m in models]
    means = [0.15, -0.1, 0.05, 0.2, -0.2]
    weights = evaluation.weights(conditions)
    truth = sum(weights[c.name] * mean for c, mean in zip(conditions, means, strict=True))
    estimates = []
    for _ in range(400):
        tallies = {c.name: Tally() for c in conditions}

        def add(model, mean, games):
            drawn = evaluation_design._draw_head_to_head(model, mean, games, generator)
            tally = tallies[model.condition.name]
            tally.outcomes = [a + b for a, b in zip(tally.outcomes, drawn, strict=True)]
            tally.games += games
            tally.steady_games += games
            tally.steady_seconds += games * model.seconds

        for model, mean in zip(models, means, strict=True):
            add(model, mean, 16)
        summary = evaluation.summarise(conditions, tallies)
        wanted = evaluation.targets(
            conditions, tallies, summary, 1e-4, {c.name: 16 for c in conditions}
        )
        for model, mean in zip(models, means, strict=True):
            add(model, mean, wanted[model.condition.name] - tallies[model.condition.name].games)
        estimates.append(evaluation.summarise(conditions, tallies).overall[0])
    values = torch.tensor(estimates, dtype=torch.float64)
    assert abs(float(values.mean()) - truth) < 3 * float(values.std()) / math.sqrt(len(values))


def test_the_allocation_puts_games_where_they_are_cheap_and_reaches_the_variance():
    models = _models([0.1, 1.0, 10.0], [0.2, 0.2, 0.2])
    conditions = [m.condition for m in models]
    tallies = {}
    for model in models:
        tally = Tally(outcomes=[32.0, 0.0, 32.0], games=64, steady_games=64)
        tally.steady_seconds = 64 * model.seconds
        tallies[model.condition.name] = tally
    summary = evaluation.summarise(conditions, tallies)
    wanted = evaluation.targets(
        conditions, tallies, summary, 1e-4, {c.name: 64 for c in conditions}
    )
    games = [wanted[c.name] for c in conditions]
    assert games[0] > games[1] > games[2]
    weights = evaluation.weights(conditions)
    variance = sum(
        weights[c.name] ** 2 * summary.conditions[c.name].per_game / wanted[c.name]
        for c in conditions
    )
    assert variance <= 1e-4
    # cost-optimal games go as w sigma / sqrt(cost): about sqrt(10) between neighbours here
    assert 2.0 < games[0] / games[1] < 5.0


def test_the_cost_per_information_is_the_cost_optimal_allocations():
    conditions = [c for c in CONDITIONS if c.kind == HEAD_TO_HEAD][:3]
    per_game = {c.name: 0.2 for c in conditions}
    seconds = {c.name: s for c, s in zip(conditions, [0.1, 1.0, 10.0], strict=True)}
    weights = evaluation.weights(conditions)
    found = evaluation.cost_per_information(conditions, per_game, seconds)
    # n_c = lambda w sigma / sqrt(cost) at variance 1 (information 1): the cost is found
    scale = sum(weights[n] * math.sqrt(per_game[n] * seconds[n]) for n in per_game)
    games = {n: scale * weights[n] * math.sqrt(per_game[n] / seconds[n]) for n in per_game}
    variance = sum(weights[n] ** 2 * per_game[n] / games[n] for n in per_game)
    assert math.isclose(variance, 1.0)
    assert math.isclose(sum(games[n] * seconds[n] for n in per_game), found)


# ---- the sequential test


def test_one_look_is_the_fixed_sample_test():
    drift, found = sequential.plan(sequential.SequentialDesign(looks=1, alpha=0.05, beta=0.2))
    assert math.isclose(found.upper[0], 1.959964, abs_tol=1e-3)
    assert math.isclose(drift, 1.959964 + 0.841621, abs_tol=2e-3)
    assert found.inner == found.upper


def test_the_spending_functions_spend_alpha_and_beta_by_the_last_look():
    assert math.isclose(sequential.alpha_spent(1.0, 0.05), 0.05)
    assert math.isclose(sequential.beta_spent(1.0, 0.2), 0.2)
    assert sequential.alpha_spent(0.25, 0.05) < 1e-3  # O'Brien-Fleming: little early


@pytest.mark.parametrize("looks", [2, 4])
def test_the_boundaries_keep_their_error_rates_under_the_null_and_at_delta(looks):
    design = sequential.SequentialDesign(looks=looks, alpha=0.05, beta=0.2, delta=0.02)
    drift, found = sequential.plan(design)
    generator = torch.Generator().manual_seed(looks)
    paths = 200_000
    null = sequential.simulate_paths(found, 0.0, paths, generator)
    margin = 4 * math.sqrt(0.05 * 0.95 / paths)
    assert abs(null.a_wins + null.b_wins - 0.05) < margin
    assert abs(null.a_wins - null.b_wins) < margin
    at_delta = sequential.simulate_paths(found, drift, paths, generator)
    assert abs(at_delta.tie - 0.2) < 4 * math.sqrt(0.2 * 0.8 / paths)
    assert at_delta.b_wins < 1e-3
    # the grid computation agrees with the paths
    exact = sequential.operating_characteristics(found, drift)
    assert abs(exact.tie - at_delta.tie) < 4 * math.sqrt(0.2 * 0.8 / paths)
    assert abs(exact.expected_fraction - at_delta.expected_fraction) < 0.01
    # stopping early saves information when the difference is large
    assert sequential.operating_characteristics(found, 2 * drift).expected_fraction < 0.7


def test_the_boundaries_at_unplanned_fractions_still_spend_alpha():
    design = sequential.SequentialDesign(looks=3)
    drift, _ = sequential.plan(design)
    found = sequential.boundaries(design, [0.4, 0.55, 1.1], drift)
    null = sequential.operating_characteristics(found, 0.0)
    assert abs(null.a_wins + null.b_wins - 0.05) < 2e-3
    assert found.decision(2, 3.0) == "A" and found.decision(2, -3.0) == "B"
    assert found.decision(2, 0.0) == "tie"
    early = sequential.boundaries(design, [0.4], drift, final=False)
    assert early.decision(0, 1.0) == "continue"


# ---- looks and the gate


def _summary(modes: dict[str, tuple[float, float]]) -> evaluation.Summary:
    """A summary with one condition per mode at these (score, standard error)."""
    conditions = [Condition(f"{mode}_test", mode, HEAD_TO_HEAD, 5) for mode in modes]
    found = {
        f"{mode}_test": Estimate(score, stderr**2, stderr**2 * 100)
        for mode, (score, stderr) in modes.items()
    }
    return evaluation.combine(conditions, found)


def test_the_gate_holds_a_winner_worse_in_a_mode_to_a_tie():
    config = EvaluationConfig(looks=1, budget_hours=1.0)
    maximum = evaluation.maximum_information(config)
    clear = _summary(
        dict(frontline=(0.06, 0.01), deathmatch=(0.06, 0.01), elimination=(0.03, 0.01))
    )
    found = evaluation.look(config, clear, [], 0.0, maximum)
    assert (
        found.test == "A"
        and found.decision == "A"
        and found.gate == dict(A=[], B=["frontline", "deathmatch", "elimination"])
    )
    split = _summary(
        dict(frontline=(0.12, 0.01), deathmatch=(0.12, 0.01), elimination=(-0.03, 0.01))
    )
    held = evaluation.look(config, split, [], 0.0, maximum)
    assert held.test == "A" and held.gate["A"] == ["elimination"] and held.decision == "tie"


def test_a_look_continues_until_the_information_or_the_budget_is_reached():
    config = EvaluationConfig(looks=4, budget_hours=1.0)
    maximum = evaluation.maximum_information(config)
    overall = math.sqrt(1 / (0.3 * maximum))  # fraction 0.3
    mode = (1.5 * overall, math.sqrt(3) * overall)  # Z 1.5: between the boundaries
    noisy = _summary(dict(frontline=mode, deathmatch=mode, elimination=mode))
    found = evaluation.look(config, noisy, [], 0.0, maximum)
    assert found.decision == "continue" and not found.last and found.fraction < 1.0
    spent = evaluation.look(config, noisy, [], 3600.0, maximum)
    assert spent.last and spent.decision in ("A", "B", "tie")
    with pytest.raises(ValueError, match="no new information"):
        evaluation.look(config, noisy, [found.fraction + 0.1], 0.0, maximum)


# ---- the procedure simulated


@pytest.mark.slow
def test_the_simulated_procedure_keeps_its_error_rates():
    """The whole suite simulated (estimated variances, batch steps, the allocation): a
    winner about alpha of the time with no difference, a tie about beta of the time at delta."""
    config = EvaluationConfig(
        conditions=tuple(c.name for c in CONDITIONS if c.kind == HEAD_TO_HEAD)[:6], looks=3
    )
    models = _models([0.05, 0.05, 0.03, 0.01, 0.02, 0.01], [0.2] * 6, batch=128)
    generator = torch.Generator().manual_seed(1)
    null = evaluation_design.simulate(config, models, 0.0, 600, generator)
    assert null.a_wins + null.b_wins < 0.05 + 3 * math.sqrt(0.05 * 0.95 / 600)
    at_delta = evaluation_design.simulate(config, models, config.delta, 600, generator)
    assert abs(at_delta.tie - 0.2) < 3 * math.sqrt(0.2 * 0.8 / 600) + 0.02


def test_the_simulated_procedure_decides_a_large_difference_early():
    config = EvaluationConfig(
        conditions=tuple(c.name for c in CONDITIONS if c.kind == HEAD_TO_HEAD)[:3], looks=3
    )
    models = _models([0.05, 0.05, 0.03], [0.2] * 3, batch=128)
    found = evaluation_design.simulate(config, models, 0.1, 20, torch.Generator().manual_seed(2))
    assert found.a_wins == 1.0 and found.looks < 2.0


# ---- records


def _config(**changes) -> EvaluationConfig:
    """Tiny games: one head-to-head condition per mode at 1v1 or 2v2 and deathmatch's
    break-even with the weaker side at 3 ships (a 3-a-side game)."""
    return EvaluationConfig(
        **{
            **dict(
                conditions=("deathmatch_1v1", "elimination_1v1", "deathmatch_break_even"),
                presets=("tiny_frontline", "tiny_deathmatch", "tiny_elimination"),
                lineup_types=("Fighter", "Heavy"),
                fixed_count=3,
                bracket=(1, 2, 3),
                head_to_head_blocks=2,
                head_to_head_environments=8,
                break_even_blocks=4,
                break_even_environments=8,
                looks=2,
                delta=0.2,
                budget_hours=0.05,
                initial_batch_limit=20,
            ),
            **changes,
        }
    )


@pytest.fixture
def tiny_presets(monkeypatch):
    """Tiny presets of the three modes (3 ships a side, 12 s, one kill per ship)."""
    for mode, name in (
        ("frontline", "tiny_frontline"),
        ("team_deathmatch", "tiny_deathmatch"),
        ("elimination", "tiny_elimination"),
    ):
        respawn = None if mode == "elimination" else 1.0
        game = GameConfig(
            3, 2, 400.0, episode_seconds=12.0, respawn_seconds=respawn, kills_per_ship=1, mode=mode
        )
        monkeypatch.setitem(api.PRESETS, name, game)


def _line(condition: str, batch: int, writer: str, **extra) -> dict:
    return dict(
        condition=condition,
        batch=batch,
        writer=writer,
        games=8,
        seconds=1.0,
        first=False,
        blocks=[dict(wins=4, draws=0, losses=4)],
        time="2026-11-15T00:00:00+00:00",
        **extra,
    )


def test_the_records_merge_every_writer_once(tmp_path):
    batches = tmp_path / "batches"
    batches.mkdir()
    (batches / "one.jsonl").write_text(
        json.dumps(_line("deathmatch_1v1", 0, "one"))
        + "\n"
        + json.dumps(_line("deathmatch_1v1", 1, "one"))
        + "\n"
    )
    (batches / "two.jsonl").write_text(
        json.dumps(_line("elimination_1v1", 0, "two"))
        + "\n"
        + json.dumps(_line("deathmatch_1v1", 1, "two"))  # a repeat of one's batch 1
        + "\n"
        + '{"condition": "elim'  # a write in progress
    )
    found, damaged, repeats = evaluation.read_batches(tmp_path)
    assert len(found) == 3 and damaged == 0 and repeats == 1
    tallies = evaluation.tallies_of(found)
    assert tallies["deathmatch_1v1"].games == 16 and tallies["elimination_1v1"].games == 8
    assert tallies["deathmatch_1v1"].seconds_per_game() == 1 / 8


def test_settings_are_written_once_and_checked(tmp_path, tiny_presets):
    players = dict(A=player("scripted"), B=player("rung_T2"))
    entry = evaluation.settings_entry(players, _config())
    evaluation.open_directory(tmp_path, entry)
    evaluation.open_directory(tmp_path, entry)  # the same: fine
    _, config = evaluation.read_settings(tmp_path)
    assert config == _config()
    with pytest.raises(ValueError, match="config"):
        evaluation.open_directory(tmp_path, evaluation.settings_entry(players, _config(looks=3)))
    swapped = evaluation.settings_entry(dict(A=player("scripted"), B=player("rung_T3")), _config())
    with pytest.raises(ValueError, match="player B"):
        evaluation.open_directory(tmp_path, swapped)


@pytest.mark.slow
@pytest.mark.usefixtures("global_generator")
def test_a_pilot_records_every_batch_and_resumes_past_it(tmp_path, tiny_presets):
    players = dict(A=player("scripted"), B=player("rung_T2"))
    config = _config(conditions=("deathmatch_1v1", "deathmatch_break_even"))
    printed = []
    suite = evaluation.Evaluation(
        tmp_path, players, player("scripted"), config, CPU, progress=printed.append, compiled=False
    )
    suite.fixed(16)
    found, _, _ = evaluation.read_batches(tmp_path)
    # a head-to-head batch is 2 x 8 games, a break-even batch 4 x 8
    assert [(b["condition"], b["batch"]) for b in found] == [
        ("deathmatch_1v1", 0),
        ("deathmatch_break_even", 0),
    ]
    assert found[0]["first"] and found[1]["first"]
    sides = [block["side"] for block in found[1]["blocks"]]
    assert sides == ["A", "A", "B", "B"]
    assert all(b["wins"] + b["draws"] + b["losses"] == 8 for line in found for b in line["blocks"])
    # another process continues each condition past its recorded batches
    again = evaluation.Evaluation(
        tmp_path, players, player("scripted"), config, CPU, progress=printed.append, compiled=False
    )
    again.fixed(48)
    found, _, repeats = evaluation.read_batches(tmp_path)
    assert repeats == 0
    numbers = sorted(b["batch"] for b in found if b["condition"] == "deathmatch_1v1")
    assert numbers == [0, 1, 2]
    assert any("ETA" in text for text in printed)
    assert [b["first"] for b in found if b["condition"] == "deathmatch_1v1"] == [True, True, False]
    assert len({line["writer"] for line in found}) == 2


@pytest.mark.usefixtures("global_generator")
def test_the_lineups_are_mirrored_and_keyed_by_condition_and_batch(tiny_presets):
    config = _config()
    condition = evaluation_settings_condition("deathmatch_1v1")
    first = evaluation.lineups(config, condition, 3, 16, (0, 1))
    assert torch.equal(first, evaluation.lineups(config, condition, 3, 16, (0, 1)))
    assert not torch.equal(first, evaluation.lineups(config, condition, 4, 16, (0, 1)))
    other = evaluation_settings_condition("elimination_1v1")
    many = evaluation.lineups(config, other, 3, 256, (0, 1))
    assert 0.4 < float(many.float().mean()) < 0.6
    assert torch.equal(
        crossover.lineup_draw(0, 3, 16, 1, (0, 1)),
        crossover.lineup_draw(0, 3, 16, 1, (0, 1), stream="crossover_lineup"),
    )


def test_every_condition_plays_its_presets_own_arena_unless_the_density_rule_is_asked():
    fixed = EvaluationConfig()
    for condition in CONDITIONS:
        base = api.PRESETS[fixed.presets[evaluation.MODES.index(condition.mode)]]
        game = evaluation.game_of(condition, fixed)
        assert game.arena_radius == base.arena_radius
        size = condition.ships_per_team or fixed.crossover_config().ships_per_team
        assert game.ships_per_team == size
    density = EvaluationConfig(map_rule="density")
    small = evaluation.game_of(evaluation_settings_condition("elimination_1v1"), density)
    base = api.PRESETS["real_scale_elimination"]
    assert small.arena_radius == api.arena_radius_for(base, 1) < base.arena_radius
    assert evaluation.game_of(evaluation_settings_condition("elimination_5v5"), density) == base
    # break-even's density map is the fixed side's fleet's
    wide = evaluation.game_of(evaluation_settings_condition("elimination_break_even"), density)
    assert wide.arena_radius == api.arena_radius_for(base, density.fixed_count)
    with pytest.raises(ValueError, match="map rule"):
        EvaluationConfig(map_rule="scaled")


def test_an_unscored_condition_is_neither_scored_nor_played(tmp_path, tiny_presets):
    default = EvaluationConfig()
    assert default.unscored == ()
    assert len(default.scored()) == 13
    assert math.isclose(evaluation.weights(default.scored())["frontline_5v5"], 1 / 9)
    left_out = EvaluationConfig(unscored=("frontline_break_even",))
    names = [condition.name for condition in left_out.scored()]
    assert len(names) == 12 and "frontline_break_even" not in names
    weights = evaluation.weights(left_out.scored())
    assert math.isclose(weights["frontline_5v5"], 1 / 6)
    with pytest.raises(ValueError, match="unknown"):
        EvaluationConfig(unscored=("frontline_1v1",))
    with pytest.raises(ValueError, match="every one is unscored"):
        EvaluationConfig(conditions=("frontline_break_even",), unscored=("frontline_break_even",))
    config = _config(unscored=("deathmatch_break_even",))
    players = dict(A=player("scripted"), B=player("rung_T2"))
    suite = evaluation.Evaluation(tmp_path, players, player("scripted"), config, CPU)
    assert "deathmatch_break_even" not in [c.name for c in suite.conditions]
    with pytest.raises(ValueError, match="not among the scored"):
        evaluation.Evaluation(
            tmp_path, players, player("scripted"), config, CPU, own=("deathmatch_break_even",)
        )


def evaluation_settings_condition(name: str) -> Condition:
    from bnb.league.evaluation_settings import BY_NAME

    return BY_NAME[name]


@pytest.mark.slow
@pytest.mark.usefixtures("global_generator")
def test_a_small_evaluation_decides_and_decide_reads_it(tmp_path, tiny_presets):
    players = dict(A=player("scripted"), B=player("rung_T2"))
    config = _config()
    printed = []
    suite = evaluation.Evaluation(
        tmp_path, players, player("scripted"), config, CPU, progress=printed.append, compiled=False
    )
    result = suite.run(poll_seconds=0.0, sleep=lambda seconds: None)
    assert result.decision in ("A", "B", "tie")
    looks = evaluation.read_looks(tmp_path)
    assert looks[-1]["decision"] == result.decision
    summary, again, tallies = evaluation.decide(tmp_path)
    assert again == result and summary.overall is not None
    assert len(evaluation.read_looks(tmp_path)) == len(looks)  # a decided test adds no look
    lines = evaluation.table(config.scored(), summary, tallies)
    assert lines[-1].startswith("overall")


@pytest.mark.usefixtures("global_generator")
def test_a_process_with_nothing_of_its_own_to_play_waits(tmp_path, tiny_presets):
    players = dict(A=player("scripted"), B=player("rung_T2"))
    config = _config(conditions=("deathmatch_1v1", "elimination_1v1"))
    printed, sleeps = [], []
    suite = evaluation.Evaluation(
        tmp_path,
        players,
        player("scripted"),
        config,
        CPU,
        own=("deathmatch_1v1",),
        progress=printed.append,
        compiled=False,
    )

    def sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt  # stop the test at the first wait

    with pytest.raises(KeyboardInterrupt):
        suite.run(poll_seconds=5.0, sleep=sleep)
    assert sleeps == [5.0] and "elimination_1v1" in printed[-1]
    found, _, _ = evaluation.read_batches(tmp_path)
    assert {line["condition"] for line in found} == {"deathmatch_1v1"}


@pytest.mark.slow
@pytest.mark.usefixtures("global_generator")
def test_networks_play_both_kinds_of_condition(tmp_path, tiny_presets):
    game = api.PRESETS["tiny_deathmatch"]
    paths = []
    for seed in (1, 2):
        directory = tmp_path / f"policy_{seed}"
        checkpoint.save_policy(directory, tiny_model("yemong", seed=seed, game=game))
        paths.append(str(directory))
    players = dict(A=player(paths[0]), B=player(paths[1]))
    config = _config(conditions=("deathmatch_1v1", "deathmatch_break_even"))
    for name, lanes in (("deathmatch_1v1", (4,)), ("deathmatch_break_even", (4,))):
        condition = evaluation_settings_condition(name)
        field = [players["A"], players["B"]] + (
            [player("scripted")] if condition.kind == BREAK_EVEN else []
        )
        blocks = config.layout(condition).blocks
        assert evaluation._lanes(field, condition, blocks) == lanes
    suite = evaluation.Evaluation(
        tmp_path / "out",
        players,
        player("scripted"),
        config,
        CPU,
        progress=lambda text: None,
        compiled=False,
    )
    suite.fixed(8)
    held = json.loads((tmp_path / "out" / "settings.json").read_text())
    assert held["players"]["A"]["identity"].startswith("sha256:")


# ---- the command


def test_the_command_parses_its_options_with_the_decided_defaults():
    import tyro

    from bnb.cli import Evaluate, EvaluateDecide, EvaluateDesign

    options = tyro.cli(Evaluate, args=["A", "B", "--out", "DIR", "--play", "deathmatch_1v1"])
    assert options.settings == EvaluationConfig() and options.play == ("deathmatch_1v1",)
    assert options.settings.alpha == 0.05 and options.settings.delta == 0.02
    assert options.settings.fixed_count == 20 and options.settings.gate_stderrs == 2.0
    assert options.settings.minimum_count == 1 and options.settings.map_rule == "preset"
    changed = tyro.cli(Evaluate, args=["A", "B", "--out", "DIR", "--looks", "3"])
    assert changed.settings.looks == 3
    assert tyro.cli(EvaluateDecide, args=["DIR", "--peek"]).peek
    assert tyro.cli(EvaluateDesign, args=["D1", "D2"]).directories == ("D1", "D2")


def test_the_design_reads_a_pilot(tmp_path, tiny_presets):
    players = dict(A=player("scripted"), B=player("rung_T2"))
    config = _config(conditions=("deathmatch_1v1", "elimination_1v1"))
    evaluation.open_directory(tmp_path, evaluation.settings_entry(players, config))
    batches = tmp_path / "batches"
    batches.mkdir()
    lines = [
        dict(_line("deathmatch_1v1", 0, "w"), blocks=[dict(wins=5, draws=2, losses=1)]),
        dict(
            _line("elimination_1v1", 0, "w"), seconds=4.0, blocks=[dict(wins=3, draws=0, losses=5)]
        ),
    ]
    (batches / "w.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    _, models, summary = evaluation_design.pilot_models(tmp_path)
    assert [m.condition.name for m in models] == ["deathmatch_1v1", "elimination_1v1"]
    assert math.isclose(models[0].draws, 0.25) and math.isclose(models[1].seconds, 0.5)
    rows, plans = evaluation_design.design_table(config, models, (2, 3), (0.0, 0.2))
    assert len(rows) == 4 and set(plans) == {2, 3}
    at_null = [row for row in rows if row.difference == 0.0]
    assert all(abs(row.a_wins + row.b_wins - 0.05) < 2e-3 for row in at_null)
    assert all(row.expected_hours <= row.maximum_hours for row in rows)
    assert summary.overall is not None
