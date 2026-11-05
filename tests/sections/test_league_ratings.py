"""The league's ratings (league spec 4, 6 and acceptance 2): recovery of known ratings from
synthetic matches, the live gauge's two stages, the averages' window and sweeps."""

import math

import torch

from bnb.league import gauge as gauges
from bnb.league.gauge import Gauge, Record, gauge_rating
from bnb.league.ratings import fit_ratings, rate_against, rate_live, win_probability
from bnb.runtime.opponents import Rung

TRIALS = 200
COVERAGE = 0.9  # acceptance 2: within two standard errors in at least 90% of trials


def _results(rating: torch.Tensor, opponents: torch.Tensor, games: torch.Tensor, generator):
    """Wins, losses and ties (each (K,)) of a player rated `rating` against `opponents`, with
    `games` (K,) per opponent. A game ties with probability `t = 0.2 min(p, 1 - p)` and is
    otherwise won with probability `(p - t/2) / (1 - t)`, so the expected score, a tie
    counting half, is the rating model's `p`."""
    probability = win_probability(rating, opponents)
    tie = 0.2 * torch.minimum(probability, 1.0 - probability)
    tied = torch.binomial(games, tie, generator=generator)
    decided = games - tied
    wins = torch.binomial(decided, (probability - tie / 2) / (1.0 - tie), generator=generator)
    return wins, decided - wins, tied


def test_the_live_rating_recovers_a_known_rating():
    generator = torch.Generator().manual_seed(1)
    opponents = torch.tensor([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1150.0], dtype=torch.float64)
    games = torch.full((7,), 12.0, dtype=torch.float64)
    covered = 0
    for _ in range(TRIALS):
        truth = torch.rand((), generator=generator, dtype=torch.float64) * 800.0 + 200.0
        wins, losses, ties = _results(truth, opponents, games, generator)
        rating, stderr = rate_live(opponents, wins, losses, ties)
        covered += math.isfinite(rating) and abs(rating - float(truth)) < 2.0 * stderr
    assert covered >= COVERAGE * TRIALS, covered


def test_the_offline_fit_recovers_known_ratings():
    generator = torch.Generator().manual_seed(2)
    players, anchor = 6, 2
    covered = total = 0
    for _ in range(TRIALS):
        truth = torch.rand(players, generator=generator, dtype=torch.float64) * 800.0 - 400.0
        truth -= truth[anchor].clone()
        wins = torch.zeros((players, players), dtype=torch.float64)
        for first in range(players):
            for second in range(first + 1, players):
                won, lost, tied = _results(
                    truth[first : first + 1],
                    truth[second : second + 1],
                    torch.tensor([30.0], dtype=torch.float64),
                    generator,
                )
                wins[first, second] = won[0] + tied[0] / 2
                wins[second, first] = lost[0] + tied[0] / 2
        ratings, stderr = fit_ratings(wins, anchor, 1.0)
        others = torch.arange(players) != anchor
        error = (ratings - truth).abs()[others]
        covered += int((error < 2.0 * stderr[others]).sum())
        total += players - 1
        assert float(ratings[anchor]) == 0.0 and float(stderr[anchor]) == 0.0
    assert covered >= COVERAGE * total, (covered, total)


def _ladder_gauge() -> Gauge:
    gauge = Gauge()
    # the priors 1000 / T: random 0, T = 2 at 500, the scripted agent 1000
    for name, temperature in (("0.0", math.inf), ("0.5", 2.0), ("1.0", 1.0)):
        gauge.add(f"ladder/{name}", "ladder", gauge_rating(Rung(temperature)))
    return gauge


def test_the_ladder_keeps_its_fixed_ratings():
    gauge = _ladder_gauge()
    gauge.update({"ladder/0.0": Record(wins=9), "ladder/0.5": Record(wins=5, losses=4)})
    gauge.add("checkpoint/1", "checkpoint")
    gauge.credit("checkpoint/1", {"ladder/0.0": Record(wins=9), "ladder/1.0": Record(losses=8)})
    gauge.update({"ladder/1.0": Record(wins=3, losses=3)})
    assert [gauge.entries[f"ladder/{p}"].rating for p in (0.0, 0.5, 1.0)] == [0.0, 500.0, 1000.0]
    assert 0.0 < gauge.entries["checkpoint/1"].rating < 1000.0


def test_a_new_snapshot_starts_at_the_live_rating_of_its_update():
    gauge = _ladder_gauge()
    gauge.update({"ladder/0.5": Record(wins=6, losses=4), "ladder/1.0": Record(wins=1, losses=5)})
    gauge.add("checkpoint/1", "checkpoint")
    assert gauge.entries["checkpoint/1"].rating == gauge.live.rating
    assert 500.0 < gauge.live.rating < 1000.0


def test_the_live_rating_uses_only_this_update():
    gauge = _ladder_gauge()
    gauge.update({"ladder/0.5": Record(wins=1, losses=9)})
    first = gauge.live.rating
    gauge.update({"ladder/0.5": Record(wins=9, losses=1)})
    expected, _ = rate_live([500.0], [9.0], [1.0], [0.0])
    assert gauge.live.rating == expected and first < 500.0 < expected


def test_a_sweep_reports_infinity_and_holds_the_last_finite_rating():
    gauge = _ladder_gauge()
    gauge.update({"ladder/0.5": Record(wins=6, losses=4)})
    held = gauge.live.rating
    gauge.update({"ladder/0.5": Record(wins=10)})
    assert gauge.live.reported == math.inf and gauge.live.stderr == math.inf
    assert gauge.live.rating == held


def test_an_average_without_games_takes_the_live_rating():
    gauge = _ladder_gauge()
    gauge.add("average/ema", "average")
    gauge.update({"ladder/0.5": Record(wins=7, losses=3)})
    assert gauge.entries["average/ema"].rating == gauge.live.rating


def test_an_average_is_rated_on_its_window_against_the_live_ratings():
    gauge = _ladder_gauge()
    gauge.add("average/ema", "average")
    for _ in range(30):  # 300 games: the window keeps the newest AVERAGE_WINDOW
        gauge.update(
            {"ladder/0.5": Record(wins=5, losses=5), "average/ema": Record(wins=2, losses=8)}
        )
    window = list(gauge.windows["average/ema"])
    assert len(window) == gauges.AVERAGE_WINDOW
    # the newest games, each against the live rating of its own update
    opponents = torch.tensor([live for live, _ in window], dtype=torch.float64)
    score = torch.tensor([result for _, result in window], dtype=torch.float64)
    assert opponents[-1] == gauge.live.rating and float(score.mean()) == 0.8
    expected, _ = rate_against(opponents, score, 1.0 - score)
    assert gauge.entries["average/ema"].rating == expected > float(opponents.max())


def test_averages_stay_out_of_the_ladder_record():
    gauge = _ladder_gauge()
    gauge.add("average/ema", "average")
    gauge.add("checkpoint/1", "checkpoint")
    gauge.credit("checkpoint/1", {"average/ema": Record(wins=5), "ladder/0.5": Record(wins=2)})
    assert set(gauge.wins) == {("checkpoint/1", "ladder/0.5"), ("ladder/0.5", "checkpoint/1")}


def test_the_gauge_state_round_trips():
    gauge = _ladder_gauge()
    gauge.add("average/ema", "average")
    gauge.add("checkpoint/1", "checkpoint")
    gauge.credit("checkpoint/1", {"ladder/0.5": Record(wins=3, losses=1, ties=1)})
    gauge.update({"ladder/0.5": Record(wins=3, losses=2), "average/ema": Record(wins=1, ties=1)})
    restored = Gauge.from_state(gauge.state())
    assert restored.state() == gauge.state()
