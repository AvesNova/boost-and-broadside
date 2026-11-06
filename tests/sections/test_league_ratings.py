"""The league's ratings (league spec 4, 6 and acceptance 2): recovery of known ratings from
synthetic matches, and the live gauge, which reads the run's fit and holds between fits."""

import math

import torch

from bnb.league.gauge import Gauge, gauge_rating
from bnb.league.ratings import fit_ratings, win_probability
from bnb.runtime.lanes import LIVE
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
    for name, temperature in (("random", math.inf), ("rung_T2", 2.0), ("scripted", 1.0)):
        gauge.add(name, "ladder", gauge_rating(Rung(temperature)))
    return gauge


def test_the_ladder_holds_its_priors_until_the_fit_rates_it():
    gauge = _ladder_gauge()
    assert [rated.rating for rated in gauge.entries.values()] == [0.0, 500.0, 1000.0]
    gauge.update({"rung_T2": (640.0, 30.0), "scripted": (1000.0, 0.0), LIVE: (900.0, 40.0)})
    assert [rated.rating for rated in gauge.entries.values()] == [0.0, 640.0, 1000.0]
    assert gauge.entries["rung_T2"].stderr == 30.0
    assert math.isinf(gauge.entries["random"].stderr)
    gauge.update({LIVE: (950.0, 40.0)})  # a fit that no longer rates the rung
    assert gauge.entries["rung_T2"].rating == 640.0


def test_the_live_rating_is_the_fits_and_holds_between_fits():
    gauge = Gauge(initial_live=850.0)  # a warm start's parent's
    assert gauge.live.rating == 850.0 and math.isinf(gauge.live.stderr)
    gauge.update({"scripted": (1000.0, 0.0)})
    assert gauge.live.rating == 850.0
    gauge.update({LIVE: (700.0, 25.0)})
    assert (gauge.live.rating, gauge.live.stderr) == (700.0, 25.0)
    gauge.update({})
    assert gauge.live.rating == 700.0


def test_a_new_snapshot_starts_at_the_live_rating_until_the_fit_rates_it():
    gauge = _ladder_gauge()
    gauge.update({LIVE: (720.0, 30.0)})
    gauge.add("snapshot_100", "checkpoint")
    assert gauge.entries["snapshot_100"].rating == 720.0
    gauge.update({LIVE: (760.0, 30.0), "snapshot_100": (710.0, 35.0)})
    assert gauge.entries["snapshot_100"].rating == 710.0


def test_an_average_takes_the_live_rating_until_the_fit_rates_it():
    gauge = _ladder_gauge()
    gauge.add("exponential_average", "average")
    gauge.update({LIVE: (700.0, 30.0)})
    assert gauge.entries["exponential_average"].rating == 700.0
    gauge.update({LIVE: (800.0, 30.0), "exponential_average": (690.0, 40.0)})
    gauge.update({LIVE: (850.0, 30.0)})  # its games have left the fit's window
    assert gauge.entries["exponential_average"].rating == 690.0


def test_the_sampling_reads_the_live_policys_odds_against_each_entry():
    gauge = _ladder_gauge()
    gauge.update({LIVE: (500.0, 30.0)})
    expected = win_probability(torch.tensor(500.0), torch.tensor([0.0, 500.0, 1000.0]))
    assert torch.allclose(gauge.win_probabilities(), expected.double())


def test_the_gauge_state_round_trips():
    gauge = _ladder_gauge()
    gauge.add("exponential_average", "average")
    gauge.add("snapshot_100", "checkpoint")
    gauge.update({LIVE: (700.0, 30.0), "snapshot_100": (650.0, 35.0)})
    restored = Gauge.from_state(gauge.state())
    assert restored.state() == gauge.state()
