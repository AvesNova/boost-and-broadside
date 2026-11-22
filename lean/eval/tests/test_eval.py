import torch

from lean.eval import crossover, head_to_head, wilson_interval
from lean.learn.tests import standins


def test_crossover_returns_the_known_break_even_of_a_stand_in():
    # The network beats a scripted team of up to 1.5 times its own size, and no larger.
    def match(network_size, scripted_size, games):
        return games if scripted_size <= 1.5 * network_size else 0

    report = crossover(None, None, None, [8, 2, 4], 10, None, match=match)["sizes"]
    assert [report[size]["beats_up_to"] for size in (2, 4, 8)] == [3, 6, 12]
    assert [report[size]["crossover"] for size in (2, 4, 8)] == [4, 7, 13]
    assert report[4]["ratio"] == 1.5
    assert report[4]["matches"][7] == {"wins": 0, "games": 10, "win_rate": 0.0}


def test_crossover_walks_down_and_counts_exact_half_as_beaten():
    def match(network_size, scripted_size, games):
        return games // 2 if scripted_size < network_size else games // 2 - 1

    report = crossover(None, None, None, [4], 10, None, match=match)["sizes"]
    assert report[4]["beats_up_to"] == 3 and report[4]["crossover"] == 4


def test_head_to_head_counts_sum_to_games_and_ties_are_draws():
    generator = torch.Generator().manual_seed(0)
    policy = standins.TinyPolicy()
    report = head_to_head(standins.GameConfig(), policy, None, (2, 2), 32, generator, game=standins.game, scripted=standins.scripted)
    assert report["wins"] + report["draws"] + report["losses"] == 32
    assert report["wins"] == 0  # the stand-in scripted agent always fires, so it can be tied but not beaten
    low, high = wilson_interval(16, 32)
    assert low < 0.5 < high and abs((low + high) / 2 - 0.5) < 1e-9
