import torch

from lean import game, model, scripted
from lean.eval import crossover, head_to_head, play_matches, play_results, wilson_interval

GAME = game.GameConfig(clock_ticks=20)  # short games


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


def test_head_to_head_counts_sum_to_games():
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    policy = model.Policy(model.ModelConfig())
    report = head_to_head(GAME, policy, scripted.ScriptedConfig(), (2, 2), 16, generator)
    assert report["wins"] + report["draws"] + report["losses"] == 16
    assert report["win_rate_low"] <= report["win_rate"] <= report["win_rate_high"]
    low, high = wilson_interval(16, 32)
    assert low < 0.5 < high and abs((low + high) / 2 - 0.5) < 1e-9


def test_crossover_plays_every_size_in_rounds_of_batched_matches():
    # Every round is one batch: a walk ending near parity needs one round, a far one more.
    report = crossover(None, None, None, [2, 4, 8], 10, None, match=lambda n, s, g: g if s <= 1.5 * n else 0)
    assert report["rounds"] == 2
    near = crossover(None, None, None, [2, 4, 8], 10, None, match=lambda n, s, g: g if s <= n else 0)
    assert near["rounds"] == 1 and [near["sizes"][n]["beats_up_to"] for n in (2, 4, 8)] == [2, 4, 8]
    # The start's neighbours are played and kept: `spread` (2) sizes either side of parity.
    assert sorted(near["sizes"][4]["matches"]) == [2, 3, 4, 5, 6]
    # Started from the previous evaluation's ratios: the same answer in one round.
    ratios = {n: entry["ratio"] for n, entry in report["sizes"].items()}
    again = crossover(None, None, None, [2, 4, 8], 10, None, start_ratio=ratios, match=lambda n, s, g: g if s <= 1.5 * n else 0)
    assert again["rounds"] == 1
    assert [again["sizes"][n]["beats_up_to"] for n in (2, 4, 8)] == [report["sizes"][n]["beats_up_to"] for n in (2, 4, 8)]
    losing = crossover(None, None, None, [4], 10, None, start_ratio={4: 0.0}, match=lambda n, s, g: 0)
    assert losing["rounds"] == 1 and losing["sizes"][4]["beats_up_to"] == 0


def test_play_matches_counts_each_size_of_a_mixed_batch():
    matches = [(1, 3, 6), (3, 1, 6), (2, 2, 4)]
    torch.manual_seed(0)
    policy = model.Policy(model.ModelConfig())
    opponent = scripted.ScriptedConfig()
    wins = play_matches(GAME, policy, opponent, matches, torch.Generator().manual_seed(3))
    # The same draws as one batch of environments, each keeping its own team sizes.
    sizes = torch.tensor([[1, 3]] * 6 + [[3, 1]] * 6 + [[2, 2]] * 4)
    result = play_results(GAME, policy, opponent, sizes, torch.Generator().manual_seed(3))
    assert (result >= 0).all()  # every game ended at the clock or before
    assert wins == [int((result[:6] == 0).sum()), int((result[6:12] == 0).sum()), int((result[12:] == 0).sum())]
    # A cap on the batch splits between matches, never inside one.
    split = play_matches(GAME, policy, opponent, matches, torch.Generator().manual_seed(3), maximum_games=8)
    assert len(split) == 3 and all(0 <= won <= games for won, (_, _, games) in zip(split, matches, strict=True))
