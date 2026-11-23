import math

import torch

import lean.eval.store as store_module
from lean import game, model, scripted
from lean.eval import (
    Evaluator,
    NetworkAgent,
    Queue,
    ScriptedAgent,
    SobolStream,
    Store,
    allocate,
    bradley_terry,
    crossover,
    play,
    standard_error,
)
from lean.eval.runner import A_WON, DRAW, NOT_PLAYED
from lean.eval.store import conditions_of
from lean.scripted.parameters import PARAMETERS, rows_of, vector

GAME = game.GameConfig(clock_ticks=20)  # short games


def stand_in(match):
    """A crossover round from a per-match rule (network size, scripted size, games) -> wins."""
    return lambda wanted: [(match(n, s, g), g) for n, s, g in wanted]


def test_crossover_returns_the_known_break_even_of_a_stand_in():
    # The network beats a scripted team of up to 1.5 times its own size, and no larger.
    report = crossover(stand_in(lambda n, s, g: g if s <= 1.5 * n else 0), [8, 2, 4], 10)["sizes"]
    assert [report[size]["beats_up_to"] for size in (2, 4, 8)] == [3, 6, 12]
    assert [report[size]["crossover"] for size in (2, 4, 8)] == [4, 7, 13]
    assert report[4]["ratio"] == 1.5
    assert report[4]["matches"][7] == {"wins": 0, "games": 10, "win_rate": 0.0}


def test_crossover_walks_down_and_counts_exact_half_as_beaten():
    report = crossover(stand_in(lambda n, s, g: g // 2 if s < n else g // 2 - 1), [4], 10)["sizes"]
    assert report[4]["beats_up_to"] == 3 and report[4]["crossover"] == 4


def test_crossover_plays_every_size_in_rounds_of_batched_matches():
    # Every round is one call: a walk ending near parity needs one round, a far one more.
    far = stand_in(lambda n, s, g: g if s <= 1.5 * n else 0)
    report = crossover(far, [2, 4, 8], 10)
    assert report["rounds"] == 2
    near = crossover(stand_in(lambda n, s, g: g if s <= n else 0), [2, 4, 8], 10)
    assert near["rounds"] == 1 and [near["sizes"][n]["beats_up_to"] for n in (2, 4, 8)] == [2, 4, 8]
    # The start's neighbours are played and kept: `spread` (2) sizes either side of parity.
    assert sorted(near["sizes"][4]["matches"]) == [2, 3, 4, 5, 6]
    # Started from the previous evaluation's ratios: the same answer in one round.
    ratios = {n: entry["ratio"] for n, entry in report["sizes"].items()}
    again = crossover(far, [2, 4, 8], 10, start_ratio=ratios)
    assert again["rounds"] == 1
    assert [again["sizes"][n]["beats_up_to"] for n in (2, 4, 8)] == [report["sizes"][n]["beats_up_to"] for n in (2, 4, 8)]
    losing = crossover(stand_in(lambda n, s, g: 0), [4], 10, start_ratio={4: 0.0})
    assert losing["rounds"] == 1 and losing["sizes"][4]["beats_up_to"] == 0


def test_full_rows_at_the_defaults_decide_as_the_yardstick():
    # Every row a per-ship tensor (the runner's form) decides as the yardstick's numbers do.
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    state = game.reset(game.GameConfig(), 4, torch.tensor([[3, 2]] * 4), generator, "cpu")
    memory = scripted.fresh_memory(4, state.exists.shape[1], generator, "cpu")
    uniforms = scripted.draw_uniforms(4, state.exists.shape[1], generator, "cpu")
    table = torch.tensor([vector(scripted.ScriptedConfig())])
    agent = torch.zeros(state.exists.shape, dtype=torch.long)
    rows = rows_of(table, agent)
    assert set(rows) == set(PARAMETERS)
    _, _, yardstick = scripted.decide(scripted.ScriptedConfig(), state, None, None, uniforms, memory)
    _, _, full = scripted.decide(scripted.ScriptedConfig(), state, None, rows, uniforms, memory)
    for name in ("stick", "power", "fire"):
        torch.testing.assert_close(getattr(full, name), getattr(yardstick, name), atol=1e-5, rtol=1e-5)


def queue_of(games: int, size: int = 2) -> Queue:
    zeros = torch.zeros(games, dtype=torch.long)
    return Queue(zeros, zeros + 1, torch.arange(games) % 2, zeros + size, zeros + size)


def test_runner_rolls_through_the_queue_and_plays_every_game_whole():
    table = torch.tensor([vector(scripted.ScriptedConfig()), vector(scripted.ScriptedConfig().replace(fight_speed=1.3))])
    outcome = play(GAME, queue_of(24), table, envs=8, segment=8)
    assert (outcome.result >= 0).all()  # 24 games on 8 slots: every slot took several
    assert ((outcome.length > 0) & (outcome.length <= 20)).all()


def test_the_drain_counts_every_started_game_and_none_after_the_cutoff():
    # A cutoff at the first segment's end: the played games are a prefix of the queue, each
    # finished (a length), and the rest are not played.
    outcome = play(GAME, queue_of(40), torch.tensor([vector(scripted.ScriptedConfig())] * 2), envs=8, segment=8,
                   seconds=0.0)
    played = outcome.result != NOT_PLAYED
    count = int(played.sum())
    assert 8 <= count < 40
    assert played[:count].all() and not played[count:].any()
    assert (outcome.length[:count] > 0).all() and (outcome.length[count:] == 0).all()


def test_runner_seats_a_network():
    torch.manual_seed(0)
    policy = model.Policy(model.ModelConfig()).eval()
    outcome = play(GAME, queue_of(6, 1), torch.tensor([vector(scripted.ScriptedConfig())] * 2), policy, envs=4, segment=8)
    assert set(outcome.result.tolist()) <= {A_WON, 1, DRAW}


def test_store_reuses_games_and_keeps_conditions_apart(tmp_path, monkeypatch):
    store = Store(tmp_path / "games.sqlite")
    evaluator = Evaluator(store, GAME, device="cpu", envs=8, segment=8)
    yardstick, other = ScriptedAgent.of(), ScriptedAgent.of(scripted.ScriptedConfig().replace(fight_speed=1.3))
    first = evaluator.ensure([(yardstick, other, (2, 2), 10)])[0]
    assert first["games"] == 10 and first["wins"] + first["draws"] + first["losses"] == 10
    assert first["as_team_zero"] == 5  # sides alternated
    total = store.total_games()
    # Asked again, in either orientation: nothing is played.
    assert evaluator.ensure([(other, yardstick, (2, 2), 10)])[0]["games"] == 10
    assert store.total_games() == total
    flipped = evaluator.counts(other, yardstick, (2, 2))
    assert (flipped["wins"], flipped["losses"]) == (first["losses"], first["wins"])
    # More games: only the missing ones.
    assert evaluator.ensure([(yardstick, other, (2, 2), 14)])[0]["games"] == 14 and store.total_games() == total + 4
    # Other sizes, another game config or another game version: other conditions.
    assert evaluator.counts(yardstick, other, (2, 3))["games"] == 0
    assert Evaluator(store, game.GameConfig(clock_ticks=21), device="cpu").counts(yardstick, other, (2, 2))["games"] == 0
    identity, _ = conditions_of(GAME, (2, 2))
    monkeypatch.setattr(store_module, "game_version", lambda: "another")
    assert conditions_of(GAME, (2, 2))[0] != identity
    monkeypatch.undo()
    # The counts are a view of the games.
    row = store.connection.execute("SELECT SUM(games) FROM counts").fetchone()
    assert row[0] == store.total_games()


def test_agent_ids(tmp_path):
    assert ScriptedAgent.of().id == ScriptedAgent.of(scripted.ScriptedConfig()).id
    assert ScriptedAgent.of().id != ScriptedAgent.of(scripted.ScriptedConfig().replace(turn=0.003)).id
    path = tmp_path / "a.pt"
    path.write_bytes(b"one")
    assert NetworkAgent(str(path), "ema").id != NetworkAgent(str(path), "live").id
    store = Store(tmp_path / "s.sqlite")
    store.add_agent(ScriptedAgent.of())
    assert store.agent(ScriptedAgent.of().id) == ScriptedAgent.of()


def test_sobol_resumes_and_keeps_its_first_points(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    first = SobolStream(store, "screen", 5, seed=3).next(8)
    more = SobolStream(store, "screen", 5).next(8)  # the stored seed, continued
    whole = torch.quasirandom.SobolEngine(5, scramble=True, seed=3).draw(16, dtype=torch.float64)
    assert torch.equal(torch.cat([first, more]), whole)
    assert SobolStream(store, "screen", 5).used == 16


def test_allocation_goes_where_the_error_is_largest():
    # A new pairing first, then the one with fewer games; lots of two.
    added = allocate([(50.0, 100), (5.0, 10), (0.0, 0)], 10)
    assert sum(added) == 10 and all(extra % 2 == 0 for extra in added)
    assert added[2] >= added[1] >= added[0]
    assert standard_error(0.0, 0) == math.inf and standard_error(10.0, 10) > 0


def test_bradley_terry_recovers_a_known_gap():
    # 76% for A over B is 200 Elo; C loses every game to A (finite through the prior).
    ratings, errors = bradley_terry([(0, 1, 1000, 760.0), (0, 2, 20, 20.0)], 3)
    assert ratings[0] == 1000.0 and errors[0] == 0.0
    assert abs((ratings[0] - ratings[1]) - 200.0) < 5.0
    assert ratings[2] < ratings[1] and math.isfinite(ratings[2])


def test_adaptive_allocation_spreads_over_pairings_without_games(tmp_path, monkeypatch):
    """Review B2: a pairing with no games has no score rate (NaN); its allocation must come
    from its wins and draws, not the rate times the games. Five new pairings share a round
    evenly."""
    evaluator = Evaluator(Store(tmp_path / "s.sqlite"), GAME, device="cpu", envs=8)
    agents = [ScriptedAgent.of(scripted.ScriptedConfig().replace(fight_speed=1.0 + 0.1 * index)) for index in range(6)]
    pairings = [(agents[0], agent, (2, 2)) for agent in agents[1:]]
    asked = []
    monkeypatch.setattr(evaluator, "ensure", lambda requests: asked.extend(requests))
    evaluator.adaptive(pairings, budget=100, round_games=100)
    assert sorted(games for _, _, _, games in asked) == [20] * 5


def test_the_runner_keeps_the_callers_random_state():
    """runner.play seeds torch's default generators for its own draws (a compiled tick cannot
    take a generator) inside a fork: the caller's random state is as it was, and a seed
    replays the same games."""
    table = torch.tensor([vector(scripted.ScriptedConfig())] * 2)
    torch.manual_seed(3)
    before = torch.get_rng_state()
    first = play(GAME, queue_of(8), table, envs=4, segment=8, seed=5)
    assert torch.equal(torch.get_rng_state(), before)
    again = play(GAME, queue_of(8), table, envs=4, segment=8, seed=5)
    assert torch.equal(first.result, again.result) and torch.equal(first.length, again.length)
