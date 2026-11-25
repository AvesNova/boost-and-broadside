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
    runner,
    standard_error,
)
from lean.eval.runner import A_WON, DRAW, NOT_PLAYED, SCRIPTED
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
    assert [again["sizes"][n]["beats_up_to"] for n in (2, 4, 8)] == [
        report["sizes"][n]["beats_up_to"] for n in (2, 4, 8)
    ]
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
    _, _, yardstick = scripted.decide(
        scripted.ScriptedConfig(), state, None, None, uniforms, memory
    )
    _, _, full = scripted.decide(scripted.ScriptedConfig(), state, None, rows, uniforms, memory)
    for name in ("stick", "power", "fire"):
        torch.testing.assert_close(
            getattr(full, name), getattr(yardstick, name), atol=1e-5, rtol=1e-5
        )


def queue_of(games: int, size: int = 2) -> Queue:
    zeros = torch.zeros(games, dtype=torch.long)
    return Queue(zeros, zeros + 1, torch.arange(games) % 2, zeros + size, zeros + size)


def test_runner_rolls_through_the_queue_and_plays_every_game_whole():
    table = torch.tensor(
        [
            vector(scripted.ScriptedConfig()),
            vector(scripted.ScriptedConfig().replace(fight_speed=1.3)),
        ]
    )
    outcome = play(GAME, queue_of(24), table, envs=8, segment=8)
    assert (outcome.result >= 0).all()  # 24 games on 8 slots: every slot took several
    assert ((outcome.length > 0) & (outcome.length <= 20)).all()


def test_the_drain_counts_every_started_game_and_none_after_the_cutoff():
    # A cutoff at the first segment's end: the played games are a prefix of the queue, each
    # finished (a length), and the rest are not played.
    outcome = play(
        GAME,
        queue_of(40),
        torch.tensor([vector(scripted.ScriptedConfig())] * 2),
        envs=8,
        segment=8,
        seconds=0.0,
    )
    played = outcome.result != NOT_PLAYED
    count = int(played.sum())
    assert 8 <= count < 40
    assert played[:count].all() and not played[count:].any()
    assert (outcome.length[:count] > 0).all() and (outcome.length[count:] == 0).all()


def test_runner_seats_many_networks_and_scripted_agents_in_one_queue():
    """One call holds games of several pairings (network against scripted, network against
    network, scripted against scripted) and unequal team sizes: every game played whole."""
    torch.manual_seed(0)
    policies = [model.Policy(model.ModelConfig()).eval() for _ in range(2)]
    games = 12
    pairing = torch.arange(games) % 3  # 0: net 0 v scripted, 1: net 1 v net 0, 2: scripted
    net_a = torch.tensor([0, 1, SCRIPTED])[pairing]
    net_b = torch.tensor([SCRIPTED, 0, SCRIPTED])[pairing]
    zeros = torch.zeros(games, dtype=torch.long)
    queue = Queue(zeros, zeros + 1, torch.arange(games) % 2, zeros + 1, zeros + 2, net_a, net_b)
    outcome = play(
        GAME, queue, torch.tensor([vector(scripted.ScriptedConfig())] * 2), policies, envs=4,
        segment=8, minimum_envs=4,
    )  # fmt: skip
    assert set(outcome.result.tolist()) <= {A_WON, 1, DRAW}
    assert ((outcome.length > 0) & (outcome.length <= GAME.clock_ticks)).all()


def test_each_seat_gets_its_own_networks_step():
    """The runner's routing (lean.model.agent.routed_step, the league's): each seat a network
    plays gets the carry that network's own step gives on the seat's view; scripted and idle
    seats keep theirs."""
    from lean.model.agent import network as as_network

    torch.manual_seed(0)
    policies = [model.Policy(model.ModelConfig()).eval() for _ in range(2)]
    envs = 4
    state = game.reset(GAME, envs, torch.tensor([[2, 1]] * envs), None, "cpu")
    ships = state.exists.shape[1]
    queue = Queue(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([2, 1, 2, 1]),
        torch.tensor([1, 2, 1, 2]),
        torch.tensor([0, 1, SCRIPTED, 1]),
        torch.tensor([1, SCRIPTED, 0, 1]),
    )
    carry = torch.randn_like(policies[0].initial_carry(2 * envs, ships, "cpu"))
    slots = runner.Slots(
        torch.tensor([0, 1, 2, -1]),  # the last slot idle
        scripted.fresh_memory(envs, ships, None, "cpu"),
        carry,
        torch.randn(2 * envs, 2),
        torch.zeros(envs, 2, len(runner.style_counters.SUMS)),
    )
    table = torch.tensor([vector(scripted.ScriptedConfig())])
    decision = runner.decide(GAME, state, slots, queue, table, True)
    seat_action, carry_out = decision.seat_action, decision.carry_out
    for index, policy in enumerate(policies):
        seat_action, carry_out = runner.seat_moves(
            as_network(policy, False), torch.tensor(index), torch.arange(3), decision, carry,
            slots.offset, seat_action, carry_out,
        )  # fmt: skip
    expected_net = [0, 1, SCRIPTED, SCRIPTED, 1, SCRIPTED, 0, SCRIPTED]
    assert decision.seat_net.tolist() == expected_net
    from lean.model.agent import seat_view

    for seat, net in enumerate(expected_net):
        if net == SCRIPTED:
            assert torch.equal(carry_out[:, seat], carry[:, seat])
            assert (seat_action[seat] == 0).all()
            continue
        view = seat_view(
            decision.views, decision.seat_environment[seat : seat + 1],
            decision.seat_team[seat : seat + 1],
        )  # fmt: skip
        _, _, own = policies[net].step(
            view, carry[:, seat : seat + 1], slots.offset[seat : seat + 1]
        )
        assert torch.allclose(carry_out[:, seat], own[:, 0], atol=1e-4)


def test_one_job_is_one_runner_call_and_keeps_the_stored_orientation(tmp_path, monkeypatch):
    """A job's pairings, networks and scripted agents alike, share one queue; a scripted
    agent in seat A against a network stays in seat A."""
    from lean.learn import LearnConfig, WeightAverages, save_checkpoint

    torch.manual_seed(0)
    nets = []
    for index in range(2):
        policy = model.Policy(model.ModelConfig())
        path = tmp_path / f"net{index}.pt"
        save_checkpoint(path, policy, WeightAverages(policy, LearnConfig()), LearnConfig())
        nets.append(NetworkAgent(str(path), "live"))
    calls = []
    real = runner.play

    def counting(*arguments, **keywords):
        calls.append(len(arguments[1]))
        return real(*arguments, **keywords)

    monkeypatch.setattr(runner, "play", counting)
    store = Store(tmp_path / "s.sqlite")
    evaluator = Evaluator(store, GAME, device="cpu", envs=4, segment=8)
    yardstick = ScriptedAgent.of()
    other = ScriptedAgent.of(scripted.ScriptedConfig().replace(fight_speed=1.3))
    requests = [
        (yardstick, nets[0], (1, 2), 4),
        (nets[0], nets[1], (1, 1), 4),
        (yardstick, other, (1, 1), 4),
    ]
    counts = evaluator.ensure(requests)
    assert calls == [12] and [entry["games"] for entry in counts] == [4, 4, 4]
    rows = store.connection.execute(
        "SELECT agent_a, agent_b, COUNT(*) FROM games GROUP BY agent_a, agent_b"
    ).fetchall()
    assert (yardstick.id, nets[0].id, 4) in rows


def test_store_reuses_games_and_keeps_conditions_apart(tmp_path, monkeypatch):
    store = Store(tmp_path / "games.sqlite")
    evaluator = Evaluator(store, GAME, device="cpu", envs=8, segment=8)
    yardstick, other = (
        ScriptedAgent.of(),
        ScriptedAgent.of(scripted.ScriptedConfig().replace(fight_speed=1.3)),
    )
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
    assert (
        evaluator.ensure([(yardstick, other, (2, 2), 14)])[0]["games"] == 14
        and store.total_games() == total + 4
    )
    # Other sizes, another game config or another game version: other conditions.
    assert evaluator.counts(yardstick, other, (2, 3))["games"] == 0
    assert (
        Evaluator(store, game.GameConfig(clock_ticks=21), device="cpu").counts(
            yardstick, other, (2, 2)
        )["games"]
        == 0
    )
    identity, _ = conditions_of(GAME, (2, 2))
    monkeypatch.setattr(store_module, "game_version", lambda: "another")
    assert conditions_of(GAME, (2, 2))[0] != identity
    monkeypatch.undo()
    # The counts are a view of the games.
    row = store.connection.execute("SELECT SUM(games) FROM counts").fetchone()
    assert row[0] == store.total_games()


def test_agent_ids(tmp_path):
    assert ScriptedAgent.of().id == ScriptedAgent.of(scripted.ScriptedConfig()).id
    assert (
        ScriptedAgent.of().id != ScriptedAgent.of(scripted.ScriptedConfig().replace(turn=0.003)).id
    )
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
    agents = [
        ScriptedAgent.of(scripted.ScriptedConfig().replace(fight_speed=1.0 + 0.1 * index))
        for index in range(6)
    ]
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


def test_a_network_plays_at_the_precision_it_was_trained_with(tmp_path, monkeypatch):
    """Review B5: evaluation runs a network under the same precision helper as training
    (lean.model.agent.precision) with the checkpoint's own bf16 setting, not a fixed bf16."""
    import lean.model.agent as agent_module
    from lean.learn import LearnConfig, WeightAverages, save_checkpoint

    torch.manual_seed(0)
    policy = model.Policy(model.ModelConfig())
    config = LearnConfig(bf16=False)
    path = tmp_path / "net.pt"
    save_checkpoint(path, policy, WeightAverages(policy, config), config)
    seen = []
    precision = agent_module.precision

    def recording(device, bf16):
        seen.append(bf16)
        return precision(device, bf16)

    monkeypatch.setattr(agent_module, "precision", recording)
    evaluator = Evaluator(Store(tmp_path / "s.sqlite"), GAME, device="cpu", envs=4, segment=8)
    evaluator.ensure([(NetworkAgent(str(path), "live"), ScriptedAgent.of(), (1, 1), 4)])
    assert seen and set(seen) == {False}


def test_versions_cover_what_decides_a_game_and_how_a_network_acts(tmp_path, monkeypatch):
    """Review B3: the game version hashes the game's source, with tuning.py since it derives
    the card (2026-11-24); a network agent's id covers its inference path (lean/model with its
    agent step, the runner), so a change there starts a new agent."""
    names = {str(path.relative_to(store_module.LEAN)) for path in store_module.game_files()}
    assert {"game/core.py", "game/card.py", "scripted/opponent.py", "contract.py"} <= names
    assert "game/tuning.py" in names and not any("tests" in name for name in names)
    inference = {
        str(path.relative_to(store_module.LEAN)) for path in store_module.inference_files()
    }
    assert {"model/policy.py", "model/agent.py", "eval/runner.py"} <= inference
    path = tmp_path / "a.pt"
    path.write_bytes(b"one")
    before = NetworkAgent(str(path), "ema").id
    monkeypatch.setattr(store_module, "inference_version", lambda: "another")
    assert NetworkAgent(str(path), "ema").id != before


def test_bradley_terry_with_only_self_play_returns_the_anchor():
    """An agent against itself has no rating information; the fit must not crash."""
    from lean.eval.rating import bradley_terry

    ratings, errors = bradley_terry([(0, 0, 10, 5.0)], agents=1)
    assert ratings == [1000.0] and errors == [0.0]
    ratings, errors = bradley_terry([(0, 0, 10, 5.0), (1, 1, 4, 2.0)], agents=2)
    assert ratings == [1000.0, 1000.0]
