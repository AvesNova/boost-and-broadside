"""The games' series (diagnostics audit, groups 6, 12 to 15): the rollout's per-tick sums
and the outcome ledger against hand-computed cases, the fog slices, and a real rollout's
sums against its records."""

import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import ASPECTS, REWARD_LEVELS, Events
from bnb.contracts.obs import GLOBAL, GLOBAL_COLUMNS, SHIP, SHIP_COLUMNS
from bnb.learn.rewards import REWARD_EVENTS
from bnb.runtime import analytics, metrics

TICK = 0.5  # s, round numbers
B, N, S = 2, 2, 1  # environments, ships (one a side)


def _events(**given) -> Events:
    fields = dict(
        damage=torch.zeros((B, N, N)),
        boundary_damage=torch.zeros((B, N)),
        died=torch.zeros((B, N), dtype=torch.bool),
        fired=torch.zeros((B, N), dtype=torch.bool),
        spawned=torch.zeros((B, N), dtype=torch.bool),
        killer=torch.full((B, N), -1, dtype=torch.int32),
        done=torch.zeros(B, dtype=torch.bool),
        shield_lost=torch.zeros((B, N)),
        shield_restored=torch.zeros((B, N)),
        objective_progress=torch.zeros((B, N)),
        objective=torch.zeros((B, N)),
        result=torch.zeros(B, dtype=torch.int32),
        enemy_damage_by_aspect=torch.zeros((B, N, len(ASPECTS))),
        kill_aspect=torch.full((B, N), -1, dtype=torch.int32),
    )
    fields.update(given)
    return Events(**fields)


def _rows(speed: float = 0.0, energy: float = 1.0) -> torch.Tensor:
    rows = torch.zeros((B, N, len(SHIP_COLUMNS)))
    rows[..., SHIP["alive"]] = 1.0
    rows[..., SHIP["velocity_x"]] = speed
    rows[..., SHIP["energy"]] = energy
    return rows


def _two_slices() -> analytics.Analytics:
    """Two environments, one pool each ("a" and "b"), one learner view each."""
    return analytics.Analytics(
        ("a", "b"), torch.tensor([0, 1]), torch.tensor([0, 1]), TICK, "squad"
    )


def test_the_resolve_sums_match_a_hand_computed_episode():
    """Environment 0: ship 0 hits ship 1 for 30 and kills it on tick 2, then the episode
    ends on tick 3 with team 0's win; ship 0 flies at 4 m/s and fires every tick.
    Environment 1: nothing happens and its episode does not end."""
    game, sums = analytics.blank_game(B, N, "cpu"), analytics.blank_game_sums(B, "cpu")
    reward = torch.zeros((B, N, len(REWARD_LEVELS)))
    event_rewards = torch.zeros((B, N, len(REWARD_EVENTS)))
    fired = torch.tensor([[True, False], [False, False]])
    for tick in range(3):
        rows = _rows()
        rows[0, 0, SHIP["velocity_x"]] = 4.0
        damage = torch.zeros((B, N, N))
        died = torch.zeros((B, N), dtype=torch.bool)
        if tick == 1:
            damage[0, 0, 1] = 30.0
            died[0, 1] = True
        if tick == 2:
            rows[0, 1, SHIP["alive"]] = 0.0  # dead since tick 1
        reward[0, 0, 0] = 1.0 if tick == 1 else 0.0  # the kill's damage reward, say
        reward[0, 1, 0] = -1.0 if tick == 1 else 0.0
        event_rewards[0, 0, 1] = reward[0, 0, 0]
        event_rewards[0, 1, 0] = reward[0, 1, 0]
        done = torch.tensor([tick == 2, False])
        result = torch.tensor([1 if tick == 2 else 0, 0], dtype=torch.int32)
        events = _events(damage=damage, died=died, fired=fired, done=done, result=result)
        analytics.resolve_tick(game, sums, events, reward, event_rewards, rows, TICK)
    series = metrics.read(_two_slices().games(sums, None))
    # lives ended: ship 1 at its death (2 ticks), ship 0 with the episode (3 ticks)
    assert series["episode/lifespan_seconds/a"] == pytest.approx((2 + 3) * TICK / 2)
    assert "episode/lifespan_seconds/b" not in series  # no life ended there
    assert series["episode/event/shots_fired/a"] == 3.0
    assert series["episode/event/hits/a"] == 1.0
    assert series["episode/event/combat_deaths/a"] == 1.0
    assert series["episode/reward/damage/a"] == 0.0  # +1 and -1 over the two ships
    assert series["episode/reward_max/a"] == 1.0 and series["episode/reward_min/a"] == -1.0
    assert series["episode/result/team_0"] == 1 and series["episode/result/team_1"] == 0
    # 11 living ship-ticks (6 + 5): 5.5 ship-seconds
    assert series["physics/damage_per_ship_second/combat"] == pytest.approx(30.0 / 5.5)
    assert series["physics/deaths_per_ship_minute/combat"] == pytest.approx(60.0 / 5.5)
    assert series["physics/speed_metres_per_second"] == pytest.approx(3 * 4.0 / 11)
    assert series["reward/event_share/damage_dealt"] == pytest.approx(0.5)
    assert series["reward/event_share/damage_taken"] == pytest.approx(0.5)
    assert series["reward/zero_sum_residual/damage"] == 0.0
    # the running state restarted with the episode in environment 0 and goes on in 1
    assert game["life_ticks"][0].tolist() == [0.0, 0.0]
    assert game["life_ticks"][1].tolist() == [3.0, 3.0]


def test_the_aspect_shares_count_enemy_damage_and_enemy_kills():
    """Environment 0: ship 1 takes 10 mutual head-on and 30 from the tail, and dies to the
    tail hit; environment 1: ship 0 takes 20 on the flank and dies to its teammate's (itself
    here: a killer on its own team is no enemy kill)."""
    game, sums = analytics.blank_game(B, N, "cpu"), analytics.blank_game_sums(B, "cpu")
    by_aspect = torch.zeros((B, N, len(ASPECTS)))
    by_aspect[0, 1] = torch.tensor([10.0, 0.0, 0.0, 30.0])
    by_aspect[1, 0, 2] = 20.0
    died = torch.tensor([[False, True], [True, False]])
    killer = torch.tensor([[-1, 0], [0, -1]], dtype=torch.int32)
    kill_aspect = torch.tensor([[-1, 3], [2, -1]], dtype=torch.int32)
    events = _events(
        died=died, killer=killer, kill_aspect=kill_aspect, enemy_damage_by_aspect=by_aspect
    )
    zero = torch.zeros((B, N, len(REWARD_LEVELS)))
    analytics.resolve_tick(
        game, sums, events, zero, torch.zeros((B, N, len(REWARD_EVENTS))), _rows(), TICK
    )
    series = metrics.read(_two_slices().games(sums, None))
    assert series["combat/damage_share/mutual_head_on/a"] == pytest.approx(0.25)
    assert series["combat/damage_share/tail/a"] == pytest.approx(0.75)
    assert series["combat/damage_share/flank/b"] == pytest.approx(1.0)
    assert series["combat/damage_share/flank"] == pytest.approx(20.0 / 60.0)
    assert series["combat/kill_share/tail/a"] == pytest.approx(1.0)
    assert series["combat/kill_share/tail"] == pytest.approx(1.0)  # b's kill is no enemy's
    assert "combat/kill_share/flank/b" not in series


def test_the_perceive_sums_match_a_hand_computed_case():
    """Two fogged views of one environment and one clear environment: in the fogged one
    team 0's enemy is reacquired after a 1.5 s spell and team 1's enemy is hidden at age 2 s."""
    T = 2
    previous = torch.zeros((B, T, N, len(SHIP_COLUMNS)))
    ships = previous.clone()
    truth = previous.clone()
    truth[..., SHIP["alive"]] = 1.0
    ships[..., SHIP["visible"]] = 1.0
    previous[..., SHIP["visible"]] = 1.0
    previous[0, 0, 1, SHIP["visible"]] = 0.0
    previous[0, 0, 1, SHIP["time_since_seen"]] = 1.0  # seen 1 s ago, plus this tick's 0.5 s
    ships[0, 1, 1, SHIP["visible"]] = 0.0
    ships[0, 1, 1, SHIP["time_since_seen"]] = 2.0
    globals_ = torch.zeros((B, T, len(GLOBAL_COLUMNS)))
    globals_[0, :, GLOBAL["fogged"]] = 1.0
    globals_[:, :, GLOBAL["arena_radius"]] = 3000.0
    sums = analytics.blank_game_sums(B, "cpu")
    analytics.perceive_tick(sums, previous, ships, truth, globals_, globals_, S, TICK, 2000.0)
    series = metrics.read(_two_slices().games(sums, None))
    assert series["fog/visible_share/all"] == 0.5  # fogged environment only
    assert series["fog/mean_hidden_age_seconds"] == 2.0
    assert series["fog/occlusion_share/0.5_2_seconds"] == 1.0
    assert series["fog/reacquisitions_per_slot_minute"] == pytest.approx(1 / (2 * TICK / 60))
    assert series["map_scale/mean"] == 1.5
    assert series["map_scale/slot_share/1_2"] == 1.0


def test_the_outcome_ledger_scores_every_prediction_against_the_result():
    """One view predicts P(win | resolved) 0.8 at 0.5 s and 0.2 at 1.5 s of a 10 s
    episode it wins; another 0.5 at 0.5 s of a 3 s episode it loses."""
    ledger = analytics.blank_outcome(2, 12, "cpu")
    tables = analytics.blank_outcome_tables("cpu")

    def probabilities(win: float) -> torch.Tensor:
        return torch.tensor([win * 0.5, 0.0, (1 - win) * 0.5, 0.5])  # half unresolved

    analytics.outcome_record(
        ledger, torch.stack([probabilities(0.8), probabilities(0.5)]), torch.tensor([0.5, 0.5])
    )
    analytics.outcome_record(
        ledger, torch.stack([probabilities(0.2), probabilities(0.5)]), torch.tensor([1.5, 1.6])
    )
    done = torch.tensor([True, True])
    analytics.outcome_score(
        ledger, tables, done, torch.tensor([True, False]), torch.tensor([10.0, 3.0])
    )
    assert ledger["blocks"].abs().sum() == 0  # cleared
    out = metrics.read(analytics.outcome_series(tables))
    # view 0's predictions are 9.5 s and 8.5 s from the end (5-15 s), base rate 1: no skill
    assert "outcome/brier_skill/5_15_seconds" not in out
    # view 1's: 2.5 and 1.5 s from the end, both 0.5 for a loss: Brier 0.25
    assert "outcome/brier_skill/under_5_seconds" not in out  # its base rate is 0 too
    total_bias = (0.8 + 0.2 + 0.5 + 0.5) - 2.0
    assert out["outcome/bias"] == pytest.approx(total_bias / 4)
    # bins: 0.8 (won), 0.2 (won), 0.5 and 0.5 (lost): |0.8 - 1| + |0.2 - 1| + |1.0 - 0|
    assert out["outcome/expected_calibration_error"] == pytest.approx((0.2 + 0.8 + 1.0) / 4)
    table = analytics.reliability_table(tables["calibration"].tolist())
    assert [row[-1] for row in table.rows] == [1, 2, 1]


def test_the_brier_skill_is_one_minus_the_brier_over_the_base_rates():
    tables = analytics.blank_outcome_tables("cpu")
    # under 5 s: 4 predictions, 2 wins (base rate 0.5, reference Brier 0.25), Brier sum 0.5
    tables["skill"][0] = torch.tensor([4.0, 2.0, 0.5, 2.0])
    out = metrics.read(analytics.outcome_series(tables))
    assert out["outcome/brier_skill/under_5_seconds"] == pytest.approx(1 - (0.5 / 4) / 0.25)


def test_a_pool_with_fogged_and_clear_games_is_sliced_by_fog():
    member = analytics.Analytics(
        ("a", "b"),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0, 1]),
        TICK,
        "squad",
        view_fogged=torch.tensor([True, False, True]),
        environment_fogged=torch.tensor([True, False, True]),
    )
    assert member.slices == ("a", "b", "a/fogged", "a/clear")
    sums = torch.tensor([[1.0, 0.0, 10.0], [1.0, 1.0, 20.0], [0.0, 0.0, 0.0]])
    out = metrics.read(member.episodes(sums))
    assert out["episode/count/a"] == 2 and out["episode/count/a/fogged"] == 1
    assert out["episode/end/timeout/a/clear"] == 1.0
    assert "episode/count/b/fogged" not in out  # b is fogged only


@pytest.mark.slow
def test_a_rollouts_game_sums_agree_with_its_counters(global_generator):
    """Shots fired, resets and the sides' results from the games' sums equal the rollout's
    own counters over the same ticks."""
    net = cases.model()
    runner = lc.rollout(net, runs=1)
    runner.take_games()
    before = runner.counters()
    runner.run()
    after = runner.counters()
    sums, tables = runner.take_games()
    resolve = dict(zip(analytics.RESOLVE_COLUMNS, sums["resolve"].sum(0).tolist(), strict=True))
    resets = after["resets"] - before["resets"]
    assert resolve["team_0_wins"] + resolve["team_1_wins"] + resolve["team_draws"] == resets
    assert resolve["environment_ticks"] == runner.environments * runner.length
    assert tables is not None and tables["calibration"][:, 0].sum() > 0
    assert math.isfinite(resolve["speed"]) and resolve["alive_ship_ticks"] > 0
