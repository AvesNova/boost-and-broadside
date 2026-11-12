"""The game loop's GPU harness (`bench.game_loop`), on the CPU uncompiled: the layout, the
card variants and a short batch of games with its statistics."""

import math

import torch

from bench.game_loop import (
    FILLER,
    ROW_LINEUPS,
    Group,
    Job,
    Loop,
    Player,
    base_card,
    batches,
    job_summary,
    layout,
    ratings,
    run,
    variant_types,
)


def test_variants_multiply_or_set_the_base_rows():
    base = base_card("cpu")
    types = variant_types(
        base, [{}, {"hit_damage": 0.5, "explosive_share": {"set": 0.4}}], [{}, {}]
    )
    assert types.hit_damage.shape == (4,)
    assert math.isclose(float(types.hit_damage[1]), 0.5 * float(base.hit_damage[0]), rel_tol=1e-6)
    assert math.isclose(float(types.explosive_share[1]), 0.4, rel_tol=1e-6)
    assert float(types.hit_damage[2]) == float(base.hit_damage[1])


def test_the_layout_gives_each_side_its_doctrines_and_seats():
    hand = Player("x", temperature=2.0)
    placed = layout([Group(1, "energy v turn", hand, Player("off"), 4)], 2, "cpu")
    # side A (Yongma, variant 1 -> row 1) on team 0 in the even environments
    assert placed["ship_type"][0].tolist() == [1] * 5 + [3] * 5
    assert placed["ship_type"][1].tolist() == [3] * 5 + [1] * 5
    assert placed["a_on_zero"].tolist() == [True, False, True, False]
    assert placed["temperature"].tolist() == [2.0, 1.0, 1.0, 2.0] * 2
    assert placed["doctrines"].shape[:2] == (8, 4)


def test_a_row_lineup_places_the_rows_themselves():
    ROW_LINEUPS["test rows"] = ((0, 2, 3, 1), (1, 1, 3, 3))
    try:
        placed = layout([Group(0, "test rows", Player("a"), Player("b"), 2)], 2, "cpu", 4)
    finally:
        del ROW_LINEUPS["test rows"]
    assert placed["ship_type"][0].tolist() == [0, 2, 3, 1, 1, 1, 3, 3]
    assert placed["ship_type"][1].tolist() == [1, 1, 3, 3, 0, 2, 3, 1]


def test_batches_pad_with_filler():
    plan = batches([Job(0, "mixed", Player("a"), Player("b"), 6)], 4)
    assert [sum(g.environments for _, g in batch) for batch in plan] == [4, 4]
    assert plan[1][-1][0] == -1 and plan[1][-1][1].a is FILLER


def test_a_short_batch_counts_its_games(global_generator):
    loop = Loop(4, 1, 2.0, device="cpu", compiled=False)
    jobs = [Job(0, "mixed", Player("off"), Player("off"), 4)]
    (out,) = run(loop, jobs)
    assert out["kills_side"].shape == (4, 2)
    # the kills by type row are the kills by side
    assert torch.equal(out["kills_by_row"].sum((2, 3)), out["kills_side"])
    row = job_summary(jobs[0], out, 2.0)
    assert row["games"] == 4 and 0.0 <= row["score"][0] <= 1.0


def test_ratings_pin_the_anchor():
    rows = [dict(a="off", b="x", games=100, wins=75.0, draws=0.0)]
    elo = ratings(rows)
    assert elo["off"][0] == 1000.0 and elo["x"][0] < 1000.0
    assert torch.isfinite(torch.tensor(elo["x"][1]))


def test_the_pace_counts_the_ships_a_game():
    from bench.game_loop import PACE_BINS, pace

    B = 2
    out = dict(
        gaps=torch.zeros((B, PACE_BINS)),
        deaths=torch.tensor([3.0, 3.0]),
        alive_ticks=torch.zeros(B),
        engage=torch.zeros((B, PACE_BINS)),
        first_hit_tick=torch.full((B,), -1.0),
        first_kill_tick=torch.full((B,), -1.0),
        ships=torch.full((B,), 2.0),
    )
    mask = torch.ones(B, dtype=torch.bool)
    # 6 deaths over 2 games of 2 ships for a minute: 1.5 a minute a ship
    assert math.isclose(pace(out, mask, 60.0)["kills_per_minute_per_ship"], 1.5)
    del out["ships"]  # a 5v5 record from before the count: 10 ships
    assert math.isclose(pace(out, mask, 60.0)["kills_per_minute_per_ship"], 0.3)
