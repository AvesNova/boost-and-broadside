"""`bench.mode_dynamics.elimination_result`: an elimination game's result comes from the
survivors, so the death that ends the game counts (the trace's score misses it)."""

from bench.mode_dynamics import elimination_result


def test_the_team_with_a_survivor_wins():
    assert elimination_result([120, -1, 300, 450]) == 1
    assert elimination_result([120, 90, -1, 450]) == -1


def test_a_1v1_wipe_is_a_win_not_a_draw():
    assert elimination_result([3716, -1]) == -1
    assert elimination_result([-1, 3716]) == 1


def test_both_teams_wiped_or_both_alive_is_a_draw():
    assert elimination_result([500, 500]) == 0
    assert elimination_result([-1, 80, -1, 90]) == 0
