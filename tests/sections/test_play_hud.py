"""The HUD's text (`bnb.play.render.hud`)."""

import numpy as np

from bnb.play.render.hud import ship_label


def test_the_flown_ship_is_named_by_team_and_slot():
    team = np.repeat(np.arange(2), 5)  # five ships a team, team-major
    assert ship_label(team, 2) == "team 0, ship 2"
    assert ship_label(team, 7) == "team 1, ship 2"
