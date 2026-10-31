"""The observation hides the seat: views depend on team only through team_rel."""

import math

import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import match
from bnb.game.perception.observe import perceive
from bnb.game.state import State, apply, replace


def _game(S=3, M=6, R=1800.0, B=16, warm=60):
    cfg = GameConfig(S, M, R)
    p = match.rules_of(cfg)
    torch.manual_seed(1)
    st = match.reset(p, match.uniforms(p, B))
    for _ in range(warm):  # some motion, firing and sight so the views are not trivial
        a = (torch.rand(B, cfg.num_ships, 3) * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32)
        st, _ = match.step(p, st, Action.unpack(a), match.spread(p, B), match.uniforms(p, B))
        st, _, _ = perceive(p, st)
    return p, st


def _swap_teams(st: State, S: int) -> State:
    """The same world with the two teams' sim indices exchanged."""
    def roll(v):
        return v.roll(S, dims=1)

    return replace(st, ships=apply(roll, st.ships), bullets=apply(roll, st.bullets),
                   match=replace(st.match, last_seen=st.match.last_seen.flip(1).roll(S, dims=2)))


def test_views_swap_with_the_teams():
    p, st = _game()
    _, obs, _ = perceive(p, st)
    _, swapped, _ = perceive(p, _swap_teams(st, p.ships_per_team))
    for part in ("ships", "fields", "globals"):
        a, b = getattr(obs, part), getattr(swapped, part)
        assert torch.equal(a[:, 0], b[:, 1]) and torch.equal(a[:, 1], b[:, 0]), part


def test_own_rows_first_and_hidden_rows_blank():
    p, st = _game()
    _, obs, truth = perceive(p, st)
    S = p.ships_per_team
    rel = obs.ships[..., SHIP["team_rel"]]
    assert (rel[..., :S] == 0).all() and (rel[..., S:] == 1).all()
    assert (obs.ships[..., :S, SHIP["visible"]] == 1).all()  # a team always sees itself
    hidden = obs.ships[..., SHIP["visible"]] == 0
    assert hidden.any(), "the scenario should hide some enemies"
    assert (obs.ships[hidden][:, : SHIP["alive"]] == 0).all()
    assert torch.equal(truth.ships[..., :S, :], obs.ships[..., :S, :])


def test_spawn_layout_does_not_reveal_the_seat():
    cfg = GameConfig(3, 4, 2600.0)
    p = match.rules_of(cfg)
    _, obs, _ = perceive(p, match.reset(p, match.uniforms(p, 512)))
    S = p.ships_per_team
    own = obs.ships[..., :S, :]
    dx = own[..., 0, SHIP["x"]] - own[..., SHIP["x"]].mean(-1)
    dy = own[..., 0, SHIP["y"]] - own[..., SHIP["y"]].mean(-1)
    angle = torch.atan2(dy, dx)  # world-frame direction of slot 0 from its team's centre
    spread = 1.0 - torch.hypot(angle.cos().mean(0), angle.sin().mean(0))  # circular spread
    assert (spread > 0.5).all(), f"slot 0's direction is fixed per view: {spread} (uniform ~1)"
    assert math.isfinite(float(spread.sum()))
