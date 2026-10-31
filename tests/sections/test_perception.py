"""The observation hides the seat: views depend on team only through team_rel."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import match
from bnb.game.perception.observe import perceive


def _game(S=3, M=6, R=1800.0, B=16, warm=60):
    cfg = GameConfig(S, M, R)
    p = match.preset(cfg)
    torch.manual_seed(1)
    st = match.reset(p, torch.rand(B, p.num_uniforms))
    for _ in range(warm):  # some motion, firing and sight so the views are not trivial
        a = (torch.rand(B, cfg.num_ships, 3) * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32)
        moved, origin = match.move(p, st, a, torch.randn(B, cfg.num_ships, 2))
        st, _ = match.resolve(p, moved, origin, match.new_map(p, torch.rand(B, p.num_uniforms)))
        st, _, _ = perceive(p, st)
    return p, st


def _swap_teams(st: dict, S: int) -> dict:
    """The same world with the two teams' sim indices exchanged."""
    out = {}
    for k, v in st.items():
        if k == "last_seen":
            out[k] = v.flip(1).roll(S, dims=2)
        elif v.dim() >= 2 and v.shape[1] == 2 * S and k not in ("fx", "fy", "fr", "fw", "flevel"):
            out[k] = v.roll(S, dims=1)
        else:
            out[k] = v
    return out


def test_views_swap_with_the_teams():
    p, st = _game()
    _, obs, _ = perceive(p, st)
    _, swapped, _ = perceive(p, _swap_teams(st, p.half))
    for part in ("ships", "fields", "globals"):
        a, b = getattr(obs, part), getattr(swapped, part)
        assert torch.equal(a[:, 0], b[:, 1]) and torch.equal(a[:, 1], b[:, 0]), part


def test_own_rows_first_and_hidden_rows_blank():
    p, st = _game()
    _, obs, truth = perceive(p, st)
    S = p.half
    rel = obs.ships[..., SHIP["team_rel"]]
    assert (rel[..., :S] == 0).all() and (rel[..., S:] == 1).all()
    assert (obs.ships[..., :S, SHIP["visible"]] == 1).all()  # a team always sees itself
    hidden = obs.ships[..., SHIP["visible"]] == 0
    assert hidden.any(), "the scenario should hide some enemies"
    assert (obs.ships[hidden][:, : SHIP["alive"]] == 0).all()
    assert torch.equal(truth.ships[..., :S, :], obs.ships[..., :S, :])


@pytest.mark.xfail(strict=True, reason="race spawn phase uses the sim ship index: team 0's slot 0 "
                   "always spawns due east of its centre, so a reset reveals the seat. Fix in the "
                   "physics section (spawn by slot, pattern relative to the spawn axis).")
def test_spawn_layout_does_not_reveal_the_seat():
    cfg = GameConfig(3, 4, 2600.0)
    p = match.preset(cfg)
    _, obs, _ = perceive(p, match.reset(p, torch.rand(512, p.num_uniforms)))
    S = p.half
    own = obs.ships[..., :S, :]
    dx = own[..., 0, SHIP["x"]] - own[..., SHIP["x"]].mean(-1)
    dy = own[..., 0, SHIP["y"]] - own[..., SHIP["y"]].mean(-1)
    angle = torch.atan2(dy, dx)  # world-frame direction of slot 0 from its team's centre
    spread = 1.0 - torch.hypot(angle.cos().mean(0), angle.sin().mean(0))  # circular spread
    assert (spread > 0.5).all(), f"slot 0's direction is fixed per view: {spread} (uniform ~1)"
    assert math.isfinite(float(spread.sum()))
