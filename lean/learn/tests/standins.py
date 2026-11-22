"""Stand-ins with the contract's signatures: a counting game, a scripted agent and a tiny policy.

The game: every living ship that fires adds one point to its team; an episode lasts
`config.length` ticks; the team with more points wins. The scripted agent always fires.
Every ship emits a bullet every tick; the bullet emitted on episode tick s ends on tick
s + config.flight with outcome (s + slot) % 3 (`bullet_outcome`), one bullet slot. No
damage, deaths or boundary.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from torch import nn

from lean.learn.config import LEVELS

SHIP_FEATURES = 3
MODEL_CONFIG = SimpleNamespace(offset_span=100.0, next_state_fields=(("x", 1.0), ("speed", 1.0)))


@dataclass
class State:
    x: torch.Tensor
    y: torch.Tensor
    heading_cos: torch.Tensor
    heading_sin: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    speed: torch.Tensor
    shield: torch.Tensor
    alive: torch.Tensor
    exists: torch.Tensor
    team: torch.Tensor
    points: torch.Tensor  # (E, 2)
    tick: torch.Tensor


@dataclass
class StepOut:
    damage_dealt: torch.Tensor
    damage_taken: torch.Tensor
    friendly_damage: torch.Tensor
    died: torch.Tensor
    fired: torch.Tensor
    killer: torch.Tensor
    done: torch.Tensor
    result: torch.Tensor
    shot_outcome: torch.Tensor
    shot_tick: torch.Tensor
    hit_damage: torch.Tensor
    boundary_damage: torch.Tensor
    shield_restored: torch.Tensor


@dataclass
class Obs:
    ship: torch.Tensor
    pos: torch.Tensor
    heading: torch.Tensor
    key_mask: torch.Tensor
    own: torch.Tensor
    globals: torch.Tensor


@dataclass
class GameConfig:
    slots: int = 4
    length: int = 6
    flight: int = 3  # ticks from a bullet's emission to its end
    arena_radius: float = 1e9  # no boundary
    boundary_ramp_width: float = 1.0
    tick_length: float = 1.0 / 30.0


def bullet_outcome(fired_tick, slot):
    """The stand-in's outcome of the bullet emitted on episode tick `fired_tick` by `slot`."""
    return (fired_tick + slot) % 3


def reset(config, envs, team_sizes, generator, device):
    index = torch.arange(config.slots, device=device).expand(envs, -1)
    first, second = team_sizes[:, :1], team_sizes[:, 1:]
    zeros = torch.zeros(envs, config.slots, device=device)
    exists = index < first + second
    return State(
        x=index.float(), y=zeros, heading_cos=zeros + 1, heading_sin=zeros, angle_cos=zeros + 1, angle_sin=zeros,
        speed=zeros, shield=zeros, alive=exists, exists=exists, team=(index >= first).long(),
        points=torch.zeros(envs, 2, device=device), tick=torch.zeros(envs, dtype=torch.long, device=device),
    )


def draw_noise(config, envs, slots, generator, device):
    return {}


def step(config, state, action, noise):
    fired = (action[..., 2] == 1) & state.alive
    points = state.points + torch.stack([(fired & (state.team == side)).sum(1).float() for side in (0, 1)], dim=1)
    tick = state.tick + 1
    done = tick >= config.length
    lead = points[:, 0] - points[:, 1]
    result = torch.where(lead > 0, 0, torch.where(lead < 0, 1, 2))
    zeros = torch.zeros_like(state.x)
    fired_tick = (state.tick - config.flight)[:, None].expand_as(state.team)
    slot = torch.arange(state.team.shape[1], device=state.team.device).expand_as(state.team)
    ended = fired_tick >= 0
    out = StepOut(
        damage_dealt=fired.float() * 0.1, damage_taken=zeros, friendly_damage=zeros, died=torch.zeros_like(fired),
        fired=fired, killer=torch.full_like(state.team, -1), done=done, result=torch.where(done, result, -1),
        shot_outcome=torch.where(ended, bullet_outcome(fired_tick, slot), -1)[..., None],
        shot_tick=fired_tick[..., None],
        hit_damage=zeros[..., None].expand(-1, -1, zeros.shape[-1]).clone(),
        boundary_damage=zeros, shield_restored=zeros,
    )
    return replace(state, points=points, tick=tick), out


def deal(config, envs, slots, team_sizes, generator, device):
    return reset(config, envs, team_sizes, generator, device)


def take_deal(state, done, fresh, index=None):
    changes = {}
    for name in State.__dataclass_fields__:
        old, new = getattr(state, name), getattr(fresh, name)
        new = new if index is None else new.index_select(0, index)
        changes[name] = torch.where(done.view(-1, *[1] * (old.dim() - 1)), new, old)
    return State(**changes)


def reset_done(config, state, done, team_sizes, generator):
    return take_deal(state, done, reset(config, done.shape[0], team_sizes, generator, done.device))


def observe(config, state, team):
    own = state.alive & (state.team == team)
    ship = torch.stack([own.float(), state.alive.float(), (state.tick[:, None] / config.length).expand_as(state.x)], -1)
    return Obs(
        ship=ship, pos=torch.stack([state.x, state.y], -1), heading=torch.zeros_like(state.x),
        key_mask=state.alive, own=own, globals=state.points / config.length,
    )


def personas(config, envs, slots, generator, device, spread):
    return {"aggression": torch.ones(envs, slots, device=device)}


def act(config, state, team, persona, uniforms):
    action = torch.zeros(*state.team.shape, 3, dtype=torch.long, device=state.team.device)
    action[..., 2] = 1 if team is None else (state.team == team).long()
    return action


game = SimpleNamespace(reset=reset, step=step, draw_noise=draw_noise, deal=deal, take_deal=take_deal, reset_done=reset_done, observe=observe)
scripted = SimpleNamespace(personas=personas, act=act)


class TinyPolicy(nn.Module):
    """A per-ship MLP with a leaky running mean as its carry; same methods as lean.model.Policy."""

    def __init__(self, levels=LEVELS, bins=21, width=32):
        super().__init__()
        self.width, self.levels, self.bins = width, levels, bins
        self.config = MODEL_CONFIG
        self.body = nn.Sequential(nn.Linear(SHIP_FEATURES + width, width), nn.Tanh())
        self.remember = nn.Linear(SHIP_FEATURES, width)
        self.action = nn.Linear(width, 42)
        self.value = nn.Linear(width, levels * bins)
        self.next_state = nn.Linear(width, len(MODEL_CONFIG.next_state_fields))
        self.hit = nn.Linear(width, 3)

    def initial_carry(self, envs, slots, device):
        return torch.zeros(envs, slots, self.width, device=device)

    def step(self, obs, carry, position_offset=None):
        carry = 0.5 * carry + 0.5 * torch.tanh(self.remember(obs.ship))
        hidden = self.body(torch.cat([obs.ship, carry], -1))
        return self.action(hidden), self.value(hidden).unflatten(-1, (self.levels, self.bins)), carry

    def sequence(self, obs, carry, reset, position_offset=None):
        logits, values, nexts, hits = [], [], [], []
        for tick in range(reset.shape[0]):
            carry = carry * (~reset[tick])[:, None, None]
            carry = 0.5 * carry + 0.5 * torch.tanh(self.remember(obs.ship[tick]))
            hidden = self.body(torch.cat([obs.ship[tick], carry], -1))
            logits.append(self.action(hidden))
            values.append(self.value(hidden).unflatten(-1, (self.levels, self.bins)))
            nexts.append(self.next_state(hidden))
            hits.append(self.hit(hidden))
        return torch.stack(logits), torch.stack(values), torch.stack(nexts), torch.stack(hits), carry
