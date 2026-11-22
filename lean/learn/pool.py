"""The league: the scripted agent and past snapshots of the policy, with win-rate tallies.

Member 0 is the scripted agent; members 1..capacity are snapshot slots, filled in turn
and then replaced oldest first. Opponents for league games are drawn by prioritised
fictitious self-play with weight (1 - p)^2 (AlphaStar's `squared`), p the live policy's
win rate against the member, mixed with a uniform floor over the members present:

    probability = (1 - floor) * (1 - p)^2 / sum + floor / members, floor = 1/8.

Win rates are kept as in AlphaStar's pseudocode (`Payoff`): per member, decaying tallies
of games, wins and draws; each game first decays that member's tallies by `decay` (0.99)
and then adds itself; p = (wins + draws / 2) / games, and an unplayed member reads 0.5.
The tallies live on the run's device and are updated without reading back to the host.
"""

import copy

import torch
from torch import Tensor, nn

SCRIPTED = 0


class Stepper(nn.Module):
    """A policy whose forward is its `step`, so that torch.func.functional_call can run
    the step with stacked snapshot weights."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, carry, position_offset):
        return self.policy.step(obs, carry, position_offset)


class League:
    """The opponents of league games.

    The snapshot members' weights are stacked: `weights` (parameters, buffers), each
    tensor with a leading axis of `capacity` slots, slot k - 1 holding member k (a slot not
    yet filled repeats the first snapshot's weights and is never drawn); `template` is a
    weightless (meta) Stepper of the same policy for torch.func.functional_call. Every
    snapshot member runs in one batched call that way (lean.learn.collect.league_actions).
    `present` (members,) bool marks who can be drawn; the scripted agent is always present.
    """

    def __init__(self, every: int, capacity: int, device=None, floor: float = 1.0 / 8.0, decay: float = 0.99):
        self.every = every
        self.capacity = capacity
        self.floor = floor
        self.decay = decay
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.members = capacity + 1
        self.weights: tuple[dict, dict] | None = None
        self.template: nn.Module | None = None
        self.present = torch.zeros(self.members, dtype=torch.bool, device=self.device)
        self.present[SCRIPTED] = True
        self.games = torch.zeros(self.members, device=self.device)
        self.wins = torch.zeros(self.members, device=self.device)
        self.draws = torch.zeros(self.members, device=self.device)
        self.added = 0

    @property
    def snapshots(self) -> int:
        """Snapshot members present."""
        return min(self.added, self.capacity)

    def maybe_add(self, policy, update_index: int) -> int | None:
        """Store the current weights as a member when update_index is a multiple of
        `every`. Returns the member index written (its tallies start unplayed), or None.
        A replaced member's games still in progress must be abandoned by the caller
        (`lean.learn.collect.abandon`), since its weights change under them."""
        if self.capacity == 0 or update_index % self.every != 0:
            return None
        slot = self.added % self.capacity
        stepper = Stepper(policy)
        if self.weights is None:
            self.template = copy.deepcopy(stepper).to("meta").eval()
            self.weights = tuple(
                {name: tensor.detach()[None].repeat(self.capacity, *[1] * tensor.dim()) for name, tensor in named}
                for named in (stepper.named_parameters(), stepper.named_buffers())
            )
        with torch.no_grad():
            for stacked, named in zip(self.weights, (stepper.named_parameters(), stepper.named_buffers()), strict=True):
                for name, tensor in named:
                    stacked[name][slot].copy_(tensor)
        member = slot + 1
        self.present[member] = True
        self.games[member] = 0.0
        self.wins[member] = 0.0
        self.draws[member] = 0.0
        self.added += 1
        return member

    def win_rates(self) -> Tensor:
        """(members,) the live policy's win rate against each member, draws as half,
        0.5 where unplayed."""
        played = self.games > 0
        rate = (self.wins + 0.5 * self.draws) / self.games.clamp(min=1e-12)
        return torch.where(played, rate, torch.full_like(rate, 0.5))

    def scripted_win_rate(self) -> float:
        """Win rate against the scripted agent for the BC gate; 0 until a game is played
        (so the BC term starts at full weight), not the sampler's 0.5 prior."""
        if float(self.games[SCRIPTED]) <= 0.0:
            return 0.0
        return float(self.win_rates()[SCRIPTED])

    def probabilities(self) -> Tensor:
        """(members,) sampling probabilities: (1 - floor) by (1 - p)^2, floor uniform,
        both over the members present. Uniform in place of the first part when its
        weights sum below 1e-10 (AlphaStar's rule)."""
        present = self.present.to(self.games.dtype)
        uniform = present / present.sum()
        weight = (1.0 - self.win_rates()) ** 2 * present
        total = weight.sum()
        prioritised = torch.where(total > 1e-10, weight / total.clamp(min=1e-10), uniform)
        return (1.0 - self.floor) * prioritised + self.floor * uniform

    def draw(self, count: int, generator) -> Tensor:
        """`count` member indices (long), independent draws from `probabilities`."""
        return torch.multinomial(self.probabilities(), count, replacement=True, generator=generator)

    def record(self, member: Tensor, finished: Tensor, won: Tensor, drawn: Tensor) -> None:
        """Add the league games that ended: member, finished, won, drawn are (L,) over the
        league environments. Each game decays its member's tallies by `decay` before it
        counts. Games of one member ending on the same tick are taken in every order
        equally: n games give games * d^n + (1 - d^n) / (1 - d), and each win counts
        (1 - d^n) / (n (1 - d)), which is exact for n = 1."""
        finished = finished.to(self.games.dtype)
        count = torch.zeros_like(self.games).index_add_(0, member, finished)
        wins = torch.zeros_like(self.games).index_add_(0, member, finished * won.to(finished.dtype))
        draws = torch.zeros_like(self.games).index_add_(0, member, finished * drawn.to(finished.dtype))
        factor = self.decay**count
        added = (1.0 - factor) / (1.0 - self.decay)
        per_game = added / count.clamp(min=1.0)
        self.games = self.games * factor + added
        self.wins = self.wins * factor + wins * per_game
        self.draws = self.draws * factor + draws * per_game
