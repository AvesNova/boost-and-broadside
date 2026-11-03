"""The rollout's series (diagnostics audit, groups 4, 12 and 15, and checklist C5): what the
learner's stored segments say about the games, read from the store shard by shard on the
device, with no host synchronisation.

Per pool (a pool's name is its game and its opponent, "frontline/league", or the opponent
alone in a single-game lineup):
- `episode/count`, `episode/length_seconds`, `episode/end/decided`, `episode/end/timeout`:
  the games that ended in the update, from the rollout's per-environment sums;
- `episode/wins`, `episode/draws`, `episode/losses`: the learner views' results (a self-play
  game ends in two views, a win in one and a loss in the other);
- `reward/rate/{level}`: the learner's reward per ship and second.
Over the batch: `episode/alive_share`, `reward/level_share/{level}`, the learner's tokens
by game and by opponent (`batch/token_share/mode/{mode}`, `.../opponent/{opponent}`), the
committed actions' use (`policy/usage/{factor}/{option}`) and how often a ship changes a
factor from one step to the next (`policy/switch_rate/{factor}`).
"""

import torch

from bnb.contracts import REWARD_LEVELS
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.series import Mean

# What the analytics read of a stored shard.
NAMES = ("reward", "done", "result", "alive", "action")
# The rollout's per-environment episode sums (`bnb.runtime.rollout.Rollout.episodes`):
# episodes ended, ended at the time limit, and their lengths in ticks.
EPISODE_COLUMNS = ("ends", "timeouts", "ticks")
FACTORS = (("power", NUM_POWER), ("stick", NUM_STICK), ("shoot", NUM_SHOOT))


def pool_parts(name: str, mode: str) -> tuple[str, str]:
    """(game, opponent) of a pool's name: "frontline/league" is ("frontline", "league"); a
    single-game lineup's "league" is (mode, "league")."""
    return tuple(name.split("/", 1)) if "/" in name else (mode, name)


class Analytics:
    """The rollout's series for a lineup. pools: the pool names; view_pool (R,) each learner
    view's pool index; environment_pool (B,) each environment's; tick_length in seconds;
    mode: the run's mode (for a single-game lineup's names)."""

    def __init__(
        self,
        pools: tuple[str, ...],
        view_pool: torch.Tensor,
        environment_pool: torch.Tensor,
        tick_length: float,
        mode: str,
    ) -> None:
        self.pools, self.tick_length = pools, tick_length
        self.member = torch.nn.functional.one_hot(view_pool.long(), len(pools)).float()  # (R, P)
        self.environment_member = torch.nn.functional.one_hot(
            environment_pool.long(), len(pools)
        ).float()  # (B, P)
        parts = [pool_parts(name, mode) for name in pools]
        self.modes = tuple(dict.fromkeys(game for game, _ in parts))
        self.opponents = tuple(dict.fromkeys(opponent for _, opponent in parts))
        device = self.member.device
        self.mode_of_pool = torch.tensor(
            [[float(game == mode) for mode in self.modes] for game, _ in parts], device=device
        )  # (P, modes)
        self.opponent_of_pool = torch.tensor(
            [[float(other == kind) for kind in self.opponents] for _, other in parts],
            device=device,
        )  # (P, opponents)

    def episodes(self, sums: torch.Tensor) -> dict:
        """Per pool, from the rollout's per-environment sums (B, EPISODE_COLUMNS) over the
        update: the games ended (`episode/count`), their mean length and the shares that
        ended at the time limit or were decided before it."""
        columns = (self.environment_member.transpose(0, 1) @ sums).unbind(-1)  # (P,) each
        ends, timeouts, ticks = (columns[EPISODE_COLUMNS.index(name)] for name in EPISODE_COLUMNS)
        out = {}
        for index, pool in enumerate(self.pools):
            count = ends[index]
            out[f"episode/count/{pool}"] = count.int()
            out[f"episode/length_seconds/{pool}"] = Mean(ticks[index] * self.tick_length, count)
            out[f"episode/end/timeout/{pool}"] = Mean(timeouts[index], count)
            out[f"episode/end/decided/{pool}"] = Mean(count - timeouts[index], count)
        return out

    def shard(self, stored: dict) -> dict:
        """The series of one shard (`stored`: the store's NAMES of it, (L, R, ...)), to be
        summed over the update's shards."""
        reward, done, result = stored["reward"], stored["done"], stored["result"]
        S = reward.shape[2]
        own_alive = stored["alive"][..., :S]
        member = self.member
        out = {}

        def per_pool(values: torch.Tensor) -> list[torch.Tensor]:
            """(L, R) summed over steps, then over each pool's views: one tensor per pool."""
            return list((values.sum(0) @ member).unbind())

        columns = dict(
            wins=per_pool((done & (result > 0)).float()),
            draws=per_pool((done & (result == 0)).float()),
            losses=per_pool((done & (result < 0)).float()),
        )
        steps = per_pool(torch.ones_like(done, dtype=torch.float32))  # view-steps
        rates = (reward.mean(2).sum(0).transpose(0, 1) @ member).unbind(-1)  # (levels,) per pool
        for index, pool in enumerate(self.pools):
            for name in ("wins", "draws", "losses"):
                out[f"episode/{name}/{pool}"] = columns[name][index].int()
            for level, rate in zip(REWARD_LEVELS, rates[index].unbind(), strict=True):
                out[f"reward/rate/{level}/{pool}"] = Mean(rate / self.tick_length, steps[index])
        tokens = own_alive.float().sum((0, 2))  # (R,) the learner's tokens per view
        out["episode/alive_share"] = Mean(tokens.sum(), float(own_alive.numel()))
        size = reward.abs().sum((0, 1, 2))  # (levels,)
        for level, value in zip(REWARD_LEVELS, size.unbind(), strict=True):
            out[f"reward/level_share/{level}"] = Mean(value, size.sum())
        by_pool = tokens @ member  # (P,)
        for mode, value in zip(self.modes, (by_pool @ self.mode_of_pool).unbind(), strict=True):
            out[f"batch/token_share/mode/{mode}"] = Mean(value, tokens.sum())
        for kind, value in zip(
            self.opponents, (by_pool @ self.opponent_of_pool).unbind(), strict=True
        ):
            out[f"batch/token_share/opponent/{kind}"] = Mean(value, tokens.sum())
        out.update(self._actions(stored, own_alive))
        return out

    def _actions(self, stored: dict, own_alive: torch.Tensor) -> dict:
        """Use per option and switch rate per factor over the learner's living ships; a
        switch counts where the ship lives at both steps and its environment did not reset
        in between (the segment's first step has no previous one)."""
        action, S = stored["action"], own_alive.shape[-1]
        out = {}
        alive = own_alive.float()
        tokens = alive.sum()
        ended = stored["done"][:-1, :, None]  # the environment resets after an end
        pairs = own_alive[1:] & own_alive[:-1] & ~ended  # (L - 1, R, S)
        pair_count = pairs.float().sum()
        for factor, options in FACTORS:
            chosen = getattr(action, factor)[..., :S].long()  # (L, R, S)
            usage = (torch.nn.functional.one_hot(chosen, options).float() * alive[..., None]).sum(
                (0, 1, 2)
            )  # (options,)
            for option, value in enumerate(usage.unbind()):
                out[f"policy/usage/{factor}/{option}"] = Mean(value, tokens)
            switched = (chosen[1:] != chosen[:-1]) & pairs
            out[f"policy/switch_rate/{factor}"] = Mean(switched.float().sum(), pair_count)
        return out
