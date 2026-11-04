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

import math

import torch

from bnb.contracts import REWARD_LEVELS
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.contracts.series import Mean, Table
from bnb.learn.rewards import REWARD_EVENTS

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
        view_fogged: torch.Tensor | None = None,
        environment_fogged: torch.Tensor | None = None,
        ships: int = 1,
    ) -> None:
        """view_fogged (R,) and environment_fogged (B,): which views' and environments'
        games have fog of war; a pool with both kinds has its series sliced by fog as well
        (`{pool}/fogged`, `{pool}/clear`). ships: per environment."""
        self.pools, self.tick_length, self.ships = pools, tick_length, ships
        view_member = torch.nn.functional.one_hot(view_pool.long(), len(pools)).float()  # (R, P)
        environment_member = torch.nn.functional.one_hot(
            environment_pool.long(), len(pools)
        ).float()  # (B, P)
        # the slices: each pool, then fogged and clear for each pool that has both
        names, view_columns, environment_columns = list(pools), [view_member], [environment_member]
        if view_fogged is not None and environment_fogged is not None:
            for index, pool in enumerate(pools):
                inside = environment_member[:, index] > 0
                if bool((environment_fogged & inside).any()) and bool(
                    (~environment_fogged & inside).any()
                ):
                    for suffix, view_mask, environment_mask in (
                        ("fogged", view_fogged, environment_fogged),
                        ("clear", ~view_fogged, ~environment_fogged),
                    ):
                        names.append(f"{pool}/{suffix}")
                        view_columns.append((view_member[:, index] * view_mask.float())[:, None])
                        environment_columns.append(
                            (environment_member[:, index] * environment_mask.float())[:, None]
                        )
        self.slices = tuple(names)
        self.member = torch.cat(view_columns, 1)  # (R, slices)
        self.environment_member = torch.cat(environment_columns, 1)  # (B, slices)
        self.fogged_share = (
            float(environment_fogged.float().mean()) if environment_fogged is not None else 0.0
        )
        pool_of_slice = list(range(len(pools))) + [
            pools.index(name.rsplit("/", 1)[0]) for name in names[len(pools) :]
        ]
        # the slices with learner views (a rating lane's pool has none): the views' series
        self.viewed = tuple(bool(column.any()) for column in self.member.unbind(1))
        parts = [
            pool_parts(name, mode)
            for name, viewed in zip(pools, self.viewed[: len(pools)], strict=True)
            if viewed
        ]
        self.modes = tuple(dict.fromkeys(game for game, _ in parts))
        self.opponents = tuple(dict.fromkeys(opponent for _, opponent in parts))
        device = self.member.device
        all_parts = [pool_parts(name, mode) for name in pools]
        pool_parts_of_slice = [all_parts[index] for index in pool_of_slice]
        whole = [float(index < len(pools)) for index in range(len(names))]  # the pools alone
        self.mode_of_pool = torch.tensor(
            [
                [weight * float(game == mode) for mode in self.modes]
                for weight, (game, _) in zip(whole, pool_parts_of_slice, strict=True)
            ],
            device=device,
        )  # (slices, modes)
        self.opponent_of_pool = torch.tensor(
            [
                [weight * float(other == kind) for kind in self.opponents]
                for weight, (_, other) in zip(whole, pool_parts_of_slice, strict=True)
            ],
            device=device,
        )  # (slices, opponents)

    def episodes(self, sums: torch.Tensor) -> dict:
        """Per pool, from the rollout's per-environment sums (B, EPISODE_COLUMNS) over the
        update: the games ended (`episode/count`), their mean length and the shares that
        ended at the time limit or were decided before it."""
        columns = (self.environment_member.transpose(0, 1) @ sums).unbind(-1)  # (P,) each
        ends, timeouts, ticks = (columns[EPISODE_COLUMNS.index(name)] for name in EPISODE_COLUMNS)
        out = {}
        for index, pool in enumerate(self.slices):
            count = ends[index]
            out[f"episode/count/{pool}"] = count.int()
            out[f"episode/length_seconds/{pool}"] = Mean(ticks[index] * self.tick_length, count)
            out[f"episode/end/timeout/{pool}"] = Mean(timeouts[index], count)
            out[f"episode/end/decided/{pool}"] = Mean(count - timeouts[index], count)
        return out

    def games(self, sums: dict, tables: dict | None) -> dict:
        """The games' series of an update, from the rollout's sums (`blank_game_sums`) and its
        outcome tables (`blank_outcome_tables`, or None without an outcome head): per slice,
        the lifespan, the ship-episode reward (mean, per level, lowest and highest) and the
        events per episode; over every game, the results by side; over every environment,
        physics, the front, map scale and the reward's events; over the fogged environments,
        fog; over the learner's views, the outcome ledger."""
        tick = self.tick_length
        resolve = dict(zip(RESOLVE_COLUMNS, sums["resolve"].unbind(-1), strict=True))  # (B,)
        perceive = dict(zip(PERCEIVE_COLUMNS, sums["perceive"].unbind(-1), strict=True))
        member = self.environment_member  # (B, slices)

        def per_slice(values: torch.Tensor) -> list[torch.Tensor]:
            return list((values @ member).unbind())

        def total(values: torch.Tensor) -> torch.Tensor:
            return values.sum()

        out = {}
        ended = resolve["team_0_wins"] + resolve["team_1_wins"] + resolve["team_draws"]
        columns = {
            name: per_slice(resolve[name])
            for name in ("lives_ended", "life_seconds", "ship_episodes")
        }
        episodes = per_slice(ended)
        rewards = {level: per_slice(resolve[f"episode_reward/{level}"]) for level in REWARD_LEVELS}
        events = {name: per_slice(resolve[f"episode_event/{name}"]) for name in EPISODE_EVENTS}
        low, high = sums["extremes"].unbind(-1)
        inside = member > 0
        lowest = torch.where(inside, low[:, None], math.inf).amin(0)  # (slices,)
        highest = torch.where(inside, high[:, None], -math.inf).amax(0)
        for index, name in enumerate(self.slices):
            out[f"episode/lifespan_seconds/{name}"] = Mean(
                columns["life_seconds"][index], columns["lives_ended"][index]
            )
            ship_episodes = columns["ship_episodes"][index]
            out[f"episode/reward_mean/{name}"] = Mean(
                sum(rewards[level][index] for level in REWARD_LEVELS), ship_episodes
            )
            for level in REWARD_LEVELS:
                out[f"episode/reward/{level}/{name}"] = Mean(rewards[level][index], ship_episodes)
            for event in EPISODE_EVENTS:
                out[f"episode/event/{event}/{name}"] = Mean(events[event][index], episodes[index])
            finite = torch.isfinite(lowest[index])
            out[f"episode/reward_min/{name}"] = Mean(
                torch.where(finite, lowest[index], 0.0), finite.float()
            )
            out[f"episode/reward_max/{name}"] = Mean(
                torch.where(finite, highest[index], 0.0), finite.float()
            )
        out["episode/result/team_0"] = total(resolve["team_0_wins"]).int()
        out["episode/result/team_1"] = total(resolve["team_1_wins"]).int()
        out["episode/result/draw"] = total(resolve["team_draws"]).int()
        ship_seconds = total(resolve["alive_ship_ticks"]) * tick
        for kind in ("combat", "boundary"):
            out[f"physics/damage_per_ship_second/{kind}"] = Mean(
                total(resolve[f"{kind}_damage"]), ship_seconds
            )
            out[f"physics/deaths_per_ship_minute/{kind}"] = Mean(
                total(resolve[f"{kind}_deaths"]), ship_seconds / 60.0
            )
        ship_ticks = total(resolve["alive_ship_ticks"])
        out["physics/speed_metres_per_second"] = Mean(total(resolve["speed"]), ship_ticks)
        out["physics/energy_mean"] = Mean(total(resolve["energy"]), ship_ticks)
        out["physics/out_of_energy_share"] = Mean(total(resolve["out_of_energy"]), ship_ticks)
        out["physics/inside_field_share"] = Mean(total(resolve["in_field"]), ship_ticks)
        out["physics/shield_recharge_per_ship_second"] = Mean(
            total(resolve["shield_restored"]), ship_seconds
        )
        out["frontline/front_moves_per_environment_minute"] = Mean(
            total(perceive["front_moves"]), total(perceive["frontline_ticks"]) * tick / 60.0
        )
        environment_ticks = total(resolve["environment_ticks"])
        out["map_scale/mean"] = Mean(total(perceive["map_scale"]), total(perceive["observations"]))
        for name, _, _ in SCALE_BUCKETS:
            out[f"map_scale/slot_share/{name}"] = Mean(
                total(perceive[f"enemy_slots/{name}"]), total(perceive["enemy_slots"])
            )
        level_size = {level: 0.0 for level in REWARD_LEVELS}
        for level, event in REWARD_EVENTS:
            level_size[level] = level_size[level] + total(resolve[f"event/{event}"])
        for level, event in REWARD_EVENTS:
            out[f"reward/event_share/{event}"] = Mean(
                total(resolve[f"event/{event}"]), level_size[level]
            )
        for level in REWARD_LEVELS:
            out[f"reward/zero_sum_residual/{level}"] = Mean(
                total(resolve[f"residual/{level}"]), environment_ticks * tick
            )
        out["fog/fogged_environment_share"] = self.fogged_share
        slots, seen = total(perceive["fog/enemy_slots"]), total(perceive["fog/visible"])
        out["fog/visible_share/all"] = Mean(seen, slots)
        for name, _, _ in SCALE_BUCKETS:
            out[f"fog/visible_share/scale_{name}"] = Mean(
                total(perceive[f"fog/visible/{name}"]), total(perceive[f"fog/enemy_slots/{name}"])
            )
        out["fog/mean_hidden_age_seconds"] = Mean(
            total(perceive["fog/hidden_age_seconds"]), slots - seen
        )
        reacquisitions = total(perceive["fog/reacquisitions"])
        out["fog/reacquisitions_per_slot_minute"] = Mean(reacquisitions, slots * tick / 60.0)
        for name, _, _ in OCCLUSION_BUCKETS:
            out[f"fog/occlusion_share/{name}"] = Mean(
                total(perceive[f"fog/occlusion/{name}"]), reacquisitions
            )
        if tables is not None:
            out.update(outcome_series(tables))
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
        for index, pool in enumerate(self.slices):
            if not self.viewed[index]:
                continue
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


# ---------------------------------------------------------------------- the games' sums
#
# Per-environment sums the rollout adds every tick inside its captured graph (no host
# synchronisation) and the trainer takes once per update (diagnostics audit, groups 6, 12,
# 13, 14 and 15). `resolve_tick` reads the tick's events and rewards and the privileged
# rows at its start; `perceive_tick` the observation before and after the tick;
# `outcome_record` and `outcome_score` keep the outcome ledger of the learner's views.

EPISODE_EVENTS = ("shots_fired", "hits", "combat_deaths", "boundary_deaths", "spawns")
RESOLVE_COLUMNS = (
    "environment_ticks",
    "alive_ship_ticks",
    "combat_damage",
    "boundary_damage",
    "combat_deaths",
    "boundary_deaths",
    "speed",
    "energy",
    "out_of_energy",
    "in_field",
    "shield_restored",
    *(f"residual/{level}" for level in REWARD_LEVELS),
    *(f"event/{event}" for _, event in REWARD_EVENTS),
    "lives_ended",
    "life_seconds",
    "ship_episodes",
    *(f"episode_reward/{level}" for level in REWARD_LEVELS),
    *(f"episode_event/{name}" for name in EPISODE_EVENTS),
    "team_0_wins",
    "team_1_wins",
    "team_draws",
)
# Hidden spells' lengths at reacquisition, in seconds (lower bound inclusive).
OCCLUSION_BUCKETS = (
    ("under_0.5_seconds", 0.0, 0.5),
    ("0.5_2_seconds", 0.5, 2.0),
    ("2_5_seconds", 2.0, 5.0),
    ("5_10_seconds", 5.0, 10.0),
    ("10_30_seconds", 10.0, 30.0),
    ("over_30_seconds", 30.0, math.inf),
)
# Map scale (arena radius over MAP_SCALE_RADIUS) buckets, upper bound inclusive.
SCALE_BUCKETS = (("1", 0.0, 1.0), ("1_2", 1.0, 2.0), ("2_plus", 2.0, math.inf))
PERCEIVE_COLUMNS = (
    "fog/enemy_slots",
    "fog/visible",
    "fog/hidden_age_seconds",
    "fog/reacquisitions",
    *(f"fog/occlusion/{name}" for name, _, _ in OCCLUSION_BUCKETS),
    *(f"fog/enemy_slots/{name}" for name, _, _ in SCALE_BUCKETS),
    *(f"fog/visible/{name}" for name, _, _ in SCALE_BUCKETS),
    "observations",
    "map_scale",
    "enemy_slots",
    *(f"enemy_slots/{name}" for name, _, _ in SCALE_BUCKETS),
    "front_moves",
    "frontline_ticks",
)
# The outcome ledger: time-to-end buckets (seconds, lower bound inclusive) and the
# calibration's probability bins; the predictions are summed per second of elapsed time.
OUTCOME_BUCKETS = (
    ("under_5_seconds", 0.0, 5.0),
    ("5_15_seconds", 5.0, 15.0),
    ("15_30_seconds", 15.0, 30.0),
    ("30_60_seconds", 30.0, 60.0),
    ("60_120_seconds", 60.0, 120.0),
    ("over_120_seconds", 120.0, math.inf),
)
CALIBRATION_BINS = 10
# Outcome probability columns (`bnb.contracts.controller.Outputs.outcome`).
WIN, TIE, LOSS = 0, 1, 2


def blank_game(environments: int, ships: int, device) -> dict:
    """The running per-episode state of the games' sums: each ship's life so far (ticks),
    its reward in the episode per level, and each environment's episode events."""
    return dict(
        life_ticks=torch.zeros((environments, ships), device=device),
        episode_reward=torch.zeros((environments, ships, len(REWARD_LEVELS)), device=device),
        episode_events=torch.zeros((environments, len(EPISODE_EVENTS)), device=device),
    )


def blank_game_sums(environments: int, device) -> dict:
    """An update's games' sums: per environment, RESOLVE_COLUMNS and PERCEIVE_COLUMNS, and
    the lowest and highest ship-episode reward (summed over the levels) of the episodes
    ended."""
    return dict(
        resolve=torch.zeros((environments, len(RESOLVE_COLUMNS)), device=device),
        perceive=torch.zeros((environments, len(PERCEIVE_COLUMNS)), device=device),
        extremes=torch.tensor([math.inf, -math.inf], device=device).repeat(environments, 1),
    )


def reset_game_sums(sums: dict) -> None:
    """Empty an update's games' sums in place (no host copy: the buffers stay the ones a
    captured graph writes)."""
    sums["resolve"].zero_()
    sums["perceive"].zero_()
    sums["extremes"][:, 0].fill_(math.inf)
    sums["extremes"][:, 1].fill_(-math.inf)


def _bucket_masks(values: torch.Tensor, buckets) -> list[torch.Tensor]:
    return [(values >= low) & (values < high) for _, low, high in buckets]


@torch.no_grad()
def resolve_tick(
    game: dict,
    sums: dict,
    events,
    reward: torch.Tensor,
    event_rewards: torch.Tensor,
    rows: torch.Tensor,
    tick_length: float,
) -> None:
    """Add one tick to the sums, in place. events: the tick's Events (sim order); reward
    (B, N, levels) and event_rewards (B, N, len(REWARD_EVENTS)): the tick's, per ship;
    rows (B, N, columns): the privileged rows at the tick's start, in sim order (team 0's
    view)."""
    alive = rows[..., SHIP["alive"]] > 0.5
    living = alive.float()
    hit = events.damage.sum(1)  # (B, N) raw hit damage per target
    combat_death = events.died & (hit > 0.0)
    boundary_death = events.died & ~combat_death
    speed = torch.sqrt(rows[..., SHIP["velocity_x"]] ** 2 + rows[..., SHIP["velocity_y"]] ** 2)
    energy = rows[..., SHIP["energy"]]
    in_field = rows[..., SHIP["log_refractive_index"]].abs() > 1e-6
    done = events.done
    ended = done.float()
    # lives: a ship's life ends at its death or with its episode
    life = game["life_ticks"] + living
    life_ends = events.died | (done[:, None] & alive)
    episode_reward = game["episode_reward"] + reward
    ship_totals = episode_reward.sum(-1)  # (B, N)
    tick_events = torch.stack(
        [
            events.fired.float().sum(1),
            (events.damage > 0.0).float().sum((1, 2)),
            combat_death.float().sum(1),
            boundary_death.float().sum(1),
            events.spawned.float().sum(1),
        ],
        -1,
    )
    episode_events = game["episode_events"] + tick_events
    result = events.result
    columns = dict(
        environment_ticks=torch.ones_like(ended),
        alive_ship_ticks=living.sum(1),
        combat_damage=events.damage.sum((1, 2)),
        boundary_damage=events.boundary_damage.sum(1),
        combat_deaths=combat_death.float().sum(1),
        boundary_deaths=boundary_death.float().sum(1),
        speed=(speed * living).sum(1),
        energy=(energy * living).sum(1),
        out_of_energy=((energy <= 0.0).float() * living).sum(1),
        in_field=(in_field.float() * living).sum(1),
        shield_restored=events.shield_restored.sum(1),
        lives_ended=life_ends.float().sum(1),
        life_seconds=torch.where(life_ends, life, 0.0).sum(1) * tick_length,
        ship_episodes=ended * reward.shape[1],
        team_0_wins=ended * (result > 0).float(),
        team_1_wins=ended * (result < 0).float(),
        team_draws=ended * (result == 0).float(),
    )
    for index, level in enumerate(REWARD_LEVELS):
        columns[f"residual/{level}"] = reward[..., index].sum(1).abs()
        columns[f"episode_reward/{level}"] = ended * episode_reward[..., index].sum(1)
    for index, (_, event) in enumerate(REWARD_EVENTS):
        columns[f"event/{event}"] = event_rewards[..., index].abs().sum(1)
    for index, name in enumerate(EPISODE_EVENTS):
        columns[f"episode_event/{name}"] = ended * episode_events[:, index]
    sums["resolve"].add_(torch.stack([columns[name] for name in RESOLVE_COLUMNS], -1))
    low = torch.where(done, ship_totals.amin(1), math.inf)
    high = torch.where(done, ship_totals.amax(1), -math.inf)
    extremes = sums["extremes"]
    extremes.copy_(torch.stack([extremes[:, 0].minimum(low), extremes[:, 1].maximum(high)], -1))
    game["life_ticks"].copy_(torch.where(life_ends, 0.0, life))
    game["episode_reward"].copy_(torch.where(done[:, None, None], 0.0, episode_reward))
    game["episode_events"].copy_(torch.where(done[:, None], 0.0, episode_events))


@torch.no_grad()
def perceive_tick(
    sums: dict,
    previous: torch.Tensor,
    ships: torch.Tensor,
    truth: torch.Tensor,
    globals_: torch.Tensor,
    previous_globals: torch.Tensor,
    ships_per_team: int,
    tick_length: float,
    map_scale_radius: float,
) -> None:
    """Add one observation to the sums, in place. previous, ships (B, T, N, columns): each
    view's ship rows before and after the tick; truth: the privileged rows after it;
    globals_, previous_globals (B, T, columns): the views' globals after and before. The fog
    columns count fogged environments only; a reset's observation counts no
    reacquisition and no front move."""
    S = ships_per_team
    fogged = globals_[:, 0, GLOBAL["fogged"]] > 0.5  # (B,)
    reset = globals_[:, 0, GLOBAL["reset"]] > 0.5
    alive = truth[..., S:, SHIP["alive"]] > 0.5  # (B, T, E) living enemies of each view
    visible = ships[..., S:, SHIP["visible"]] > 0.5
    was_visible = previous[..., S:, SHIP["visible"]] > 0.5
    hidden = alive & ~visible
    reacquired = alive & visible & ~was_visible & ~reset[:, None, None]
    spell = previous[..., S:, SHIP["time_since_seen"]] + tick_length  # the hidden spell's length
    scale = globals_[:, 0, GLOBAL["arena_radius"]] / map_scale_radius  # (B,)
    fog = fogged.float()

    def count(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().sum((1, 2))

    slots = count(alive)
    columns = {
        "fog/enemy_slots": fog * slots,
        "fog/visible": fog * count(alive & visible),
        "fog/hidden_age_seconds": fog
        * torch.where(hidden, ships[..., S:, SHIP["time_since_seen"]], 0.0).sum((1, 2)),
        "fog/reacquisitions": fog * count(reacquired),
        "observations": torch.ones_like(scale),
        "map_scale": scale,
        "enemy_slots": slots,
    }
    for (name, _, _), member in zip(
        OCCLUSION_BUCKETS, _bucket_masks(spell, OCCLUSION_BUCKETS), strict=True
    ):
        columns[f"fog/occlusion/{name}"] = fog * count(reacquired & member)
    for name, low, high in SCALE_BUCKETS:
        inside = ((scale > low) & (scale <= high)).float()
        columns[f"enemy_slots/{name}"] = inside * slots
        columns[f"fog/enemy_slots/{name}"] = fog * inside * slots
        columns[f"fog/visible/{name}"] = fog * inside * count(alive & visible)
    frontline = (globals_[:, 0, GLOBAL["mode_frontline"]] > 0.5) & ~reset
    moved = (globals_[:, 0, GLOBAL["front"]] - previous_globals[:, 0, GLOBAL["front"]]).abs()
    columns["front_moves"] = torch.where(frontline, moved, 0.0)
    columns["frontline_ticks"] = frontline.float()
    sums["perceive"].add_(torch.stack([columns[name] for name in PERCEIVE_COLUMNS], -1))


def blank_outcome(views: int, blocks: int, device) -> dict:
    """The outcome ledger's running state: per learner view and second of elapsed time in
    its episode, the predictions' count, sum and sum of squares of P(win | resolved), and
    per calibration bin their count and sum."""
    return dict(
        blocks=torch.zeros((views, blocks, 3), device=device),
        bins=torch.zeros((views, CALIBRATION_BINS, 2), device=device),
    )


def blank_outcome_tables(device) -> dict:
    """An update's outcome tables: per time-to-end bucket (count, sum p, Brier sum, wins)
    and per calibration bin (count, sum p, wins)."""
    return dict(
        skill=torch.zeros((len(OUTCOME_BUCKETS), 4), device=device),
        calibration=torch.zeros((CALIBRATION_BINS, 3), device=device),
    )


def win_given_resolved(probabilities: torch.Tensor) -> torch.Tensor:
    """P(win | the match resolves) of outcome probabilities (..., 4)."""
    resolved = probabilities[..., WIN] + probabilities[..., TIE] + probabilities[..., LOSS]
    return probabilities[..., WIN] / resolved.clamp_min(torch.finfo(probabilities.dtype).tiny)


@torch.no_grad()
def outcome_record(
    ledger: dict, probabilities: torch.Tensor, elapsed_seconds: torch.Tensor
) -> None:
    """Add each view's prediction (R, 4) at elapsed_seconds (R,) into its ledger, in place."""
    blocks, bins = ledger["blocks"], ledger["bins"]
    p = win_given_resolved(probabilities.float())
    block = elapsed_seconds.long().clamp(0, blocks.shape[1] - 1)
    one = torch.ones_like(p)
    blocks.scatter_add_(
        1, block[:, None, None].expand(-1, 1, 3), torch.stack([one, p, p * p], -1)[:, None]
    )
    bin_index = (p * CALIBRATION_BINS).long().clamp(0, CALIBRATION_BINS - 1)
    bins.scatter_add_(
        1, bin_index[:, None, None].expand(-1, 1, 2), torch.stack([one, p], -1)[:, None]
    )


@torch.no_grad()
def outcome_score(
    ledger: dict, tables: dict, done: torch.Tensor, won: torch.Tensor, end_seconds: torch.Tensor
) -> None:
    """Score the ended views' ledgers (done (R,), won (R,) the view's team won, end_seconds
    (R,) the episode's length) into the tables and clear them, in place. A block's
    predictions lie at most a second from its centre."""
    blocks, bins = ledger["blocks"], ledger["bins"]
    w = (won & done).float()[:, None]  # (R, 1)
    scored = done.float()[:, None]
    count, total, square = blocks.unbind(-1)  # (R, K)
    centre = torch.arange(blocks.shape[1], device=blocks.device, dtype=blocks.dtype) + 0.5
    to_end = end_seconds[:, None] - centre
    brier = square - 2.0 * w * total + w * count
    values = torch.stack([count, total, brier, w * count], -1) * scored[..., None]  # (R, K, 4)
    for index, member in enumerate(_bucket_masks(to_end, OUTCOME_BUCKETS)):
        tables["skill"][index] += (values * member[..., None]).sum((0, 1))
    bin_count, bin_total = bins.unbind(-1)
    tables["calibration"] += (
        torch.stack([bin_count, bin_total, w * bin_count], -1) * scored[..., None]
    ).sum(0)
    keep = ~done
    blocks.copy_(blocks * keep[:, None, None])
    bins.copy_(bins * keep[:, None, None])


def outcome_series(tables: dict) -> dict:
    """The outcome ledger's series (diagnostics audit, group 6): per time-to-end bucket the
    Brier skill score, 1 - Brier / the base rate's Brier (absent where the base rate is 0 or
    1); over every scored prediction the expected calibration error over CALIBRATION_BINS
    bins and the bias, mean predicted less realised P(win)."""
    out = {}
    count, total, brier, wins = tables["skill"].unbind(-1)  # (buckets,) each
    rate = wins / count.clamp_min(1.0)
    reference = rate * (1.0 - rate)
    usable = (count > 0) & (reference > 0)
    skill = 1.0 - (brier / count.clamp_min(1.0)) / torch.where(usable, reference, 1.0)
    for index, (name, _, _) in enumerate(OUTCOME_BUCKETS):
        out[f"outcome/brier_skill/{name}"] = Mean(
            torch.where(usable[index], skill[index], 0.0), usable[index].float()
        )
    bin_count, bin_total, bin_wins = tables["calibration"].unbind(-1)
    predictions = bin_count.sum()
    gap = (bin_total - bin_wins).abs()  # count x |mean predicted - realised| per bin
    out["outcome/expected_calibration_error"] = Mean(gap.sum(), predictions)
    out["outcome/bias"] = Mean(bin_total.sum() - bin_wins.sum(), predictions)
    return out


def reliability_table(calibration: list[list[float]]) -> Table:
    """The reliability diagram's table (host rows of the calibration table, (bins, 3)): per
    bin its range, the mean predicted and realised P(win) and the count."""
    rows = []
    for index, (count, total, wins) in enumerate(calibration):
        if count:
            low = index / CALIBRATION_BINS
            rows.append((low, low + 1 / CALIBRATION_BINS, total / count, wins / count, int(count)))
    return Table(("low", "high", "predicted", "realised", "count"), tuple(rows))
