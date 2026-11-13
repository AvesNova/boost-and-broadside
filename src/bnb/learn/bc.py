"""Behaviour cloning (scripted spec B): section 6's learner with the policy's loss replaced
by the cross-entropy of the student's joint distribution against a teacher's.

Everything else is the learner's: the segments, streaming, `unroll`, the critic on the
student's own returns, the auxiliary heads, micro-batching, mixed batches and the return
scale. No policy gradient, no entropy bonus and no KL early stop. The teacher labels the
stored observations at update time, on the device, so nothing beyond the observation is
stored; the scripted agent's own temperature and a uniform floor set the labels' entropy.
BC's scored opponent plays the labels' distribution (`TeacherOpponent`), so the live score
is against the teacher the labels describe.
"""

import collections
import dataclasses
import math
from collections.abc import Callable

import torch
from torch.utils import _pytree

from bnb.contracts import Action, Obs, Outputs, Trajectory
from bnb.contracts.action import NUM_JOINT, NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL
from bnb.contracts.series import Mean
from bnb.controllers.scripted import Scripted, uniform_actions
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.controllers.scripted_parameters import ModeParameters, ScriptedParameters
from bnb.game.handling.characteristics import Characteristics
from bnb.models.yemong import policy

from . import losses
from .ppo import Learner, LearnerConfig, factor_sums

# The student's agreement with the labels' likeliest action: per factor, and jointly.
AGREEMENTS = ("power", "stick", "shoot", "joint")
# BC's per-pool sums over the living own ship-steps, in order.
SUMS = ("kl", "teacher", "student", "tokens", *(f"agreement_{name}" for name in AGREEMENTS))
TEACHER_LABELS = "teacher_labels"  # the derived name of the labels (L, V, S, 42)
# the carry's field names, taken once: torch.compile cannot trace dataclasses.fields


MINIBATCHES = 256
# BC's returns per reward level (REWARD_LEVELS' order), in place of PPO's: the outcome
# undiscounted and every level's trace near 1, as every BC arm ran from 2026-11-06 (with
# PPO's values the outcome head's Brier skill was negative in the first 10M steps), the two
# combat levels' discounts and traces squared for the x2 game (game v13).
DISCOUNT = (0.982, 0.990, 0.995, 0.999, 1.0)
TRACE_DECAY = (0.98, 0.98, 0.99, 0.99, 1.0)


def config(**overrides) -> LearnerConfig:
    """BC's learner settings (scripted spec B1): the BC coefficient 0.025 in nats per token
    in place of the surrogate's, no entropy term, no KL early stop, 256 minibatches, and
    BC's returns (`DISCOUNT`, `TRACE_DECAY`); the rest section 6's. Cloning's progress
    follows its optimizer steps, so at the 8,192-view batch (2,048 x 4) it takes
    minibatches of 32 views, not PPO's 32 per epoch: 8x the steps per environment step for
    the same compute."""
    defaults = dict(minibatches=MINIBATCHES, discount=DISCOUNT, trace_decay=TRACE_DECAY)
    return dataclasses.replace(
        LearnerConfig(),
        policy=0.025,
        entropy_power=0.0,
        entropy_stick=0.0,
        entropy_shoot=0.0,
        target_kl=None,
        **{**defaults, **overrides},
    )


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """The teacher: the scripted agent at its own temperature T (`Scripted`'s), its labels
    then mixed with uniform, (1 - floor) q_T + floor / 42. The default, T = 1 and floor 0,
    is the scripted agent itself: it gives no living ship's action probability 0 but a
    shot at its gates (its floors, `bnb.controllers.scripted_parameters.EPSILONS`), so the
    KL needs no floor of its own."""

    temperature: float = 1.0
    floor: float = 0.0

    def __post_init__(self) -> None:
        if not self.temperature > 0.0 or not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"need temperature > 0 and floor in [0, 1], got {self}")


def labels(probabilities: torch.Tensor, floor: float) -> torch.Tensor:
    """The labels from the teacher's distribution at its temperature (..., 42): mixed with
    uniform by `floor`."""
    if floor != 0.0:
        probabilities = (1.0 - floor) * probabilities + floor / NUM_JOINT
    return probabilities


def teacher_agent(
    characteristics: Characteristics,
    ships_per_team: int,
    teacher: TeacherConfig,
    parameters: ScriptedParameters | ModeParameters = MODE_DEFAULTS,
) -> Scripted:
    """The scripted agent at the teacher's temperature."""
    return Scripted(characteristics, ships_per_team, parameters, teacher.temperature)


class TeacherOpponent:
    """The scripted agent as BC's opponent, playing the labels' distribution: the agent at
    the teacher's temperature, and with probability `floor` a uniform joint action instead.
    At floor 0 it is the agent at that temperature."""

    reads = Scripted.reads

    def __init__(
        self,
        characteristics: Characteristics,
        ships_per_team: int,
        teacher: TeacherConfig,
        parameters: ScriptedParameters | ModeParameters = MODE_DEFAULTS,
    ) -> None:
        self.agent = teacher_agent(characteristics, ships_per_team, teacher, parameters)
        self.teacher = teacher

    def init_carry(self, views: int, device: torch.device) -> ScriptedCarry:
        """The agent's carry."""
        return self.agent.init_carry(views, device)

    def action_distribution(self, obs: Obs, carry: ScriptedCarry | None = None) -> torch.Tensor:
        """(R, S, NUM_JOINT) the distribution `step` samples from given a carry it returned
        (None: memoryless): the labels'."""
        return labels(self.agent.action_distribution(obs, carry), self.teacher.floor)

    def step(self, carry: ScriptedCarry, obs: Obs, reset: torch.Tensor):
        """One decision for views (R, ...), drawn from the labels' distribution; see
        `Controller.step`."""
        action, carry, _ = self.agent.step(carry, obs, reset)
        action, floor = action.packed(), self.teacher.floor
        if floor != 0.0:
            uniform = torch.rand(action.shape[:-1], device=action.device) < floor
            action = torch.where(
                uniform[..., None], uniform_actions(action.shape[:-1], action.device), action
            )
        return Action.unpack(action), carry, Outputs()


def factor_marginals(probabilities: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The power, stick and shoot marginals of joint probabilities (..., NUM_JOINT), in
    `bnb.contracts.action.to_joint`'s order, and the joint itself."""
    joint = probabilities.unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    return joint.sum((-2, -1)), joint.sum((-3, -1)), joint.sum((-3, -2)), probabilities


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """In nats, over the last axis (0 log 0 = 0)."""
    return -torch.special.xlogy(probabilities, probabilities).sum(-1)


def flat_views(obs: Obs) -> Obs:
    """Stored observations (L, n, ...) as views (L n, ...)."""
    parts = {field.name: getattr(obs, field.name) for field in dataclasses.fields(obs)}
    return Obs(
        **{name: None if value is None else value.flatten(0, 1) for name, value in parts.items()}
    )


@torch.no_grad()
def teacher_carries(
    teacher: Scripted, obs: Obs, carry: ScriptedCarry
) -> tuple[ScriptedCarry, ScriptedCarry]:
    """The teacher run over stored observations (L, n, ...) in time order from its carry
    (n, ...), each tick's reset read from the observation: (its carry after every tick,
    (L, n, ...), and after the last, (n, ...))."""
    steps = []
    for tick in range(obs.ships.shape[0]):
        view = _pytree.tree_map_only(torch.Tensor, lambda x, t=tick: x[t], obs)
        reset = view.globals[:, GLOBAL["reset"]] > 0.5
        _, carry = teacher.decide(carry, view, reset)
        steps.append(carry)
    return _pytree.tree_map(lambda *leaves: torch.stack(leaves), *steps), carry


def teacher_distribution(teacher: Scripted, obs: Obs, carries: ScriptedCarry) -> torch.Tensor:
    """The teacher on stored observations (L, n, ...) given its carry at each
    (`teacher_carries`): (L, n, S, 42)."""
    length, views = obs.ships.shape[:2]
    flat = _pytree.tree_map(lambda x: x.flatten(0, 1), carries)
    return teacher.action_distribution(flat_views(obs), flat).unflatten(0, (length, views))


class BehaviourCloning(Learner):
    """teacher: the scripted agent at the teacher's temperature; labels: its distribution
    given its carry, transformed. pools: the pool names, view_pool (V,) each stored view's
    pool index (the metrics are per pool). The other arguments are the Learner's."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: LearnerConfig,
        rules,
        types,
        outcome_weight: float,
        device: torch.device,
        teacher: Scripted,
        teacher_config: TeacherConfig,
        pools: tuple[str, ...],
        view_pool: torch.Tensor,
        stage: Callable = lambda function: function,
        seed: int = 0,
        diagnostics: bool = True,
        split: bool = False,
    ) -> None:
        self.teacher, self.teacher_config, self.pools = teacher, teacher_config, pools
        self.teacher_carry: ScriptedCarry | None = None  # (R, ...) after the last update
        # (KL, teacher entropy, student entropy, tokens, agreements) per pool.
        self.STATISTICS = tuple(f"{name}/{pool}" for name in SUMS for pool in pools)
        self.pool_one_hot = torch.nn.functional.one_hot(view_pool.to(device), len(pools)).float()[
            None
        ]
        super().__init__(
            model, config, rules, types, outcome_weight, device, stage, seed, diagnostics, split
        )

    def prepare(self, segments, bootstrap) -> dict:
        """The Learner's prepare, with each view's pool one-hot (1, V, pools) added to derived for
        the per-pool metrics."""
        prepared = super().prepare(segments, bootstrap)
        prepared["derived"]["pool"] = self.pool_one_hot  # (1, V, pools)
        prepared["derived"][TEACHER_LABELS] = self._teacher_labels(segments)
        return prepared

    @torch.no_grad()
    def _teacher_labels(self, segments) -> torch.Tensor:
        """The labels at every stored tick, (L, V, S, 42): the teacher run over each shard
        in time order from the last update's carry (`teacher_carries`), its distribution
        given its carry at each tick, transformed (`labels`). Made once per update, outside
        the compiled loss: the labels do not change between epochs, and the loss graph then
        holds no copy of the scripted agent."""
        rows = segments.views // segments.shards
        carry = self.teacher_carry
        if carry is None:
            carry = self.teacher.init_carry(rows, self.device)
        # as many views a pass as a loss micro-batch held when the loss made them
        ticks = max(1, self.micro_views * segments.length // rows)
        shards = []
        for shard in range(segments.shards):
            obs = segments.shard(("obs",), shard)["obs"]
            carries, carry = teacher_carries(self.teacher, obs, carry)
            chunks = []
            for start in range(0, segments.length, ticks):
                part = slice(start, start + ticks)
                chunk_obs = _pytree.tree_map_only(torch.Tensor, lambda x, p=part: x[p], obs)
                chunk_carries = _pytree.tree_map(lambda x, p=part: x[p], carries)
                chunks.append(
                    labels(
                        teacher_distribution(self.teacher, chunk_obs, chunk_carries),
                        self.teacher_config.floor,
                    )
                )
            shards.append(torch.cat(chunks))
        self.teacher_carry = carry
        return torch.cat(shards, 1)

    def _policy(
        self,
        unrolled,
        trajectory: Trajectory,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """(the cross-entropy against the labels, the student's mean entropy's factor parts
        (3,), per-pool sums
        of KL(labels || student), the labels' and the student's entropy and the tokens),
        over own ships alive."""
        own_alive = trajectory.alive[..., : trajectory.log_probability.shape[-1]]
        target = derived[TEACHER_LABELS]
        logits = unrolled.policy_logits.float()
        cross = losses.cross_entropy(logits, target)
        teacher_entropy = entropy(target)
        # PPO's formula (`bnb.models.yemong.policy.evaluate`): from log-probabilities, so a
        # probability that underflows to 0 gives 0 x a finite number in the gradient, where
        # p log p of the probabilities would give 0 x infinity.
        log_probabilities = torch.log_softmax(logits, -1)
        student_entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
        per_view = torch.stack(
            [
                torch.where(own_alive, value, 0.0).sum((0, 2))
                for value in (
                    cross - teacher_entropy,
                    teacher_entropy,
                    student_entropy,
                    own_alive.float(),
                )
            ]
        )
        agreements = [
            (student.argmax(-1) == teacher.argmax(-1)).float()
            for student, teacher in zip(
                factor_marginals(log_probabilities.exp()), factor_marginals(target), strict=True
            )
        ]
        per_view = torch.cat(
            [
                per_view,
                torch.stack(
                    [torch.where(own_alive, value, 0.0).sum((0, 2)) for value in agreements]
                ),
            ]
        )
        statistics = (per_view @ derived["pool"][0]).flatten()  # (SUMS x pools,)
        return (
            losses.masked_sum(cross, own_alive) / denominator["policy"],
            factor_sums(policy.factor_entropies(logits), own_alive) / denominator["policy"],
            statistics,
        )

    def report(
        self, first_epoch: torch.Tensor, last_epoch: torch.Tensor, mean: torch.Tensor
    ) -> dict:
        """Over the first epoch (the score stop reads `bc/kl`), per living own ship-step:
        the BC KL, both entropies and the student's agreement with the labels' likeliest
        action per factor and jointly (`bc/agreement/{factor}`), overall and the KL and
        entropies per pool (`bc/kl_by_pool/{pool}`, ...)."""
        sums = first_epoch.reshape(len(SUMS), len(self.pools))
        named = dict(zip(SUMS, sums.unbind(), strict=True))
        tokens = named["tokens"]
        out = {
            "bc/kl": Mean(named["kl"].sum(), tokens.sum()),
            "bc/entropy/target": Mean(named["teacher"].sum(), tokens.sum()),
            "bc/entropy/student": Mean(named["student"].sum(), tokens.sum()),
        }
        for factor in AGREEMENTS:
            out[f"bc/agreement/{factor}"] = Mean(named[f"agreement_{factor}"].sum(), tokens.sum())
        for index, pool in enumerate(self.pools):
            out[f"bc/kl_by_pool/{pool}"] = Mean(named["kl"][index], tokens[index])
            out[f"bc/entropy_by_pool/target/{pool}"] = Mean(named["teacher"][index], tokens[index])
            out[f"bc/entropy_by_pool/student/{pool}"] = Mean(named["student"][index], tokens[index])
        return out


def mean_entropy(probabilities: torch.Tensor, floor: float = 0.0) -> float:
    """The labels' mean entropy over rows of teacher distributions (tokens, 42)."""
    return float(entropy(labels(probabilities, floor)).mean())


def calibrate(
    distributions: Callable[[float], torch.Tensor],
    target: float,
    floor: float = 0.0,
    tolerance: float = 1e-3,
    low: float = 1e-2,
    high: float = 1e2,
) -> TeacherConfig:
    """The temperature (floor fixed) at which the labels' mean entropy is `target` nats over
    the teacher's distributions, `distributions(T)` (tokens, 42) for the scripted agent at
    temperature T on fixed observations (living own ship-steps), by bisection on log T.
    Fails when the target lies outside what [low, high] reaches (raise the floor when it is
    above what the temperature alone can give)."""

    def at(temperature: float) -> float:
        return mean_entropy(distributions(temperature), floor)

    reach = [at(t) for t in (low, high)]
    if not reach[0] <= target <= reach[1]:
        raise ValueError(
            f"target entropy {target} nats is outside [{reach[0]:.4f}, "
            f"{reach[1]:.4f}] for temperatures [{low}, {high}] at floor {floor}"
        )
    log_low, log_high = math.log(low), math.log(high)
    while True:
        middle = math.exp(0.5 * (log_low + log_high))
        value = at(middle)
        if abs(value - target) <= tolerance or log_high - log_low < 1e-6:
            return TeacherConfig(middle, floor)
        if value < target:
            log_low = math.log(middle)
        else:
            log_high = math.log(middle)


@dataclasses.dataclass(frozen=True)
class StopConfig:
    """When BC stops (scripted spec B4), whichever comes first: BC's imitation has
    plateaued (`Plateau`: the BC KL's trailing mean over plateau_smoothing_steps improved
    by less than plateau_fraction of its improvement since plateau_from_steps over the
    last plateau_window_steps, at every update for plateau_hold_steps; the default rule,
    None turns it off); the score against the raw scripted agent in the rating lanes,
    over the last score_window finished games, of the better of the players `reads` names
    (the averages of the live weights; decisions: BC round three), reaches score_target
    (None, the default, turns it off: the windows are still kept and logged); or
    budget_steps environment steps (a run stopped by its budget resumes with a larger one).
    `reads` is in order of preference: an equal score goes to the earlier. The windows need
    rating lanes: with none (`LaneConfig.environments` 0) they never fill and the score
    stop never fires. Which policy RL starts from is not the stop's choice: it is always
    the exponential average (`bnb.runs.session.HANDOFF_PLAYER`)."""

    # the hub's BC runs (2026-11-09, 2026-11-10): offline, 5% over 10M steps since 5M
    # matched the calibrated Elo's plateau (author, 2026-11-11); online, a held condition
    # stands for the offline "stays below thereafter" (docs/explanation/behaviour-cloning.md)
    plateau_fraction: float | None = 0.05
    plateau_window_steps: int = 10_000_000
    plateau_from_steps: int = 5_000_000
    plateau_smoothing_steps: int = 5_000_000
    plateau_hold_steps: int = 10_000_000
    # parity with the scripted agent was 0.5 (author, 2026-11-04); off since the plateau
    score_target: float | None = None
    score_window: int = 1000
    # v1's time to parity (author, 2026-11-04)
    budget_steps: int = 50_000_000
    # the lane players `bnb.runtime.lanes.EXPONENTIAL_AVERAGE` and `UNIFORM_AVERAGE`
    reads: tuple[str, ...] = ("exponential_average", "uniform_average")

    def __post_init__(self) -> None:
        if self.score_window < 2:
            raise ValueError(f"the score window needs two games for its spread, got {self}")
        if not self.reads or len(set(self.reads)) != len(self.reads):
            raise ValueError(f"the score stop reads distinct players, got {self.reads}")
        if self.plateau_fraction is not None and not 0.0 < self.plateau_fraction < 1.0:
            raise ValueError(f"the plateau fraction is in (0, 1) or None, got {self}")
        if self.plateau_window_steps < 1 or self.plateau_smoothing_steps < 1:
            raise ValueError(f"the plateau's window and smoothing need a step, got {self}")
        if self.plateau_from_steps < 0 or self.plateau_hold_steps < 0:
            raise ValueError(f"the plateau's start and hold are not negative, got {self}")


class Plateau:
    """BC's plateau rule on the BC KL, fed once per update (`observe`) with the run's
    environment steps and the update's KL; it reads only the past.

    The KL is smoothed by its trailing mean over the updates of the last `smoothing` steps
    (`StopConfig.plateau_smoothing_steps`). Its improvement since `from`
    (`plateau_from_steps`) is measured from a baseline, the plain mean of the updates
    within half a smoothing on either side of `from` (fixed once the updates past it are
    in; the first update's KL, the untrained policy's, lies before it); its recent
    improvement is the fall of the trailing mean since the newest update at least
    `window` steps back. The condition: the improvement since `from` is positive and the
    recent one is below `fraction` of it. The rule fires once the condition has held at
    every update for `hold` steps: offline, the condition was read on centred means and
    had to stay true for the rest of the run; online, a hold stands for that, and keeps
    a run from stopping in a temporary plateau (every hub run has one at 20-30M steps,
    4M steps long).

    Its state (`state`: the updates the smoothing still reads, the trailing means the
    window still reads, the baseline and the step the condition began holding) is plain
    JSON, kept in the training checkpoint so that a resume decides as one process would."""

    def __init__(self, config: StopConfig, state: dict | None = None) -> None:
        self.config = config
        self.values: list[tuple[int, float]] = []  # (environment steps, the update's KL)
        self.means: list[tuple[int, float]] = []  # (environment steps, the trailing mean)
        self.baseline: float | None = None
        self.held_from: int | None = None  # the first step of the condition's current hold
        self.fraction: float | None = None  # recent improvement / improvement since `from`
        if state is not None:
            self.values = [(steps, value) for steps, value in state["values"]]
            self.means = [(steps, mean) for steps, mean in state["means"]]
            self.baseline, self.held_from = state["baseline"], state["held_from"]
            self.fraction = state["fraction"]

    @property
    def state(self) -> dict:
        """The rule's state as JSON (floats round-trip exactly through JSON)."""
        return dict(
            values=[list(entry) for entry in self.values],
            means=[list(entry) for entry in self.means],
            baseline=self.baseline,
            held_from=self.held_from,
            fraction=self.fraction,
        )

    @property
    def mean(self) -> float | None:
        """The KL's trailing mean after the last update."""
        return self.means[-1][1] if self.means else None

    def held_steps(self, environment_steps: int) -> int:
        """How long the condition has held, in environment steps (0 when it does not)."""
        return 0 if self.held_from is None else environment_steps - self.held_from

    def observe(self, environment_steps: int, kl: float) -> bool:
        """environment_steps: the run's total after this update; kl: its BC KL. Whether the
        rule fires."""
        config = self.config
        half = config.plateau_smoothing_steps / 2
        start = config.plateau_from_steps
        self.values.append((environment_steps, kl))
        if self.baseline is None and environment_steps >= start + half:
            around = [value for steps, value in self.values if start - half < steps <= start + half]
            self.baseline = sum(around) / len(around) if around else kl
        recent_values = [
            entry
            for entry in self.values
            if entry[0] > environment_steps - config.plateau_smoothing_steps
        ]
        mean = sum(value for _, value in recent_values) / len(recent_values)
        self.means.append((environment_steps, mean))
        cutoff = environment_steps - config.plateau_window_steps
        past = [average for steps, average in self.means if steps <= cutoff]
        self.fraction, holds = None, False
        if self.baseline is not None and past:
            total = self.baseline - mean
            if total > 0.0:
                self.fraction = (past[-1] - mean) / total
                holds = config.plateau_fraction is not None and (
                    self.fraction < config.plateau_fraction
                )
        if holds:
            self.held_from = environment_steps if self.held_from is None else self.held_from
        else:
            self.held_from = None
        # The window reads the newest mean at or before the cutoff, which only rises, and
        # the smoothing the values of the last smoothing steps (the baseline's lie among
        # them until it is fixed): the rest is never read again.
        while len(self.means) > 1 and self.means[1][0] <= cutoff:
            self.means.pop(0)
        self.values = recent_values
        return holds and self.held_steps(environment_steps) >= config.plateau_hold_steps


class ScoreWindow:
    """One player's score over its last `games` finished games, fed once per update.

    It keeps per-update totals (games, points, squared points; a game scores 1, 0.5 or 0):
    it drops the oldest updates whole while the rest still fill it, then counts the oldest
    kept update by the exact share of its games the window still needs, so trimming
    favours no result. Its state (`updates`) is plain JSON."""

    def __init__(self, games: int, updates: list | None = None) -> None:
        self.games = games
        self.updates: collections.deque[tuple[float, float, float]] = collections.deque(
            tuple(update) for update in updates or ()
        )

    def add(self, wins: int, draws: int, losses: int) -> None:
        """One update's finished games."""
        games = wins + draws + losses
        if games > 0:
            self.updates.append((float(games), wins + 0.5 * draws, wins + 0.25 * draws))
        while (
            len(self.updates) > 1
            and sum(update[0] for update in self.updates) - self.updates[0][0] >= self.games
        ):
            self.updates.popleft()

    def _totals(self) -> tuple[float, float] | None:
        """(points, squared points) over the last `games` games, once there are that many."""
        games = sum(update[0] for update in self.updates)
        if games < self.games:
            return None
        oldest_games, oldest_points, oldest_squares = self.updates[0]
        share = (games - self.games) / oldest_games  # of the oldest update, outside the window
        return (
            sum(update[1] for update in self.updates) - oldest_points * share,
            sum(update[2] for update in self.updates) - oldest_squares * share,
        )

    @property
    def score(self) -> float | None:
        """Wins plus half the draws over the window, once it is full."""
        totals = self._totals()
        return None if totals is None else totals[0] / self.games

    @property
    def standard_error(self) -> float | None:
        """The window score's standard error, from the spread of its own per-game scores
        (the sample variance over games - 1)."""
        totals = self._totals()
        if totals is None:
            return None
        points, squares = totals
        variance = max(squares - points * points / self.games, 0.0) / (self.games - 1)
        return math.sqrt(variance / self.games)

    @property
    def state(self) -> list:
        """The per-update totals as JSON (floats round-trip exactly)."""
        return [list(update) for update in self.updates]


class StopRule:
    """Fed once per update; `observe` names the rule that fires, or returns None.

    The plateau rule reads the BC KL (`Plateau`). Each lane player's games against the
    scripted agent fill a score window of its own (`ScoreWindow`, `windows` by player).
    The score stop, when it has a target, reads the window of the best of the players
    `StopConfig.reads` names whose window is full (`read`; an equal score goes to the
    earlier in `reads`); the others' windows, the live clone's among them, are logged.

    Its state (`state`: the windows and the plateau's) is plain JSON, kept in the training
    checkpoint so that a resume continues the same windows."""

    def __init__(self, config: StopConfig, state: dict | None = None) -> None:
        self.config = config
        self.windows: dict[str, ScoreWindow] = {}
        self.plateau = Plateau(config, None if state is None else state["plateau"])
        if state is not None:
            self.windows = {
                player: ScoreWindow(config.score_window, updates)
                for player, updates in state["windows"].items()
            }

    @property
    def state(self) -> dict:
        """The rule's state as JSON: each player's window and the plateau's state."""
        return dict(
            windows={player: window.state for player, window in self.windows.items()},
            plateau=self.plateau.state,
        )

    @property
    def read(self) -> str | None:
        """The player the score stop reads: of `StopConfig.reads` with a full window, the
        best scoring (an equal score goes to the earlier); None while none is full."""
        scores = [
            (self.windows[player].score, -index, player)
            for index, player in enumerate(self.config.reads)
            if player in self.windows and self.windows[player].score is not None
        ]
        return max(scores)[2] if scores else None

    @property
    def score(self) -> float | None:
        """The read player's window score, once a window is full."""
        player = self.read
        return None if player is None else self.windows[player].score

    @property
    def standard_error(self) -> float | None:
        """The read player's window score's standard error; reported beside the score."""
        player = self.read
        return None if player is None else self.windows[player].standard_error

    def observe(
        self,
        environment_steps: int,
        results: dict[str, tuple[int, int, int]],
        kl: float,
    ) -> str | None:
        """environment_steps: the total so far; results: each lane player's games finished
        against the raw scripted agent this update, (wins, draws, losses) from its side (a
        player left out played none that count); kl: this update's BC KL. The rule that
        fires: "plateau", "score" or "budget", or None."""
        config = self.config
        for player, games in results.items():
            self.windows.setdefault(player, ScoreWindow(config.score_window)).add(*games)
        if self.plateau.observe(environment_steps, kl):
            return "plateau"
        target = config.score_target
        if target is not None and self.score is not None and self.score >= target:
            return "score"
        if environment_steps >= config.budget_steps:
            return "budget"
        return None
