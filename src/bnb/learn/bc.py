"""Behaviour cloning (scripted spec B): section 6's learner with the policy's loss replaced
by the cross-entropy of the student's joint distribution against a teacher's.

Everything else is the learner's: the segments, streaming, `unroll`, the critic on the
student's own returns, the auxiliary heads, micro-batching, mixed batches and the return
scale. No policy gradient, no entropy bonus and no KL early stop. The teacher labels the
stored observations at update time, on the device, so nothing beyond the observation is
stored; a temperature and a uniform floor set the labels' entropy. BC's scored opponent
plays the labels' distribution (`TeacherOpponent`), so the live score is against the
teacher the labels describe; the scripted agent itself is untouched.
"""

import collections
import dataclasses
import math
from collections.abc import Callable

import torch

from bnb.contracts import Action, Obs, Outputs, Trajectory
from bnb.contracts.action import NUM_JOINT
from bnb.controllers.scripted import Scripted, marginals, sample, uniform_actions

from . import losses
from .ppo import Learner, LearnerConfig

Teacher = Callable[[Obs], torch.Tensor]  # views (R, ...) -> (R, S, 42) probabilities
# The labels' default uniform floor (author): no label gives an action probability 0, so the
# clone can sample every action in reinforcement learning.
FLOOR = 0.01


def config(**overrides) -> LearnerConfig:
    """BC's learner settings (scripted spec B1): the BC coefficient 0.025 in nats per token
    in place of the surrogate's, no entropy term, no KL early stop; the rest section 6's."""
    return dataclasses.replace(LearnerConfig(), policy=0.025, entropy=0.0, target_kl=None,
                               **overrides)


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """The labels' transformation: q_T proportional to q^(1/T), then (1 - floor) q_T +
    floor / 42. The floor defaults to `FLOOR`; T = 1 and floor 0 leave them as they are."""

    temperature: float = 1.0
    floor: float = FLOOR

    def __post_init__(self) -> None:
        if not self.temperature > 0.0 or not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"need temperature > 0 and floor in [0, 1], got {self}")


def labels(probabilities: torch.Tensor, teacher: TeacherConfig) -> torch.Tensor:
    """The teacher's labels from its distribution (..., 42). A structural zero stays zero
    under the temperature alone."""
    if teacher.temperature != 1.0:
        probabilities = torch.softmax(torch.log(probabilities) / teacher.temperature, -1)
    if teacher.floor != 0.0:
        probabilities = (1.0 - teacher.floor) * probabilities + teacher.floor / NUM_JOINT
    return probabilities


class TeacherOpponent:
    """The scripted agent as BC's opponent, playing the labels' distribution: each part's
    marginal tempered (q^(1/T) normalised per part tempers the joint, their product), then
    with probability `floor` a uniform joint action instead. At T = 1 and floor 0 it draws
    exactly as `agent` does. agent: a scripted agent at scripted probability 1."""

    reads = Scripted.reads

    def __init__(self, agent: Scripted, teacher: TeacherConfig) -> None:
        if not (isinstance(agent.scripted_probability, float)
                and agent.scripted_probability == 1.0):
            raise ValueError("the teacher is the scripted agent at scripted probability 1")
        self.agent, self.teacher = agent, teacher

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def action_distribution(self, obs: Obs) -> torch.Tensor:
        """(R, S, 42) the distribution `step` samples from: the labels'."""
        return labels(self.agent.action_distribution(obs), self.teacher)

    def step(self, carry: None, obs: Obs, done: torch.Tensor):
        parts = marginals(obs, self.agent.characteristics, self.agent.parameters)
        temperature, floor = self.teacher.temperature, self.teacher.floor
        if temperature != 1.0:
            parts = [torch.softmax(torch.log(part) / temperature, -1) for part in parts]
        action = torch.stack([sample(part) for part in parts], -1).to(torch.int32)
        if floor != 0.0:
            uniform = torch.rand(action.shape[:-1], device=action.device) < floor
            action = torch.where(uniform[..., None],
                                 uniform_actions(action.shape[:-1], action.device), action)
        return Action.unpack(action), carry, Outputs()


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """In nats, over the last axis (0 log 0 = 0)."""
    return -torch.special.xlogy(probabilities, probabilities).sum(-1)


def flat_views(obs: Obs) -> Obs:
    """Stored observations (L, n, ...) as views (L n, ...)."""
    parts = {field.name: getattr(obs, field.name) for field in dataclasses.fields(obs)}
    return Obs(**{name: None if value is None else value.flatten(0, 1)
                  for name, value in parts.items()})


def teacher_distribution(teacher: Teacher, obs: Obs) -> torch.Tensor:
    """The teacher on stored observations (L, n, ...): (L, n, S, 42)."""
    length, views = obs.ships.shape[:2]
    return teacher(flat_views(obs)).unflatten(0, (length, views))


class BehaviourCloning(Learner):
    """teacher: the teacher's distribution on views; labels: its transformation.
    pools: the pool names, view_pool (V,) each stored view's pool index (the metrics are
    per pool). The other arguments are the Learner's."""

    def __init__(self, model: torch.nn.Module, config: LearnerConfig, rules, types,
                 outcome_weight: float, device: torch.device, teacher: Teacher,
                 teacher_config: TeacherConfig, pools: tuple[str, ...],
                 view_pool: torch.Tensor, stage: Callable = lambda function: function,
                 seed: int = 0) -> None:
        self.teacher, self.teacher_config, self.pools = teacher, teacher_config, pools
        # (KL, teacher entropy, student entropy, tokens) per pool.
        self.STATISTICS = tuple(f"{name}/{pool}" for name in ("kl", "teacher", "student",
                                                               "tokens") for pool in pools)
        self.pool_one_hot = torch.nn.functional.one_hot(view_pool.to(device),
                                                        len(pools)).float()[None]
        super().__init__(model, config, rules, types, outcome_weight, device, stage, seed)

    def prepare(self, segments, bootstraps: list) -> dict:
        prepared = super().prepare(segments, bootstraps)
        prepared["derived"]["pool"] = self.pool_one_hot  # (1, V, pools)
        return prepared

    def _policy(self, unrolled, trajectory: Trajectory, derived: dict, denominator: dict,
                scale: torch.Tensor):
        """(the cross-entropy against the labels, the student's mean entropy, per-pool sums
        of KL(labels || student), the labels' and the student's entropy and the tokens),
        over own ships alive."""
        own_alive = trajectory.alive[..., :trajectory.logp.shape[-1]]
        target = labels(teacher_distribution(self.teacher, trajectory.obs), self.teacher_config)
        logits = unrolled.policy_logits.float()
        cross = losses.cross_entropy(logits, target)
        teacher_entropy = entropy(target)
        # PPO's formula (`bnb.models.yemong.policy.evaluate`): from log-probabilities, so a
        # probability that underflows to 0 gives 0 x a finite number in the gradient, where
        # p log p of the probabilities would give 0 x infinity.
        log_probabilities = torch.log_softmax(logits, -1)
        student_entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
        per_view = torch.stack([torch.where(own_alive, value, 0.0).sum((0, 2))
                                for value in (cross - teacher_entropy, teacher_entropy,
                                              student_entropy, own_alive.float())])
        statistics = (per_view @ derived["pool"][0]).flatten()  # (4 pools,)
        return (losses.masked_sum(cross, own_alive) / denominator["policy"],
                losses.masked_sum(student_entropy, own_alive) / denominator["policy"],
                statistics)

    def report(self, statistics: list[float]) -> dict:
        """The BC KL and both entropies, per living own ship-step: overall and per pool."""
        pools = len(self.pools)
        kl, teacher, student, tokens = (statistics[i * pools:(i + 1) * pools] for i in range(4))
        total = max(sum(tokens), 1e-12)
        out = dict(bc_kl=sum(kl) / total, teacher_entropy=sum(teacher) / total,
                   student_entropy=sum(student) / total)
        for index, pool in enumerate(self.pools):
            count = max(tokens[index], 1e-12)
            out[f"bc_kl/{pool}"] = kl[index] / count
            out[f"teacher_entropy/{pool}"] = teacher[index] / count
            out[f"student_entropy/{pool}"] = student[index] / count
        return out


def mean_entropy(probabilities: torch.Tensor, teacher: TeacherConfig) -> float:
    """The labels' mean entropy over rows of teacher distributions (tokens, 42)."""
    return float(entropy(labels(probabilities, teacher)).mean())


def calibrate(probabilities: torch.Tensor, target: float, floor: float = FLOOR,
              tolerance: float = 1e-3, low: float = 1e-2, high: float = 1e2) -> TeacherConfig:
    """The temperature (floor fixed) whose labels' mean entropy over these teacher
    distributions (tokens, 42), living own ship-steps, is `target` nats, by bisection on
    log T. Fails when the target lies outside what [low, high] reaches (raise the floor
    when it is above what the temperature alone can give)."""
    reach = [mean_entropy(probabilities, TeacherConfig(t, floor)) for t in (low, high)]
    if not reach[0] <= target <= reach[1]:
        raise ValueError(f"target entropy {target} nats is outside [{reach[0]:.4f}, "
                         f"{reach[1]:.4f}] for temperatures [{low}, {high}] at floor {floor}")
    log_low, log_high = torch.log(torch.tensor(low)), torch.log(torch.tensor(high))
    while True:
        middle = float(torch.exp(0.5 * (log_low + log_high)))
        value = mean_entropy(probabilities, TeacherConfig(middle, floor))
        if abs(value - target) <= tolerance or float(log_high - log_low) < 1e-6:
            return TeacherConfig(middle, floor)
        if value < target:
            log_low = torch.log(torch.tensor(middle))
        else:
            log_high = torch.log(torch.tensor(middle))


@dataclasses.dataclass(frozen=True)
class StopConfig:
    """When BC stops (scripted spec B4), whichever comes first: the live score against the
    teacher over the last score_window finished games, less score_errors of its standard
    errors, reaches score_target; the KL's EMA (kl_decay per update) improved by less than
    plateau_fraction over the last plateau_steps environment steps; or budget_steps
    environment steps."""

    score_target: float = 0.45
    score_window: int = 1000
    score_errors: float = 2.0
    plateau_fraction: float = 0.01
    plateau_steps: int = 20_000_000
    kl_decay: float = 0.9
    budget_steps: int = 500_000_000

    def __post_init__(self) -> None:
        if self.score_window < 2:
            raise ValueError(f"the score window needs two games for its spread, got {self}")


class StopRule:
    """Fed once per update; `observe` names the rule that fires, or returns None.

    The score window keeps per-update totals (games, points, squared points; a game scores
    1, 0.5 or 0): it drops the oldest updates whole while the rest still fill it, then
    counts the oldest kept update by the exact share of its games the window still needs,
    so trimming favours no result. The score stop reads the window's lower confidence
    bound, so a window that is lucky by chance does not stop BC."""

    def __init__(self, config: StopConfig) -> None:
        self.config = config
        self.updates: collections.deque[tuple[float, float, float]] = collections.deque()
        self.kl_average: float | None = None
        self.history: list[tuple[int, float]] = []  # (environment steps, the KL's EMA)

    def _window(self) -> tuple[float, float] | None:
        """(points, squared points) over the last score_window games, once there are that
        many."""
        window = self.config.score_window
        games = sum(update[0] for update in self.updates)
        if games < window:
            return None
        oldest_games, oldest_points, oldest_squares = self.updates[0]
        share = (games - window) / oldest_games  # of the oldest update, outside the window
        return (sum(update[1] for update in self.updates) - oldest_points * share,
                sum(update[2] for update in self.updates) - oldest_squares * share)

    @property
    def score(self) -> float | None:
        """Wins plus half the draws over the last score_window games, once there are that
        many."""
        totals = self._window()
        return None if totals is None else totals[0] / self.config.score_window

    @property
    def standard_error(self) -> float | None:
        """The window score's standard error, from the spread of its own per-game scores
        (the sample variance over score_window - 1)."""
        totals = self._window()
        if totals is None:
            return None
        window = self.config.score_window
        points, squares = totals
        variance = max(squares - points * points / window, 0.0) / (window - 1)
        return math.sqrt(variance / window)

    @property
    def lower_bound(self) -> float | None:
        """The window score less score_errors standard errors: what the score stop reads."""
        score = self.score
        return None if score is None else \
            score - self.config.score_errors * self.standard_error

    def observe(self, env_steps: int, wins: int, draws: int, losses: int,
                kl: float) -> str | None:
        """env_steps: the total so far; wins, draws, losses: games finished against the
        teacher this update; kl: this update's BC KL."""
        config = self.config
        games = wins + draws + losses
        if games > 0:
            self.updates.append((float(games), wins + 0.5 * draws, wins + 0.25 * draws))
        while (len(self.updates) > 1
               and sum(update[0] for update in self.updates) - self.updates[0][0]
               >= config.score_window):
            self.updates.popleft()
        decay = config.kl_decay
        self.kl_average = kl if self.kl_average is None else \
            decay * self.kl_average + (1.0 - decay) * kl
        self.history.append((env_steps, self.kl_average))
        if self.lower_bound is not None and self.lower_bound >= config.score_target:
            return "score"
        past = [average for steps, average in self.history
                if steps <= env_steps - config.plateau_steps]
        if past and self.kl_average > (1.0 - config.plateau_fraction) * past[-1]:
            return "kl_plateau"
        if env_steps >= config.budget_steps:
            return "budget"
        return None
