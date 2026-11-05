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
TEACHER_CARRY = "teacher_carry/"  # the derived names of the teacher's carry's fields
# the carry's field names, taken once: torch.compile cannot trace dataclasses.fields
CARRY_FIELDS = tuple(field.name for field in dataclasses.fields(ScriptedCarry))


MINIBATCHES = 256


def config(**overrides) -> LearnerConfig:
    """BC's learner settings (scripted spec B1): the BC coefficient 0.025 in nats per token
    in place of the surrogate's, no entropy term, no KL early stop, and 256 minibatches; the
    rest section 6's. Cloning's progress follows its optimizer steps, so at the 8,192-view
    batch (2,048 x 4) it takes minibatches of 32 views, not PPO's 32 per epoch: 8x the
    steps per environment step for the same compute."""
    return dataclasses.replace(
        LearnerConfig(),
        policy=0.025,
        entropy_power=0.0,
        entropy_stick=0.0,
        entropy_shoot=0.0,
        target_kl=None,
        **{"minibatches": MINIBATCHES, **overrides},
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
    ) -> None:
        self.teacher, self.teacher_config, self.pools = teacher, teacher_config, pools
        self.teacher_carry: ScriptedCarry | None = None  # (R, ...) after the last update
        # (KL, teacher entropy, student entropy, tokens, agreements) per pool.
        self.STATISTICS = tuple(f"{name}/{pool}" for name in SUMS for pool in pools)
        self.pool_one_hot = torch.nn.functional.one_hot(view_pool.to(device), len(pools)).float()[
            None
        ]
        super().__init__(
            model, config, rules, types, outcome_weight, device, stage, seed, diagnostics
        )

    def prepare(self, segments, bootstrap) -> dict:
        """The Learner's prepare, with each view's pool one-hot (1, V, pools) added to derived for
        the per-pool metrics."""
        prepared = super().prepare(segments, bootstrap)
        prepared["derived"]["pool"] = self.pool_one_hot  # (1, V, pools)
        prepared["derived"].update(self._teacher_carries(segments))
        return prepared

    def _teacher_carries(self, segments) -> dict:
        """The teacher's carry at every stored tick, each field (L, V, ...) under
        TEACHER_CARRY + its name: the shards in time order, from the last update's carry."""
        rows = segments.views // segments.shards
        carry = self.teacher_carry
        if carry is None:
            carry = self.teacher.init_carry(rows, self.device)
        shards = []
        for shard in range(segments.shards):
            obs = segments.shard(("obs",), shard)["obs"]
            carries, carry = teacher_carries(self.teacher, obs, carry)
            shards.append(carries)
        self.teacher_carry = carry
        return {
            TEACHER_CARRY + name: torch.cat([getattr(c, name) for c in shards], 1)
            for name in CARRY_FIELDS
        }

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
        carries = ScriptedCarry(**{name: derived[TEACHER_CARRY + name] for name in CARRY_FIELDS})
        target = labels(
            teacher_distribution(self.teacher, trajectory.obs, carries), self.teacher_config.floor
        )
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
    """When BC stops (scripted spec B4), whichever comes first: the score against the raw
    scripted agent in the rating lanes, over the last score_window finished games, of the
    better of the players `reads` names (the averages of the live weights; decisions: BC
    round three), reaches score_target; the KL's EMA (kl_decay per update) improved by less
    than plateau_fraction over the last plateau_steps environment steps; or budget_steps
    environment steps (a run stopped by its budget resumes with a larger one). `reads` is
    in order of preference: an equal score goes to the earlier."""

    # parity with the scripted agent (author, 2026-11-04), a plain score
    score_target: float = 0.5
    score_window: int = 1000
    plateau_fraction: float = 0.01
    plateau_steps: int = 20_000_000
    kl_decay: float = 0.9
    # v1's time to parity (author, 2026-11-04)
    budget_steps: int = 50_000_000
    # the lane players `bnb.runtime.lanes.EXPONENTIAL_AVERAGE` and `UNIFORM_AVERAGE`
    reads: tuple[str, ...] = ("exponential_average", "uniform_average")

    def __post_init__(self) -> None:
        if self.score_window < 2:
            raise ValueError(f"the score window needs two games for its spread, got {self}")
        if not self.reads or len(set(self.reads)) != len(self.reads):
            raise ValueError(f"the score stop reads distinct players, got {self.reads}")


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

    Each lane player's games against the scripted agent fill a score window of its own
    (`ScoreWindow`, `windows` by player). The score stop reads the window of the best of
    the players `StopConfig.reads` names whose window is full (`read`; an equal score goes
    to the earlier in `reads`); the others' windows, the live clone's among them, are
    logged.

    Its state (`state`: the windows, the KL's EMA and its history) is plain JSON, kept in
    the training checkpoint so that a resume continues the same windows."""

    def __init__(self, config: StopConfig, state: dict | None = None) -> None:
        self.config = config
        self.windows: dict[str, ScoreWindow] = {}
        self.kl_average: float | None = None
        self.history: list[tuple[int, float]] = []  # (environment steps, the KL's EMA)
        if state is not None:
            self.windows = {
                player: ScoreWindow(config.score_window, updates)
                for player, updates in state["windows"].items()
            }
            self.kl_average = state["kl_average"]
            self.history = [(steps, average) for steps, average in state["history"]]

    @property
    def state(self) -> dict:
        """The rule's state as JSON: each player's window, the KL's EMA and its history
        (floats round-trip exactly through JSON)."""
        return dict(
            windows={player: window.state for player, window in self.windows.items()},
            kl_average=self.kl_average,
            history=[list(entry) for entry in self.history],
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
        player left out played none that count); kl: this update's BC KL."""
        config = self.config
        for player, games in results.items():
            self.windows.setdefault(player, ScoreWindow(config.score_window)).add(*games)
        decay = config.kl_decay
        self.kl_average = (
            kl if self.kl_average is None else decay * self.kl_average + (1.0 - decay) * kl
        )
        self.history.append((environment_steps, self.kl_average))
        if self.score is not None and self.score >= config.score_target:
            return "score"
        cutoff = environment_steps - config.plateau_steps
        past = [average for steps, average in self.history if steps <= cutoff]
        # the plateau reads the newest average at or before the cutoff, which only rises:
        # the older ones are never read again
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.pop(0)
        if past and self.kl_average > (1.0 - config.plateau_fraction) * past[-1]:
            return "kl_plateau"
        if environment_steps >= config.budget_steps:
            return "budget"
        return None
