"""The settings of the decision suite, `bnb evaluate` (`bnb.league.evaluation`): its
conditions and its settings, kept free of torch so the command line starts without it."""

import dataclasses
from typing import Literal

MODES = ("frontline", "deathmatch", "elimination")
HEAD_TO_HEAD, BREAK_EVEN = "head_to_head", "break_even"
# The map rules (`bnb.league.crossover.sized_game`): "preset", every game on its preset's
# own arena radius at every team size (the default: the author's rule, 2026-11-15, every map
# fixed-size until an agent has trained successfully on variable-size maps); "density", a
# size other than the preset's on the density rule's radius (`bnb.game.api.arena_radius_for`),
# for the map-size arm.
MAP_RULES = ("preset", "density")


@dataclasses.dataclass(frozen=True)
class Condition:
    """One condition: its name, mode, kind (HEAD_TO_HEAD or BREAK_EVEN) and ships per team
    (a head-to-head's team size; None for break-even, whose game is the larger of
    `EvaluationConfig.opponent_count` and `maximum_count` a side)."""

    name: str
    mode: str
    kind: str
    ships_per_team: int | None = None


def _head_to_head(mode: str, size: int) -> Condition:
    return Condition(f"{mode}_{size}v{size}", mode, HEAD_TO_HEAD, size)


CONDITIONS = (
    _head_to_head("frontline", 5),
    _head_to_head("deathmatch", 5),
    _head_to_head("elimination", 5),
    _head_to_head("deathmatch", 1),
    _head_to_head("deathmatch", 2),
    _head_to_head("elimination", 1),
    _head_to_head("elimination", 2),
    _head_to_head("frontline", 10),
    _head_to_head("deathmatch", 10),
    _head_to_head("elimination", 10),
    Condition("frontline_break_even", "frontline", BREAK_EVEN),
    Condition("deathmatch_break_even", "deathmatch", BREAK_EVEN),
    Condition("elimination_break_even", "elimination", BREAK_EVEN),
)
BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """The suite's settings, recorded in settings.json.

    conditions: the scored conditions' names; presets: each mode's base preset
    (5v5); map_rule: one of `MAP_RULES` ("preset": every condition on its preset's own
    arena radius at every size; "density": the density rule's radius at other sizes, for
    the map-size arm); lineup_types: the ship types' names a slot is drawn from;
    opponent and opponent_count: break-even's opponent and its ships; minimum_count and
    maximum_count: the players' counts there (up to 30, above the opponent's 20, so a
    player weaker than the opponent at parity still crosses inside the range); bracket: its
    first batch's counts;
    head_to_head_blocks and head_to_head_environments, break_even_blocks and
    break_even_environments: the engines' layouts (a batch is blocks x environments games;
    break-even's blocks split between A's and B's curves); seed: the games' keys;
    initial_batches: each condition's batches before the allocation; initial_batch_limit:
    break-even's most batches waiting for both fits to cross; looks, alpha, beta, delta:
    the sequential design (`bnb.league.sequential.SequentialDesign`); budget_hours: the
    most engine hours (every process's batches' seconds), after which the next look is
    the last; gate_stderrs: the gate's standard errors."""

    conditions: tuple[str, ...] = tuple(condition.name for condition in CONDITIONS)
    presets: tuple[str, ...] = (
        "real_scale_frontline",
        "real_scale_deathmatch",
        "real_scale_elimination",
    )
    map_rule: Literal["preset", "density"] = "preset"
    lineup_types: tuple[str, ...] = ("Yongma", "Jujak")
    opponent: str = "imugi"
    opponent_count: int = 20
    minimum_count: int = 1
    maximum_count: int = 30
    bracket: tuple[int, ...] = (6, 10, 14, 20)
    head_to_head_blocks: int = 8
    head_to_head_environments: int = 64
    break_even_blocks: int = 8
    break_even_environments: int = 32
    seed: int = 0
    initial_batches: int = 1
    initial_batch_limit: int = 8
    looks: int = 4
    alpha: float = 0.05
    beta: float = 0.2
    delta: float = 0.02
    budget_hours: float = 24.0
    gate_stderrs: float = 2.0

    def __post_init__(self) -> None:
        unknown = [name for name in self.conditions if name not in BY_NAME]
        if unknown:
            raise ValueError(f"unknown conditions {unknown}; choose from {list(BY_NAME)}")
        if len(self.presets) != len(MODES):
            raise ValueError(f"presets {self.presets}: one per mode, {MODES}")
        if self.map_rule not in MAP_RULES:
            raise ValueError(f"map rule {self.map_rule!r}: one of {MAP_RULES}")
        if self.break_even_blocks < 2 or self.break_even_blocks % 2:
            raise ValueError(f"{self.break_even_blocks} break-even blocks: even, A's and B's")
        if self.initial_batches < 1 or self.budget_hours <= 0.0:
            raise ValueError("at least one initial batch and a positive budget")
        self.design()  # checks the design's numbers

    def design(self):
        """The sequential design's settings (`bnb.league.sequential.SequentialDesign`)."""
        from .sequential import SequentialDesign

        return SequentialDesign(self.looks, self.alpha, self.beta, self.delta)

    def scored(self) -> list[Condition]:
        """The scored conditions, in `CONDITIONS`' order."""
        return [condition for condition in CONDITIONS if condition.name in self.conditions]

    def crossover_config(self):
        """Break-even's measurement settings, its fit's and allocation's
        (`bnb.league.crossover.CrossoverConfig`)."""
        from .crossover import CrossoverConfig

        return CrossoverConfig(
            opponent_count=self.opponent_count,
            minimum_count=self.minimum_count,
            maximum_count=self.maximum_count,
            bracket=self.bracket,
        )

    def layout(self, condition: Condition):
        """A condition's engine layout (`bnb.league.matches.ArenaConfig`)."""
        from .matches import ArenaConfig

        if condition.kind == BREAK_EVEN:
            blocks, environments = self.break_even_blocks, self.break_even_environments
        else:
            blocks, environments = self.head_to_head_blocks, self.head_to_head_environments
        return ArenaConfig(blocks=blocks, block_environments=environments, seed=self.seed)
