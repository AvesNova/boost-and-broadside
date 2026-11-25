"""A match to watch: one lean game at batch size 1, every ship the default scripted agent
(ScriptedConfig(), the yardstick), both teams in one call as in evaluation, stepped tick by
tick into snapshots. The same loop as docs/rewrite/sections/replay/replay.py.

Setup (`MatchSettings`) changes this process's game only: the vision range and eccentricity
set the process's card (lean.game.card.CARD) before the first deal, as the replay script
does; nothing on disk changes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import torch

from lean import game, scripted
from lean.game import card, core

from .index_map import IndexGrid, build
from .snapshot import Fields, Snapshot, fields_of, snapshot


@dataclasses.dataclass(frozen=True)
class MatchSettings:
    """One match: ships per team, seed, fields on or off, the tracking circle on or off, and
    the vision range (m, the sight ellipse's side reach) and eccentricity for every type (None:
    the card's)."""

    ships_per_team: tuple[int, int] = (4, 4)
    seed: int = 0
    fields: bool = True
    tracking: bool = True
    vision_range: float | None = None
    vision_eccentricity: float | None = None

    def game_config(self) -> core.GameConfig:
        """The core game (GameConfig()) with this match's switches."""
        config = core.GameConfig()
        if not self.fields:
            config = dataclasses.replace(config, field_count=0)
        if not self.tracking:
            config = dataclasses.replace(config, fog_tracking=False)
        return config

    def apply_card(self) -> None:
        """Set this process's card for the vision range and eccentricity (before the first
        deal builds the ship card)."""
        rows = card.CARD
        if self.vision_range is not None:
            scale = self.vision_range / card.VISION_RANGE
            rows = {**rows, "vision_range_scale": tuple(scale for _ in rows["vision_range_scale"])}
        if self.vision_eccentricity is not None:
            if not 0.0 <= self.vision_eccentricity < 1.0:
                raise ValueError(f"vision eccentricity {self.vision_eccentricity}: 0 <= e < 1")
            rows = {
                **rows,
                "vision_eccentricity": tuple(
                    self.vision_eccentricity for _ in rows["vision_eccentricity"]
                ),
            }
        card.CARD = rows


class Match:
    """A scripted-against-scripted game on the CPU. `snapshots()` yields the deal's snapshot,
    then one per tick to the end (the last carries the result); `index` is the deal's index
    texture for the fields."""

    def __init__(self, settings: MatchSettings) -> None:
        settings.apply_card()
        self.settings = settings
        self.config = settings.game_config()
        self.generator = torch.Generator().manual_seed(settings.seed)
        sizes = settings.ships_per_team
        self.slots = sum(sizes)
        self.state = game.deal(
            self.config, 1, self.slots, torch.tensor([list(sizes)]), self.generator, "cpu"
        )
        self.memory = scripted.fresh_memory(1, self.slots, self.generator, "cpu")
        self.agent = scripted.ScriptedConfig()
        self.fields: Fields = fields_of(self.state)
        self.index: IndexGrid = build(self.fields, self.config.arena_radius)

    @torch.no_grad()
    def snapshots(self) -> Iterator[Snapshot]:
        """The deal, then every tick until the game ends."""
        config, state = self.config, self.state
        yield snapshot(config, state, fields=self.fields)
        while True:
            uniforms = scripted.draw_uniforms(1, self.slots, self.generator, "cpu")
            action, self.memory, _ = scripted.decide(
                self.agent, state, None, None, uniforms, self.memory
            )
            previous = state
            state, out = game.tick(config, state, action, self.generator)
            self.state = state
            yield snapshot(config, state, step_out=out, previous=previous, fields=self.fields)
            if bool(out.done[0]):
                return


def record(settings: MatchSettings) -> tuple[list[Snapshot], IndexGrid]:
    """A whole match's snapshots and its index texture."""
    match = Match(settings)
    return list(match.snapshots()), match.index
