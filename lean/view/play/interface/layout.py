"""Where every card sits: rectangles in pixels for a screen size, an interface scale and
which cards are open. The cards are drawn inside these rectangles (at most this large),
so everything outside them is battlefield; a test renders with and without the interface
and checks those pixels are unchanged.

    top bar           the full width, one row
    fleets (left)     under the top bar, down to the bottom (or the scrubber)
    selection (right) under the top bar, down to the minimap
    kill feed         upper right, under the top bar, left of an open selection card
    minimap           lower right
    scrubber          the bottom strip, in replays
    flying HUD        bottom centre, while the player flies a ship
    notice            a passing message, centred under the top bar

A collapsed card leaves a slim tab on its edge.
"""

import dataclasses

TOP_BAR = 44.0  # height, pixels at scale 1
MARGIN = 8.0
FLEETS_WIDTH = 300.0
SELECTION_WIDTH = 290.0
MINIMAP = 220.0  # the map's side; the card adds its padding
MINIMAP_PADDING = 6.0
SCRUBBER = 64.0
KILL_FEED_WIDTH = 270.0
KILL_FEED_HEIGHT = 130.0
TAB = (96.0, 30.0)  # a collapsed card's tab
FLYING_HUD = (500.0, 78.0)
NOTICE = (900.0, 40.0)  # the area of a passing notice, under the top bar
CARDS = ("fleets", "selection", "minimap")


@dataclasses.dataclass(frozen=True)
class Rect:
    """A rectangle in pixels, y down: its top left corner and its size."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """The x of the right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """The y of the bottom edge."""
        return self.y + self.height

    def contains(self, px: float, py: float) -> bool:
        """Whether the pixel (px, py) is inside."""
        return self.x <= px < self.right and self.y <= py < self.bottom

    def centre(self) -> tuple[float, float]:
        """The middle point."""
        return self.x + 0.5 * self.width, self.y + 0.5 * self.height


@dataclasses.dataclass(frozen=True)
class Layout:
    """The rectangles of one frame's interface. open: which cards are open (the others
    show their tab); replay: the scrubber shows; flying: the flying HUD shows; selected:
    the selection card has something to show; fleets_width: the fleets card's width at
    scale 1 (wider than FLEETS_WIDTH when long ship names need it)."""

    width: float
    height: float
    scale: float = 1.0
    open: frozenset = frozenset(CARDS)
    replay: bool = False
    flying: bool = False
    selected: bool = True
    fleets_width: float = FLEETS_WIDTH

    def _size(self, value: float) -> float:
        return value * self.scale

    @property
    def top_bar(self) -> Rect:
        """The top bar: the full width."""
        return Rect(0.0, 0.0, self.width, self._size(TOP_BAR))

    @property
    def scrubber(self) -> Rect | None:
        """The bottom strip in replays."""
        if not self.replay:
            return None
        height = self._size(SCRUBBER)
        return Rect(0.0, self.height - height, self.width, height)

    @property
    def _bottom(self) -> float:
        """The lowest y the side cards reach."""
        scrubber = self.scrubber
        return (self.height if scrubber is None else scrubber.y) - self._size(MARGIN)

    @property
    def _top(self) -> float:
        return self.top_bar.bottom + self._size(MARGIN)

    def _tab(self, left: bool, y: float) -> Rect:
        width, height = self._size(TAB[0]), self._size(TAB[1])
        return Rect(0.0 if left else self.width - width, y, width, height)

    @property
    def fleets(self) -> Rect:
        """The fleets card (left), or its tab."""
        if "fleets" not in self.open:
            return self._tab(True, self._top)
        margin = self._size(MARGIN)
        return Rect(margin, self._top, self._size(self.fleets_width), self._bottom - self._top)

    @property
    def minimap(self) -> Rect:
        """The minimap card (lower right), or its tab."""
        side = self._size(MINIMAP + 2 * MINIMAP_PADDING)
        if "minimap" not in self.open:
            return self._tab(False, self._bottom - self._size(TAB[1]))
        margin = self._size(MARGIN)
        return Rect(self.width - margin - side, self._bottom - side, side, side)

    @property
    def minimap_map(self) -> Rect:
        """The map's square inside the minimap card."""
        card, padding = self.minimap, self._size(MINIMAP_PADDING)
        return Rect(
            card.x + padding, card.y + padding, card.width - 2 * padding, card.height - 2 * padding
        )

    @property
    def selection(self) -> Rect | None:
        """The selection card (right), its tab, or None when nothing is selected."""
        if not self.selected:
            return None
        if "selection" not in self.open:
            return self._tab(False, self._top)
        margin = self._size(MARGIN)
        width = self._size(SELECTION_WIDTH)
        bottom = self.minimap.y - margin
        return Rect(self.width - margin - width, self._top, width, bottom - self._top)

    @property
    def kill_feed(self) -> Rect:
        """The kill feed's area: upper right, left of an open selection card."""
        margin = self._size(MARGIN)
        right = self.width - margin
        selection = self.selection
        if selection is not None and "selection" in self.open:
            right = selection.x - margin
        width = self._size(KILL_FEED_WIDTH)
        return Rect(right - width, self._top, width, self._size(KILL_FEED_HEIGHT))

    @property
    def flying_hud(self) -> Rect | None:
        """The flying HUD's area (bottom centre)."""
        if not self.flying:
            return None
        width, height = self._size(FLYING_HUD[0]), self._size(FLYING_HUD[1])
        return Rect(0.5 * (self.width - width), self._bottom - height, width, height)

    @property
    def notice(self) -> Rect:
        """Where a passing notice shows: centred under the top bar."""
        width = min(self._size(NOTICE[0]), self.width - 2 * self._size(self.fleets_width))
        top = self.top_bar.bottom + self._size(MARGIN)
        return Rect(0.5 * (self.width - width), top, width, self._size(NOTICE[1]))

    def rects(self) -> list[Rect]:
        """Every rectangle the interface may draw in this frame (menus aside)."""
        found = [self.top_bar, self.fleets, self.minimap, self.kill_feed, self.notice]
        found += [rect for rect in (self.selection, self.scrubber, self.flying_hud) if rect]
        return found

    def covers(self, px: float, py: float) -> bool:
        """Whether the pixel is under any of the interface's rectangles."""
        return any(rect.contains(px, py) for rect in self.rects())
