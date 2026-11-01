---
covers: [src/bnb/game/handling/, src/bnb/cli.py]
---

# Designing a ship

A ship type is designed by its handles: what it should do in flight and in combat.
`bnb.game.handling.solve` finds the card that does it, and `bnb ship-card` charts the
result. [The ship card](../explanation/game/ship-card.md) explains the handles and the
solver.

## 1. Chart the current card

The charts need matplotlib, in the optional `tools`{.plain} extra:

<!-- not-runnable: changes the environment and takes about a minute -->
```bash
uv sync --extra tools
uv run --no-sync bnb ship-card
```

The command prints every handle, every card field and the shown quantities per type, then
writes `ps_curve.png`{.plain}, `doghouse.png`{.plain} and `similarity.png`{.plain} into the directory given by its
`--out`{.plain} option (`ship-card` by default). `--card`{.plain} names a JSON file of card
fields to chart instead of the default card (see step 4).

## 2. Read the default card's handles

<!-- not-runnable: flies for about half a minute -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling import measure

rules = api.rules_of(GameConfig(ships_per_team=1, num_fields=1, arena_radius=1000.0))
handles = measure(api.ship_types(dtype=torch.float64), rules)
print(handles.sustained_turn_rate, handles.bleed_rate)
```

Every field of `bnb.game.handling.Handles` has one value per type (fighter, heavy). Angles
are in radians, times in seconds.

## 3. Change handles and solve

Change the handles the design calls for and keep the others. Every handle not changed is
held: the solver moves whatever parameters it must.

<!-- not-runnable: solves for several minutes on the CPU -->
```python
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling import measure, solve

rules = api.rules_of(GameConfig(ships_per_team=1, num_fields=1, arena_radius=1000.0))
handles = measure(api.ship_types(dtype=torch.float64), rules)
faster = handles.replace(cruise_speed=handles.cruise_speed * torch.tensor([1.1, 1.0]))
solution = solve(faster, rules)
print(solution.error.amax(-1), solution.condition)
types = solution.types  # a float32 card for the game
```

- `Solution.error` is each flown handle's remaining relative error. A solve succeeds when
  every one is within the solver's `ACCEPTANCE`, the spec's bar; it stops early once all are
  within its much smaller `TOLERANCE`. `Solution.condition` is the conditioning of each
  type's solve.
- A handle set that no card meets raises `bnb.game.handling.Infeasible` with the reason. A
  batch of cards (handles shaped `(cards, types)`) can be solved with `strict=False`, which
  returns the failures in `Solution.failures` instead.
- A `bnb.game.handling.RedundantHandles` warning names two handles that pull in nearly the
  same direction. A low-speed acceleration beyond what the engine gives at the stall speed
  is one way to get it: the static-thrust cap no longer acts, so the low-speed acceleration
  moves only with the engine power, which the cruise sets.
- The zero-lift drag is a handle the solver holds, not one it solves for: under the cruise
  fix, the cruise is met at any drag by the power that pays for it, so the cruise cannot
  set the drag.

## 4. Save the card and chart it

<!-- not-runnable: continues the example above -->
```python
from bnb.game.handling.chart import save_card

save_card(types, "my-card.json")
```

Then chart it with `bnb ship-card` and the `--card my-card.json`{.plain} option. Check the
printed handles against the design, and the cruise speeds and endurance in the shown
quantities.

## 5. Check it at every tick rate

The fire rate and the recharge delay must be whole numbers of ticks at the rules' tick:
solving with `api.rules_of(..., tick_length=1.0 / 25)` refuses the fighter's 15 Hz fire
rate, and so does the game's reset. Flight
handles agree across 25, 30 and 60 Hz; `tests/sections/test_ship_card.py` checks this for
the default and varied cards.
