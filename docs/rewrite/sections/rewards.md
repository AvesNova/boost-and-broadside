# Section 3b: rewards (events to reward levels, learning side)

Status: decisions settled, 2026-11-01. Spec next.

## Decided (author, 2026-11-01)

- **Starting point:** v1's reward math (zero-sum levels, ally blame, kill shares,
  charge-back) in new code, with the changes below.

### Levels

Five levels, from the least to the most directly tied to winning. Each has its own critic
head, discount and GAE lambda:

| level | what it pays |
|---|---|
| `damage` | damage taken and dealt, with the charge-back on recharge |
| `kill_death` | deaths, and the kills credited for them |
| `objective_progress` | progress on the mode's main objective: meter movement on a zone, damage to a fortress or capital ship, ... |
| `objective` | completing a step of the objective: a capture, a destroyed point, a destroyed capital ship, ... |
| `outcome` | the match result, ±1 or 0 |

- **The two objective levels are generic.** A mode decides what counts and emits it as
  events with per-ship attribution: who earned it, and whom it is charged to. Rewards
  price and route those events zero-sum like every other level. This keeps the seam: the
  game emits events, never rewards.
- **Every mode has the outcome, kill/death and damage levels.** Team deathmatch has no
  objective, so its objective levels are always zero.

### Prices

- **Damage is damage:** priced per point, in a fixed unit of 100 points. Emptying a
  160-point heavy costs 1.6 units. Type differences come from play, not from prices.
- **A kill is a kill:** one death price for every ship type. Ship types are meant to be
  balanced (Overwatch-style), never better or worse.
- **Every level is zero-sum by construction.** Payout ratios and the offensive-bias
  schedule are dropped (v1 kept the mechanism but ran it at 1:1). The tests assert
  zero-sum unconditionally.
- **Starting values:** v1's weights (win 1, death 0.283, a full shield of damage 0.274,
  the objective levels 2 each) and per-level γ/λ (outcome 0.9997/0.97, objective
  0.999/0.97, objective progress 0.995/0.95, kill/death 0.995/0.95, damage 0.991/0.90).

### Attribution

- **The game emits the applied shield loss and the shield restored,** per ship per tick
  (a small change to the game's `Events`).
  - The game computes both exactly inside the tick, so `Events` becomes a complete record
    of what happened in it.
  - Rewards then never re-derive the shield rules: overkill ignored, respawn to full,
    resets, dead ships holding their shield, and later per-type recharge or pickups.
  - Applied loss is split over the tick's hits and its boundary damage in proportion to
    their raw amounts.
- **The ledger:** for each (attacker, target) pair, the damage done that has not healed
  yet. It is reward state, carried per environment.
  - It has a **boundary pseudo-attacker** row, so the ledger covers the whole shield
    deficit. Recharge is attributed pro rata over the ledger alone, with no per-type
    maximum shield needed (the same math as v1's "unattributed recharge").
  - **Kill shares:** half by the final tick's raw hits, half by outstanding damage.
  - **Charge-back:** recharge reverses, pro rata, the damage payment the healed damage
    earned. Damage that heals did not change the game, so a duel that ends with the
    target fully healed nets the attacker zero, never less.
- **On death,** the dead ship's ledger **column** (damage done to it) is cleared after its
  kill shares are paid. Its death made that damage stick, so respawning to full shield
  triggers no charge-back, and a later death does not credit attackers from a previous
  life.
- **The dead ship's row (damage it did to others) is kept (v2, a change from v1).** v1
  cleared it at respawn. Keeping it means a ship keeps its claims across its own death
  and respawn:
  - its share of a later kill of a ship it damaged;
  - its charge-back when that ship heals.
- **A reset clears the ledger.**

### Where it runs

- A pure function from (ledger, events, mode result) to (rewards per level per ship, new
  ledger), called in the rollout tick. The pairwise damage is a per-tick transient, so
  this cannot wait for the learner.
- Rewards are per sim ship, routed to views as the skeleton's stub does.

## Related (modes section)

- **Match length is a mode parameter.** Train team deathmatch at about 5 minutes (v1
  trained at about 290 s). The later target is about 10 minutes, with Halo and Overwatch
  pacing: an encounter every 10–30 s, which suits the kill and damage discounts as they
  are. The outcome discount is revisited once measured. The skeleton's 60 s is a
  placeholder.
- **The mode supplies each environment's result** on its final tick. The win rule belongs
  to the modes section.

## Acceptance

- **Equivalence:** fixtures from v1's own reward components on synthetic states, with the
  v2 changes stated.
- **Zero-sum:** every level sums to zero on every tick, with friendly fire, boundary deaths
  and respawns in play.
- **Ledger conservation:** outstanding damage (the boundary row included) equals each
  living ship's shield deficit.
- **A mutation check** on the tests.
