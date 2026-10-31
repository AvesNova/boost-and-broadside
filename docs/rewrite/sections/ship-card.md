# Section 1b: ship card (handles to physics parameters)

Status: design concluded, 2026-10-31. Implementation follows the game section
(it needs the per-type parameter tensors).

## Purpose

Designers set **handles**, the quantities a fighter's designers would specify. A solver
computes the physics parameters that make the simulated ship meet them.

- **Independence is the requirement.** The handles are coordinates: changing one
  re-solves with every other handle held at its value, and the physics parameters move
  however the math requires. A whole new set of handles is the same solve. The number of
  handles equals the number of free parameters (13 per type: `C_max`, `a_max`, `C_D0`, the charge rate `a_0` (P follows from the cruise fix), `F_s`, `E_max`, `k`, `e`, `omega_c`, `a_dive`, `a_climb`, `alpha_s`, vision range), and the solver checks the conditioning:
  nearly redundant handles give a warning.
- **Infeasible handle sets fail with a reason** (for example a g limit below 1), never a
  silent nearest fit.
- **Accuracy target 0.5%, soft:** a technique that is twice as fast at 1% may be worth
  it. Integration accuracy matters more (energy, tick-rate robustness).

## The lens: energy-maneuverability

The store is effectively altitude:
- boost is a dive and reverse is a climb;
- the engine charging the store is climbing;
- engine power falling with a fuller store is the real loss above an engine's rated
  altitude.

So Boyd's E-M metrics (specific excess power, sustained and instantaneous turn) carry
over almost directly, and the handles are the specifications a WW2 or F-16 designer would
recognise.

## Flight handles (13, per ship type)

| # | Handle | Real-world analogue |
|---|---|---|
| 1 | stall speed | 1 g stall speed |
| 2 | g limit | structural load factor; corner speed = stall speed x sqrt(g limit), **shown** |
| 3 | max level speed (cruise: equilibrium with no input) | max level speed |
| 4 | climb: time to fill the store while coasting | time to climb |
| 5 | ceiling: store size | service ceiling |
| 6 | low-speed acceleration | acceleration, thrust at low speed |
| 7 | sustained turn rate | sustained turn (Ps = 0 at load) |
| 8 | **bleed rate**: Ps at corner speed in a max-g turn | energy bleed in a hard turn (E-M) |
| 8b | roll: nose reversal time | roll rate |
| 9 | dive acceleration (boost strength) | dive |
| 10 | zoom: reverse/brake strength | zoom climb |
| 11 | AoA limit (stall angle; sets the gun offset in turns) | angle-of-attack limit |
| 12 | vision range | pilot visibility, sensors |

Notes:
- Stall speed and minimum radius are not independent (both from `C_max`: radius about
  u_s^2 / g), so the radius is shown, not set.
- **The AoA limit** changes no equations: `alpha_s` becomes per type.
  - It sets how far the gun leads the flight path at each stick level: 3.4/7.5/15 deg
    today, 5.6/12.5/25 deg at 25 deg.
  - The thrust and nose-travel side effects are absorbed by the solver, because the
    sustained-turn and roll handles stay pinned.
- **Vision range** becomes per type (the observer's type sets its range). The game agent
  was told to keep the sight test compatible.
- **Bleed rate (author, 2026-10-31: "turns hard but bleeds fast" vs "turns gently but
  keeps energy").** Induced drag becomes `k |C|^e` with a per-type **bleed exponent**
  `e` (2 today), or `k1 C^2 + k2 C^4` if cheaper.
  - The sustained turn and the bleed rate pin `k` and `e` independently. A high `e`
    makes hard turns very expensive and gentle ones cheap.
  - This is a small physics change (one formula, one per-type value), made in this
    section after the game section merges.
- **AoA limit and nose reversal time** are the two "nose" dials: crisp and wide vs slow
  and narrow.
- **Snapshots come from the three nose and energy handles (author, 2026-10-31).** A
  quick nose (reversal time) with a wide AoA limit gets most of a large nose swing from
  the angle of attack within a few tenths of a second, before the flight path has turned
  much. It pays the bleed rate for only that short burst of maximum lift. A narrow-AoA
  ship must turn its path instead, which is slower and bleeds for longer.
  - No post-stall mechanic is needed.
  - Above corner speed, the g limit shrinks the usable AoA (roughly as (corner/speed)^2),
    so wide deflection is a near-corner ability.
  - Nose damping `zeta` stays shared; it is a possible later "crispness" handle.

## Combat handles (per ship type)

- Damage and fire interval (DPS shown), muzzle speed and spread.
- Range, which sets the bullet drag, with a shared 0.4 s life.
- Shots from a full store, which sets the energy per shot.
- Shield capacity, recharge rate and delay.
- Head-on fraction.
- Hit radius.

## Physics change: cruise does not depend on the store (author, 2026-10-31)

Cruise is the steady state with no input, at the standard field index, and must not vary
with the store.
- **Drive power is constant:** the null power, equal to drag x speed at cruise.
- **Only charging tapers:** surplus power fills the store, flat and then falling as it
  fills.
- **At a full store** the engine delivers just the null power.

In the current model this is exact with two constraints:
- `P_min` = null power;
- `P` = null power + the charging power at cruise, i.e. `m a_0 u_cruise`.

The solver imposes them; full-store speed is no longer a separate quantity. The
low-speed thrust cap shaves about 2% off thrust at cruise. Either the solver includes
it, or the spec moves the cap out of the drive term (decided when the ship card is
specified).

## How the solve works

1. A closed-form first guess in log space (most handles are near products of powers of
   the parameters), as v1's `config/handling.py` did.
2. A correction against the **simulated** model: fly standard manoeuvres in the real game
   code, measure the handles, and Newton-iterate to the target. The sim is batched, so
   every candidate flies at once, one per environment, through the per-type tensors.
3. Fail with a reason on infeasible or ill-conditioned sets.

## Acceptance

- Round trip: handles -> parameters -> measured handles, within the soft 0.5%.
- Independence: perturb each handle and every other measured handle stays put.
- **Tick-rate robustness:** the same parameters give handles that agree within 0.5% at
  25, 30 and 60 Hz, and energy conservation holds at each rate. Time-native rules must be
  whole ticks at every supported rate. Fire interval 1/15 s is 2 ticks at 30 Hz but not
  whole at 25 Hz, so weapon cadence is validated per rate.

## The reference chart (tool)

A command that prints and plots everything about a designed ship:
- every handle and parameter;
- corner speed, minimum radius, instantaneous peak turn, gun offset per stick level;
- firing endurance;
- the Ps curve and the E-M "doghouse" chart (turn rate against speed with the stall
  boundary, the g line, the sustained curve and corner speed);
- similarity ratios against a reference aircraft (`ship-card.md` has the P-51D set).
