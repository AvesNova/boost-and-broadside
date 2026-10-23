# Flight model (proposed for the physics-first work)

Status: **proposed**, designed 2026-10-22/23 in discussion with the author. Prototype:
`docs/rewrite/physics-checks/flight8.py` (x/y), checks in `final_check.py`. The design
history and every measurement are in `decisions.md` ("Tick-rate accuracy findings" and the
entries after it). This file is the clean specification.

## Units and scale

- SI, game scale. The standard ship has **mass 1 kg**, so force = acceleration (N = m/s²
  per kg), stored energy is in J, engine output in W.
- Game gravity g = 27 m/s² ("1 g": the lift needed to hold level flight).
- Against a real P-51D: speeds x1.48, times x4.1, lengths x6.1 (the game is ~4x faster and
  ~6x smaller). Anchors: game g -> 9.81 m/s², game stall -> the P-51D's ~47 m/s.

## State per ship

Position (x, y) in the map-local frame (no wrap); heading as a unit pair (cos, sin);
speed u and angle of attack α (velocity stored in polar form, fastest); nose rate r;
energy store E (J); refractive index n and ∇ln n at the ship (cached from the last field
evaluation).

## Controls (42 joint actions, unchanged count)

- **Stick (7 levels):** straight, or one of three levels to either side. A level asks for a
  fixed fraction of the ship's max lift coefficient sideways: **0.4, 0.75, 1.0** (decided;
  the same for every ship type), capped by the g limit. Below corner speed each level flies
  a fixed multiple of the minimum radius (2.5x, 1.33x, 1x). A turn's induced drag is paid
  from speed (kinetic energy), not the store (decided).
- **Power (3):** coast, boost, reverse. These are not thrust settings: each commands an
  **energy exchange** along the flight path (no implied pitch angle): coast +a0 = 3 m/s²
  (a small climb that charges E), reverse +27 m/s² (full climb), boost -27 m/s² (full
  dive). dE/dt = m u a_x; the speed loses a_x.
- **Shoot (2):** one bullet per decision step when held; each costs energy.

## Forces and rules

- **Engine:** power P = 1200 W, all of it thrust along the nose: F = P/u, softly
  capped at 20 N at low speed. *Decided 2026-10-23:* power falls as the store fills (an
  engine above its critical altitude), from P below a knee to the power that holds cruise
  at a full store (863 W for this ship: level cruise at 104 m/s with nothing going to the
  store). Shape: smoothstep from the knee to full (decided); the knee is a per-ship input.
  The kick evaluates it at the predicted mid-kick store (keeps second order).
  The coast charge a0 scales with the spare power (decided), so coasting holds cruise
  all the way up; the store's top is approached asymptotically. Static thrust per kg stays below the climb rate, so reverse
  always slows the ship (thrust-to-weight < 1).
- **Lift budget:** L(u) = softmin(C_Lmax u², a_max), C_Lmax = 0.027 1/m, a_max = 216 m/s²
  (8 g), soft-min exponent p = 16 (smooth but nearly hard, so the instantaneous turn curve
  matches a P-51's).
- **One lift vector, coupled (p = 2):** lateral = stick x L, vertical = g (to hold level);
  if their norm exceeds L both are scaled down along their direction. The exchange command
  is scaled by the lift left after turning, √(1 - (lateral/L)²). When the vertical lift
  cannot hold level (below stall, or a hard turn at low speed) the ship is forced into a
  descent (sink), paid from E.
- **Store ends:** near a full store positive exchange fades (ceiling); near an empty one
  negative exchange fades (floor). Soft bands of 2% of the store.
- **Drag:** (C_D0 + k (C_lat² + C_vert²)) u², with C_D0 = 6.96e-4 1/m (set so coasting
  cruises at ~104 m/s while charging) and induced factor k = 11.4 (3x the P-51-derived
  value, so the no-boost sustained turn matches a P-51's).
- **Nose:** a rigid body in the plane: r' = ω(u)² (α_cmd - α) - 2ζ ω(u) α', ω(u) =
  22 rad/s x u / u_corner (twice the first tuning), ζ = 0.7. α_cmd inverts the lift curve
  C_L(α) = C_Lmax (2x - x|x|), x = α / 15°. The gun points along the nose (offset α from the
  flight path).
- **Thrust along the nose** has a lateral part (T sin α); dive and climb act along the
  flight path.
- **Fields:** refractive index n(x) is curved space (metric n² I). The ship flies the
  same model in proper time (dτ = dt/n) with proper velocity w = n v; the field adds only
  geodesic bending of w at rate |v| (v̂ x ∇ln n), which leaves |w| unchanged.
- **Bullets:** one per decision step (30/s), 29 J each (1.2x the engine's maximum excess
  power / 30, so continuous fire always drains the store: no shooting forever).
  Proposed ballistics: 600 m/s, 0.4 s lifetime (12 decisions), hitbox radius 3 m,
  12 slots per ship. Hits: relative swept test (decided). Shot labels: learner-side
  re-simulation (decided).

## Integrator (per 30 Hz tick; physics tick = decision tick, decided)

Half kick | drift | field evaluation | half kick.

- **Kick** (nose rate, speed, angle of attack, energy; pose held): the fast nose /
  angle-of-attack oscillator is solved exactly (closed-form 2x2 matrix exponential,
  polynomial form; lift slope linearized; coefficients at the predicted mid-kick speed);
  the velocity direction is set exactly, so lift and field bending do no work; speed and
  energy by a two-stage update.
- **Drift:** the pose moves along an exact arc (rotation r h; displacement factor
  (e^{iφ} - 1)/(iφ) by series); the field index at the step midpoint from
  ln n + ½ ∇ln n · Δx.
- One field evaluation per tick. Second order throughout.

## Measured (prototype, 30 Hz, no fields unless noted)

| Stat | Today | Proposed | P-51D at game scale |
|---|---|---|---|
| Stall (min level speed) | none (control fades below 35 m/s) | 31.6 m/s | 31.6 |
| Cruise (coast) | 100 m/s | 104 charging (312 W), 117 store full | 109 |
| Dive peak from a full store | 229 (boost) | 164 m/s, store empty in 2.6 s | ~152 |
| Corner speed | 100 m/s | 89 | ~90 |
| Instantaneous peak | 151 deg/s | 124 deg/s | ~137 |
| Nose response (90%) | 0.07 s | 0.13 s | - |
| Sustained turn, coast | 70 deg/s at 81 m/s | 70 deg/s at 47 m/s, r 39 m | 70 at 49 |
| Sustained turn, boost, unlimited store | 119 deg/s at 73 (power-limited) | 85 deg/s at 57 m/s | - |
| Min instantaneous radius | 37 m | 37 m | ~38 |
| Best climb (max excess power) | - | ~65 m/s, 728 W (27 m/s of climb) | ~51, 11 |
| Store fill from empty, coasting | - | 31 s | - |
| Hold reverse (unlimited store) | - | 83 m/s after 1 s, 50 after 3 s, settles ~27 m/s banking 357 W | - |
| Continuous fire from a full store | - | coasting 18 s, climbing 40 s, diving 2.2 s, hard turn 11 s | - |
| Turn levels (coast datum) | 1/3: 81 m/s r 67; 3/3: 39 m/s r 35 | 0.4: 79 m/s, 50 deg/s, r 91; 0.75: 52, 63, r 48; 1.0: 47, 70, r 39 | 0.4: 80, 50, r 91; 0.75: 55, 66, r 48; 1.0: 51, 76, r 39 |
| Gun offset at the datum | 5 / 10 / 15 deg | 2.7 / 6.3 / 10.3 deg (measured with lift thirds; re-measure) | - |

Accuracy (mean position error after 2 s of random inputs): 0.25 / 0.064 / 0.017 m at
30 / 60 / 120 Hz without fields, 0.66 / 0.15 / 0.04 m with fields (today's model: ~3.9 m
at 30 Hz without fields). Energy drift with drag, engine and fire off: 8e-7 (1e-6 with
fields). Cost (JAX CPU, 40,960 ships, per tick): 5-6 ms vs today's 4.8 ms in the same run
(CPU timings vary ~20%); per-ship work is elementwise and small next to pairwise work.

## Open

1. *Decided 2026-10-23:* turn levels are lift-coefficient fractions 0.4 / 0.75 / 1.0
   (see Controls); turn energy comes from speed.
2. **Ship card and handling solver.** The handles change (see "Superseded"). New
   per-type parameters: C_Lmax (sets stall with g and the min radius), a_max (corner),
   C_D0 and k (drag), engine power and static thrust, a0 / climb / dive rates, store size,
   nose frequency and damping, stall angle (gun offset), bullet cost. Stall speed and min
   radius are no longer independent for a fixed g (both come from C_Lmax). The solver and
   the card need rewriting around these.
3. **Field composition.** The prototype sums profiles in log space; the env uses the
   union rule (`compose_refractive_index`). Port the real rule. Bullets through fields
   need the same curved-space treatment.
4. **Bullets and hitbox values** (600 m/s, 0.4 s, 3 m, 29 J) are proposals; not yet in a
   prototype with collisions.
5. **Where it lands first:** the physics-first plan (below in `decisions.md`) implements
   it in the PyTorch env so the training stack can use it and the JAX port stays a
   translation; alternatively implement it directly in JAX with the prototype checks as
   the specification.
6. **Observation fields** the contracts must carry: speed, angle of attack (or nose
   heading), nose rate, energy store (J), per-type parameters.
