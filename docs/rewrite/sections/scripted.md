# Section 4: the scripted agent

Status: decisions settled, 2026-11-01. Spec: `scripted-bc-spec.md`.

## Already decided (training review 4c)

- **One function, three uses:** league opponent, the semi-random rungs (scripted action
  with probability p, else uniform) and the BC teacher.
- **Memoryless.**
- **The 42-way action:** it emits a distribution over the joint action, the product of
  its power, stick and shoot marginals.
- **Entropy through the teacher:** a knob on the teacher's distribution (a temperature,
  or a mix with uniform) sets the entropy the clone starts RL with. No entropy bonus.

## Decided (author, 2026-11-01)

- **It reads only the observation,** exactly what the learner sees: fog respected, no
  game state. The two then certainly share their information, so the scripted agent
  cannot cheat, and the student can match the teacher's entropy.
- **Port v1's structure, retune its numbers.**
  - The structure: the probability ramps, interception (target leading), the team
    target blend, and the Frontline layer (zone need, local strength, separation,
    recovery at low health).
  - The interception geometry carries over essentially unchanged.
  - The new turn level: v1 chose among 5 turns (none, normal, sharp, per side); v2's
    stick has 3 levels per side, and the agent uses all three.
- **Every threshold is relative to the ship card:** ranges as fractions of bullet range,
  speeds as fractions of the ship's characteristic speeds, turn levels from its turn
  rates. The agent follows ship-card changes and other ship types, so the ship card (1b)
  does not block it.
- **Modes:** team deathmatch and Frontline. Sandbox needs none.
- **Acceptance:**
  - **Behavioural tests:** the stick level matches the angle error; it fires only when
    aligned and in range; it boosts when far; in Frontline it heads for the zone that
    needs ships and falls back when hurt.
  - **Against random:** it essentially always wins. The bar is a small buffer for a
    one-in-a-million game, not a tolerance: a steady 99% would be a failure to
    investigate.
  - **Against the chaser:** a win rate of at least 70% in both modes.
  - **Speed:** "dummy fast": well under the game's own tick cost, since BC runs it on
    every learner environment.
  - **No v1 Elo comparison:** v1 and v2 are different games.
- **v1 fixtures:** none, apart from possibly the interception geometry. The retuned
  numbers make the rest a behavioural question.
