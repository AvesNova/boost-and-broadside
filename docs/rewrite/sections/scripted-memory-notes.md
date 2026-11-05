# Scripted memory: notes

Brief: `scripted-memory-brief.md` (with the author's amendment of 2026-11-05: one parameter
set per game mode). Round one: `scripted-redesign-notes.md`.

## The memory's form

`bnb.controllers.scripted_memory.ScriptedCarry`, per view: each own ship's committed target
(an enemy row, or none), turn side per layer (combat, Frontline), Frontline goal (defend or
attack), recovery, a `fresh` flag and an agent flag (a mixture rung's draw); each enemy's
last seen position and velocity and whether it is known. One code path: memory off
(`Scripted(memory=False)`) makes every ship fresh and forgets every hidden enemy at every
decision, which is the memoryless agent. On the CPU over round one's own games (three
segments of 64 ticks, 8 games per preset, T = 1 and 2), memory off with round one's
parameters gives round one's distribution exactly (largest difference 0.0).

- **Switches.** A plan choice switches to the best alternative with probability
  `1 − exp(−rate dt)`, `rate = r_max σ((advantage − centre) / (scale T))`. Advantages: the
  target's, the log ratio of its distance over its weight to the best's; the turn side's,
  how much shorter the other way round is (radians); the Frontline goal's, the other goal's
  rule margin in ship strengths; recovery's, the other choice's log margin. A fresh ship or a
  lost target (dead, forgotten) re-chooses outright. The Frontline/combat blend is a ramp
  already, so it takes no hysteresis.
- **The turn side** is committed only past an angle: the committed side outweighs the
  error's sign by a logistic in the error (`side_commit_in_stall_angle`).
- **Dead reckoning.** Last seen position plus last seen velocity times the time since seen,
  held inside the arena; its weight as a target a falling logistic in log time
  (`memory_fade_in_seconds`, centre bounded to 0.5–2 s); forgotten below a weight of 1%, on
  its death, or when the prediction falls inside an own living ship's vision.
- **Shooting at a prediction** is allowed, P(shoot) times the target's weight: the same
  fade, so a target just lost (behind a core) still draws a shot and a stale one does not;
  beyond vision range (1024 m, the gun's range about 210 m) the in-range logistic already
  gives a hidden target almost none.
- **BC labels** are the teacher's distribution given its carry: before each update the
  learner runs the teacher over the stored shards in time order, on from the last update's
  carry, and keeps the carry per tick in the update's derived tensors (re-reading a carry
  without switching gives the same distribution; `tests/sections/test_bc.py`).
- **Per mode** (amendment): `ModeParameters`, a set per training preset, picked per view
  from the observation (Frontline's flag; the wide set beyond the geometric mean of the two
  deathmatch arenas). Every use plays the mode's set.
