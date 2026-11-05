# Brief: a scripted agent with memory (soft hysteresis and dead reckoning), round two

The soft scripted agent has landed (`docs/rewrite/sections/scripted-redesign-notes.md`).
It is level with the old raw agent and beats the old BC teacher 0.63. It is still
memoryless: every choice is a function of the current observation. That leaves two
problems:

1. **Plan choices flip at hard edges.** The target is the nearest visible living enemy
   (`scripted_combat.combat`, an argmin). The goal choices (target, zone, recovery) are
   hard selections. The turn side comes from a sign. Near a tie they flip from tick to
   tick.
2. **BC puts no pressure on the clone's memory.** The best clone of a memoryless teacher
   is memoryless.

The author's plan (2026-11-05): soft hysteresis on plan choices; dead reckoning of
enemies that fog hides, for a short time; then one more search, until it plateaus.

## 1. Per-ship memory (the controller's carry)

`Scripted` gains a carry: per ship, its committed plan choices, and per enemy, the last
seen position, velocity and time.
- It resets cleanly on the ship's death and respawn, on episode reset, and when an
  enemy dies.
- It works in the CUDA-graph rollout, in the lanes, in the head-to-head and in the BC
  teacher. The labels are the agent's distribution given its carry.
- Keep the memoryless agent reachable for the ablation below: a flag or a parameter
  value that turns the memory off, not a second code path.

## 2. Soft hysteresis: a switching rate, not thresholds

For each plan choice (the target, the Frontline/combat layer or goal, and the turn side
when the error is near ±180°), the agent keeps its committed option. Each tick it
switches with probability

  rate × dt,  rate = r_max · σ((advantage − centre) / (scale · T)),

where advantage is how much better the best alternative is than the committed one, in a
natural unit for that choice. For the target, for example, the log ratio of distances or
of a target score. r_max, centre and scale are searched within bounds. The rate rises
smoothly: near-equal options almost never switch, and a clearly better one switches
within a fraction of a second.
- **No hard thresholds.** Keep the existing per-decision floors on the actions. The plan
  choices take no floor beyond the rate.
- **The turn side:** commit to a direction once the error passes some angle. Use the
  switching rate to reconsider, with advantage being how much shorter the other way is.
  Near-zero errors need no side.
- **Losing the committed target** (dead, or faded from memory) forces a re-choice, as
  today.

## 3. Dead reckoning of hidden enemies

When an enemy leaves the agent's view, the agent keeps a predicted position: the last
seen position plus last seen velocity × time since seen, kept inside the arena.
- **The prediction's weight as a target fades** smoothly, as a logistic in time since
  seen. The centre is searched within tight bounds, about 0.5–2 s.
- **A prediction that falls inside the agent's own view with no enemy there is dropped:**
  it was wrong.
- **A predicted enemy can be the committed target.** It competes with the visible ones
  through the hysteresis, its distance discounted by its fade weight.
- **Aim and fire at a predicted target follow the same logistics.** The shot's gates
  still hold: no shot without a target, or while the gun can't fire. Decide whether a
  faded prediction may be shot at, and say why.

## 4. The ladder's bottom

Temperature alone does not reach random (the round-one notes: T = 8–32 still beats random
about 1.000). Fill the gap between random and the warmest temperature rung with a few
mixture rungs: per game or per ship, the agent at a moderate T with probability p, else
random. Keep the names clear, and keep the lanes' ladder chain calibrating them.

## 5. One more search, until it plateaus

- **Same population round robin** (`bench/scripted_search.py`): 32 candidates, all three
  presets, temperatures {1, 1.5, 2, 3}.
- **Start from round one's defaults.** Add the new parameters within bounds; the others
  keep their bounds and scale floors.
- **References:** the old raw agent (the frozen copy from round one) and round one's agent
  (freeze it the same way before you change the defaults).

## Acceptance, to report

128 games per preset each:

- The new agent at T = 1 against round one's agent, and against the old raw agent.
- **The memory ablation:** the new agent against itself with memory off, both at T = 1.
  Memory should not make it weaker. If it does, report that, not a workaround.
- The temperature curve (T = 1.5, 2, 3, 4 against T = 1), and the new bottom rungs against
  random and against each other.
- The softness audit, rerun. Add one row per plan choice: switches per ship-second, and
  the share of ticks within 0.1 s of a switch.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-scripted-memory`, branch `feat/scripted-memory`
  (from v2). Work only there. Scratch in `/tmp/bnb-scratch-scripted-memory`. Do not read
  other directories under `/home/vizia/avesnova`, apart from the round-one scratch,
  `/tmp/bnb-scratch-scripted-redesign`, where the frozen old agent and the audit script
  are.
- **Python:** `uv sync --all-extras`, then always `uv run --no-sync ...`.
- **GPU:** you are the only GPU job. Hold the bench lock for every GPU command:
  `uv run --no-sync python ~/.local/share/bnb/locked.py <command...>`.
- **W&B:** none.
- **Progress notes:** append brief milestone notes to
  `/tmp/bnb-scratch-scripted-memory/progress.md`, with a `## HH:MM` heading (from `date`)
  and 1–3 bullets. Write one when a part lands (with its commit), when a measurement or
  generation summary is worth showing, when something fails, and when you change plan.
  Start with a plan note.
- **Tests:** CPU tests for what you change, plus the quick tier; ruff clean; docs updated.
  The GPU acceptance file (`tests/sections/test_scripted_acceptance.py`) is the only GPU
  test to run. No dummy training run.

## Done

- Commit and push to `feat/scripted-memory` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Final message, one page:**
  - the memory's form;
  - the search's run;
  - the acceptance tables, including the memory ablation;
  - anything left open.
