---
covers: []
---

# The scripted agent (in progress)

Section 4 is specified (`docs/rewrite/sections/scripted-bc-spec.md`, part A) and not built.
Today the scripted controllers are `bnb.controllers.scripted.Random` and
`bnb.controllers.scripted.Chaser`.

It will cover one memoryless scripted controller with three uses: league opponent, the
semi-random ladder rungs, and the teacher for behaviour cloning. It reads only the
observation, exactly what the learner sees, and gives a distribution over the joint
action. Its thresholds are relative to the ship card, so it follows card changes.
