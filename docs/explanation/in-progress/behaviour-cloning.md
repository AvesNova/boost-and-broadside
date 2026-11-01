---
covers: []
---

# Behaviour cloning (in progress)

Section 7 is specified (`docs/rewrite/sections/scripted-bc-spec.md`, part B) and not built.

It will cover on-policy imitation of the scripted agent: the student flies the learner's
views and the teacher labels the states it visits. Behaviour cloning gives every head a
head start, after which reinforcement learning starts warm from its weights. It will also
cover the teacher's temperature, which sets the entropy the clone starts with, and the rule
for when cloning stops.
