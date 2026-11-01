---
covers: []
---

# League and ratings (in progress)

Section 8 has its decisions recorded (`docs/rewrite/sections/league.md`); its spec is to
follow. Today the lineup has one league slot, a copy of the learner refreshed every few
updates (see [The training loop](../training-loop.md)).

It will cover one opponent pool for every opponent kind (checkpoints, averaged policies, the
scripted agent, its ladder rungs and random), sampling by predicted win probability, and
ratings: a live gauge from the training games themselves, and published ratings fitted
after training, per mode and map.
