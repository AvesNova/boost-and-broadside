# Section: Yemong (the model)

Status: walkthrough in progress, 2026-10-31.

## Decided (author, 2026-10-31)

- **Starting point:** v1 Yemong (same ideas and math, new code), because it is known to
  train: about 1500 Elo with the scripted agent at 1000. The stack has room to do much
  better, but the rewrite starts from the known good.
- **Head-to-head comparison later:** the Yemong-review design is added as an option once
  v1 works (Griffin-only memory, the dual next-state head, damped rotary encodings for
  hidden ships). When the feedback loops are removed for that comparison, hidden ships
  still need a position for rotary encoding: probably the forecast, decided then.
- **No bullets:** inputs and the bullet encoder are removed entirely.
- **The model owns its inputs:**
  - the belief store and the enemy-action feedback live in the model's carry;
  - the model composes its own input from the raw observation;
  - nothing is split between the model and the environment wrapper.
- **Actions:** 42-way joint action and enemy-action heads (3 power x 7 stick x 2
  shoot), converted to the factored contract.
- **Critic:**
  - keep the reward levels, each with its own gamma and lambda;
  - per-ship categorical values per level, plus the outcome on the global token;
  - the `Outputs` contract grows to carry them.
- **Density grid:** scaled with the arena. The hexes grow with the arena radius, which
  is on the global token.
- **Units:** the ship-state codes are rebuilt in the new units (J, W, m, s) and for the
  new state (angle of attack included).
- **Mode features:** the design leaves room for them; what they are is decided with the
  modes.
- **Counterfactual shot heads:**
  - train on all, or most, points along each bullet's trajectory (labels come nearly
    free from re-simulation; query the head in parallel over them);
  - give the trajectory head a second hidden layer (it takes time as an input).
- **Food for thought:** separate encoders for allies and enemies. Rows are ordered by
  `team_rel`, so this is a contiguous slice and cheap. Candidate ablation.

## Related decisions (outside the model)

- **Fog of war is toggleable per environment,** with a mix of fogged and clear maps in
  training (the decision log's mixed fog) and the flag on the global token. It also makes
  memoryless test models meaningful.
- **Simple test models,** for checking the pipeline learns:
  - fixed-size MLPs (1v1, 2v2, 5v5) with relational inputs (relative heading, distance,
    bearing). They cannot scale, but learn easily;
  - a spatial-transformer-only model (Yemong's spatial layer, no temporal and no
    memory), most useful with fog off.

  Yemong's own rule (train at 5v5, evaluate at 50v50) does not bind these.

## Open

- Equivalence fixtures from v1 modules vs letting the agent read the v1 code.
- Contract additions for the inputs v1 reads (is-shooting, shield wait, local index and
  gradient, field band width). Names per the full-words rule.
