"""Yemong, the model (spec: docs/rewrite/sections/yemong-spec.md).

    model       Yemong, YemongConfig, the Carry, the Replay record; step and unroll
    inputs      the tokens a view and its belief compose into
    belief      the belief store: observe (sight) and advance (the forecast)
    state_code  the ship-state code: belief means and spreads <-> softmax groups
    codes       code primitives: circular, position, scalar, velocity, grouped softmax
    network     the encoder and the trunk; spatial, temporal and layers its sublayers
    heads       MLP heads, the next-state head, critic values, the shot trajectory head
    policy      the 42-way joint policy: sampling, log-probability, entropy
    critic      value bins, two-hot, outcome classes and targets, the return scale
    density     the hex density target and its Poisson terms
    shots       the shot trajectory and outcome codes, age features
    features    small token features: symlog, respawn features, local presence
"""
