"""The belief store: assimilating sight and storing the forecast."""

from yemong_helpers import TIGHT, case, check


def belief_of(inputs):
    return {key[len("belief_"):]: value for key, value in inputs.items() if key.startswith("belief_")}


def test_observe(arm):
    golden = case("belief_observe")
    inputs = golden.inputs
    out = arm.belief_observe(belief_of(inputs), inputs["truth"], inputs["visible"],
                             inputs["spawned"], golden.config["tick_seconds"])
    for name, expected in golden.outputs.items():
        check(out[name], expected, name, **TIGHT)


def test_advance(arm):
    golden = case("belief_advance")
    inputs = golden.inputs
    out = arm.belief_advance(belief_of(inputs), inputs["prediction"], inputs["enemy_action_logits"])
    for name, expected in golden.outputs.items():
        check(out[name], expected, name, **TIGHT)


def test_advance_without_action_forecast(arm):
    golden = case("belief_advance_without_actions")
    inputs = golden.inputs
    out = arm.belief_advance(belief_of(inputs), inputs["prediction"], None)
    for name, expected in golden.outputs.items():
        check(out[name], expected, name, **TIGHT)
