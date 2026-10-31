"""Targets and their codes: the critic, the categorical codes, density, shots, features."""

import pytest
from yemong_helpers import TIGHT, case, check

# --------------------------------------------------------------------------- critic


def test_value_bins_two_hot_and_expectation(arm):
    golden = case("critic_two_hot")
    bins = arm.value_bins(golden.config["count"], golden.config["limit"])
    check(bins, golden.outputs["bins"], "bins", **TIGHT)
    check(arm.two_hot(golden.inputs["values"], bins), golden.outputs["two_hot"], "two_hot", **TIGHT)
    check(arm.expectation(golden.inputs["logits"], bins), golden.outputs["expectation"],
          "expectation", **TIGHT)


def test_outcome_values_and_classes(arm):
    golden = case("critic_outcome_classes")
    check(arm.outcome_values(golden.inputs["probabilities"]), golden.outputs["values"], "values",
          **TIGHT)
    check(arm.outcome_class(golden.inputs["result"]), golden.outputs["classes"], "classes")


def test_outcome_targets(arm):
    golden = case("critic_outcome_targets")
    inputs = golden.inputs
    targets = arm.outcome_targets(
        inputs["probabilities"], inputs["result"], inputs["terminated"],
        inputs["next_probabilities"], golden.config["gamma"], golden.config["lam"],
    )
    check(targets, golden.outputs["targets"], "targets", **TIGHT)


def test_return_scale(arm):
    golden = case("critic_return_scale")
    scales = arm.return_scales(golden.inputs["returns"], golden.inputs["masks"],
                               golden.config["decay"], golden.config["floor"])
    check(scales, golden.outputs["scales"], "scales", **TIGHT)


# --------------------------------------------------------------------------- codes


@pytest.mark.parametrize("name", ["codes_circular_three", "codes_circular_four"])
def test_circular_code(arm, name):
    golden = case(name)
    period, colours, levels = (golden.config[k] for k in ("period", "colours", "levels"))
    code = arm.circular_encode(period, colours, levels, golden.inputs["mean"], golden.inputs["sigma"])
    check(code, golden.outputs["code"], "code", **TIGHT)
    mean, spread = arm.circular_decode(period, colours, levels, golden.outputs["code"])
    check(mean, golden.outputs["mean"], "mean", **TIGHT)
    check(spread, golden.outputs["spread"], "spread", **TIGHT)
    mean, spread = arm.circular_decode(period, colours, levels, golden.inputs["noisy"])
    check(mean, golden.outputs["noisy_mean"], "noisy_mean", **TIGHT)
    check(spread, golden.outputs["noisy_spread"], "noisy_spread", **TIGHT)


def test_position_code(arm):
    golden = case("codes_position")
    size, levels = golden.config["size"].tolist(), golden.config["levels"]
    code = arm.position_encode(size, levels, golden.inputs["mean"], golden.inputs["sigma"])
    check(code, golden.outputs["code"], "code", **TIGHT)
    for source, prefix in ((golden.outputs["code"], ""), (golden.inputs["noisy"], "noisy_")):
        mean, spread = arm.position_decode(size, levels, source)
        check(mean, golden.outputs[prefix + "mean"], prefix + "mean", **TIGHT)
        check(spread, golden.outputs[prefix + "spread"], prefix + "spread", **TIGHT)


def test_scalar_code(arm):
    golden = case("codes_scalar")
    low, high, bins = (golden.config[k] for k in ("low", "high", "bins"))
    check(arm.scalar_sharp(low, high, bins, golden.inputs["values"]), golden.outputs["sharp"],
          "sharp", **TIGHT)
    code = arm.scalar_encode(low, high, bins, golden.inputs["mean"], golden.inputs["sigma"])
    check(code, golden.outputs["code"], "code", **TIGHT)
    for source, prefix in ((golden.outputs["code"], ""), (golden.inputs["noisy"], "noisy_")):
        mean, spread = arm.scalar_decode(low, high, bins, source)
        check(mean, golden.outputs[prefix + "mean"], prefix + "mean", **TIGHT)
        check(spread, golden.outputs[prefix + "spread"], prefix + "spread", **TIGHT)


def test_velocity_code(arm):
    golden = case("codes_velocity")
    knee, bins, spacing = (golden.config[k] for k in ("knee", "bins", "spacing"))
    velocity, covariance = golden.inputs["velocity"], golden.inputs["covariance"]
    check(arm.compress_velocity(velocity, knee), golden.outputs["compressed"], "compressed", **TIGHT)
    check(arm.expand_velocity(golden.outputs["compressed"], knee), golden.outputs["expanded"],
          "expanded", **TIGHT)
    jacobian, inverse = arm.velocity_jacobian(velocity, knee)
    check(jacobian, golden.outputs["jacobian"], "jacobian", **TIGHT)
    check(inverse, golden.outputs["inverse_jacobian"], "inverse_jacobian", **TIGHT)
    check(arm.velocity_sharp(bins, spacing, knee, velocity), golden.outputs["sharp"], "sharp",
          **TIGHT)
    code = arm.velocity_encode(bins, spacing, knee, velocity, covariance)
    check(code, golden.outputs["code"], "code", **TIGHT)
    for source, prefix in ((golden.outputs["code"], ""), (golden.inputs["noisy"], "noisy_")):
        mean, decoded = arm.velocity_decode(bins, spacing, knee, source)
        check(mean, golden.outputs[prefix + "mean"], prefix + "mean", **TIGHT)
        check(decoded, golden.outputs[prefix + "covariance"], prefix + "covariance", **TIGHT)


def test_grouped_softmax_codes(arm):
    golden = case("codes_grouped")
    runs = golden.config["runs"].tolist()
    logits, target = golden.inputs["logits"], golden.inputs["target"]
    check(arm.grouped_log_softmax(logits, runs), golden.outputs["log_probabilities"],
          "log_probabilities", **TIGHT)
    check(arm.grouped_cross_entropy(logits, target, runs), golden.outputs["cross_entropy"],
          "cross_entropy", **TIGHT)
    probabilities = arm.remove_baseline_floor(logits, runs, golden.config["floor"])
    check(probabilities, golden.outputs["probabilities"], "probabilities", **TIGHT)


# --------------------------------------------------------------------------- density


@pytest.mark.parametrize("rings", [3, 10])
def test_hex_cells(arm, rings):
    golden = case(f"density_cells_{rings}")
    axial, offsets = arm.hex_cells(rings)
    check(axial, golden.outputs["axial"], "axial")
    check(offsets, golden.outputs["offsets"], "offsets", **TIGHT)


def test_hex_counts(arm):
    golden = case("density_counts")
    inputs = golden.inputs
    counts = arm.hex_counts(inputs["x"], inputs["y"], inputs["is_ally"], inputs["alive"],
                            inputs["arena_radius"], golden.config["rings"])
    check(counts, golden.outputs["counts"], "counts", **TIGHT)


def test_poisson_terms(arm):
    golden = case("density_poisson")
    per_cell, saturated = arm.poisson_terms(golden.inputs["log_rate"], golden.inputs["counts"])
    check(per_cell, golden.outputs["per_cell"], "per_cell", **TIGHT)
    check(saturated, golden.outputs["saturated"], "saturated", **TIGHT)


# --------------------------------------------------------------------------- shots


def test_shot_trajectory_code(arm):
    golden = case("shots_trajectory_code")
    code = arm.shot_trajectory_code(golden.config, golden.inputs["snapshots"])
    check(code, golden.outputs["code"], "code", **TIGHT)


def test_shot_outcome_code(arm):
    golden = case("shots_outcome_code")
    inputs = golden.inputs
    code = arm.shot_outcome_code(golden.config, inputs["hit"], inputs["distance"], inputs["side"])
    check(code, golden.outputs["code"], "code", **TIGHT)


def test_shot_time_features(arm):
    golden = case("shots_time_features")
    features = arm.shot_time_features(golden.inputs["ages"], golden.config["max_age"],
                                      golden.config["harmonics"])
    check(features, golden.outputs["features"], "features", **TIGHT)


# --------------------------------------------------------------------------- features


def test_scalar_features(arm):
    golden = case("features_scalars")
    check(arm.symlog(golden.inputs["values"]), golden.outputs["symlog"], "symlog", **TIGHT)
    respawn = arm.respawn_features(golden.inputs["seconds"], golden.config["horizon"],
                                   golden.config["harmonics"])
    check(respawn, golden.outputs["respawn"], "respawn", **TIGHT)


def test_local_presence(arm):
    golden = case("features_presence")
    inputs = golden.inputs
    presence = arm.local_presence(inputs["position"], inputs["team"], inputs["source"],
                                  golden.config["radius"])
    check(presence, golden.outputs["presence"], "presence", **TIGHT)
