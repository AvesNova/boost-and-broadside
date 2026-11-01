"""The network: temporal and spatial sublayers, the encoder, the trunk, the heads."""

import pytest
import torch
from yemong_helpers import LOOSE, TIGHT, case, check, to_float32

PRECISIONS = [pytest.param(False, id="float64"), pytest.param(True, id="float32")]


def prepared(name, single):
    golden = case(name)
    inputs, weights = golden.inputs, golden.weights
    if single:
        inputs, weights = to_float32(inputs), to_float32(weights)
    return golden, inputs, weights, LOOSE if single else TIGHT


# --------------------------------------------------------------------------- temporal


def test_parallel_scan(arm):
    golden = case("temporal_scan")
    inputs = golden.inputs
    outputs, final = arm.parallel_scan(inputs["decay"], inputs["contribution"], inputs["initial"])
    check(outputs, golden.outputs["outputs"], "outputs", **TIGHT)
    check(final, golden.outputs["final"], "final", **TIGHT)


@pytest.mark.parametrize("single", PRECISIONS)
def test_rg_lru(arm, single):
    golden, inputs, weights, tolerance = prepared("temporal_rg_lru", single)
    outputs, final = arm.rg_lru(weights, inputs["inputs"], inputs["initial"], inputs["done"])
    check(outputs, golden.outputs["outputs"], "outputs", **tolerance)
    check(final, golden.outputs["final"], "final", **tolerance)


@pytest.mark.parametrize("single", PRECISIONS)
def test_temporal_block(arm, single):
    golden, inputs, weights, tolerance = prepared("temporal_block", single)
    out, state, buffer = arm.temporal_block(
        weights, inputs["inputs"], inputs["initial"], inputs["convolution_buffer"], inputs["done"]
    )
    check(out, golden.outputs["outputs"], "outputs", **tolerance)
    check(state, golden.outputs["final_state"], "final_state", **tolerance)
    check(buffer, golden.outputs["final_convolution_buffer"], "final_convolution_buffer", **tolerance)


def test_temporal_block_without_episode_ends(arm):
    """No done flags at all is the same as all-false ones."""
    golden = case("temporal_block")
    inputs = golden.inputs
    done = torch.zeros_like(inputs["done"])
    expected = arm.temporal_block(
        golden.weights, inputs["inputs"], inputs["initial"], inputs["convolution_buffer"], done
    )
    actual = arm.temporal_block(
        golden.weights, inputs["inputs"], inputs["initial"], inputs["convolution_buffer"], None
    )
    for a, e, name in zip(actual, expected, ("outputs", "state", "buffer"), strict=True):
        check(a, e, name, **TIGHT)


# --------------------------------------------------------------------------- spatial


def test_rotary_tables(arm):
    golden = case("spatial_rotary_tables")
    inputs = golden.inputs
    cos, sin = arm.rotary_tables(
        inputs["frequencies_x"], inputs["frequencies_y"], inputs["frequencies_heading"],
        inputs["position"], inputs["heading"],
    )
    check(cos, golden.outputs["cos"], "cos", **TIGHT)
    check(sin, golden.outputs["sin"], "sin", **TIGHT)


def test_apply_rotary(arm):
    golden = case("spatial_apply_rotary")
    inputs = golden.inputs
    rotated = arm.apply_rotary(inputs["values"], inputs["cos"], inputs["sin"])
    check(rotated, golden.outputs["rotated"], "rotated", **TIGHT)


def test_gated_mlp(arm):
    golden = case("spatial_gated_mlp")
    out = arm.gated_mlp(golden.weights, golden.inputs["inputs"])
    check(out, golden.outputs["outputs"], "outputs", **TIGHT)


@pytest.mark.parametrize("single", PRECISIONS)
def test_spatial_block(arm, single):
    golden, inputs, weights, tolerance = prepared("spatial_block", single)
    out = arm.spatial_block(
        weights,
        golden.config["heads"],
        inputs["tokens"],
        inputs["map_memory"],
        (inputs["token_cos"], inputs["token_sin"]),
        (inputs["map_cos"], inputs["map_sin"]),
    )
    check(out, golden.outputs["outputs"], "outputs", **tolerance)


# --------------------------------------------------------------------------- model


@pytest.mark.parametrize("single", PRECISIONS)
def test_encoder(arm, single):
    golden, inputs, weights, tolerance = prepared("model_encoder", single)
    queries, memory = arm.encoder(weights, inputs["ship"], inputs["global_token"], inputs["field"])
    check(queries, golden.outputs["queries"], "queries", **tolerance)
    check(memory, golden.outputs["map_memory"], "map_memory", **tolerance)


def trunk_arguments(inputs):
    return (
        inputs["map_memory"],
        (inputs["token_cos"], inputs["token_sin"]),
        (inputs["map_cos"], inputs["map_sin"]),
    )


@pytest.mark.parametrize("single", PRECISIONS)
def test_trunk_sequence(arm, single):
    """The update path: a whole rollout segment at once, carries reset at episode ends."""
    golden, inputs, weights, tolerance = prepared("model_trunk", single)
    out = arm.trunk_sequence(
        weights,
        golden.config,
        inputs["tokens"],
        *trunk_arguments(inputs),
        inputs["initial_state"],
        inputs["initial_convolution_buffer"],
        inputs["done"],
    )
    check(out, golden.outputs["outputs"], "outputs", **tolerance)


@pytest.mark.parametrize("single", PRECISIONS)
def test_trunk_steps(arm, single):
    """The rollout path: one decision at a time, the carry cleared after each episode end,
    gives the update path's outputs and the final carry."""
    golden, inputs, weights, tolerance = prepared("model_trunk", single)
    memory, (token_cos, token_sin), (map_cos, map_sin) = trunk_arguments(inputs)
    state = inputs["initial_state"]
    buffer = inputs["initial_convolution_buffer"]
    done = inputs["done"]
    outputs = []
    for t in range(inputs["tokens"].shape[0]):
        out, state, buffer = arm.trunk_step(
            weights,
            golden.config,
            inputs["tokens"][t],
            memory[t],
            (token_cos[t], token_sin[t]),
            (map_cos[t], map_sin[t]),
            state,
            buffer,
        )
        outputs.append(out)
        keep = (~done[t]).to(state.dtype).view(1, -1, 1, 1)
        state, buffer = state * keep, buffer * keep.unsqueeze(-1)
    check(torch.stack(outputs), golden.outputs["outputs"], "outputs", **tolerance)
    check(state, golden.outputs["final_state"], "final_state", **tolerance)
    check(buffer, golden.outputs["final_convolution_buffer"], "final_convolution_buffer", **tolerance)


# --------------------------------------------------------------------------- heads


@pytest.mark.parametrize("single", PRECISIONS)
def test_mlp_head(arm, single):
    golden, inputs, weights, tolerance = prepared("heads_mlp", single)
    check(arm.mlp_head(weights, inputs["inputs"]), golden.outputs["outputs"], "outputs", **tolerance)


def test_next_state_head(arm):
    golden = case("heads_next_state")
    logits = arm.next_state_head(golden.weights, golden.inputs["inputs"], golden.inputs["code"])
    check(logits, golden.outputs["logits"], "logits", **TIGHT)


def test_critic_values(arm):
    golden = case("heads_critic_values")
    values = arm.critic_values(
        golden.inputs["local_logits"],
        golden.inputs["outcome_logits"],
        golden.inputs["bins"],
        golden.config["local_levels"].tolist(),
        golden.config["outcome_level"],
    )
    check(values, golden.outputs["values"], "values", **TIGHT)
