"""The Yemong fixture interface over bnb's model code (see README.md).

Each method is a translation onto `bnb.models.yemong`: fixture weight names to module
parameters, the fixtures' (batch, time) and (blocks, batch) layouts to the model's
time-first and batch-first ones, and the interface's `done` ("ended at this step") to the
model's episode starts ("an episode starts at this step").
"""

import math

import torch

from bnb.models.yemong import belief, codes, critic, density, features, heads, layers, shots
from bnb.models.yemong.network import Encoder, Trunk
from bnb.models.yemong.spatial import SpatialBlock
from bnb.models.yemong.temporal import RGLRU, TemporalBlock, parallel_scan

def _loaded(module: torch.nn.Module, weights: dict) -> torch.nn.Module:
    dtype = next(iter(weights.values())).dtype
    module = module.to(dtype)
    module.load_state_dict(weights, strict=True)
    return module


def _fused(weights: dict, parts: tuple[str, ...], into: str, prefix: str = "") -> dict:
    """Concatenate the fixture's separate matrices `prefix + part + .weight/.bias` (in order)
    into our one matrix `prefix + into`."""
    out = dict(weights)
    for kind in ("weight", "bias"):
        names = [f"{prefix}{part}.{kind}" for part in parts]
        if names[0] in out:
            out[f"{prefix}{into}.{kind}"] = torch.cat([out.pop(name) for name in names])
    return out


def _temporal_names(weights: dict, prefix: str = "") -> dict:
    """The fixtures list the RG-LRU's parameters beside the block's, with its two gates
    apart; ours nest them under `recurrence` with the gates as one matrix."""
    out = _fused(weights, ("recurrence_gate", "input_gate"), "recurrence.gates", prefix)
    out = _mlp_names(out, prefix + "feedforward.")
    if prefix + "log_decay" in out:
        out[prefix + "recurrence.log_decay"] = out.pop(prefix + "log_decay")
    return out


def _mlp_names(weights: dict, prefix: str = "") -> dict:
    """The gated MLP's gate and up matrices are one matrix in ours."""
    return _fused(weights, ("gate", "up"), "gate_up", prefix)


def _spatial_names(weights: dict, prefix: str = "") -> dict:
    return _mlp_names(weights, prefix + "feedforward.")


def _head_names(weights: dict) -> dict:
    """One MLP head as a group of one."""
    renamed = {"norm.weight": "norms.0.weight", "last.weight": "lasts.0.weight",
               "last.bias": "lasts.0.bias"}
    return {renamed.get(name, name): value for name, value in weights.items()}


def _starts(done: torch.Tensor, time_axis: int) -> torch.Tensor:
    """`done` (ended at t) as starts (an episode starts at t + 1), time-first."""
    done = done.movedim(time_axis, 0)
    return torch.cat([torch.zeros_like(done[:1]), done[:-1]])


class Arm:
    # ------------------------------------------------------------------ temporal

    def parallel_scan(self, decay, contribution, initial):
        outputs, final = parallel_scan(decay.transpose(0, 1), contribution.transpose(0, 1),
                                       initial)
        return outputs.transpose(0, 1), final

    def rg_lru(self, weights, inputs, initial, done):
        unit = _loaded(RGLRU(inputs.shape[-1]),
                       _fused(weights, ("recurrence_gate", "input_gate"), "gates"))
        outputs, final = unit.sequence(inputs.transpose(0, 1), initial, _starts(done, 1))
        return outputs.transpose(0, 1), final

    def temporal_block(self, weights, inputs, initial, convolution_buffer, done):
        block = _loaded(TemporalBlock(inputs.shape[-1]), _temporal_names(weights))
        if done is None:
            done = torch.zeros(inputs.shape[:2], dtype=torch.bool)
        outputs, state, buffer = block.sequence(inputs.transpose(0, 1), initial,
                                                convolution_buffer, _starts(done, 1))
        return outputs.transpose(0, 1), state, buffer

    # ------------------------------------------------------------------ spatial

    def rotary_tables(self, frequencies_x, frequencies_y, frequencies_heading, position,
                      heading):
        angle = torch.atan2(heading[..., 1], heading[..., 0])  # 0 for (0, 0)
        # Ours takes the position frequencies as periods, 2 pi / f.
        return layers.rotary_tables(position, angle, 2 * math.pi / frequencies_x,
                                    2 * math.pi / frequencies_y, frequencies_heading)

    def apply_rotary(self, values, cos, sin):
        return layers.apply_rotary(values, cos, sin)

    def gated_mlp(self, weights, inputs):
        width, hidden = inputs.shape[-1], weights["gate.weight"].shape[0]
        return _loaded(layers.GatedMLP(width, hidden), _mlp_names(weights))(inputs)

    def spatial_block(self, weights, heads_count, tokens, map_memory, token_tables, map_tables):
        block = _loaded(SpatialBlock(tokens.shape[-1], heads_count, map_memory.shape[-1]),
                        _spatial_names(weights))
        return block(tokens, map_memory, token_tables, map_tables)

    # ------------------------------------------------------------------ model

    def encoder(self, weights, ship, global_token, field):
        width = weights["shared.weight"].shape[0]
        encoder = Encoder(ship.shape[-1], global_token.shape[-1], field.shape[-1], width,
                          weights["map_memory.weight"].shape[0])
        # `global` is a Python keyword, so our global kind is `global_token`.
        names = {("global_token." + name[len("global."):] if name.startswith("global.") else name):
                 value for name, value in weights.items()}
        return _loaded(encoder, names)(ship, global_token, field)

    def _trunk(self, weights, config):
        trunk = Trunk(config["width"], config["heads"], config["map_width"], config["blocks"],
                      config["spatial_per_block"])
        names = weights
        for index in range(config["blocks"]):
            names = _temporal_names(names, f"blocks.{index}.temporal.")
            for layer in range(config["spatial_per_block"]):
                names = _spatial_names(names, f"blocks.{index}.spatial.{layer}.")
        return _loaded(trunk, names)

    def trunk_sequence(self, weights, config, tokens, map_memory, token_tables, map_tables,
                       state, convolution_buffer, done):
        trunk = self._trunk(weights, config)
        outputs, _, _ = trunk.sequence(tokens, map_memory, token_tables, map_tables,
                                       state.movedim(0, 1), convolution_buffer.movedim(0, 1),
                                       _starts(done, 0))
        return outputs

    def trunk_step(self, weights, config, tokens, map_memory, token_tables, map_tables, state,
                   convolution_buffer):
        trunk = self._trunk(weights, config)
        outputs, state, buffer = trunk.step(tokens, map_memory, token_tables, map_tables,
                                            state.movedim(0, 1), convolution_buffer.movedim(0, 1))
        return outputs, state.movedim(1, 0), buffer.movedim(1, 0)

    # ------------------------------------------------------------------ heads

    def mlp_head(self, weights, inputs):
        hidden, outputs = weights["first.weight"].shape[0], weights["last.weight"].shape[0]
        group = _loaded(heads.HeadGroup(inputs.shape[-1], hidden, (outputs,)), _head_names(weights))
        return group(inputs)[0]

    def next_state_head(self, weights, inputs, code):
        return heads.next_state_logits(code, self.mlp_head(weights, inputs))

    def critic_values(self, local_logits, outcome_logits, bins, local_levels, outcome_level):
        return heads.critic_values(local_logits, outcome_logits, bins, tuple(local_levels),
                                   outcome_level)

    # ------------------------------------------------------------------ critic

    def value_bins(self, count, limit):
        return critic.value_bins(count, limit, dtype=torch.float64)

    def two_hot(self, values, bins):
        return critic.two_hot(values, bins)

    def expectation(self, logits, bins):
        return critic.expectation(logits, bins)

    def outcome_values(self, probabilities):
        return critic.outcome_values(probabilities)

    def outcome_class(self, result):
        return critic.outcome_class(result)

    def outcome_targets(self, probabilities, result, terminated, next_probabilities, gamma, lam):
        return critic.outcome_targets(probabilities, result, terminated, next_probabilities,
                                      gamma, lam)

    def return_scales(self, returns, masks, decay, floor):
        scale = critic.ReturnScale(decay, floor, dtype=returns.dtype)
        return torch.stack([scale.update(batch, mask) for batch, mask in zip(returns, masks)])

    # ------------------------------------------------------------------ codes

    def circular_encode(self, period, colours, levels, mean, sigma):
        return codes.circular_encode(mean, sigma, period, colours, levels)

    def circular_decode(self, period, colours, levels, probabilities):
        return codes.circular_decode(probabilities, period, colours, levels)

    def position_encode(self, size, levels, mean, sigma):
        return codes.position_encode(mean, sigma, tuple(size), levels)

    def position_decode(self, size, levels, probabilities):
        return codes.position_decode(probabilities, tuple(size), levels)

    def scalar_sharp(self, low, high, bins, values):
        return codes.scalar_sharp(values, low, high, bins)

    def scalar_encode(self, low, high, bins, mean, sigma):
        return codes.scalar_encode(mean, sigma, low, high, bins)

    def scalar_decode(self, low, high, bins, probabilities):
        return codes.scalar_decode(probabilities, low, high, bins)

    def compress_velocity(self, velocity, knee):
        return codes.compress_velocity(velocity, knee)

    def expand_velocity(self, compressed, knee):
        return codes.expand_velocity(compressed, knee)

    def velocity_jacobian(self, velocity, knee):
        return codes.velocity_jacobian(velocity, knee)

    def velocity_sharp(self, bins, spacing, knee, velocity):
        return codes.velocity_sharp(velocity, bins, spacing, knee)

    def velocity_encode(self, bins, spacing, knee, mean, covariance):
        return codes.velocity_encode(mean, covariance, bins, spacing, knee)

    def velocity_decode(self, bins, spacing, knee, probabilities):
        return codes.velocity_decode(probabilities, bins, spacing, knee)

    def grouped_log_softmax(self, logits, runs):
        return codes.grouped_log_softmax(logits, tuple(map(tuple, runs)))

    def grouped_cross_entropy(self, logits, target, runs):
        return codes.grouped_cross_entropy(logits, target, tuple(map(tuple, runs)))

    def remove_baseline_floor(self, logits, runs, floor):
        return codes.remove_baseline_floor(logits, tuple(map(tuple, runs)), floor)

    # ------------------------------------------------------------------ density

    def hex_cells(self, rings):
        return density.hex_cells(rings, dtype=torch.float64)

    def hex_counts(self, x, y, is_ally, alive, arena_radius, rings):
        return density.HexGrid(rings).counts(x, y, is_ally, alive, arena_radius)

    def poisson_terms(self, log_rate, counts):
        return density.poisson_terms(log_rate, counts)

    # ------------------------------------------------------------------ shots

    def shot_trajectory_code(self, parameters, snapshots):
        return shots.trajectory_code(shots.ShotCodeParameters(**parameters), snapshots)

    def shot_outcome_code(self, parameters, hit, distance, side):
        return shots.outcome_code(shots.ShotCodeParameters(**parameters), hit, distance, side)

    def shot_time_features(self, ages, max_age, harmonics):
        return shots.time_features(ages, max_age, harmonics)

    # ------------------------------------------------------------------ features

    def symlog(self, values):
        return features.symlog(values)

    def respawn_features(self, seconds, horizon, harmonics):
        return features.respawn_features(seconds, horizon, harmonics)

    def local_presence(self, position, team, source, radius):
        return features.local_presence(position, team, source, radius)

    # ------------------------------------------------------------------ belief

    @staticmethod
    def _belief(state: dict) -> belief.Belief:
        return belief.Belief(valid=state["valid"], age=state["age"], means=state["means"],
                             spreads=state["uncertainty"], action=state["action"])

    def belief_observe(self, state, truth, visible, spawned, tick_seconds):
        out = belief.observe(self._belief(state), truth, visible, spawned)
        return dict(valid=out.valid, age=out.age, means=out.means, uncertainty=out.spreads,
                    action=out.action,
                    time_since_seen=out.age.to(truth.dtype) * tick_seconds)

    def belief_advance(self, state, prediction, enemy_action_logits):
        prior = self._belief(state)
        if enemy_action_logits is None:  # no forecast yet: uniform, the softmax of zeros
            enemy_action_logits = torch.zeros_like(prior.action)
        out, nonfinite = belief.advance(prior, prediction, enemy_action_logits)
        return dict(means=out.means, uncertainty=out.spreads, action=out.action,
                    nonfinite=nonfinite.sum())


def make_arm():
    return Arm()
