"""The adapter from the fixture interface (README.md) to `bnb.models.swift`.

`make_arm()` returns an object with the README's methods. Fixture sequences are batch-major
(B, T, ...) with `done[t]` = "the episode ended on step t"; the implementation is time-major
with `starts[t]` = "an episode starts at t", so `starts[t] = done[t - 1]`.
"""

import torch

from bnb.models.swift import belief, codes, network, targets
from bnb.models.swift.scan import linear_scan


def done_to_starts(done):
    """done (..., T) -> starts (..., T): starts[t] = done[t - 1], starts[0] = False."""
    return torch.nn.functional.pad(done[..., :-1], (1, 0), value=False)


def loaded(module, weights, rename=None):
    first = next(iter(weights.values()))
    module = module.to(first.dtype)
    module.load_state_dict(rename(weights) if rename else weights)
    return module


class Arm:
    # ----------------------------------------------------------------- temporal

    def parallel_scan(self, decay, contribution, initial):
        outputs = linear_scan(decay.transpose(0, 1), contribution.transpose(0, 1), initial)
        return outputs.transpose(0, 1), outputs[-1]

    def rg_lru(self, weights, inputs, initial, done):
        module = loaded(network.RGLRU(inputs.shape[-1]), weights)
        starts = done_to_starts(done).transpose(0, 1)
        outputs, final = module(inputs.transpose(0, 1), initial, starts)
        return outputs.transpose(0, 1), final

    def temporal_block(self, weights, inputs, initial, convolution_buffer, done):
        module = loaded(network.TemporalBlock(inputs.shape[-1]), weights)
        starts = None if done is None else done_to_starts(done).transpose(0, 1)
        outputs, state, buffer = module(inputs.transpose(0, 1), initial, convolution_buffer, starts)
        return outputs.transpose(0, 1), state, buffer

    # ----------------------------------------------------------------- spatial

    def rotary_tables(self, frequencies_x, frequencies_y, frequencies_heading, position, heading):
        angle = torch.atan2(heading[..., 1], heading[..., 0])
        return network.rotary_tables(frequencies_x, frequencies_y, frequencies_heading, position,
                                     angle)

    def apply_rotary(self, values, cos, sin):
        return network.apply_rotary(values, cos, sin)

    def gated_mlp(self, weights, inputs):
        hidden, width = weights["gate.weight"].shape
        return loaded(network.GatedMLP(width, hidden), weights)(inputs)

    def spatial_block(self, weights, heads, tokens, map_memory, token_tables, map_tables):
        width, map_width = tokens.shape[-1], map_memory.shape[-1]
        module = loaded(network.SpatialBlock(width, int(heads), map_width), weights)
        return module(tokens, map_memory, token_tables, map_tables)

    # ----------------------------------------------------------------- model

    def encoder(self, weights, ship, global_token, field):
        width = weights["shared.weight"].shape[0]
        map_width = weights["map_memory.weight"].shape[0]
        module = network.Encoder(ship.shape[-1], global_token.shape[-1], field.shape[-1], width,
                                 map_width)
        module = loaded(module, weights, module.fixture_state)
        return module(ship, global_token, field)

    def trunk(self, weights, config):
        module = network.Trunk(int(config["width"]), int(config["heads"]),
                               int(config["map_width"]), int(config["blocks"]),
                               int(config["spatial_per_block"]))
        return loaded(module, weights)

    def trunk_sequence(self, weights, config, tokens, map_memory, token_tables, map_tables, state,
                       convolution_buffer, done):
        module = self.trunk(weights, config)
        starts = done_to_starts(done.transpose(0, 1)).transpose(0, 1)
        out, _, _ = module(tokens, map_memory, token_tables, map_tables, state,
                           convolution_buffer, starts)
        return out

    def trunk_step(self, weights, config, tokens, map_memory, token_tables, map_tables, state,
                   convolution_buffer):
        module = self.trunk(weights, config)
        tables = tuple(t.unsqueeze(0) for t in token_tables)
        map_tables_ = tuple(t.unsqueeze(0) for t in map_tables)
        out, state, buffer = module(tokens.unsqueeze(0), map_memory.unsqueeze(0), tables,
                                    map_tables_, state, convolution_buffer)
        return out[0], state, buffer

    # ----------------------------------------------------------------- heads

    def mlp_head(self, weights, inputs):
        hidden, width = weights["first.weight"].shape
        outputs = weights["last.weight"].shape[0]
        return loaded(network.MLPHead(width, hidden, outputs), weights)(inputs)

    def next_state_head(self, weights, inputs, code):
        return codes.code_baseline(code) + self.mlp_head(weights, inputs)

    def critic_values(self, local_logits, outcome_logits, bins, local_levels, outcome_level):
        return targets.critic_values(local_logits, outcome_logits, bins, local_levels,
                                     int(outcome_level))

    # ----------------------------------------------------------------- critic

    def value_bins(self, count, limit):
        return targets.value_bins(int(count), float(limit))

    def two_hot(self, values, bins):
        return targets.two_hot(values, bins)

    def expectation(self, logits, bins):
        return targets.expectation(logits, bins)

    def outcome_values(self, probabilities):
        return targets.outcome_values(probabilities)

    def outcome_class(self, result):
        return targets.outcome_class(result)

    def outcome_targets(self, probabilities, result, terminated, next_probabilities, gamma, lam):
        return targets.outcome_targets(probabilities, result, terminated, next_probabilities,
                                       gamma, lam)

    def return_scales(self, returns, masks, decay, floor):
        scale = targets.ReturnScale(decay, floor)
        values = [scale.update(r, m) for r, m in zip(returns, masks, strict=True)]
        return torch.tensor(values, dtype=returns.dtype)

    # ----------------------------------------------------------------- codes

    def circular_encode(self, period, colours, levels, mean, sigma):
        return codes.circular_encode(period, int(colours), int(levels), mean, sigma)

    def circular_decode(self, period, colours, levels, probabilities):
        return codes.circular_decode(period, int(colours), int(levels), probabilities)

    def position_encode(self, size, levels, mean, sigma):
        return codes.position_encode(size, int(levels), mean, sigma)

    def position_decode(self, size, levels, probabilities):
        return codes.position_decode(size, int(levels), probabilities)

    def scalar_sharp(self, low, high, bins, values):
        return codes.scalar_sharp(low, high, int(bins), values)

    def scalar_encode(self, low, high, bins, mean, sigma):
        return codes.scalar_encode(low, high, int(bins), mean, sigma)

    def scalar_decode(self, low, high, bins, probabilities):
        return codes.scalar_decode(low, high, int(bins), probabilities)

    def compress_velocity(self, velocity, knee):
        return codes.compress_velocity(velocity, knee)

    def expand_velocity(self, compressed, knee):
        return codes.expand_velocity(compressed, knee)

    def velocity_jacobian(self, velocity, knee):
        return codes.velocity_jacobian(velocity, knee)

    def velocity_sharp(self, bins, spacing, knee, velocity):
        return codes.velocity_sharp(int(bins), spacing, knee, velocity)

    def velocity_encode(self, bins, spacing, knee, mean, covariance):
        return codes.velocity_encode(int(bins), spacing, knee, mean, covariance)

    def velocity_decode(self, bins, spacing, knee, probabilities):
        return codes.velocity_decode(int(bins), spacing, knee, probabilities)

    def grouped_log_softmax(self, logits, runs):
        return codes.grouped_log_softmax(logits, runs)

    def grouped_cross_entropy(self, logits, target, runs):
        return codes.grouped_cross_entropy(logits, target, runs)

    def remove_baseline_floor(self, logits, runs, floor):
        return codes.remove_baseline_floor(logits, runs, floor)

    # ----------------------------------------------------------------- density

    def hex_cells(self, rings):
        return targets.hex_cells(int(rings))

    def hex_counts(self, x, y, is_ally, alive, arena_radius, rings):
        return targets.hex_counts(x, y, is_ally, alive, arena_radius, int(rings))

    def poisson_terms(self, log_rate, counts):
        return targets.poisson_terms(log_rate, counts)

    # ----------------------------------------------------------------- shots, features

    def shot_trajectory_code(self, parameters, snapshots):
        return targets.shot_trajectory_code(parameters, snapshots)

    def shot_outcome_code(self, parameters, hit, distance, side):
        return targets.shot_outcome_code(parameters, hit, distance, side)

    def shot_time_features(self, ages, max_age, harmonics):
        return targets.shot_time_features(ages, max_age, int(harmonics))

    def symlog(self, values):
        return targets.symlog(values)

    def respawn_features(self, seconds, horizon, harmonics):
        return targets.respawn_features(seconds, horizon, int(harmonics))

    def local_presence(self, position, team, source, radius):
        return targets.local_presence(position, team, source, radius)

    # ----------------------------------------------------------------- belief

    def belief_observe(self, belief_state, truth, visible, spawned, tick_seconds):
        valid, age, means, spreads, action = belief.observe(
            belief_state["valid"], belief_state["age"], belief_state["means"],
            belief_state["uncertainty"], belief_state["action"], truth, visible, spawned)
        return {"valid": valid, "age": age, "means": means, "uncertainty": spreads,
                "action": action, "time_since_seen": age.to(truth.dtype) * tick_seconds}

    def belief_advance(self, belief_state, prediction, enemy_action_logits):
        means, spreads, action, nonfinite = belief.advance(
            belief_state["valid"], belief_state["means"], belief_state["uncertainty"],
            belief_state["action"], prediction, enemy_action_logits)
        return {"means": means, "uncertainty": spreads, "action": action,
                "nonfinite": nonfinite.sum()}


def make_arm():
    return Arm()
