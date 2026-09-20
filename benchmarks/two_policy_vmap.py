"""Correctness prototype for vmap-ing two distinct Yemong policy weights.

This is deliberately not wired into interactive inference.  It uses
``stack_module_state`` plus ``functional_call`` on a wrapper whose inputs are
the tensor-dictionary part of :class:`YemongObservation`; dictionaries are a
supported pytree, while the observation dataclass itself is reconstructed
inside the mapped call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.models.yemong import policy as policy_module
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.policy_io import build_policy


@dataclass(frozen=True)
class DeterministicOutputs:
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    prediction: torch.Tensor
    hidden: torch.Tensor


def _argmax_sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling-free stand-in used only to validate the vectorized network."""

    action = torch.stack(
        (
            logits[..., :5].argmax(-1),
            logits[..., 5:10].argmax(-1),
            logits[..., 10:].argmax(-1),
        ),
        dim=-1,
    )
    logprob = torch.log_softmax(logits[..., :5], -1).gather(-1, action[..., 0:1]).squeeze(-1)
    logprob += torch.log_softmax(logits[..., 5:10], -1).gather(-1, action[..., 1:2]).squeeze(-1)
    logprob += torch.log_softmax(logits[..., 10:], -1).gather(-1, action[..., 2:3]).squeeze(-1)
    return action, logprob


@contextmanager
def deterministic_sampling() -> Iterator[None]:
    """Temporarily replace Categorical.sample; never use in production calls."""

    original = policy_module._sample_action
    policy_module._sample_action = _argmax_sample
    try:
        yield
    finally:
        policy_module._sample_action = original


class _PolicyStep(nn.Module):
    """Make ``get_action_and_value`` reachable through functional_call.forward."""

    def __init__(self, policy: YemongPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self, observation_data: dict, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.get_action_and_value(YemongObservation(data=observation_data), hidden)


def vmap_deterministic_step(
    policies: tuple[YemongPolicy, YemongPolicy],
    observations: tuple[YemongObservation, YemongObservation],
    hidden: tuple[torch.Tensor, torch.Tensor],
) -> DeterministicOutputs:
    """Run two same-architecture distinct policies with stacked module state.

    This validates all deterministic network components with argmax sampling,
    which permits exact per-policy output comparisons.
    """

    modules = tuple(_PolicyStep(policy) for policy in policies)
    params, buffers = stack_module_state(modules)
    base = modules[0]
    stacked_hidden = torch.stack(hidden)
    observation_data = _stack_team_observations(observations)

    def call(parameter_state, buffer_state, policy_observation, policy_hidden):
        return functional_call(
            base, (parameter_state, buffer_state), (policy_observation, policy_hidden)
        )

    with deterministic_sampling():
        action, logprob, value, prediction, new_hidden = vmap(
            call, in_dims=(0, 0, 0, 0), randomness="error"
        )(params, buffers, observation_data, stacked_hidden)
    return DeterministicOutputs(action, logprob, value, prediction, new_hidden)


def vmap_sampled_step(
    policies: tuple[YemongPolicy, YemongPolicy],
    observations: tuple[YemongObservation, YemongObservation],
    hidden: tuple[torch.Tensor, torch.Tensor],
) -> DeterministicOutputs:
    """Run normal policy sampling with a distinct vmap RNG stream per policy.

    ``Categorical.sample`` reaches ``aten::multinomial``. Current PyTorch
    supports it under ``randomness='different'`` (verified by the CPU fixture),
    so no semantics-changing external sampling split is necessary for this
    prototype. A future compiled/CUDA path must re-check that support.
    """

    modules = tuple(_PolicyStep(policy) for policy in policies)
    params, buffers = stack_module_state(modules)
    base = modules[0]
    stacked_hidden = torch.stack(hidden)
    observation_data = _stack_team_observations(observations)

    def call(parameter_state, buffer_state, policy_observation, policy_hidden):
        return functional_call(
            base, (parameter_state, buffer_state), (policy_observation, policy_hidden)
        )

    action, logprob, value, prediction, new_hidden = vmap(
        call, in_dims=(0, 0, 0, 0), randomness="different"
    )(params, buffers, observation_data, stacked_hidden)
    return DeterministicOutputs(action, logprob, value, prediction, new_hidden)


def _stack_team_observations(
    observations: tuple[YemongObservation, YemongObservation],
) -> dict:
    """Stack every independent team-view data tensor on the mapped policy axis."""

    first, second = observations
    if first.data.keys() != second.data.keys():
        raise ValueError("team views must expose the same observation keys")
    return {key: torch.stack((first.data[key], second.data[key])) for key in first.data}


def build_frontline_50v50_policies() -> tuple[YemongPolicy, YemongPolicy]:
    """Build the actual 50v50 architecture for an external CUDA experiment."""

    profile = PROFILES["rl"]
    kwargs = dict(
        model_config=profile.model_config,
        ship_config=profile.ship_config,
        num_value_components=12,
        num_ships=100,
        team_pma_k=(0, 1),
    )
    return build_policy(**kwargs), build_policy(**kwargs)


def small_cpu_fixture() -> tuple[
    tuple[YemongPolicy, YemongPolicy],
    tuple[YemongObservation, YemongObservation],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Cheap real-wrapper observation fixture for tests, not a benchmark."""

    ship_config = ShipConfig()
    model_config = ModelConfig(d_model=16, n_heads=2, n_yemong_blocks=1)
    env = YemongEnvWrapper(
        1,
        ship_config,
        EnvConfig(num_ships=2, num_fields=0, max_bullets=1, max_episode_steps=4),
        REWARDS,
        "cpu",
        include_bullets=False,
        perceive_bullets=False,
    )
    root = env.reset()
    observations = (root.for_team(0), root.for_team(1))
    policies = tuple(
        build_policy(model_config, ship_config, num_value_components=2, num_ships=2, team_pma_k=())
        for _ in range(2)
    )
    # Force distinct weights while retaining identical architecture.
    with torch.no_grad():
        next(policies[1].parameters()).add_(0.125)
    hidden = tuple(policy.initial_hidden(1, 2, torch.device("cpu")) for policy in policies)
    return policies, observations, hidden


def _frontline_50v50_inputs(device: torch.device):
    """Real 50v50 views/hidden state for the opt-in CUDA policy-call harness."""

    profile = PROFILES["rl"]
    scale = 7000.0 / 2600.0
    frontline = replace(
        profile.frontline,
        zone_radius=profile.frontline.zone_radius * scale,
        zone_ring_radius=profile.frontline.zone_ring_radius * scale,
        playable_radius=profile.frontline.playable_radius * scale,
    )
    config = EnvConfig(
        num_ships=100,
        num_fields=72,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        action_repeat=profile.action_repeat,
        frontline=frontline,
        vision_range=profile.vision_range * scale,
        zones_occlude=profile.zones_occlude,
    )
    wrapper = YemongEnvWrapper(
        1,
        profile.ship_config,
        config,
        REWARDS,
        device,
        include_bullets=False,
        perceive_bullets=True,
    )
    root = wrapper.reset()
    policies = tuple(policy.to(device).eval() for policy in build_frontline_50v50_policies())
    hidden = tuple(policy.initial_hidden(1, 100, device) for policy in policies)
    return policies, (root.for_team(0), root.for_team(1)), hidden


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int((len(ordered) - 1) * 0.95)],
        "p99_ms": ordered[int((len(ordered) - 1) * 0.99)],
        "max_ms": ordered[-1],
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def _run_arm(
    name: str, device: torch.device, warmup: int, samples: int, compile_mode: str | None
) -> dict:
    """Measure policy calls only; no environment step, renderer, or display."""

    torch.manual_seed(20260920)
    policies, observations, hidden = _frontline_50v50_inputs(device)
    if name == "separate":

        def call():
            return (
                policies[0].get_action_and_value(observations[0], hidden[0]),
                policies[1].get_action_and_value(observations[1], hidden[1]),
            )

    else:

        def call():
            return vmap_sampled_step(policies, observations, hidden)

    compile_start = time.perf_counter()
    compiled_call = torch.compile(call, mode=compile_mode) if compile_mode else call
    compile_setup_ms = 1000.0 * (time.perf_counter() - compile_start)
    first_start = time.perf_counter()
    compiled_call()
    _sync(device)
    first_call_ms = 1000.0 * (time.perf_counter() - first_start)
    for _ in range(warmup):
        compiled_call()
    _sync(device)
    timings = []
    for _ in range(samples):
        begin = time.perf_counter()
        compiled_call()
        _sync(device)
        timings.append(1000.0 * (time.perf_counter() - begin))
    return {
        "compile_setup_ms": compile_setup_ms,
        "first_call_ms": first_call_ms,
        "samples_ms": timings,
        **_summary(timings),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        parser.error("two-policy vmap measurement requires CUDA")
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    result = {
        "scope": "policy calls only: no environment step, renderer, snapshot, or display flip",
        "configuration": {
            "ships": 100,
            "teams": "50v50",
            "seed": 20260920,
            "compile_mode": compile_mode,
        },
        "arms": {"separate": [], "vmap": []},
    }
    _save(args.out, result)
    for pair in range(args.pairs):
        order = ("separate", "vmap") if pair % 2 == 0 else ("vmap", "separate")
        for arm in order:
            torch.cuda.reset_peak_memory_stats(device)
            row = _run_arm(arm, device, args.warmup, args.samples, compile_mode)
            row["pair"] = pair
            result["arms"][arm].append(row)
            _save(args.out, result)


if __name__ == "__main__":
    main()
