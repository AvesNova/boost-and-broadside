"""Is `belief/visible/position_px` temporally misaligned, or leaking teleports?

The next-state head audit left one hypothesis open: that
`_precompute_belief_diagnostics` pairs `rollout_predictions[t]` with
`privileged_targets[t+1]` off by one decision, which would inflate position by
an order of magnitude and leave every other channel looking healthy.

This runs the *production* rollout and the *production* diagnostic with run
747's weights in run 747's environment, then recomputes the same statistic with
one mask swapped at a time. Results land in JSON next to this file.

Usage: uv run --no-sync python <this> [envs] [steps] [warmup] [device]

The warmup matters more than it looks. Frontline ships deploy ~2400 px apart
and a decision is 1/60 s, so a few hundred decisions is a couple of seconds of
game time in which no enemy is ever in sight and every belief cell is empty.
Three thousand decisions is fifty seconds, which is where contact lives.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig  # noqa: E402
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent  # noqa: E402
from boost_and_broadside.config.resolve import LaunchOverrides, resolve_profile  # noqa: E402
from boost_and_broadside.env.observation import ObsKey, YemongObservation  # noqa: E402
from boost_and_broadside.profiles import get_profile  # noqa: E402
from boost_and_broadside.train.rl.ppo import PPOTrainer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "checkpoints" / "bright-forest-747"
OUT = Path(__file__).with_name("belief-diagnostic-alignment-sep2026.json")

NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NUM_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda"


def build() -> PPOTrainer:
    """A trainer on run 747's profile, narrowed to fit one 8 GB GPU."""

    # num_minibatches only has to divide num_envs; the update never runs here.
    profile = replace(get_profile("bc"), num_minibatches=1)
    resolved = resolve_profile(profile, LaunchOverrides(num_envs=NUM_ENVS, num_envs_source="cli"))
    train_config = replace(
        resolved.train_config,
        num_steps=NUM_STEPS,
        rollouts_per_update=1,
        num_minibatches=1,
        total_timesteps=NUM_ENVS * NUM_STEPS,
    )
    trainer = PPOTrainer(
        train_config=train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=DEVICE,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
    )
    checkpoint = sorted(RUN_DIR.glob("step_*.pt"))[-1]
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    trainer.policy.load_state_dict(payload["policy_state_dict"])
    print(f"loaded {checkpoint.relative_to(REPO)}")
    return trainer


def collect(trainer: PPOTrainer, steps: int, *, store: bool) -> None:
    runtime = trainer.probe_runtime
    slots = trainer._prepare_league_slots(runtime.num_recurrent)
    for _ in range(steps):
        (
            runtime.obs,
            runtime.hidden,
            runtime.hidden_t1,
            runtime.action_state,
            _terminated,
        ) = trainer._collect_primary_step(
            obs=runtime.obs,
            beliefs=runtime.beliefs,
            hidden=runtime.hidden,
            hidden_t1=runtime.hidden_t1,
            action_state=runtime.action_state,
            num_envs=runtime.num_envs,
            num_ships=runtime.num_ships,
            num_recurrent=runtime.num_recurrent,
            slots=slots,
            env_stream=None,
            net_stream=None,
        )
        if not store:
            trainer.buffer.ptr = 0


def alignment(trainer: PPOTrainer) -> dict[str, float]:
    """Which instant `obs[t]` and `privileged_targets[t]` each describe.

    The forecast is replaced by the believed target vector itself. Position is
    an *absolute* Fourier-moment prediction, so a zero prediction decodes to the
    origin rather than to no change -- the believed vector is the identity.
    """

    buf, coordinator = trainer.buffer, trainer.coordinator
    T, B, N = buf.num_steps, buf.num_envs, buf.num_ships
    believed = coordinator.get_target_vector(
        YemongObservation(
            data={
                key: (
                    value[:T, :, :N].reshape(T * B, N, *value.shape[3:])
                    if value.dim() > 3
                    else value[:T, :, :N].reshape(T * B, N)
                )
                for key, value in buf.obs.items()
            }
        )
    ).reshape(T, B, N, -1)
    forecast = coordinator.decode_targets(believed)
    world = torch.tensor(trainer.ship_config.world_size, device=trainer.device)
    predicted = torch.cat([forecast["position_x"], forecast["position_y"]], dim=-1)
    # Own ships are never hidden, so their believed state is truth exactly.
    own = buf.obs[ObsKey.VISIBLE][:T, :, :N].bool() & (buf.obs[ObsKey.TEAM_ID][:T, :, :N] == 0)

    out = {}
    for shift in (0, 1, 2):
        truth = coordinator.decode_targets(buf.privileged_targets[shift : shift + T])
        true_pos = torch.cat([truth["position_x"], truth["position_y"]], dim=-1)
        steps = true_pos.shape[0]
        delta = (
            torch.remainder(predicted[:steps] - true_pos + world / 2.0, world) - world / 2.0
        ).norm(dim=-1)
        mask = own[:steps]
        out[f"identity_vs_truth_t{shift}"] = float((delta * mask).sum() / mask.sum().clamp(min=1))
    return out


def bisect(trainer: PPOTrainer) -> dict[str, dict[str, float]]:
    """The production cell, and the same cell with one filter changed."""

    buf, coordinator = trainer.buffer, trainer.coordinator
    T, B, N = buf.num_steps, buf.num_envs, buf.num_ships
    believed = coordinator.get_target_vector(
        YemongObservation(
            data={
                key: (
                    value[:T, :, :N].reshape(T * B, N, *value.shape[3:])
                    if value.dim() > 3
                    else value[:T, :, :N].reshape(T * B, N)
                )
                for key, value in buf.obs.items()
            }
        )
    ).reshape(T, B, N, -1)
    forecast = coordinator.decode_targets(
        coordinator.apply_scaled_predictions(believed, buf.rollout_predictions)
    )
    truth = coordinator.decode_targets(buf.privileged_targets[1:])
    world = torch.tensor(trainer.ship_config.world_size, device=trainer.device)
    error = (
        torch.remainder(
            torch.cat([forecast["position_x"], forecast["position_y"]], dim=-1)
            - torch.cat([truth["position_x"], truth["position_y"]], dim=-1)
            + world / 2.0,
            world,
        )
        - world / 2.0
    ).norm(dim=-1)

    visible = buf.obs[ObsKey.VISIBLE][:T, :, :N].bool()
    valid = buf.obs[ObsKey.BELIEF_VALID][:T, :, :N].bool()
    enemy = buf.obs[ObsKey.TEAM_ID][:T, :, :N] == 1
    contiguous = buf.transition_contiguous
    non_terminal = ~buf.terminated.unsqueeze(-1)
    production = visible & enemy & non_terminal & contiguous
    masks = {
        "production_visible": production,
        "production_hidden": ~visible & valid & enemy & non_terminal & contiguous,
        "visible_without_contiguity_filter": visible & enemy & non_terminal,
        "visible_with_state_team_id": visible
        & (trainer.wrapper.env.state.ship_team_id[:, :N] == 1)
        & non_terminal
        & contiguous,
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        out[name] = {
            "position_px": float((error * mask).sum() / max(count, 1)),
            "tokens": count,
        }
    selected = error[production].float()
    if selected.numel():
        quantiles = torch.quantile(selected, torch.tensor([0.5, 0.9, 0.99, 1.0], device=DEVICE))
        out["production_visible"].update(
            dict(zip(("p50", "p90", "p99", "max"), (round(v, 4) for v in quantiles.tolist())))
        )
    return out


def main() -> None:
    torch.manual_seed(0)
    trainer = build()
    trainer.probe_runtime = trainer._initialize_rollout_runtime()
    print(f"warming up {WARMUP} decisions")
    collect(trainer, WARMUP, store=False)
    trainer.buffer.reset()
    trainer.buffer.store_initial_hidden(trainer.probe_runtime.hidden)
    collect(trainer, NUM_STEPS, store=True)
    runtime = trainer.probe_runtime
    final = trainer.coordinator.get_target_vector(trainer.wrapper.privileged_observation())[
        :, : runtime.num_ships
    ]
    trainer.buffer.store_final_obs(runtime.obs, privileged_targets=final)

    trainer._precompute_ns_labels(trainer.buffer)
    produced = {
        name: float(total / count.clamp(min=1.0))
        for name, (total, count) in trainer.buffer.belief_diagnostics.items()
    }
    result = {
        "conditions": {
            "checkpoint": str(sorted(RUN_DIR.glob("step_*.pt"))[-1].relative_to(REPO)),
            "profile": "bc",
            "envs": NUM_ENVS,
            "steps": NUM_STEPS,
            "warmup": WARMUP,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "production_diagnostic": produced,
        "alignment": alignment(trainer),
        "bisection": bisect(trainer),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["alignment"], indent=2))
    print(json.dumps(result["bisection"], indent=2))
    for name in (
        "belief/visible/position_px",
        "belief/visible/position_beyond_legal_frac",
        "belief/hidden/position_px",
        "belief/hidden/position_beyond_legal_frac",
    ):
        print(f"{name:46s} {produced[name]:10.4f}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
