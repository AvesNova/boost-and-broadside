"""Capture the historical forward pass of the 682 landmark policies.

The migration in ``scripts/migrate_682.py`` claims that padding the encoder's input
projection and renaming the trunk's sublayer keys leaves these weights' behavior
unchanged. That claim is only worth what it is tested against, and the thing to test
against is the code the run actually trained under.

That code is recoverable exactly: the run records its own training commit in
``wandb_export/files/wandb-metadata.json``, and that commit is an ancestor of
``main``. This script runs the *original* weights through the *original* policy,
observation builder, and environment at that commit, and writes the results to a
fixture. ``tests/migration/test_landmark_682.py`` then replays the same inputs
through the migrated checkpoints and compares, with no historical code involved.

Producing the historical checkout::

    git worktree add --detach /tmp/682-historical b4883769ca49bb60e818986586db5673a4bf83c1
    uv run --no-sync python scripts/landmark_682_reference.py \
        --historical-root /tmp/682-historical \
        --source checkpoints/resilient-resonance-682 \
        --out tests/fixtures/migration/landmark_682_reference.npz

The script imports the historical package by putting its ``src`` first on
``sys.path`` and asserts the import resolved there, so it must run in a process that
has not already imported ``boost_and_broadside``. It touches nothing in the
repository except the fixture it writes.

Two input sets are captured:

``fixed``
    Seeded synthetic ship states, deliberately covering dead ships, mixed team
    assignments, and the full range of health, power, cooldown, and previous
    actions — states a short episode would not reach.

``scenario``
    A real seeded zero-field 4v4 episode, played by the historical policy in the
    historical environment. This is the end-to-end confirmation: the recurrent state
    evolves across the whole episode under states the policy itself produced.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

FIXED_SEED = 20260811
SCENARIO_SEED = 682
NUM_ENVS = 2
NUM_SHIPS = 8
FIXED_STEPS = 6
SCENARIO_STEPS = 24

# Kept in step with scripts/migrate_682.py; asserted against it by the tests.
TRAINING_COMMIT = "b4883769ca49bb60e818986586db5673a4bf83c1"

# Ship-state channels that fully determine the eleven observation channels the two
# versions share. Everything else in TensorState is either derived (radius) or
# belongs to machinery the landmark run did not use (bullets, fields, obstacles).
STATE_CHANNELS = (
    "pos_real",
    "pos_imag",
    "vel_real",
    "vel_imag",
    "att_real",
    "att_imag",
    "ang_vel",
    "health",
    "power",
    "cooldown",
    "team_id",
    "alive",
    "prev_action",
)

# The observation channels present in both versions. The current build adds seven
# more; the migrated encoder multiplies all of them by zero.
SHARED_OBS_KEYS = (
    "pos",
    "vel",
    "att",
    "ang_vel",
    "health",
    "power",
    "cooldown",
    "team_id",
    "alive",
    "previous_action",
    "radius",
)


def _import_historical(root: Path):
    """Import the package at the training commit, and prove that is what we got."""

    if "boost_and_broadside" in sys.modules:
        raise RuntimeError(
            "boost_and_broadside is already imported; run this script in a fresh process"
        )
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    import boost_and_broadside

    resolved = Path(boost_and_broadside.__file__).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise RuntimeError(f"historical import resolved to {resolved}, not under {root}")
    return boost_and_broadside


def build_fixed_states(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Seeded synthetic ship states, deliberately spanning the awkward cases."""

    shape = (FIXED_STEPS, NUM_ENVS, NUM_SHIPS)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=shape).astype(np.float32)
    states = {
        "pos_real": rng.uniform(0.0, 1024.0, size=shape).astype(np.float32),
        "pos_imag": rng.uniform(0.0, 1024.0, size=shape).astype(np.float32),
        "vel_real": rng.uniform(-180.0, 180.0, size=shape).astype(np.float32),
        "vel_imag": rng.uniform(-180.0, 180.0, size=shape).astype(np.float32),
        "att_real": np.cos(angle),
        "att_imag": np.sin(angle),
        "ang_vel": rng.uniform(-3.0, 3.0, size=shape).astype(np.float32),
        "health": rng.uniform(0.0, 100.0, size=shape).astype(np.float32),
        "power": rng.uniform(0.0, 100.0, size=shape).astype(np.float32),
        "cooldown": rng.uniform(0.0, 0.1, size=shape).astype(np.float32),
        # Four ships a side, which is the run's own 4v4 matchup.
        "team_id": np.tile(
            np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32), (FIXED_STEPS, NUM_ENVS, 1)
        ),
        "alive": np.ones(shape, dtype=bool),
        "prev_action": np.stack(
            [
                rng.integers(0, 3, size=shape),
                rng.integers(0, 7, size=shape),
                rng.integers(0, 2, size=shape),
            ],
            axis=-1,
        ).astype(np.float32),
    }
    # Kill a spreading set of ships so the alive mask, the TeamPMA key mask, and the
    # encoder's dead-ship handling are all exercised rather than assumed.
    for step in range(FIXED_STEPS):
        for env in range(NUM_ENVS):
            dead = (step + env) % 4
            if dead:
                states["alive"][step, env, :dead] = False
    return states


def _state_arrays_from(state) -> dict[str, np.ndarray]:
    """Read the ship channels back off a live TensorState."""

    return {
        "pos_real": state.ship_pos.real.cpu().numpy(),
        "pos_imag": state.ship_pos.imag.cpu().numpy(),
        "vel_real": state.ship_vel.real.cpu().numpy(),
        "vel_imag": state.ship_vel.imag.cpu().numpy(),
        "att_real": state.ship_attitude.real.cpu().numpy(),
        "att_imag": state.ship_attitude.imag.cpu().numpy(),
        "ang_vel": state.ship_ang_vel.cpu().numpy(),
        "health": state.ship_health.cpu().numpy(),
        "power": state.ship_power.cpu().numpy(),
        "cooldown": state.ship_cooldown.cpu().numpy(),
        "team_id": state.ship_team_id.cpu().numpy(),
        "alive": state.ship_alive.cpu().numpy(),
        "prev_action": state.prev_action.cpu().numpy(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _import_historical(args.historical_root.resolve())

    # Imported after _import_historical, and deliberately not sorted into the
    # module header: every name below resolves to the training commit's package,
    # not this checkout's. `runs.shared` is where the run's ShipConfig lived before
    # profiles moved under src/.
    import torch  # noqa: I001
    from boost_and_broadside.config import EnvConfig, ModelConfig
    from boost_and_broadside.env.env import TensorEnv
    from boost_and_broadside.env.observation import ObsKey, observation_from_state
    from boost_and_broadside.models.mvp.policy import MVPPolicy
    from boost_and_broadside.train.rl.features import build_standard_coordinator
    from runs.shared import SHIP_CONFIG

    torch.use_deterministic_algorithms(True, warn_only=True)

    env_config = EnvConfig(
        num_ships=NUM_SHIPS, max_bullets=20, max_episode_steps=1024, num_obstacles=0
    )

    def make_env() -> TensorEnv:
        env = TensorEnv(NUM_ENVS, SHIP_CONFIG, env_config, "cpu")
        env.reset(seed=SCENARIO_SEED)
        return env

    def make_policy(state_dict) -> MVPPolicy:
        policy = MVPPolicy(
            ModelConfig(d_model=128, n_heads=4, n_transformer_blocks=2),
            build_standard_coordinator(SHIP_CONFIG),
            num_value_components=11,
            num_ships=NUM_SHIPS,
            team_pma_k=(0, 1),
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        policy.requires_grad_(False)
        return policy

    def apply_state(state, arrays: dict[str, np.ndarray], index: int) -> None:
        def pull(key):
            return torch.from_numpy(np.ascontiguousarray(arrays[key][index]))

        state.ship_pos = torch.complex(pull("pos_real"), pull("pos_imag"))
        state.ship_vel = torch.complex(pull("vel_real"), pull("vel_imag"))
        state.ship_attitude = torch.complex(pull("att_real"), pull("att_imag"))
        state.ship_ang_vel = pull("ang_vel")
        state.ship_health = pull("health")
        state.ship_power = pull("power")
        state.ship_cooldown = pull("cooldown")
        state.ship_team_id = pull("team_id").to(torch.int32)
        state.ship_alive = pull("alive").to(torch.bool)
        state.prev_action = pull("prev_action")

    def run_sequence(policy: MVPPolicy, arrays: dict[str, np.ndarray], steps: int):
        """Replay a state sequence, capturing every head's output and the recurrent state."""

        env = make_env()
        captured: dict[str, torch.Tensor] = {}
        handles = [
            policy.action_head.register_forward_hook(
                lambda _m, _i, out: captured.__setitem__("logits", out)
            )
        ]
        hidden = policy.initial_hidden(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
        logits, values, preds = [], [], []
        try:
            for index in range(steps):
                apply_state(env.state, arrays, index)
                obs = observation_from_state(env.state, SHIP_CONFIG)
                _action, _logprob, value, pred_next, hidden = policy.get_action_and_value(
                    obs, hidden
                )
                logits.append(captured["logits"].clone())
                values.append(value.clone())
                preds.append(pred_next.clone())
        finally:
            for handle in handles:
                handle.remove()
        return (
            torch.stack(logits).numpy(),
            torch.stack(values).numpy(),
            torch.stack(preds).numpy(),
            hidden.numpy(),
        )

    def capture_observation(arrays: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
        env = make_env()
        apply_state(env.state, arrays, index)
        obs = observation_from_state(env.state, SHIP_CONFIG)
        return {key: obs[ObsKey(key)].cpu().numpy() for key in SHARED_OBS_KEYS}

    # ------------------------------------------------------------------
    # Input sets
    # ------------------------------------------------------------------
    rng = np.random.default_rng(FIXED_SEED)
    fixed_states = build_fixed_states(rng)

    # The scenario is a real episode: the final policy plays itself in the historical
    # environment from a fixed seed, and every visited state is recorded.
    final = torch.load(
        args.source / "step_000999424000.pt", map_location="cpu", weights_only=False
    )
    scenario_policy = make_policy(final["policy_state_dict"])
    env = make_env()
    torch.manual_seed(SCENARIO_SEED)
    hidden = scenario_policy.initial_hidden(NUM_ENVS, NUM_SHIPS, torch.device("cpu"))
    visited: list[dict[str, np.ndarray]] = []
    for _ in range(SCENARIO_STEPS):
        visited.append(_state_arrays_from(env.state))
        obs = observation_from_state(env.state, SHIP_CONFIG)
        action, _logprob, _value, _pred, hidden = scenario_policy.get_action_and_value(obs, hidden)
        env.step(action)
    scenario_states = {
        key: np.stack([entry[key] for entry in visited]) for key in STATE_CHANNELS
    }

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    payload: dict[str, np.ndarray] = {
        "meta_training_commit": np.array(TRAINING_COMMIT),
        "meta_num_envs": np.array(NUM_ENVS),
        "meta_num_ships": np.array(NUM_SHIPS),
        "meta_fixed_steps": np.array(FIXED_STEPS),
        "meta_scenario_steps": np.array(SCENARIO_STEPS),
    }
    for prefix, arrays in (("fixed", fixed_states), ("scenario", scenario_states)):
        for key, value in arrays.items():
            payload[f"{prefix}_state_{key}"] = value
        for key, value in capture_observation(arrays, 0).items():
            payload[f"{prefix}_obs_{key}"] = value

    files = sorted(path.name for path in args.source.glob("*.pt"))
    payload["meta_files"] = np.array(files)
    for name in files:
        checkpoint = torch.load(args.source / name, map_location="cpu", weights_only=False)
        entries = [("", checkpoint["policy_state_dict"])]
        if "avg_policy_state_dict" in checkpoint:
            entries.append(("avg_", checkpoint["avg_policy_state_dict"]))
        for tag, state_dict in entries:
            policy = make_policy(state_dict)
            for prefix, arrays, steps in (
                ("fixed", fixed_states, FIXED_STEPS),
                ("scenario", scenario_states, SCENARIO_STEPS),
            ):
                logits, values, preds, hidden_out = run_sequence(policy, arrays, steps)
                stem = f"{prefix}_{tag}{name}"
                payload[f"{stem}_logits"] = logits
                payload[f"{stem}_values"] = values
                payload[f"{stem}_pred_next"] = preds
                payload[f"{stem}_hidden"] = hidden_out
        print(f"captured {name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
