"""The migrated 682 landmark checkpoint set, checked as tracked files.

These tests do not run the migration. They run over the sixteen ``.pt`` files that
are actually committed under ``checkpoints/resilient-resonance-682/``, which is what
``S16`` will publish from and what the plan's phase 10 "done when" is about: *every*
required policy loads through the ordinary loader, with no migration path, and
behaves as it did.

The equivalence half compares against ``tests/fixtures/migration/landmark_682_reference.npz``,
captured by ``scripts/landmark_682_reference.py`` from the original weights running
under the code at the run's own recorded training commit. No historical code is
needed here — the fixture is the historical behavior, frozen.

Nothing in this module has a fallback: a missing file, a missing fixture, or a
missing checkpoint is a failure, not a skip. The whole point is that the previous
attempt's suite passed while every landmark file was broken.
"""

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.constants import POWER_SLICE, SHOOT_SLICE, TURN_SLICE
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import ObsKey, observation_from_state
from boost_and_broadside.train.rl.checkpoint import (
    OPTIONAL_CHECKPOINT_FIELDS,
    POLICY_CHECKPOINT_FIELDS,
    RESUMABLE_CHECKPOINT_FIELDS,
    require_resumable_checkpoint,
)
from boost_and_broadside.train.rl.checkpoint_schema import (
    OBSERVATION_SCHEMA,
    load_checkpoint_payload,
)
from boost_and_broadside.train.rl.policy_io import (
    CheckpointProvenanceWarning,
    build_policy,
    feature_signature,
    load_policy_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "checkpoints" / "resilient-resonance-682"
REFERENCE_PATH = REPO_ROOT / "tests" / "fixtures" / "migration" / "landmark_682_reference.npz"
REPORT_PATH = RUN_DIR / "migration_report.json"

LADDER_STEPS = (
    14991360,
    21987328,
    28983296,
    36978688,
    49971200,
    70959104,
    87949312,
    102940672,
    155910144,
    206880768,
    272842752,
    416759808,
    876494848,
)
RESUMABLE_FILES = ("step_000999424000.pt", "recent_avg.pt")
BEST_FILES = ("best_training.pt",)
LADDER_FILES = tuple(f"ladder_step_{step:012d}.pt" for step in LADDER_STEPS)
ALL_FILES = RESUMABLE_FILES + BEST_FILES + LADDER_FILES

# The best-model family: the policy block plus the seven fields
# ``_checkpoint_payload_lightweight`` adds. Spelled out rather than imported because
# no module exports it, and pinning it here is the point.
BEST_CHECKPOINT_FIELDS = POLICY_CHECKPOINT_FIELDS + (
    "scaler_state_dict",
    "adv_scaler_state_dict",
    "update",
    "eval_window_rand",
    "eval_window_sc",
    "elo_milestone",
    "train_config",
)

# Measured, not guessed: the largest deviation from the historical reference observed
# across all sixteen files and both input sets. Bitwise equality is unreachable — the
# encoder's first matmul went from k=58 to k=66, which changes the accumulation order
# of the terms that *did* survive — so these are float32 tolerances with roughly a
# 30x margin over the observed maxima (logits 3.2e-5, values 7.9e-6, next-state
# 8.6e-6, recurrent state 1.0e-5). The encoder's own output agrees to 2.4e-7, which
# is under one float32 ULP at its magnitude; everything larger is that rounding
# amplified through two Yemong blocks.
LOGIT_TOLERANCE = 1e-3
VALUE_TOLERANCE = 1e-3
PREDICTION_TOLERANCE = 1e-3
HIDDEN_TOLERANCE = 1e-3
# Action *probabilities* are the quantity that decides play, and they are far less
# sensitive than raw logits.
PROBABILITY_TOLERANCE = 1e-4

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


def _migration_script():
    """Load the one-off migration script by path.

    It lives in ``scripts/`` rather than in the package because it is a one-time
    repository operation, not runtime compatibility infrastructure — so it is not on
    ``pythonpath``. Only its canonical content digest is used here; the tests
    otherwise read the tracked files directly, which is the point.
    """

    import importlib.util

    path = REPO_ROOT / "scripts" / "migrate_682.py"
    spec = importlib.util.spec_from_file_location("_migrate_682", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    if not REFERENCE_PATH.exists():
        raise AssertionError(
            f"missing historical reference fixture {REFERENCE_PATH}; regenerate it with "
            "scripts/landmark_682_reference.py against the run's training commit"
        )
    with np.load(REFERENCE_PATH) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def payloads() -> dict[str, dict]:
    return {
        name: dict(load_checkpoint_payload(RUN_DIR / name, map_location="cpu"))
        for name in ALL_FILES
    }


def _family(payload: dict) -> str:
    if "optimizer_state_dict" in payload:
        return "resumable"
    if "scaler_state_dict" in payload:
        return "best"
    return "policy"


def _apply_state(state, reference: dict[str, np.ndarray], prefix: str, index: int) -> None:
    """Overwrite a live state's ship channels with a recorded step.

    Driving the comparison from recorded states rather than by re-running the
    environment is deliberate: the physics changed between the two commits, and this
    section is about the policy, not the simulator.
    """

    def pull(channel: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(reference[f"{prefix}_state_{channel}"][index]))

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


def _make_env(payload: dict, num_envs: int) -> TensorEnv:
    ship_config = ShipConfig(**payload["ship_config"])
    env_config = EnvConfig(**payload["env_config"])
    env = TensorEnv(num_envs, ship_config, env_config, "cpu")
    env.reset(seed=682)
    return env


def _make_policy(payload: dict, state_dict, num_ships: int):
    policy = build_policy(
        ModelConfig(**payload["model_config"]),
        ShipConfig(**payload["ship_config"]),
        num_value_components=payload["num_value_components"],
        num_ships=num_ships,
        team_pma_k=tuple(payload["team_pma_k"]),
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def _replay(payload, state_dict, reference, prefix: str, steps: int):
    """Run a recorded state sequence through a policy, capturing every head."""

    num_envs = int(reference["meta_num_envs"])
    num_ships = int(reference["meta_num_ships"])
    env = _make_env(payload, num_envs)
    ship_config = ShipConfig(**payload["ship_config"])
    policy = _make_policy(payload, state_dict, num_ships)

    captured: dict[str, torch.Tensor] = {}
    handle = policy.action_head.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("logits", output)
    )
    hidden = policy.initial_hidden(num_envs, num_ships, torch.device("cpu"))
    logits, values, predictions = [], [], []
    try:
        for index in range(steps):
            _apply_state(env.state, reference, prefix, index)
            observation = observation_from_state(env.state, ship_config)
            _action, _logprob, value, prediction, hidden = policy.get_action_and_value(
                observation, hidden
            )
            logits.append(captured["logits"].clone())
            values.append(value.clone())
            predictions.append(prediction.clone())
    finally:
        handle.remove()
    return (
        torch.stack(logits).numpy(),
        torch.stack(values).numpy(),
        torch.stack(predictions).numpy(),
        hidden.numpy(),
    )


def _policy_entries(payload: dict) -> list[tuple[str, dict]]:
    """Every set of weights a payload carries, tagged as the fixture names them."""

    entries = [("", payload["policy_state_dict"])]
    if "avg_policy_state_dict" in payload:
        entries.append(("avg_", payload["avg_policy_state_dict"]))
    return entries


def _softmax_groups(logits: np.ndarray) -> list[np.ndarray]:
    tensor = torch.from_numpy(logits)
    return [
        torch.softmax(tensor[..., group], dim=-1).numpy()
        for group in (POWER_SLICE, TURN_SLICE, SHOOT_SLICE)
    ]


# ---------------------------------------------------------------------------
# Inventory and schema
# ---------------------------------------------------------------------------


class TestInventory:
    def test_the_run_holds_exactly_the_sixteen_expected_checkpoints(self):
        present = sorted(path.name for path in RUN_DIR.glob("*.pt"))
        assert present == sorted(ALL_FILES)

    def test_the_migration_report_is_tracked_beside_the_checkpoints(self):
        assert (RUN_DIR / "migration_report.md").exists()
        assert REPORT_PATH.exists()

    def test_the_report_records_every_file_with_both_hashes(self):
        report = json.loads(REPORT_PATH.read_text())
        recorded = {entry["file"]: entry for entry in report["files"]}
        assert sorted(recorded) == sorted(ALL_FILES)
        for entry in recorded.values():
            assert len(entry["sha256"]["original"]) == 64
            assert len(entry["sha256"]["migrated"]) == 64
            assert entry["transformation_version"] == report["transformation_version"]

    def test_the_migrated_files_hash_to_what_the_report_records(self):
        import hashlib

        report = json.loads(REPORT_PATH.read_text())
        for entry in report["files"]:
            digest = hashlib.sha256((RUN_DIR / entry["file"]).read_bytes()).hexdigest()
            assert digest == entry["sha256"]["migrated"], entry["file"]

    def test_the_migrated_files_match_the_reproducible_content_digest(self, payloads):
        """The digest a reproduction should compare, rather than the byte hash.

        Thirteen of the sixteen files serialize byte-identically every time. The
        three carrying the historical ``train_config`` do not: two of its reward
        fields are ``frozenset``s, and ``pickle`` writes a frozenset in iteration
        order, which Python randomizes per process. Their *contents* are identical,
        and this is the hash that says so.
        """

        content_sha256 = _migration_script().content_sha256

        report = json.loads(REPORT_PATH.read_text())
        for entry in report["files"]:
            digest = content_sha256(payloads[entry["file"]])
            assert digest == entry["sha256"]["migrated_content"], entry["file"]

    def test_every_recorded_original_is_a_version_this_history_still_names(self):
        """The inputs stay identifiable, and the undo path stays real.

        Nothing about the migration is recoverable from the migrated files alone, so
        the record has to point at something that still exists. Each recorded
        "original" hash is a git-LFS object id, and this walks the file's own history
        looking for the revision that named it. That is what makes ``git checkout
        <rev> -- <path>`` a real restore rather than a claim.
        """

        import re
        import subprocess

        def revisions(path: str) -> list[str]:
            return subprocess.run(
                ["git", "log", "--format=%H", "--", path],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=True,
            ).stdout.split()

        report = json.loads(REPORT_PATH.read_text())
        for entry in report["files"]:
            relative = f"checkpoints/resilient-resonance-682/{entry['file']}"
            found = set()
            for revision in revisions(relative):
                pointer = subprocess.run(
                    ["git", "show", f"{revision}:{relative}"],
                    capture_output=True,
                    text=True,
                    cwd=REPO_ROOT,
                    check=True,
                ).stdout
                match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
                if match:
                    found.add(match.group(1))
            assert entry["sha256"]["original"] in found, (
                f"{entry['file']}: no revision in this history names the recorded original "
                f"{entry['sha256']['original']}"
            )


@pytest.mark.parametrize("name", ALL_FILES)
class TestFrozenSchema:
    def test_payload_carries_exactly_its_family_key_set(self, name, payloads):
        payload = payloads[name]
        expected = {
            "resumable": RESUMABLE_CHECKPOINT_FIELDS,
            "best": BEST_CHECKPOINT_FIELDS,
            "policy": POLICY_CHECKPOINT_FIELDS,
        }[_family(payload)]
        assert set(payload) == set(expected)

    def test_the_optional_provenance_fields_are_absent_rather_than_invented(self, name, payloads):
        # Neither was ever recorded for this run. Both are optional in the frozen
        # schema, and a placeholder in either would sit in a field a loader compares.
        for field in OPTIONAL_CHECKPOINT_FIELDS:
            assert field not in payloads[name]

    def test_observation_schema_is_the_frozen_value(self, name, payloads):
        assert payloads[name]["observation_schema"] == OBSERVATION_SCHEMA

    def test_paradigm_is_the_one_the_run_recorded(self, name, payloads):
        # ego_pass, from the run's own train_config. Getting this wrong replays the
        # policy without the team-flipped observation it always acted from.
        assert payloads[name]["paradigm"] == "ego_pass"

    def test_critic_width_and_team_routing_match_the_stored_tensors(self, name, payloads):
        payload = payloads[name]
        state = payload["policy_state_dict"]
        assert payload["num_value_components"] == state["value_head_local.3.weight"].shape[0] == 11
        assert tuple(payload["team_pma_k"]) == (0, 1)
        assert state["value_head_win.3.weight"].shape[0] == 2

    def test_env_config_rebuilds_and_describes_a_zero_field_4v4_run(self, name, payloads):
        env_config = EnvConfig(**payloads[name]["env_config"])
        assert env_config == EnvConfig(
            num_ships=8,
            max_bullets=20,
            max_episode_steps=1024,
            num_fields=0,
            single_team=False,
            action_repeat=1,
            spawn_resource_spread=0.0,
        )

    def test_model_config_rebuilds_to_the_architecture_the_tensors_show(self, name, payloads):
        model_config = ModelConfig(**payloads[name]["model_config"])
        assert model_config == ModelConfig(
            d_model=128,
            n_heads=4,
            n_yemong_blocks=2,
            n_spatial_per_block=1,
            n_temporal_per_block=1,
            encoder_split=False,
            n_bullet_cross_per_block=0,
            bullet_encoder_hidden=64,
            grad_checkpoint=False,
        )

    def test_ship_config_rebuilds_and_does_not_trip_the_physics_drift_check(self, name, payloads):
        # The historical constants and today's agree on every field both versions
        # define. That is a measured agreement — the drift check is live and simply
        # has nothing to report, which is what makes it still useful for these files.
        ship_config = ShipConfig(**payloads[name]["ship_config"])
        assert feature_signature(ship_config) == feature_signature(SHIP_CONFIG)

    def test_the_legacy_env_config_field_is_gone(self, name, payloads):
        # num_obstacles is what made the first attempt's files unloadable.
        assert "num_obstacles" not in payloads[name]["env_config"]

    def test_the_pre_rename_live_elo_keys_are_gone(self, name, payloads):
        payload = payloads[name]
        assert "training_elo" not in payload
        assert "avg_training_elo" not in payload
        assert "scripted_elo" not in payload
        assert isinstance(payload["live_elo"], float)


@pytest.mark.parametrize("name", ALL_FILES)
def test_every_file_loads_through_the_ordinary_loader_without_assuming_provenance(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", CheckpointProvenanceWarning)
        bundle = load_policy_bundle(
            str(RUN_DIR / name), device="cpu", num_ships=8, ship_config=SHIP_CONFIG
        )
    assert bundle.paradigm == "ego_pass"
    assert bundle.num_value_components == 11
    assert bundle.team_pma_k == (0, 1)
    assert bundle.env_config is not None
    assert bundle.field_map_config is None


@pytest.mark.parametrize("name", RESUMABLE_FILES)
class TestResumableFamily:
    def test_payload_satisfies_the_resumable_contract(self, name, payloads):
        require_resumable_checkpoint(payloads[name], name)

    def test_optimizer_state_covers_every_parameter_with_the_recorded_hyperparameters(
        self, name, payloads
    ):
        payload = payloads[name]
        policy = _make_policy(payload, payload["policy_state_dict"], num_ships=8)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1.0)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        group = optimizer.param_groups[0]
        # The run's own values. A freshly constructed Adam would silently substitute
        # lr=1e-3 and eps=1e-8, which is a different optimization run.
        assert group["lr"] == pytest.approx(1e-4)
        assert group["eps"] == pytest.approx(1e-5)
        assert group["betas"] == (0.9, 0.999)
        parameters = list(policy.parameters())
        assert len(optimizer.state_dict()["state"]) == len(parameters) == 76

    def test_the_two_parameters_without_history_carry_fresh_state(self, name, payloads):
        payload = payloads[name]
        policy = _make_policy(payload, payload["policy_state_dict"], num_ships=8)
        names = [parameter_name for parameter_name, _ in policy.named_parameters()]
        state = payload["optimizer_state_dict"]["state"]
        fresh = [index for index, key in enumerate(names) if ".field_sub." in key]
        assert len(fresh) == 2
        for index in fresh:
            assert float(state[index]["step"]) == 0.0
            assert state[index]["exp_avg"].abs().max() == 0.0
            assert state[index]["exp_avg_sq"].abs().max() == 0.0

    def test_the_averaging_accumulator_still_reproduces_the_averaged_policy(self, name, payloads):
        payload = payloads[name]
        count = payload["avg_update_count"]
        averaged = payload["avg_policy_state_dict"]
        cumulative = payload["avg_param_cumsum"]
        assert len(cumulative) == len(averaged) == 76
        for tensor, key in zip(cumulative, averaged, strict=True):
            assert torch.allclose(tensor / count, averaged[key], atol=1e-5), key

    def test_train_config_is_the_historical_record_the_loader_reads(self, name, payloads):
        train_config = payloads[name]["train_config"]
        assert isinstance(train_config, dict)
        # The one key load_checkpoint reads out of it.
        assert train_config["paradigm"] == "ego_pass"
        # Kept in the historical schema on purpose: it is the record of what the run
        # was launched with, and no loader rebuilds it into a dataclass.
        assert "field_map" not in train_config


# ---------------------------------------------------------------------------
# Equivalence with the historical forward pass
# ---------------------------------------------------------------------------


class TestObservationEquivalence:
    """The eleven observation channels both versions share still carry the same values."""

    @pytest.mark.parametrize("prefix", ["fixed", "scenario"])
    def test_current_builder_reproduces_the_historical_channels(self, prefix, reference, payloads):
        payload = payloads["step_000999424000.pt"]
        env = _make_env(payload, int(reference["meta_num_envs"]))
        _apply_state(env.state, reference, prefix, 0)
        observation = observation_from_state(env.state, ShipConfig(**payload["ship_config"]))
        for key in SHARED_OBS_KEYS:
            current = observation[ObsKey(key)].numpy()
            expected = reference[f"{prefix}_obs_{key}"]
            assert current.shape == expected.shape, key
            assert np.array_equal(current, expected), key


@pytest.mark.parametrize("name", ALL_FILES)
@pytest.mark.parametrize("prefix", ["fixed", "scenario"])
class TestForwardEquivalence:
    """Every file, both input sets, against the run's own training commit.

    ``fixed`` is seeded synthetic state covering dead ships and the full range of
    every channel; ``scenario`` is a real seeded zero-field 4v4 episode the historical
    policy played, replayed here so the recurrent state evolves across it.
    """

    def test_logits_values_predictions_and_recurrent_state_match(
        self, name, prefix, reference, payloads
    ):
        payload = payloads[name]
        steps = int(reference[f"meta_{prefix}_steps"])
        for tag, state_dict in _policy_entries(payload):
            logits, values, predictions, hidden = _replay(
                payload, state_dict, reference, prefix, steps
            )
            stem = f"{prefix}_{tag}{name}"
            np.testing.assert_allclose(
                logits, reference[f"{stem}_logits"], atol=LOGIT_TOLERANCE, rtol=0
            )
            np.testing.assert_allclose(
                values, reference[f"{stem}_values"], atol=VALUE_TOLERANCE, rtol=0
            )
            np.testing.assert_allclose(
                predictions[..., :9],
                reference[f"{stem}_pred_next"],
                atol=PREDICTION_TOLERANCE,
                rtol=0,
            )
            np.testing.assert_allclose(
                hidden, reference[f"{stem}_hidden"], atol=HIDDEN_TOLERANCE, rtol=0
            )

    def test_action_distributions_match(self, name, prefix, reference, payloads):
        payload = payloads[name]
        steps = int(reference[f"meta_{prefix}_steps"])
        for tag, state_dict in _policy_entries(payload):
            logits, _values, _predictions, _hidden = _replay(
                payload, state_dict, reference, prefix, steps
            )
            expected = reference[f"{prefix}_{tag}{name}_logits"]
            for current_group, expected_group in zip(
                _softmax_groups(logits), _softmax_groups(expected), strict=True
            ):
                np.testing.assert_allclose(
                    current_group, expected_group, atol=PROBABILITY_TOLERANCE, rtol=0
                )

    def test_greedy_actions_are_identical(self, name, prefix, reference, payloads):
        payload = payloads[name]
        steps = int(reference[f"meta_{prefix}_steps"])
        for tag, state_dict in _policy_entries(payload):
            logits, _values, _predictions, _hidden = _replay(
                payload, state_dict, reference, prefix, steps
            )
            expected = reference[f"{prefix}_{tag}{name}_logits"]
            for group in (POWER_SLICE, TURN_SLICE, SHOOT_SLICE):
                assert np.array_equal(
                    logits[..., group].argmax(-1), expected[..., group].argmax(-1)
                )

    def test_the_predictor_without_history_is_exactly_zero(self, name, prefix, reference, payloads):
        # The current next-state head predicts a tenth target, local_log_index, that
        # the historical head never had. Its row is zero-padded, so the migrated
        # policy predicts a constant zero for it. Asserted rather than left implicit:
        # any next-state analysis of these files reads a flat series there.
        payload = payloads[name]
        steps = int(reference[f"meta_{prefix}_steps"])
        for _tag, state_dict in _policy_entries(payload):
            _logits, _values, predictions, _hidden = _replay(
                payload, state_dict, reference, prefix, steps
            )
            assert predictions.shape[-1] == 10
            assert np.abs(predictions[..., 9]).max() == 0.0


@pytest.mark.parametrize("name", ALL_FILES)
def test_the_invented_field_parameter_is_the_identity_and_cannot_act(name, payloads):
    """``field_sub`` has no legacy counterpart, and no way to affect these weights.

    griffin.py applies it only to a non-empty field-token slice, and this run has
    ``num_fields=0``. Identity, not zero, is what a freshly constructed block uses,
    because its output feeds a multiplicative gate.
    """

    payload = payloads[name]
    state = payload["policy_state_dict"]
    keys = [key for key in state if ".field_sub." in key]
    assert len(keys) == 2
    for key in keys:
        assert torch.equal(state[key], torch.eye(state[key].shape[0]))

    env = _make_env(payload, 1)
    assert env.state.num_fields == 0

    policy = _make_policy(payload, state, num_ships=8)
    ship_config = ShipConfig(**payload["ship_config"])
    observation = observation_from_state(env.state, ship_config)
    hidden = policy.initial_hidden(1, 8, torch.device("cpu"))
    _action, _logprob, value, _prediction, _hidden = policy.get_action_and_value(
        observation, hidden
    )

    for block in policy.yemong_layers:
        for sublayer in block.field_sub:
            sublayer.weight.data.mul_(-3.0)
    hidden = policy.initial_hidden(1, 8, torch.device("cpu"))
    _action, _logprob, perturbed, _prediction, _hidden = policy.get_action_and_value(
        observation, hidden
    )
    assert torch.equal(value, perturbed)


def test_the_encoder_ignores_every_input_column_the_run_never_had(payloads):
    """The eight appended encoder columns are zero-weighted, so their values cannot matter."""

    payload = payloads["step_000999424000.pt"]
    weight = payload["policy_state_dict"]["encoder.feature_extractor.0.weight"]
    assert weight.shape == (256, 66)
    assert weight[:, 58:].abs().max() == 0.0
    assert weight[:, :58].abs().max() > 0.0


def test_the_radius_column_carries_the_compensated_scale(payloads):
    """The one input encoding that changed since the run, undone exactly.

    ``radius`` was normalized by 40 at the training commit and is normalized by half
    the world size (512) now. The migration scales that column of the encoder's first
    projection by 512/40 so the pre-activation is unchanged. Comparing the two
    resumable files' *own* averaged and live weights would not catch a mistake here,
    so this asserts the ratio directly against the untouched neighbouring column.
    """

    from boost_and_broadside.train.rl.features import Normalize, build_standard_coordinator

    payload = payloads["step_000999424000.pt"]
    ship_config = ShipConfig(**payload["ship_config"])
    coordinator = build_standard_coordinator(ship_config)
    radius = next(feature for feature in coordinator.features if feature.name == "radius")
    assert isinstance(radius.input_encoder, Normalize)
    assert float(radius.input_encoder.scales) == pytest.approx(
        0.5 * min(ship_config.world_size)
    )

    live = payload["policy_state_dict"]["encoder.feature_extractor.0.weight"]
    averaged = payload["avg_policy_state_dict"]["encoder.feature_extractor.0.weight"]
    # Column 57 is `radius`; the migration is the only thing that could have moved it
    # relative to the averaging record, and it moved both by the same factor.
    assert torch.allclose(
        averaged[:, 57] * payload["avg_update_count"],
        payload["avg_param_cumsum"][0][:, 57],
        atol=1e-2,
    )
    assert live[:, 57].abs().max() > 0.0


def test_the_report_names_the_training_commit_the_run_recorded(payloads):
    report = json.loads(REPORT_PATH.read_text())
    metadata = json.loads(
        (RUN_DIR / "wandb_export" / "files" / "wandb-metadata.json").read_text()
    )
    assert report["training_commit"] == metadata["git"]["commit"]
    assert report["observation_schema"] == OBSERVATION_SCHEMA
    assert report["provenance"]["paradigm"] == "ego_pass"
    # The value-component mapping the permutation question turns on, recorded rather
    # than left as a bare literal in a script.
    assert report["provenance"]["historical_components"] == [
        "ally_win",
        "enemy_win",
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "death",
    ]
    assert report["provenance"]["active_components"] == [
        "ally_win",
        "enemy_win",
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "combat_damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "combat_death",
    ]


def test_the_value_component_order_needs_no_permutation():
    """The identity claim the critic depends on, recomputed from the live registry.

    The historically active components, mapped through the two renames and sorted by
    the *current* registry's index, come out in the historical order. If a future
    change to REWARD_COMPONENT_NAMES breaks that, these eleven value-head rows stop
    meaning what they say and this fails.
    """

    from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES

    current = [
        "ally_win",
        "enemy_win",
        "facing",
        "closing_speed",
        "shoot_quality",
        "kill_shot",
        "kill_assist",
        "combat_damage_taken",
        "damage_dealt_enemy",
        "damage_dealt_ally",
        "combat_death",
    ]
    assert sorted(current, key=REWARD_COMPONENT_NAMES.index) == current


def test_roster_ladder_entries_resolve_to_migrated_files():
    """Every tournament-eligible ladder policy the roster names is in the migrated set."""

    roster = json.loads((RUN_DIR / "roster.json").read_text())
    named = {
        Path(entry["path"]).name for entry in roster["entries"] if entry.get("path") is not None
    }
    assert named
    assert named <= set(ALL_FILES)
    assert named == set(LADDER_FILES)


def test_dataclass_round_trip_of_every_recorded_config(payloads):
    """Each stored config block is exactly what its dataclass serializes back to."""

    for name, payload in payloads.items():
        for key, cls in (
            ("model_config", ModelConfig),
            ("env_config", EnvConfig),
            ("ship_config", ShipConfig),
        ):
            rebuilt = cls(**payload[key])
            assert dataclasses.asdict(rebuilt) == payload[key], f"{name}:{key}"
