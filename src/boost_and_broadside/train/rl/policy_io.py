"""Single construction and loading path for YemongPolicy.

Every policy — the trainer's own, league opponents, ladder snapshots, and the
agents behind the eval modes — is built here. Before this module there were five
construction sites, each assembling the constructor's arguments by hand, so a
submodule added to the policy reached only the sites its author happened to edit.
``build_policy`` derives the feature pipelines from ``ship_config`` instead of
accepting them, which makes "the model's inputs match its config" true by
construction rather than by every caller remembering.

Loading adds provenance on top of that. A checkpoint records the configs it
trained under, and :func:`load_policy_bundle` rebuilds the policy from *those*
rather than from whatever the caller happens to be running — so an old or
differently-shaped checkpoint stays loadable as an opponent or a rating anchor.
"""

import dataclasses
import warnings
from dataclasses import dataclass

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.train.rl.checkpoint_schema import require_observation_schema
from boost_and_broadside.train.rl.features import (
    build_bullet_coordinator,
    build_standard_coordinator,
)

# ShipConfig fields the feature pipelines read. These set the encoders' constants
# — Fourier periods, resource normalizers, index scaling — so two runs that differ
# here have given the same weights differently-scaled inputs, and their weights do
# not mean the same thing. Sourced from every ``ship_config.<field>`` reference in
# train/rl/features.py and env/observation.py; a test pins the list to those two
# modules so a new reference cannot quietly escape the check.
FEATURE_SHIP_CONFIG_FIELDS = (
    "world_size",
    "max_health",
    "max_power",
    "firing_cooldown",
    "field_transition_width_max",
    "field_transition_width_min",
    "field_index_step",
    "field_interface_damage",
    "bullet_damage",
    "bullet_lifetime",
    "collision_radius",
)


# Process-wide escape hatch, set once by the CLI's --allow-config-drift. It can
# only loosen the per-call argument, never tighten it: the flag says "this
# invocation accepts checkpoints from before a physics change", which is a
# property of the session rather than of any one load, and threading it through
# ten mode signatures that do nothing else with it would only hide that.
_CONFIG_DRIFT_ALLOWED = False


def set_config_drift_allowed(allowed: bool = True) -> None:
    """Allow every load in this process to cross a physics change, with warnings."""
    global _CONFIG_DRIFT_ALLOWED
    _CONFIG_DRIFT_ALLOWED = allowed


class CheckpointProvenanceWarning(UserWarning):
    """A checkpoint did not record a config, so the caller's was assumed."""


class ConfigDriftError(ValueError):
    """A checkpoint's feature-relevant ShipConfig differs from the runtime one."""


@dataclass(frozen=True)
class PolicyBundle:
    """A loaded policy together with the configuration it was trained under."""

    policy: YemongPolicy
    model_config: ModelConfig
    ship_config: ShipConfig
    env_config: EnvConfig | None
    num_value_components: int
    team_pma_k: tuple[int, ...]
    global_step: int | None = None
    update: int | None = None
    # "ego_pass" | "shared_pass" — which perspectives these weights ever acted
    # from. Replaying an ego_pass policy as team 1 without mirroring its view
    # measures a policy playing the wrong side of the game.
    paradigm: str = "ego_pass"

    @property
    def reads_bullets(self) -> bool:
        """Whether this policy needs the observation's bullet axis attached."""
        return self.model_config.reads_bullets


def feature_signature(ship_config: ShipConfig) -> dict[str, object]:
    """The ShipConfig values that decide what the policy's weights mean."""
    return {field: getattr(ship_config, field) for field in FEATURE_SHIP_CONFIG_FIELDS}


def build_policy(
    model_config: ModelConfig,
    ship_config: ShipConfig,
    *,
    num_value_components: int,
    num_ships: int,
    team_pma_k: tuple[int, ...] = (),
) -> YemongPolicy:
    """Construct a policy with the feature pipelines its config implies.

    Args:
        model_config:         Architecture.
        ship_config:          Physics constants — the feature pipelines are derived
                              from these, never passed in, so a bullet-reading
                              config always gets its bullet encoder.
        num_value_components: Critic width K.
        num_ships:            N of the environment this policy will *play in*, not
                              the one it trained at. No parameter is sized by ship
                              count; N only locates the ship/field boundary for a
                              split encoder.
        team_pma_k:           Value-component indices routed through TeamPMA.
    """
    return YemongPolicy(
        model_config,
        build_standard_coordinator(ship_config),
        num_value_components=num_value_components,
        num_ships=num_ships,
        team_pma_k=tuple(team_pma_k),
        bullet_coordinator=(
            build_bullet_coordinator(ship_config) if model_config.reads_bullets else None
        ),
    )


def infer_num_value_components(ckpt: dict) -> int:
    """Return the critic width K (number of value components) for a checkpoint.

    Newer checkpoints store this directly under "num_value_components". Older ones
    predate the field, so fall back to reading the final value-head Linear's output
    width straight from the state dict — the same shape introspection every loader
    used before the field existed.
    """
    if "num_value_components" in ckpt:
        return int(ckpt["num_value_components"])
    return ckpt["policy_state_dict"]["value_head_local.3.weight"].shape[0]


def infer_team_pma_k(ckpt: dict, fallback: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Return the win/loss value-component indices for a checkpoint.

    Newer checkpoints store this directly under "team_pma_k". Older ones only carry
    the team_pma weights in the state_dict, so reconstruct the active-component
    ordering from the stored reward weights — the same filter the reward wrapper
    applies at training time. Payloads with neither (early ladder snapshots) fall
    back to the caller's ordering.
    """
    if "team_pma_k" in ckpt:
        return tuple(ckpt["team_pma_k"])
    if "team_pma.seeds" not in ckpt["policy_state_dict"]:
        return ()
    if "train_config" not in ckpt:
        if fallback is None:
            raise ValueError(
                "checkpoint has TeamPMA weights but records neither 'team_pma_k' nor "
                "'train_config'; pass team_pma_k explicitly to load it"
            )
        return tuple(fallback)

    from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES

    reward_weights = ckpt["train_config"]["rewards"]
    active = [
        name for name in REWARD_COMPONENT_NAMES if reward_weights.get(f"{name}_weight", 0.0) != 0.0
    ]
    win_k = tuple(i for i, name in enumerate(active) if name in ("ally_win", "enemy_win"))
    n_win = ckpt["policy_state_dict"]["value_head_win.3.weight"].shape[0]
    if len(win_k) != n_win:
        raise ValueError(
            f"inferred {len(win_k)} win components from checkpoint train_config but "
            f"value_head_win outputs {n_win}."
        )
    return win_k


def _resolve_paradigm(checkpoint: dict) -> str:
    """Which perspectives the weights acted from, defaulting to ego_pass.

    Recorded directly by current payloads; older full checkpoints carry it inside
    train_config, and older ladder snapshots not at all. Every run in this
    project's history has been ego_pass, so that is the fallback.
    """
    if "paradigm" in checkpoint:
        return checkpoint["paradigm"]
    return checkpoint.get("train_config", {}).get("paradigm", "ego_pass")


def _rebuild_config(stored: dict, cls, path: str):
    """Reconstruct a stored dataclass config, naming fields the class no longer has."""
    try:
        return cls(**stored)
    except TypeError as error:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(stored) - known)
        raise ValueError(
            f"checkpoint {path!r} stores {cls.__name__} fields this version does not "
            f"define: {unknown}. It predates a config change and cannot be rebuilt."
        ) from error


def _check_config_drift(
    checkpoint_ship_config: ShipConfig,
    runtime_ship_config: ShipConfig,
    path: str,
    allow_config_drift: bool,
) -> None:
    """Compare the feature-relevant physics constants, loudly."""
    stored = feature_signature(checkpoint_ship_config)
    runtime = feature_signature(runtime_ship_config)
    differing = {k: (stored[k], runtime[k]) for k in stored if stored[k] != runtime[k]}
    if not differing:
        return
    detail = ", ".join(f"{k}: checkpoint={v[0]!r} runtime={v[1]!r}" for k, v in differing.items())
    if not (allow_config_drift or _CONFIG_DRIFT_ALLOWED):
        raise ConfigDriftError(
            f"checkpoint {path!r} trained under different physics constants than the "
            f"current run ({detail}). Its weights were fitted to differently-scaled "
            "inputs. Pass allow_config_drift=True to load it anyway."
        )
    warnings.warn(
        f"loading {path!r} across a physics change ({detail}); the policy will read the "
        "environment through the constants it trained on.",
        CheckpointProvenanceWarning,
        stacklevel=2,
    )


def load_policy_bundle(
    path: str,
    *,
    device: str | torch.device,
    num_ships: int,
    ship_config: ShipConfig,
    model_config: ModelConfig | None = None,
    team_pma_k: tuple[int, ...] | None = None,
    compile_mode: str | None = None,
    allow_config_drift: bool = False,
    freeze: bool = True,
) -> PolicyBundle:
    """Rebuild a policy from a checkpoint under the config it was trained with.

    ``model_config`` and ``ship_config`` are fallbacks, used only for the fields a
    checkpoint does not record — payloads written before provenance existed. When
    one is used, a :class:`CheckpointProvenanceWarning` names it, because an
    assumed physics constant is indistinguishable from a correct one at load time
    and shows up much later as a policy that plays worse than its rating.

    Args:
        path:               A .pt file with "policy_state_dict".
        device:             Target device.
        num_ships:          N of the environment this policy will play in.
        ship_config:        Fallback physics constants, and the runtime constants
                            the checkpoint's own are checked against.
        model_config:       Fallback architecture.
        team_pma_k:         Fallback win-component indices.
        compile_mode:       torch.compile mode; None loads uncompiled. Applied
                            only when the checkpoint's architecture matches
                            ``model_config`` — a roster spanning architectures
                            would otherwise pay a fresh compile per entry, which
                            under max-autotune costs minutes each.
        allow_config_drift: Downgrade a physics mismatch from an error to a warning.
        freeze:             Put the policy in eval mode with gradients off.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    require_observation_schema(checkpoint, path)

    assumed: list[str] = []
    if "model_config" in checkpoint:
        checkpoint_model_config = _rebuild_config(checkpoint["model_config"], ModelConfig, path)
    elif model_config is not None:
        checkpoint_model_config = model_config
        assumed.append("model_config")
    else:
        raise ValueError(f"checkpoint {path!r} records no model_config and none was supplied")

    if "ship_config" in checkpoint:
        checkpoint_ship_config = _rebuild_config(checkpoint["ship_config"], ShipConfig, path)
    else:
        checkpoint_ship_config = ship_config
        assumed.append("ship_config")

    env_config = (
        _rebuild_config(checkpoint["env_config"], EnvConfig, path)
        if "env_config" in checkpoint
        else None
    )
    if assumed:
        warnings.warn(
            f"checkpoint {path!r} records no {' or '.join(assumed)}; assuming the "
            "current run's. It predates checkpoint provenance.",
            CheckpointProvenanceWarning,
            stacklevel=2,
        )

    _check_config_drift(checkpoint_ship_config, ship_config, path, allow_config_drift)

    num_value_components = infer_num_value_components(checkpoint)
    checkpoint_team_pma_k = infer_team_pma_k(checkpoint, team_pma_k)
    policy = build_policy(
        checkpoint_model_config,
        checkpoint_ship_config,
        num_value_components=num_value_components,
        num_ships=num_ships,
        team_pma_k=checkpoint_team_pma_k,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)
    if freeze:
        policy.eval()
        policy.requires_grad_(False)

    # Compiling an architecture the live run does not share buys one graph that
    # nothing else can reuse, so an odd entry in the roster runs eager instead.
    compile_it = compile_mode is not None and (
        model_config is None or checkpoint_model_config == model_config
    )
    return PolicyBundle(
        policy=torch.compile(policy, mode=compile_mode) if compile_it else policy,
        model_config=checkpoint_model_config,
        ship_config=checkpoint_ship_config,
        env_config=env_config,
        num_value_components=num_value_components,
        team_pma_k=checkpoint_team_pma_k,
        global_step=checkpoint.get("global_step"),
        update=checkpoint.get("update"),
        paradigm=_resolve_paradigm(checkpoint),
    )
