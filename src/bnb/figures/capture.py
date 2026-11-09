"""The capture: play a rollout with a policy, run its model over the rollout, and store what
it saw, what was true and what each head predicted (decoded by
`bnb.models.yemong.decode`) as one artifact file. Renderers read the artifact only.

The rollout is the training runtime's own (`bnb.runtime.rollout.Rollout`), uncompiled, with
the policy in both seats of every game, fog on, from fresh games. The model's outputs are
its `unroll` over the stored segment, as the learner computes them; the shots are the
learner's re-simulation (`bnb.learn.shots.resimulate`) of the bullet every living ship
would fire on every tick: at zero spread (the flight through the fields alone), at the
learner's random spread, and a fan of further spread samples.

The artifact is a nested dict of tensors and plain values, saved with `torch.save` and read
with `load` (weights only): `meta` (provenance and the game's numbers), `rollout` (the
segment and the heads' decoded predictions) and, when the scenario lab ran, `scenarios`.
"""

import dataclasses
import json
import pathlib
import sys

import torch

from bnb.contracts import Pool
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.bullets import launch
from bnb.learn import labels, shots
from bnb.models.yemong import decode
from bnb.models.yemong import shots as shot_codes
from bnb.models.yemong.density import HexGrid
from bnb.models.yemong.inputs import believed
from bnb.models.yemong.inputs import truth as shifted_truth
from bnb.runs import checkpoint, provenance
from bnb.runs.serialise import to_json
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import model_controller

SEAT = "policy"  # the one controller, in both seats


@dataclasses.dataclass(frozen=True)
class CaptureConfig:
    """preset: the game preset (`bnb.game.modes.rules.PRESETS`); environments: games played
    at once (two views each); ticks: the segment's length; seed: the seed `bnb figures`
    gives the process generators before it captures (the rollout and the spread draw from
    them), and the scenario lab's spread fan; spread_samples: spread draws per shot beyond
    the learner's one (the fan in the shot figure); game_segments: segments of `ticks`
    played on after the first, the same games going on, of which only the critic's record
    is kept (`games`: whole games for the outcome head's reliability)."""

    preset: str = "deathmatch"
    environments: int = 4
    ticks: int = 256
    seed: int = 0
    spread_samples: int = 4
    game_segments: int = 0


def policy_types(record: dict, device: torch.device) -> api.ShipTypes:
    """The ship card a policy was saved with (its record's `ship_card`), else the default."""
    card = record.get("ship_card") or {}
    return api.ship_types(device, **card)


def meta_of(policy: pathlib.Path, record: dict, config: CaptureConfig, model) -> dict:
    """The capture's provenance and the numbers its renderers need."""
    head, dirty = provenance.commit()
    game = PRESETS[config.preset]
    return dict(
        policy=str(policy),
        policy_commit=record.get("commit"),
        environment_steps=int(record.get("steps", {}).get("environment_steps", -1)),
        commit=head,
        dirty=dirty,
        command=" ".join(["bnb", *sys.argv[1:]]),
        time=provenance.now(),
        config=dataclasses.asdict(config),
        game=json.dumps(to_json(game)),
        ships_per_team=game.ships_per_team,
        tick_length=model.tick_length,
        bullet_moves=model.bullet_moves,
        hit_radius=model.shot_code.collision_radius,
        hex_rings=model.config.hex_rings,
    )


@torch.no_grad()
def play(model, types: api.ShipTypes, config: CaptureConfig, device: torch.device):
    """A segment of self-play: (the Rollout, its Trajectory (L, R, ...))."""
    game = PRESETS[config.preset]
    rollout = Rollout(
        game,
        types,
        (Pool("self_play", 1.0, (SEAT, SEAT)),),
        {SEAT: model_controller(model, game)},
        SEAT,
        config.environments,
        config.ticks,
        device,
        compiled=False,
        staggered=False,
    )
    return rollout, rollout.run()


def _in_chunks(function, length: int, size: int = 32) -> dict:
    """function(slice) -> dict of tensors, over [0, length) in chunks, concatenated on the
    first axis (the shot head's queries are 12 times a step's outputs)."""
    parts = [function(slice(start, start + size)) for start in range(0, length, size)]
    return {name: torch.cat([part[name] for part in parts], 0) for name in parts[0]}


@torch.no_grad()
def shot_record(model, rules, types, trajectory, unrolled, config: CaptureConfig) -> dict:
    """The shots of the segment: the launch state, the launch velocity in the launch frame
    and |grad ln n| at the launch point (1/m), the true flight at zero spread and at the
    learner's spread, the spread fan's displacements, the outcome labels, and the head's
    decoded path and outcome."""
    own = trajectory.log_probability.shape[-1]
    capture, fields, done = trajectory.capture, trajectory.obs.fields, trajectory.done
    length = done.shape[0]
    zero = torch.zeros(capture.alive.shape + (2,), device=done.device)
    exact = shots.resimulate(rules, types, capture, fields, done, zero, own)
    draws = [
        shots.resimulate(rules, types, capture, fields, done, torch.randn_like(zero), own)
        for _ in range(1 + config.spread_samples)
    ]
    learner = draws[0]  # one draw, as the learner's labels are
    start = launch(rules, types, capture.launch, zero)
    speed = start.proper_speed / torch.exp(start.log_refractive_index)
    velocity = torch.stack([start.direction_x * speed, start.direction_y * speed], -1)
    heading_cos, heading_sin = capture.launch.heading_cos, capture.launch.heading_sin
    visible = trajectory.obs.ships[..., SHIP["visible"]] > 0.5
    rows = torch.arange(visible.shape[-1], device=visible.device) < own
    ages = torch.arange(1, model.bullet_moves + 1, dtype=torch.float32, device=done.device)

    def predicted(times: slice) -> dict:
        logits = model.shot_trajectory(unrolled.shot_latents[times], ages)
        path = decode.shot_path(model.shot_code, logits)
        return {field.name: getattr(path, field.name) for field in dataclasses.fields(path)}

    outcome = decode.shot_outcome(model.shot_code, unrolled.shot_outcome_logits)
    return dict(
        launch_x=capture.launch.x,
        launch_y=capture.launch.y,
        heading_cos=heading_cos,
        heading_sin=heading_sin,
        launch_velocity=decode.world_to_launch_frame(velocity, heading_cos, heading_sin),
        launch_gradient=torch.hypot(
            capture.launch.log_index_gradient_x, capture.launch.log_index_gradient_y
        ),
        fired=trajectory.obs.ships[..., SHIP["fired"]] > 0.5,
        labelled=learner.valid & (rows | visible),
        valid=exact.valid,
        exact_snapshots=exact.snapshots,
        exact_age_valid=exact.age_valid,
        exact_hit=exact.hit,
        learner_snapshots=learner.snapshots,
        learner_age_valid=learner.age_valid,
        learner_hit=learner.hit,
        learner_distance=learner.distance,
        learner_side=learner.side,
        learner_outcome=shot_codes.outcome_code(
            model.shot_code, learner.hit, learner.distance, learner.side
        ),
        fan=torch.stack([draw.snapshots[..., :2] for draw in draws[1:]]),
        fan_age_valid=torch.stack([draw.age_valid for draw in draws[1:]]),
        predicted=_in_chunks(predicted, length),
        predicted_outcome=outcome,
        outcome_probabilities=torch.softmax(unrolled.shot_outcome_logits.float(), -1),
    )


@torch.no_grad()
def head_record(model, trajectory, unrolled) -> dict:
    """The other heads over the segment, decoded, with the truth they are scored against:
    the privileged rows, the next state, the beliefs, the enemy actions and the density.
    The critic's is `games_record`."""
    own = trajectory.log_probability.shape[-1]
    rows = labels.privileged_rows(trajectory)  # (L, R, N, columns)
    following = labels.next_rows(trajectory, rows)
    offset = trajectory.replay.offset  # (L, R, 2)
    ship_type = rows[..., SHIP["ship_type"]].long()
    position = rows[..., [SHIP["x"], SHIP["y"]]]
    _, _, believed_means, believed_spreads = believed(trajectory.obs.ships, trajectory.replay)
    replay = trajectory.replay
    age = torch.cat([torch.zeros_like(replay.age[..., :own]), replay.age], -1)
    hidden = age > 0
    forecast = decode.next_state(
        model,
        unrolled.next_state_logits,
        believed_means,
        believed_spreads,
        ship_type,
        offset,
        position,
        labels.rates(model, trajectory),
        getattr(unrolled, "correction_logits", None),
        hidden,
        age,
    )
    replay = trajectory.replay
    belief = decode.ship_forecast(
        replay.means.float(), replay.spreads.float(), offset, position[..., own:, :]
    )
    action = decode.action_distribution(unrolled.enemy_action_logits[..., own:, :])
    arena_radius = trajectory.obs.globals[..., GLOBAL["arena_radius"]]
    field = decode.density(unrolled.density_log_rates, model.config.hex_rings, arena_radius)
    grid = HexGrid(model.config.hex_rings, rows.device)
    return dict(
        rows=rows,
        next_rows=following,
        next_state_mask=labels.next_state_mask(trajectory, model),
        next_state=dict(
            position=forecast.position,
            position_sigma=forecast.position_sigma,
            velocity=forecast.velocity,
            velocity_covariance=forecast.velocity_covariance,
            heading=forecast.heading,
            heading_sigma=forecast.heading_sigma,
        ),
        belief=dict(
            valid=replay.valid,
            age=replay.age,
            position=belief.position,
            position_sigma=belief.position_sigma,
            velocity=belief.velocity,
            velocity_covariance=belief.velocity_covariance,
            shifted_truth=shifted_truth(trajectory.truth, offset),
        ),
        enemy_action=dict(
            joint=action.joint,
            taken=to_joint(trajectory.action)[..., own:],
            alive=trajectory.alive[..., own:],
        ),
        density=dict(
            ally=field.ally,
            enemy=field.enemy,
            centres=field.centres,
            cell_radius=field.cell_radius,
            counts=labels.density_targets(grid, trajectory, rows),
        ),
    )


@torch.no_grad()
def rollout_record(model, rules, types, trajectory, config: CaptureConfig) -> dict:
    """Everything the rollout's figure sets read, from one segment."""
    reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
    unrolled = model.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay, reset)
    record = dict(
        ships=trajectory.obs.ships,
        fields=trajectory.obs.fields,
        globals=trajectory.obs.globals,
        alive=trajectory.alive,
        offset=trajectory.replay.offset,
        shots=shot_record(model, rules, types, trajectory, unrolled, config),
    )
    record.update(head_record(model, trajectory, unrolled))
    return record


def capture(
    policy: str | pathlib.Path, config: CaptureConfig, device: torch.device, scenarios=None
) -> dict:
    """The artifact of a policy file: `meta` and `rollout`, and `scenarios` from
    scenarios(model, rules, types, trajectory, config) when given (the scenario lab)."""
    policy = pathlib.Path(policy)
    record = checkpoint.read_policy(policy).raw
    model = checkpoint.load_policy(policy, device).requires_grad_(False).eval()
    types = policy_types(record, device)
    rollout, trajectory = play(model, types, config, device)
    artifact = dict(
        meta=meta_of(policy, record, config, model),
        rollout=rollout_record(model, rollout.rules, types, trajectory, config),
    )
    if scenarios is not None:
        artifact["scenarios"] = scenarios(model, rollout.rules, types, trajectory, config)
    artifact["games"] = games_record(model, rollout, trajectory, config.game_segments)
    return _on_host(artifact)


@torch.no_grad()
def _critic_part(model, trajectory) -> dict:
    reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
    unrolled = model.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay, reset)
    return dict(
        outcome=dataclasses.asdict(decode.outcome(unrolled.outcome_logits)),
        values=unrolled.values.float(),
        done=trajectory.done.clone(),
        result=trajectory.result.clone(),
        alive=trajectory.alive.clone(),
    )


@torch.no_grad()
def games_record(model, rollout, trajectory, segments: int) -> dict:
    """The critic over the first segment and `segments` more of the same games (L', R, ...)
    leading: outcome (`bnb.models.yemong.decode.Outcome` by field), values (S, levels),
    done, result (the view's team's, nonzero where done) and alive (N)."""
    parts = [_critic_part(model, trajectory)]
    for _ in range(segments):
        parts.append(_critic_part(model, rollout.run()))
    return torch.utils._pytree.tree_map(lambda *x: torch.cat(x), *parts)


def _on_host(tree):
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x.detach().cpu(), tree)


def save(artifact: dict, path: str | pathlib.Path) -> pathlib.Path:
    """Write an artifact; returns its path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return path


def load(path: str | pathlib.Path) -> dict:
    """Read an artifact (tensors and plain values only)."""
    return torch.load(pathlib.Path(path), map_location="cpu", weights_only=True)
