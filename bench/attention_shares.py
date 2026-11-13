"""Attention shares of a Yemong policy: for each own living ship, the share of its softmax
attention mass that goes to each kind of key (itself, allies, enemies seen and hidden, the
global token at the map centre, the fields, the zones), averaged over heads and ticks, per
spatial layer and over all layers; and the same per layer and head (`shares_by_layer_head`,
with each head's per-query spread `spread_by_layer_head`), with `--per-head` printing the
heads' table.

    CUDA_VISIBLE_DEVICES= uv run --no-sync python -m bench.attention_shares \
        --policy NAME=DIRECTORY [--policy ...] [--preset real_scale_deathmatch ...] \
        [--lineup Yongma Yongma Jujak Jujak Jujak] [--opponent scripted|self] \
        [--environments 8] [--warmup 1] [--segments 4] [--ticks 128] [--per-head] \
        --out FILE.json

The weights are recomputed, not read from the kernel: every spatial sublayer's attention
call (`SpatialBlock._attend`, and the fused `kernels.attention.attend` where it runs) is
wrapped; the wrapper calls the original, then rebuilds the rotated queries and keys
exactly as the plain path does (the rotary tables, the autocast dtype, the benched key
mask), takes softmax(q k^T / sqrt(W)) in float32, and checks that these weights times the
values give the original output (the largest absolute difference, over the largest
output magnitude, is reported per run: about 4e-3 under the policy's bf16 autocast, the
output's own rounding; `--float32` plays without autocast, where it is about 1e-6).
Nothing in the model changes.

Key layout (`bnb.models.yemong.network`): the query tokens are the N ship rows (own S
first, then the enemies) then the global token; the keys are those N + 1 tokens then the
map memory, the M fields then the Z zone slots (an absent zone is one learned token). A
query is one own ship, counted where its row is alive. The enemy keys split by the
viewer's observation: visible, hidden (alive but not seen; the row carries the belief),
dead. The ally keys split alive and dead.

The games: the rollout of training (`bnb.runtime.rollout.Rollout`, uncompiled, fog on,
staggered clocks), the policy against the scripted agent (or itself), each environment
seating the policy on a random side. `--warmup` segments are played before recording, so
the carry (beliefs, trunk memory) is warm. Each (policy, preset) result is appended to the
output as soon as it is done.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import pathlib
import time

import torch

from bnb.contracts import Pool
from bnb.contracts.obs import SHIP, ZONE
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.models.yemong import spatial
from bnb.models.yemong.kernels import attention as attention_kernel
from bnb.models.yemong.layers import apply_rotary
from bnb.runs import checkpoint
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import lineup_indices, model_controller

KINDS = (
    "self",
    "ally_alive",
    "ally_dead",
    "enemy_visible",
    "enemy_hidden",
    "enemy_dead",
    "global",
    "field",
    "zone_present",
    "zone_absent",
)
KIND = {name: index for index, name in enumerate(KINDS)}
SHORT = {
    "self": "self",
    "ally_alive": "ally",
    "ally_dead": "ally†",
    "enemy_visible": "en.vis",
    "enemy_hidden": "en.hid",
    "enemy_dead": "en.†",
    "global": "global",
    "field": "fields",
    "zone_present": "zones",
    "zone_absent": "zone∅",
}
POLICY, OPPONENT = "policy", "opponent"


@dataclasses.dataclass
class Recorder:
    """The running sums: per layer and own ship type, each kind's attention mass and the
    count of living own ship-ticks; per layer and head, each kind's mass and its sum of
    squares over queries (the spread of a head's per-query share); the keys of each kind
    per query (mean count); the recomputation's worst relative error."""

    layers: int
    ship_types: int
    mass: torch.Tensor = None  # (layers, ship_types, kinds) float64
    queries: torch.Tensor = None  # (layers, ship_types) float64
    key_counts: torch.Tensor = None  # (kinds,) float64, summed over layer-0 queries
    head_mass: torch.Tensor = None  # (layers, heads, kinds) float64, set at the first call
    head_square: torch.Tensor = None  # (layers, heads, kinds) float64
    worst_error: float = 0.0
    calls: int = 0
    # the step's context: the policy's observation, set around each of its decisions
    obs: object = None
    own: int = 0
    layer: int = -1
    armed: bool = False  # recording segments (past the warm-up)
    active: bool = False  # inside a decision of the policy

    def __post_init__(self) -> None:
        self.mass = torch.zeros(self.layers, self.ship_types, len(KINDS), dtype=torch.float64)
        self.queries = torch.zeros(self.layers, self.ship_types, dtype=torch.float64)
        self.key_counts = torch.zeros(len(KINDS), dtype=torch.float64)

    def kinds(self, own: int, keys: int) -> torch.Tensor:
        """(R, S, K) int64 each key's kind for each own query, from the observation."""
        ships = self.obs.ships
        views, rows = ships.shape[0], ships.shape[1]
        alive = ships[..., SHIP["alive"]] > 0.5
        visible = ships[..., SHIP["visible"]] > 0.5
        device = ships.device
        ship_kind = torch.empty(views, rows, dtype=torch.long, device=device)
        ally = torch.arange(rows, device=device) < own
        ship_kind[:] = torch.where(
            ally,
            torch.where(alive, KIND["ally_alive"], KIND["ally_dead"]),
            torch.where(
                ~alive,
                KIND["enemy_dead"],
                torch.where(visible, KIND["enemy_visible"], KIND["enemy_hidden"]),
            ),
        )
        present = self.obs.zones[..., ZONE["present"]] > 0.5  # (R, Z)
        fields = keys - rows - 1 - present.shape[-1]
        memory = torch.cat(
            [
                torch.full((views, 1), KIND["global"], dtype=torch.long, device=device),
                torch.full((views, fields), KIND["field"], dtype=torch.long, device=device),
                torch.where(present, KIND["zone_present"], KIND["zone_absent"]),
            ],
            -1,
        )
        per_view = torch.cat([ship_kind, memory], -1)  # (R, K)
        kinds = per_view[:, None, :].expand(views, own, keys).clone()
        index = torch.arange(own, device=device)
        kinds[:, index, index] = KIND["self"]
        return kinds

    def add(self, weights: torch.Tensor, own: int) -> None:
        """weights (R, heads, Q, K) float32 of one spatial layer of the policy's step."""
        ships = self.obs.ships
        keys = weights.shape[-1]
        kinds = self.kinds(own, keys)  # (R, S, K)
        per_query = weights.mean(1)[:, :own, :].double()  # (R, S, K), heads averaged
        alive = (ships[:, :own, SHIP["alive"]] > 0.5).double()  # (R, S)
        ship_type = ships[:, :own, SHIP["ship_type"]].long()  # (R, S)
        mass = torch.zeros(*kinds.shape[:2], len(KINDS), dtype=torch.float64, device=kinds.device)
        mass.scatter_add_(-1, kinds, per_query)
        mass = mass * alive[..., None]
        heads = weights.shape[1]
        if self.head_mass is None:
            self.head_mass = torch.zeros(self.layers, heads, len(KINDS), dtype=torch.float64)
            self.head_square = torch.zeros_like(self.head_mass)
        per_head = weights[:, :, :own, :].double()  # (R, H, S, K)
        head_kinds = kinds[:, None].expand(-1, heads, -1, -1)
        head_mass = torch.zeros(*per_head.shape[:3], len(KINDS), dtype=torch.float64)
        head_mass = head_mass.to(kinds.device).scatter_add_(-1, head_kinds, per_head)
        head_mass = head_mass * alive[:, None, :, None]
        self.head_mass[self.layer] += head_mass.sum((0, 2)).cpu()
        self.head_square[self.layer] += head_mass.square().sum((0, 2)).cpu()
        for type_index in range(self.ship_types):
            chosen = ship_type == type_index
            self.mass[self.layer, type_index] += mass[chosen].sum(0).cpu()
            self.queries[self.layer, type_index] += alive[chosen].sum().cpu()
        if self.layer == 0:
            counts = torch.zeros_like(mass)
            counts.scatter_add_(-1, kinds, torch.ones_like(per_query))
            self.key_counts += (counts * alive[..., None]).sum((0, 1)).cpu()


RECORDER: Recorder | None = None


def explicit_weights(query, key, map_key, token_tables, map_tables, key_mask=None, dtype=None):
    """(..., heads, Q, K) float32 softmax weights, rotated as `_attend` does and cast to
    dtype (the autocast dtype `_attend` casts to; None: no cast) before the product."""
    query, key = apply_rotary(query, *token_tables), apply_rotary(key, *token_tables)
    map_key = apply_rotary(map_key, *map_tables)
    keys = torch.cat([key, map_key], -3).transpose(-3, -2)
    query = query.transpose(-3, -2)
    if dtype is not None:
        query, keys = query.to(dtype), keys.to(dtype)
    scores = (query.float() @ keys.float().transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if key_mask is not None:
        shown = key_mask.new_ones(key_mask.shape[:-1] + (keys.shape[-2] - key_mask.shape[-1],))
        mask = torch.cat([key_mask, shown], -1)[..., None, None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, -1)


def recorded(original):
    """The attention function wrapped: the original's output, with the weights recomputed,
    checked against it and recorded while the recorder is active."""

    def wrapper(query, key, value, map_key, map_value, token_tables, map_tables, *rest):
        attended = original(query, key, value, map_key, map_value, token_tables, map_tables, *rest)
        recorder = RECORDER
        if recorder is None or not recorder.active:
            return attended
        device = query.device.type
        dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else None
        with torch.autocast(device, enabled=False):
            weights = explicit_weights(
                query, key, map_key, token_tables, map_tables, rest[0] if rest else None, dtype
            )
            values = torch.cat([value, map_value], -3).transpose(-3, -2)
            values = (values if dtype is None else values.to(dtype)).float()
            rebuilt = (weights @ values).transpose(-3, -2)
            scale = attended.float().abs().max().clamp_min(1e-6)
            error = ((rebuilt - attended.float()).abs().max() / scale).item()
        recorder.worst_error = max(recorder.worst_error, error)
        recorder.calls += 1
        recorder.add(weights, recorder.own)
        return attended

    return wrapper


@contextlib.contextmanager
def patched():
    """The attention functions wrapped for the block; restored after it."""
    plain, fused = spatial.SpatialBlock._attend, attention_kernel.attend
    spatial.SpatialBlock._attend = staticmethod(recorded(plain))
    attention_kernel.attend = recorded(fused)
    try:
        yield
    finally:
        spatial.SpatialBlock._attend = staticmethod(plain)
        attention_kernel.attend = fused


class RecordingController:
    """The policy's controller, with the recorder active around each of its decisions."""

    def __init__(self, inner, own: int) -> None:
        self.inner, self.own = inner, own
        self.reads = inner.reads

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, carry, obs, reset, *arguments, **keywords):
        recorder = RECORDER
        if recorder is not None and recorder.armed:
            recorder.obs, recorder.own, recorder.active = obs, self.own, True
        try:
            return self.inner.step(carry, obs, reset, *arguments, **keywords)
        finally:
            if recorder is not None:
                recorder.active = False


def layer_hooks(model) -> list:
    """A pre-hook per spatial sublayer setting the recorder's layer index."""
    handles = []
    for index, layer in enumerate(
        m for m in model.modules() if isinstance(m, spatial.SpatialBlock)
    ):

        def hook(module, inputs, index=index):
            if RECORDER is not None:
                RECORDER.layer = index

        handles.append(layer.register_forward_pre_hook(hook))
    return handles


def measure(name, directory, preset, lineup, opponent, arguments, device) -> dict:
    """One policy on one preset: the shares and their provenance."""
    global RECORDER
    game = api.PRESETS[preset]
    record = checkpoint.read_record(directory)
    types = api.ship_types(device, ship_set=game.ship_set, **(record.get("ship_card") or {}))
    model = checkpoint.load_policy(directory, device).eval()
    if arguments.float32:  # the validation run: no bf16 autocast, so the check is tight
        model.config = dataclasses.replace(model.config, bfloat16_matmuls=False)
    S = game.ships_per_team
    spatial_layers = sum(isinstance(m, spatial.SpatialBlock) for m in model.modules())
    names = api.type_names(game.ship_set)
    controllers = {POLICY: RecordingController(model_controller(model, game), S)}
    seats = (POLICY, POLICY)
    if opponent == "scripted":
        controllers[OPPONENT] = Scripted(characteristics(types, api.rules_of(game)), S)
        seats = (POLICY, OPPONENT)
    ship_type = None
    if lineup:
        team = lineup_indices(tuple(lineup), game)
        ship_type = torch.tensor(team + team, dtype=torch.int32).repeat(arguments.environments, 1)
    rollout = Rollout(
        game,
        types,
        (Pool("attention", 1.0, seats),),
        controllers,
        POLICY,
        arguments.environments,
        arguments.ticks,
        device,
        compiled=False,
        staggered=True,
        ship_type=ship_type,
    )
    RECORDER = Recorder(spatial_layers, len(names))
    handles = layer_hooks(model)
    started = time.time()
    try:
        with patched(), torch.no_grad():
            for segment in range(arguments.warmup + arguments.segments):
                RECORDER.armed = segment >= arguments.warmup
                rollout.run()
                print(
                    f"  {name} {preset} segment {segment + 1}/"
                    f"{arguments.warmup + arguments.segments} "
                    f"({time.time() - started:.0f} s)",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    recorder, RECORDER = RECORDER, None
    by_layer = recorder.mass.sum(1) / recorder.queries.sum(1, keepdim=True)  # (layers, kinds)
    overall = recorder.mass.sum((0, 1)) / recorder.queries.sum()
    by_type = recorder.mass.sum(0) / recorder.queries.sum(0)[:, None].clamp_min(1.0)
    ship_ticks = recorder.queries[0].sum().item()
    layer_queries = recorder.queries.sum(1)[:, None, None]  # (layers, 1, 1)
    head_shares = recorder.head_mass / layer_queries
    head_spread = (recorder.head_square / layer_queries - head_shares.square()).clamp_min(0).sqrt()
    return dict(
        policy=name,
        directory=str(directory),
        preset=preset,
        lineup=list(lineup or []),
        opponent=opponent,
        environments=arguments.environments,
        ticks=arguments.ticks * arguments.segments,
        warmup_ticks=arguments.ticks * arguments.warmup,
        own_ship_ticks=ship_ticks,
        kinds=list(KINDS),
        shares=dict(zip(KINDS, overall.tolist())),
        shares_by_layer=[dict(zip(KINDS, row.tolist())) for row in by_layer],
        shares_by_ship_type={
            names[t]: dict(zip(KINDS, by_type[t].tolist()))
            for t in range(len(names))
            if recorder.queries[0, t] > 0
        },
        shares_by_layer_head=[
            [dict(zip(KINDS, head.tolist())) for head in layer] for layer in head_shares
        ],
        spread_by_layer_head=[
            [dict(zip(KINDS, head.tolist())) for head in layer] for layer in head_spread
        ],
        mean_keys_per_query=dict(zip(KINDS, (recorder.key_counts / ship_ticks).tolist())),
        recomputation_worst_relative_error=recorder.worst_error,
        attention_calls=recorder.calls,
        seconds=time.time() - started,
    )


def table(results: list[dict]) -> str:
    """The shares as a text table, a row per policy and preset."""
    header = f"{'policy':<10}{'preset':<26}" + "".join(f"{SHORT[k]:>8}" for k in KINDS)
    lines = [header]
    for result in results:
        lines.append(
            f"{result['policy']:<10}{result['preset']:<26}"
            + "".join(f"{100 * result['shares'][k]:8.1f}" for k in KINDS)
        )
    return "\n".join(lines)


def head_table(results: list[dict]) -> str:
    """The shares per layer and head, in %, a row per policy, preset, layer and head."""
    header = f"{'policy':<10}{'preset':<26}{'L':>3}{'H':>3}" + "".join(
        f"{SHORT[k]:>8}" for k in KINDS
    )
    lines = [header]
    for result in results:
        for layer, heads in enumerate(result["shares_by_layer_head"]):
            for head, shares in enumerate(heads):
                lines.append(
                    f"{result['policy']:<10}{result['preset']:<26}{layer:>3}{head:>3}"
                    + "".join(f"{100 * shares[k]:8.1f}" for k in KINDS)
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", action="append", required=True, help="NAME=DIRECTORY")
    parser.add_argument(
        "--preset",
        action="append",
        default=None,
        help="game presets (default: the three real-scale modes)",
    )
    parser.add_argument(
        "--lineup", nargs="*", default=["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]
    )
    parser.add_argument("--opponent", choices=("scripted", "self"), default="scripted")
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--float32", action="store_true", help="play without bf16 autocast (validation)"
    )
    parser.add_argument(
        "--per-head", action="store_true", help="also print the shares per layer and head"
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    presets = arguments.preset or [
        "real_scale_deathmatch",
        "real_scale_elimination",
        "real_scale_frontline",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for entry in arguments.policy:
        name, _, directory = entry.partition("=")
        for preset in presets:
            torch.manual_seed(arguments.seed)
            result = measure(
                name,
                pathlib.Path(directory),
                preset,
                arguments.lineup,
                arguments.opponent,
                arguments,
                device,
            )
            results.append(result)
            with arguments.out.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(
                f"{name} {preset}: {result['own_ship_ticks']:.0f} own ship-ticks, "
                f"recomputation error {result['recomputation_worst_relative_error']:.2e}",
                flush=True,
            )
    print(table(results))
    if arguments.per_head:
        print(head_table(results))


if __name__ == "__main__":
    main()
