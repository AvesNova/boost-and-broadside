"""The position-aliasing probe: does a trained Yemong policy read an enemy that is far away
as one that is near, because the position codes it reads are periodic?

    uv run --no-sync python -m bench.position_aliasing play --out DIR POLICY [--preset deathmatch]
    uv run --no-sync python -m bench.position_aliasing report --out DIR

Nothing is trained. Rotary attention's position periods are `rotary_period / 2^k` (8192 m
down to 64 m), so a displacement of exactly 8192 m on an axis turns every rotary pair by a
whole number of cycles; the ship token's colour code has levels at 4 3^l m (l < 9), so 8748 m
repeats every level but the coarsest (26 244 m), and 26 244 m repeats all nine. Each probe
variant moves one enemy, the same one for a whole game, by a fixed displacement in every
observation the policy reads, and nothing else:

- the policy (the driver) plays the real game, a standard preset, against itself (the
  learner seat is the same policy), and every variant reads the driver's observations with
  the chosen enemy's x, y moved; every variant's previous action is the joint action the
  driver played, and its position offset is the driver's, so the variants differ only in the
  moved enemy's position. The moved enemy is moved from the game's start (the trunk's memory,
  the belief and the motion history all saw it there); its velocity, heading and every other
  column are the real ones.
- local presence (the Gaussian count of ships within ~150 m) reads the real positions in
  every variant (`_presence_unshifted`), so the only inputs the move changes are the
  position codes: the rotary tables and the colour code (and what the belief forecasts from
  them).
- the arena radius input is the same in every variant: the preset's (`--arena actual`) or a
  larger one (`--arena 12500`) given to all of them, B included.
- with `--fog 0` (the default) no game has fog, so the moved enemy is visible whenever it is
  alive; the samples are the ticks where it is alive and visible.

At every `--stride`-th tick each living own ship is a sample: per variant the KL divergence
of its joint action distribution (42-way) from B's in both directions, the total variation
of the joint and of each factor's marginal (power, stick, shoot), the critic's value (summed
over levels) less B's, and the attention weight of its query on the moved enemy's key in
every spatial layer and head (the plain attention path, weights by an explicit softmax). The
model runs in float32 (`--bf16` for the deployed bf16 matmuls). Every segment of play is
appended to DIR/chunk_<n>.npz with a progress line (and ETA) in DIR/progress.log.

`report` pools the chunks: per variant the mean and median of each metric, with standard
errors from a bootstrap over games (each environment's games are its clusters), and by the
focal ship's distance to the moved enemy in B; DIR/report.json and DIR/report.md.
"""

import argparse
import dataclasses
import datetime
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import _pytree

from bnb.contracts import Outputs, Pool
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK, from_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.models.yemong import inputs as yemong_inputs
from bnb.models.yemong import kernels, policy
from bnb.models.yemong.layers import apply_rotary
from bnb.models.yemong.spatial import SpatialBlock
from bnb.models.yemong.state_code import POSITION_LEVELS, POSITION_PERIOD
from bnb.runs import checkpoint
from bnb.runtime.rollout import Rollout

LEARNER, SEAT = "learner", "probe"
# (name, dx, dy) in metres. B is the real position. S: genuine small moves, for scale.
# R: rotary-aliased (every rotary pair an integer number of cycles). C: colour-aliased
# (levels 0-7 at 8748 m, all nine at 26 244 m). K: controls of similar size that alias
# neither closely (chosen by `alias_table`'s cycle distances; see the report).
VARIANTS = (
    ("B", 0.0, 0.0),
    ("S250", 250.0, 0.0),
    ("S1000", 1000.0, 0.0),
    ("S3000", 3000.0, 0.0),
    ("R8192x", 8192.0, 0.0),
    ("R8192y", 0.0, 8192.0),
    ("R8192xy", 8192.0, 8192.0),
    ("C8748x", 8748.0, 0.0),
    ("C8748y", 0.0, 8748.0),
    ("C8748xy", 8748.0, 8748.0),
    ("K6559x", 6559.0, 0.0),
    ("K9911x", 9911.0, 0.0),
    ("K9911y", 0.0, 9911.0),
    ("K10930x", 10930.0, 0.0),
    ("K6559_9911", 6559.0, 9911.0),
    ("C26244x", 26244.0, 0.0),
    ("R24576x", 24576.0, 0.0),
    ("K25500x", 25500.0, 0.0),
)


def _wrapped(shift: float, period: float = 8192.0) -> float:
    """The near twin of a shift: the same rotary phases, within half the coarsest period."""
    return shift - period * round(shift / period)


# The wrapped set: each far shift F with its near twin W (the far shift less a whole number
# of 8192 m per axis, so every rotary phase is the far one's while the colour code differs),
# scored against each other: KL(W || F) near 0 says the far enemy reads as the near twin.
WRAPPED_VARIANTS = (("B", 0.0, 0.0),) + tuple(
    variant
    for name, dx, dy in (
        ("6559x", 6559.0, 0.0),
        ("9911x", 9911.0, 0.0),
        ("10930x", 10930.0, 0.0),
        ("25500x", 25500.0, 0.0),
        ("26244x", 26244.0, 0.0),
        ("8748x", 8748.0, 0.0),
        ("6559_9911", 6559.0, 9911.0),
    )
    for variant in (
        ("F" + name, dx, dy),
        ("W" + name, _wrapped(dx), _wrapped(dy), "F" + name),
    )
)
VARIANT_SETS = {"main": VARIANTS, "wrapped": WRAPPED_VARIANTS}
DISTANCE_BINS = (0.0, 500.0, 1000.0, 2000.0, 1e9)  # m, focal ship to the moved enemy in B


def alias_table(variants, rotary_period: float = 8192.0, octaves: int = 8) -> dict:
    """Per variant axis shift, the cycle distance (0 = aliased, 0.5 = opposite) of every
    rotary band and every colour level."""

    def cycles(shift: float, period: float) -> float:
        fraction = (shift / period) % 1.0
        return round(min(fraction, 1.0 - fraction), 3)

    rotary = [rotary_period / 2**k for k in range(octaves)]
    colour = [POSITION_PERIOD / 3**level for level in reversed(range(POSITION_LEVELS))]
    shifts = sorted({abs(v) for _, dx, dy, *_ in variants for v in (dx, dy) if v})
    return {
        "rotary_periods": rotary,
        "colour_periods": colour,
        "shifts": {
            str(s): {
                "rotary": [cycles(s, p) for p in rotary],
                "colour": [cycles(s, p) for p in colour],
            }
            for s in shifts
        },
    }


# ---------------------------------------------------------------- instrumentation


class Recorder:
    """What the patched model paths read and write while `active`: the per-row shift
    (rows, N, 2) the presence undoes, and every spatial layer's attention weights."""

    active = False
    shift: torch.Tensor | None = None
    weights: list = []


_original_presence = yemong_inputs.local_presence


def _presence_unshifted(position, team, source, radius):
    """local_presence on the real positions: the variants' shift taken off first."""
    if Recorder.active and Recorder.shift is not None and position.shape == Recorder.shift.shape:
        moved = position - Recorder.shift.to(position.dtype)
        position = torch.where(source[..., None], moved, position)
    return _original_presence(position, team, source, radius)


def _attend_recorded(query, key, value, map_key, map_value, token_tables, map_tables):
    """SpatialBlock._attend with an explicit softmax, its weights recorded while active."""
    query, key = apply_rotary(query, *token_tables), apply_rotary(key, *token_tables)
    map_key = apply_rotary(map_key, *map_tables)
    keys = torch.cat([key, map_key], -3).transpose(-3, -2)  # (..., heads, Q + M, width)
    values = torch.cat([value, map_value], -3).transpose(-3, -2)
    query = query.transpose(-3, -2)
    device = query.device.type
    if torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
        query, keys, values = query.to(dtype), keys.to(dtype), values.to(dtype)
    scores = (query @ keys.transpose(-1, -2)).float() / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, -1)
    if Recorder.active:
        Recorder.weights.append(weights)
    attended = weights.to(values.dtype) @ values
    return attended.transpose(-3, -2)


def instrument() -> None:
    """The plain attention path with recorded weights; presence on the real positions."""
    kernels.switches.attention = False
    SpatialBlock._attend = staticmethod(_attend_recorded)
    yemong_inputs.local_presence = _presence_unshifted


# ---------------------------------------------------------------- the probe seat


def kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return (log_p.exp() * (log_p - log_q)).sum(-1)


class Probe:
    """The probe seat: the driver plays; the variants shadow it (module docstring)."""

    reads = ("ships", "fields", "globals", "zones")

    def __init__(
        self,
        model,
        shifts: torch.Tensor,
        ships_per_team: int,
        num_ships: int,
        arena: float | None,
        stride: int,
        partners: torch.Tensor,
    ) -> None:
        self.model, self.shifts = model, shifts  # (V, 2)
        self.partners = partners  # (V,) the variant each is also scored against (B: 0)
        self.own, self.num_ships = ships_per_team, num_ships
        self.arena, self.stride = arena, stride
        self.tick = 0
        self.samples: list[dict] = []
        self.games: torch.Tensor | None = None
        self.sanity: list[float] = []

    def init_carry(self, views: int, device: torch.device):
        variants = self.shifts.shape[0]
        self.views = views
        self.driver = self.model.initial_carry(views, self.own, self.num_ships, device)
        self.variants = self.model.initial_carry(views * variants, self.own, self.num_ships, device)
        self.variants = dataclasses.replace(
            self.variants, offset=self.driver.offset.repeat(variants, 1)
        )
        # the moved enemy's row per environment: a fixed enemy for the whole run
        self.moved = self.own + torch.arange(views, device=device) % (self.num_ships - self.own)
        self.games = torch.zeros(views, dtype=torch.int64, device=device)
        return None

    def _shifted(self, obs):
        """The variants' observations, variant-major (V R, ...), and the per-row shift."""
        variants, views = self.shifts.shape[0], self.views
        repeated = _pytree.tree_map_only(
            torch.Tensor, lambda x: x.repeat((variants,) + (1,) * (x.dim() - 1)), obs
        )
        shift = torch.zeros(variants, views, self.num_ships, 2, device=obs.ships.device)
        rows = torch.arange(views, device=shift.device)
        shift[:, rows, self.moved, :] = self.shifts[:, None, :].to(shift.dtype)
        shift = shift.flatten(0, 1)
        ships = repeated.ships.clone()
        ships[..., SHIP["x"]] += shift[..., 0]
        ships[..., SHIP["y"]] += shift[..., 1]
        globals_ = repeated.globals
        if self.arena is not None:
            globals_ = globals_.clone()
            globals_[..., GLOBAL["arena_radius"]] = self.arena
        return dataclasses.replace(repeated, ships=ships, globals=globals_), shift

    @torch.no_grad()
    def step(self, carry, obs, reset):
        tick, self.tick = self.tick, self.tick + 1
        model, variants = self.model, self.shifts.shape[0]
        decision = model.step(self.driver, obs, reset)
        joint, _, _ = policy.sample(decision.policy_logits)
        self.games += reset.long()
        # the variants: the reset applied here, the driver's offset, no reset in step
        cleared = model._cleared(self.variants, reset.repeat(variants))
        cleared = dataclasses.replace(cleared, offset=decision.carry.offset.repeat(variants, 1))
        shifted, shift = self._shifted(obs)
        Recorder.active, Recorder.shift, Recorder.weights = True, shift, []
        try:
            out = model.step(cleared, shifted, torch.zeros_like(reset).repeat(variants))
        finally:
            Recorder.active, Recorder.shift = False, None
        weights, Recorder.weights = Recorder.weights, []
        self.variants = dataclasses.replace(out.carry, previous_action=joint.repeat(variants, 1))
        self.driver = dataclasses.replace(decision.carry, previous_action=joint)
        if tick % self.stride == 0 and tick > 0:
            self._record(tick, obs, reset, decision, out, weights)
        return from_joint(joint), None, Outputs()

    def _record(self, tick, obs, reset, decision, out, weights) -> None:
        variants, views, own = self.shifts.shape[0], self.views, self.own
        rows = torch.arange(views, device=obs.ships.device)
        ships = obs.ships
        moved = ships[rows, self.moved]  # (R, columns)
        focal_alive = ships[:, :own, SHIP["alive"]] > 0.5
        moved_ok = (moved[:, SHIP["alive"]] > 0.5) & (moved[:, SHIP["visible"]] > 0.5)
        mask = focal_alive & moved_ok[:, None] & ~reset[:, None]  # (R, S)
        if not bool(mask.any()):
            return
        delta = ships[:, :own, [SHIP["x"], SHIP["y"]]] - moved[:, None, [SHIP["x"], SHIP["y"]]]
        distance = torch.linalg.vector_norm(delta, dim=-1)  # (R, S)

        logits = out.policy_logits.float().unflatten(0, (variants, views))  # (V, R, S, 42)
        log_p = F.log_softmax(logits, -1)
        base = log_p[0:1]
        kl_from_base = kl(base.expand_as(log_p), log_p)  # KL(B || v)
        kl_to_base = kl(log_p, base.expand_as(log_p))  # KL(v || B)
        kl_to_partner = kl(log_p, log_p[self.partners])  # KL(v || its partner)
        p = log_p.exp()
        tv_joint = 0.5 * (p - p[0:1]).abs().sum(-1)
        factors = p.unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
        marginals = (factors.sum((-1, -2)), factors.sum((-1, -3)), factors.sum((-2, -3)))
        tv_factors = torch.stack(
            [0.5 * (m - m[0:1]).abs().sum(-1) for m in marginals], -1
        )  # (V, R, S, 3)
        value = out.values.float().sum(-1).unflatten(0, (variants, views))  # (V, R, S)
        driver_log = F.log_softmax(decision.policy_logits.float(), -1)
        sanity = kl(driver_log, log_p[0])  # B against the driver: 0 up to rounding (--arena actual)
        # attention: per layer (V R, heads, Q, K) -> the focal query's weight on the moved key
        layers = []
        for w in weights:
            w = w.unflatten(0, (variants, views))  # (V, R, H, Q, K)
            index = self.moved.view(1, views, 1, 1, 1).expand(variants, views, w.shape[2], own, 1)
            on_moved = w[..., :own, :].gather(-1, index)[..., 0]  # (V, R, H, S)
            layers.append(on_moved)
        attention = torch.stack(layers, -1).permute(0, 1, 3, 4, 2)  # (V, R, S, L, H)

        r_index, s_index = mask.nonzero(as_tuple=True)

        def take(tensor):  # (V, R, S, ...) -> (samples, V, ...)
            return tensor[:, r_index, s_index].transpose(0, 1).cpu().numpy()

        self.samples.append(
            {
                "tick": np.full(r_index.shape[0], tick, np.int64),
                "environment": r_index.cpu().numpy(),
                "game": self.games[r_index].cpu().numpy(),
                "distance": distance[r_index, s_index].cpu().numpy(),
                "entropy": (-(p[0] * log_p[0]).sum(-1))[r_index, s_index].cpu().numpy(),
                "kl_from_base": take(kl_from_base),
                "kl_to_base": take(kl_to_base),
                "kl_to_partner": take(kl_to_partner),
                "tv_joint": take(tv_joint),
                "tv_factors": take(tv_factors),
                "value_change": take(value - value[0:1]),
                "attention": take(attention).astype(np.float32),
                "sanity_kl": sanity[r_index, s_index].cpu().numpy(),
            }
        )

    def flush(self) -> dict | None:
        if not self.samples:
            return None
        merged = {k: np.concatenate([s[k] for s in self.samples]) for k in self.samples[0]}
        self.samples = []
        return merged


def play(args) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    instrument()
    game = PRESETS[args.preset]
    model = checkpoint.load_policy(args.policy, device).requires_grad_(False).eval()
    learner = checkpoint.load_policy(args.policy, device).requires_grad_(False).eval()
    if not args.bf16:
        for m in (model, learner):
            m.config = dataclasses.replace(m.config, bfloat16_matmuls=False)
    variants = VARIANT_SETS[args.variants]
    names = [v[0] for v in variants]
    shifts = torch.tensor([[v[1], v[2]] for v in variants], device=device)
    partners = torch.tensor(
        [names.index(v[3]) if len(v) > 3 else 0 for v in variants], device=device
    )
    arena = None if args.arena == "actual" else float(args.arena)
    seat = Probe(model, shifts, game.ships_per_team, game.num_ships, arena, args.stride, partners)
    length = 30
    fog = round(args.fog * args.environments) / args.environments
    rollout = Rollout(
        game,
        api.ship_types(device),
        (Pool("game", 1.0, (LEARNER, SEAT), fog),),
        {LEARNER: YemongController(learner, game.ships_per_team, game.num_ships), SEAT: seat},
        LEARNER,
        args.environments,
        length,
        device,
        compiled=False,
        staggered=False,
    )
    meta = {
        "policy": str(args.policy),
        "preset": args.preset,
        "arena": args.arena,
        "fog": fog,
        "environments": args.environments,
        "seconds": args.seconds,
        "stride": args.stride,
        "bf16": args.bf16,
        "seed": args.seed,
        "variant_set": args.variants,
        "variants": [list(v) for v in variants],
        "aliasing": alias_table(variants, model.config.rotary_period, model.config.rotary_octaves),
        "layers": "block 0 spatial 0, block 0 spatial 1, block 1 spatial 0, block 1 spatial 1",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    segments = -(-round(args.seconds / api.rules_of(game).tick_length) // length)
    started = time.perf_counter()
    log = out / "progress.log"
    for segment in range(segments):
        path = out / f"chunk_{segment:04d}.npz"
        rollout.run()
        merged = seat.flush()
        if merged is not None:
            np.savez_compressed(path, **merged)
        elapsed = time.perf_counter() - started
        eta = datetime.datetime.now() + datetime.timedelta(
            seconds=elapsed / (segment + 1) * (segments - segment - 1)
        )
        line = f"{datetime.datetime.now():%H:%M:%S} segment {segment + 1}/{segments}"
        if merged is not None:
            kls = merged["kl_to_base"].mean(0)
            line += (
                f" samples {merged['distance'].shape[0]} sanity {merged['sanity_kl'].mean():.2e} "
            )
            line += " ".join(
                f"{n} {k:.3f}" for n, k in zip(names, kls, strict=True) if n[0] in "RCKFW"
            )
        line += f"; ETA {eta:%H:%M}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- report


def _load(out: pathlib.Path) -> dict:
    chunks = sorted(out.glob("chunk_*.npz"))
    parts = [dict(np.load(c)) for c in chunks]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _bootstrap(values: np.ndarray, clusters: np.ndarray, draws: int, rng, stat=np.mean):
    """(statistic, standard error) of values (n, ...) resampling whole clusters."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    estimate = stat(values, axis=0)
    order = np.argsort(inverse)
    sorted_values = values[order]
    counts = np.bincount(inverse, minlength=unique.size)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, unique.size, unique.size)
        index = np.concatenate([np.arange(starts[c], starts[c] + counts[c]) for c in pick])
        samples.append(stat(sorted_values[index], axis=0))
    return estimate, np.std(np.stack(samples), axis=0)


def summarise(data: dict, names: list[str], draws: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    clusters = data["environment"] * 100000 + data["game"]
    attention = data["attention"]  # (n, V, L, H)
    summary = {"samples": int(data["distance"].shape[0]), "games": int(np.unique(clusters).size)}
    summary["sanity_kl_mean"] = float(data["sanity_kl"].mean())

    def block(select: np.ndarray) -> dict:
        rows = {}
        c = clusters[select]
        metrics = {
            "kl_to_base": data["kl_to_base"][select],
            "kl_from_base": data["kl_from_base"][select],
            **({"kl_to_partner": data["kl_to_partner"][select]} if "kl_to_partner" in data else {}),
            "tv_joint": data["tv_joint"][select],
            "tv_power": data["tv_factors"][select][..., 0],
            "tv_stick": data["tv_factors"][select][..., 1],
            "tv_shoot": data["tv_factors"][select][..., 2],
            "abs_value_change": np.abs(data["value_change"][select]),
        }
        # attention: mean over layers and heads of the weight on the moved enemy, and its
        # ratio to B's (of the means); per layer x head the mean weight
        att = attention[select]
        att_mean = att.mean((-1, -2))  # (n, V)
        per_head = att.reshape(att.shape[0], att.shape[1], -1)  # (n, V, L H)
        estimates = {}
        for key, values in metrics.items():
            mean, error = _bootstrap(values, c, draws, rng)
            median = np.median(values, axis=0)
            estimates[key] = (mean, error, median)
        att_m, att_e = _bootstrap(att_mean, c, draws, rng)
        head_m, head_e = _bootstrap(per_head, c, draws, rng)
        for i, name in enumerate(names):
            rows[name] = {
                key: {"mean": float(m[i]), "stderr": float(e[i]), "median": float(md[i])}
                for key, (m, e, md) in estimates.items()
            }
            rows[name]["attention_on_moved"] = {"mean": float(att_m[i]), "stderr": float(att_e[i])}
            rows[name]["attention_per_layer_head"] = {
                "mean": head_m[i].round(5).tolist(),
                "stderr": head_e[i].round(5).tolist(),
            }
        rows["_n"] = int(select.sum())
        rows["_games"] = int(np.unique(c).size)
        return rows

    everything = np.ones_like(data["distance"], dtype=bool)
    summary["all"] = block(everything)
    summary["by_distance"] = {}
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:], strict=True):
        select = (data["distance"] >= low) & (data["distance"] < high)
        if select.sum() > 50:
            summary["by_distance"][f"{low:g}-{high:g}"] = block(select)
    return summary


def _markdown(meta: dict, summary: dict) -> str:
    variants = meta["variants"]
    names = [v[0] for v in variants]
    partner = {v[0]: (v[3] if len(v) > 3 else "B") for v in variants}
    lines = [
        f"# Position-aliasing probe: {meta['policy']}",
        "",
        f"preset {meta['preset']}, arena input {meta['arena']}, fog {meta['fog']}, "
        f"{summary['samples']} samples from {summary['games']} games; "
        f"sanity KL(driver||B) {summary['sanity_kl_mean']:.2e}",
        "",
    ]

    def table(rows: dict, title: str) -> None:
        lines.extend([f"## {title} (n {rows['_n']}, games {rows['_games']})", ""])
        lines.append(
            "| variant | KL(v‖B) | median | TV joint | TV power | TV stick | TV shoot | "
            "|Δvalue| | attention on moved | ratio to B | partner | KL(v‖partner) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        base = rows["B"]["attention_on_moved"]["mean"]
        for name in names:
            r = rows[name]

            def f(key):
                return f"{r[key]['mean']:.3g} ± {r[key]['stderr']:.2g}"

            att = r["attention_on_moved"]
            lines.append(
                f"| {name} | {f('kl_to_base')} | {r['kl_to_base']['median']:.3g} | "
                f"{f('tv_joint')} | {f('tv_power')} | {f('tv_stick')} | {f('tv_shoot')} | "
                f"{f('abs_value_change')} | {att['mean']:.4f} ± {att['stderr']:.2g} | "
                f"{att['mean'] / base:.2f} | {partner[name]} | "
                + (f"{f('kl_to_partner')} |" if "kl_to_partner" in r else "- |")
            )
        lines.append("")

    table(summary["all"], "All samples")
    for key, rows in summary["by_distance"].items():
        table(rows, f"Focal ship to moved enemy in B: {key} m")
    lines.extend(["## Attention on the moved enemy per layer x head (all samples)", ""])
    heads = len(summary["all"]["B"]["attention_per_layer_head"]["mean"])
    lines.append("| variant | " + " | ".join(f"L{i // 2}H{i % 2}" for i in range(heads)) + " |")
    lines.append("|---|" + "---|" * heads)
    for name in names:
        m = summary["all"][name]["attention_per_layer_head"]["mean"]
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in m) + " |")
    lines.extend(["", "## Cycle distance of each shift (0 aliased, 0.5 opposite)", ""])
    alias = meta["aliasing"]
    lines.append(
        "rotary periods "
        + ", ".join(f"{p:g}" for p in alias["rotary_periods"])
        + "; colour periods "
        + ", ".join(f"{p:g}" for p in alias["colour_periods"])
    )
    lines.append("")
    for shift, row in alias["shifts"].items():
        lines.append(f"- {float(shift):g} m: rotary {row['rotary']}, colour {row['colour']}")
    return "\n".join(lines) + "\n"


def report(args) -> None:
    out = pathlib.Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    data = _load(out)
    summary = summarise(data, [v[0] for v in meta["variants"]], args.draws)
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(_markdown(meta, summary))
    print(_markdown(meta, summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("play")
    run.add_argument("policy", type=pathlib.Path)
    run.add_argument("--out", required=True)
    run.add_argument("--preset", default="deathmatch")
    run.add_argument(
        "--arena", default="actual", help="'actual' or a radius in m for every variant"
    )
    run.add_argument("--fog", type=float, default=0.0)
    run.add_argument("--environments", type=int, default=64)
    run.add_argument("--seconds", type=float, default=120.0)
    run.add_argument("--stride", type=int, default=15)
    run.add_argument("--bf16", action="store_true")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--variants", default="main", choices=sorted(VARIANT_SETS))
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.set_defaults(handler=play)
    summary = commands.add_parser("report")
    summary.add_argument("--out", required=True)
    summary.add_argument("--draws", type=int, default=200)
    summary.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
