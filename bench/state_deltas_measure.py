"""Measure per-tick ship-state changes from the collected segments
(bench/state_deltas_collect.py; docs/rewrite/sections/state-deltas-brief.md, part 1).

For each player and preset: every series (a quantity's change from a baseline over a
horizon, per component), its statistics, its situation splits and its histogram, from the
segments one at a time. Writes

    <scratch>/histograms/<player>/<preset>/<quantity>.json   log-spaced |change| bins, signed
    <scratch>/stats/<player>/<preset>.json                    statistics and splits
    <scratch>/samples/<player>/<preset>.npz                   a random subsample of every
                                                              1-tick series (for the code study)

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" uv run --no-sync python bench/state_deltas_measure.py
"""

import argparse
import json
import pathlib

import numpy as np

SCRATCH = pathlib.Path("/tmp/bnb-scratch-deltas")
QUANTILES = (50, 90, 99, 99.9)
EDGES = np.logspace(-9, 5, 141)  # |change| bin edges, 10 per decade
SUBSAMPLE = 400_000  # 1-tick samples kept per player and preset for the code study
MAX_SHIELD = (100.0, 160.0)  # the card's, per type (identical for both players)
MAX_ENERGY = (10000.0, 14000.0)
SHIELD_RECHARGE_RATE = 12.0  # HP/s, both types (rules.shield_recharge_rate, scale 1)
EPS32 = float(np.finfo(np.float32).eps)
TRACK_MIN_SPEED = 1.0  # m/s: below it the velocity frame is undefined (NaN)
COMPONENTS_SHARING_SCALE = ("/x", "/y", "_along", "_across")
RESPAWN_TICKS = 30  # "just respawned": within 1 s of the spawn


def wrap(angle):
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def rotate(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def valid_pairs(alive, done, k):
    """(L - k, B, N): alive at every tick t..t+k and no reset after any tick t..t+k-1."""
    dead = np.concatenate([np.zeros((1,) + alive.shape[1:], np.int32), np.cumsum(~alive, 0)])
    resets = np.concatenate([np.zeros((1, done.shape[1]), np.int32), np.cumsum(done, 0)])
    L = alive.shape[0]
    no_death = dead[k + 1 : L + 1] - dead[0 : L - k] == 0
    no_reset = resets[k:L] - resets[0 : L - k] == 0
    return no_death & no_reset[..., None]


def ticks_since_spawn(spawned, alive):
    """(L, B, N) ticks since the ship's last spawn in the segment (large if unseen)."""
    out = np.full(spawned.shape, 10_000, np.int32)
    count = np.full(spawned.shape[1:], 10_000, np.int32)
    for t in range(spawned.shape[0]):
        count = np.where(spawned[t], 0, np.minimum(count + 1, 10_000))
        out[t] = count
    return out


def segment_series(path: pathlib.Path, columns: list[str], dt: float):
    """{name: (values (M,), state magnitude (M,) for float resolution)} and the situation
    masks of the 1-tick pairs, for one segment."""
    data = np.load(path)
    rows = data["rows"].astype(np.float64)
    C = {name: i for i, name in enumerate(columns)}

    def col(name):
        return rows[..., C[name]]

    alive = col("alive") > 0.5
    done = data["done"]
    x, y = col("x"), col("y")
    vx, vy = col("velocity_x"), col("velocity_y")
    heading = np.arctan2(col("heading_sin"), col("heading_cos"))
    omega = col("nose_rate")
    ship_type = col("ship_type").astype(int)
    max_shield = np.asarray(MAX_SHIELD)[ship_type]
    max_energy = np.asarray(MAX_ENERGY)[ship_type]
    shield = col("shield") / max_shield
    energy = col("energy") / max_energy
    wait = col("shield_recharge_wait")
    cooldown = np.round(col("gun_cooldown") / dt)  # whole ticks (the observation is in s)
    log_index = col("log_refractive_index")
    gradient_x, gradient_y = col("log_index_gradient_x"), col("log_index_gradient_y")
    aoa = col("angle_of_attack")

    series, scale = {}, {}

    def add(name, values, magnitude, mask):
        series[name] = values[mask].astype(np.float32)
        if not name.endswith(COMPONENTS_SHARING_SCALE):  # they use the norm's
            scale[name] = magnitude[mask].astype(np.float32)

    def add_vector(prefix, ex, ey, magnitude, mask, nose, track_x, track_y, moving):
        """A 2-D change: world x, y, its norm, and its body-frame components, along and
        across the nose (heading at t) and along and across the velocity at t (NaN where
        the speed at t is below TRACK_MIN_SPEED). Across is positive to the left."""
        add(f"{prefix}/x", ex, magnitude, mask)
        add(f"{prefix}/y", ey, magnitude, mask)
        add(f"{prefix}/norm", np.hypot(ex, ey), magnitude, mask)
        c, s_ = np.cos(nose), np.sin(nose)
        add(f"{prefix}/nose_along", ex * c + ey * s_, magnitude, mask)
        add(f"{prefix}/nose_across", -ex * s_ + ey * c, magnitude, mask)
        add(f"{prefix}/track_along", np.where(moving, ex * track_x + ey * track_y, np.nan),
            magnitude, mask)  # fmt: skip
        add(f"{prefix}/track_across", np.where(moving, -ex * track_y + ey * track_x, np.nan),
            magnitude, mask)  # fmt: skip

    for k in (1, 8, 32):
        mask = valid_pairs(alive, done, k)
        a, b = slice(0, -k), slice(k, None)
        T = k * dt
        dx, dy = x[b] - x[a], y[b] - y[a]
        pos_mag = np.maximum(np.abs(x[a]), np.abs(x[b])) + np.maximum(np.abs(y[a]), np.abs(y[b]))
        # dead reckoning, straight and along the constant-nose-rate arc
        bases = {"persistence": (0.0, 0.0), "dead_reckoning": (vx[a] * T, vy[a] * T)}
        turn = omega[a] * T
        half = 0.5 * turn
        sinc = np.where(np.abs(half) > 1e-9, np.sin(half) / np.where(half == 0, 1, half), 1.0)
        bases["arc"] = rotate(vx[a] * T * sinc, vy[a] * T * sinc, half)
        speed = np.hypot(vx[a], vy[a])
        moving = speed >= TRACK_MIN_SPEED
        safe = np.where(moving, speed, 1.0)
        frame = (heading[a], vx[a] / safe, vy[a] / safe, moving)
        if k == 1:
            bases["trapezoid"] = (0.5 * (vx[a] + vx[b]) * T, 0.5 * (vy[a] + vy[b]) * T)
        for base, (bx, by) in bases.items():
            add_vector(f"position/{base}/{k}", dx - bx, dy - by, pos_mag, mask, *frame)
        vel_mag = np.abs(vx[a]) + np.abs(vy[a])
        add(f"_start/velocity/{k}/x", vx[a], vel_mag, mask)  # the track frame of the code study
        add(f"_start/velocity/{k}/y", vy[a], vel_mag, mask)
        rx, ry = rotate(vx[a], vy[a], turn)
        for base, (bx, by) in {"persistence": (vx[a], vy[a]), "rotated": (rx, ry)}.items():
            add_vector(f"velocity/{base}/{k}", vx[b] - bx, vy[b] - by, vel_mag, mask, *frame)
        pi_mag = np.full_like(heading[a], np.pi)
        add(f"heading/persistence/{k}/angle", wrap(heading[b] - heading[a]), pi_mag, mask)
        add(f"heading/dead_reckoning/{k}/angle", wrap(heading[b] - heading[a] - turn), pi_mag, mask)
        if k == 1:
            trap = 0.5 * (omega[a] + omega[b]) * T
            add("heading/trapezoid/1/angle", wrap(heading[b] - heading[a] - trap), pi_mag, mask)
            mag = lambda q: np.maximum(np.abs(q[a]), np.abs(q[b]))  # noqa: E731
            add("nose_rate/persistence/1/value", omega[b] - omega[a], mag(omega), mask)
            add("angle_of_attack/persistence/1/value", aoa[b] - aoa[a], mag(aoa), mask)
            add("shield_recharge_wait/persistence/1/value", wait[b] - wait[a], mag(wait), mask)
            add(
                "shield_recharge_wait/countdown/1/value",
                wait[b] - np.maximum(wait[a] - dt, 0.0),
                mag(wait),
                mask,
            )
            add("shield/persistence/1/value", shield[b] - shield[a], mag(shield), mask)
            # the deterministic recharge of an unhurt ship (combat.py phase 5)
            rate = SHIELD_RECHARGE_RATE / max_shield[a]
            recharged = np.minimum(1.0, shield[a] + rate * np.clip(dt - wait[a], 0.0, dt))
            add("shield/recharge/1/value", shield[b] - recharged, mag(shield), mask)
            add("energy/persistence/1/value", energy[b] - energy[a], mag(energy), mask)
            add("gun_cooldown/persistence/1/value", cooldown[b] - cooldown[a], mag(cooldown), mask)
            add(
                "gun_cooldown/countdown/1/value",
                cooldown[b] - np.maximum(cooldown[a] - 1.0, 0.0),
                mag(cooldown),
                mask,
            )
            add("log_refractive_index/persistence/1/value", log_index[b] - log_index[a],
                mag(log_index), mask)  # fmt: skip
            predicted = (gradient_x[a] * vx[a] + gradient_y[a] * vy[a]) * T
            add("log_refractive_index/dead_reckoning/1/value",
                log_index[b] - log_index[a] - predicted, mag(log_index), mask)  # fmt: skip
            # situations at t (1-tick pairs)
            since = ticks_since_spawn(col("spawned") > 0.5, alive)
            in_field = (log_index != 0) | (gradient_x != 0) | (gradient_y != 0)
            situations = dict(
                field=(in_field[a] | in_field[b]),
                zone=data["zone"][a],
                boost=data["power"][a] == 1,
                reverse=data["power"][a] == 2,
                coast=data["power"][a] == 0,
                respawned=since[a] < RESPAWN_TICKS,
                hit=data["hit"][a] | data["boundary"][a],
                shot=data["hit"][a],
                boundary=data["boundary"][a],
                fired=col("fired")[b] > 0.5,
                stick_centred=data["stick"][a] == 0,
                stick_full=data["stick"][a] >= 5,  # stick levels +-1.0
                slow=~moving,  # below TRACK_MIN_SPEED: no track frame
            )
            situations = {name: value[mask] for name, value in situations.items()}
            series["_ship_type"] = ship_type[a][mask].astype(np.int8)
            series["_speed"] = np.hypot(vx[a], vy[a])[mask].astype(np.float32)
            series["_heading"] = heading[a][mask].astype(np.float32)
            series["_vx"], series["_vy"] = (
                vx[a][mask].astype(np.float32),
                vy[a][mask].astype(np.float32),
            )
    return series, scale, situations


def quantity_of(name: str) -> str:
    return name.split("/")[0]


def statistics(values: np.ndarray, magnitude: np.ndarray) -> dict:
    finite = np.isfinite(values)
    skipped = float(1.0 - finite.mean()) if values.size else 0.0
    values, magnitude = values[finite], magnitude[finite]
    absolute = np.abs(values.astype(np.float64))
    if absolute.size == 0:
        return dict(n=0, skipped=skipped)
    quantiles = np.percentile(absolute, QUANTILES)
    resolution = 2.0 * EPS32 * np.maximum(magnitude.astype(np.float64), 1e-30)
    return dict(
        n=int(absolute.size),
        skipped=skipped,
        zero=float((absolute == 0).mean()),
        below_float32=float((absolute <= resolution).mean()),
        mean=float(values.astype(np.float64).mean()),
        mean_abs=float(absolute.mean()),
        **{f"p{q:g}": float(v) for q, v in zip(QUANTILES, quantiles)},
        max=float(absolute.max()),
    )


def histogram(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    v = v[np.isfinite(v)]
    positive, _ = np.histogram(v[v > 0], EDGES)
    negative, _ = np.histogram(-v[v < 0], EDGES)
    return dict(
        zero=int((v == 0).sum()),
        below_first_edge=int(((v != 0) & (np.abs(v) < EDGES[0])).sum()),
        above_last_edge=int((np.abs(v) >= EDGES[-1]).sum()),
        positive=positive.tolist(),
        negative=negative.tolist(),
    )


def measure(player: str, preset: str, rng: np.random.Generator) -> None:
    directory = SCRATCH / "segments" / player / preset
    meta = json.loads((directory / "meta.json").read_text())
    collected: dict[str, list] = {}
    scales: dict[str, list] = {}
    situations: dict[str, list] = {}
    paths = sorted(directory.glob("segment_*.npz"))
    resets = sum(int(np.load(path)["done"].sum()) for path in paths)
    for path in paths:
        series, scale, situation = segment_series(path, meta["columns"], meta["tick_length"])
        for name, value in series.items():
            collected.setdefault(name, []).append(value)
        for name, value in scale.items():
            scales.setdefault(name, []).append(value)
        for name, value in situation.items():
            situations.setdefault(name, []).append(value)
        del series, scale, situation
    collected = {name: np.concatenate(v) for name, v in collected.items()}
    scales = {name: np.concatenate(v) for name, v in scales.items()}
    situations = {name: np.concatenate(v) for name, v in situations.items()}
    stats, hists = {}, {}
    for name, values in collected.items():
        if name.startswith("_"):
            continue
        quantity, base, horizon, component = name.split("/")
        magnitude = scales.get(name)
        if magnitude is None:
            magnitude = scales[name.rsplit("/", 1)[0] + "/norm"]
        stats.setdefault(quantity, {}).setdefault(base, {}).setdefault(horizon, {})[component] = (
            statistics(values, magnitude)
        )
        hists.setdefault(quantity, {}).setdefault(f"{base}/{horizon}", {})[component] = histogram(
            values
        )
    # situation splits of each 1-tick series' main component
    splits = {}
    for name, values in collected.items():
        if name.startswith("_") or "/1/" not in name or name.endswith(("/x", "/y")):
            continue
        quantity, base, _, component = name.split("/")
        if component.endswith(("_along", "_across")):
            base = f"{base}:{component}"  # a body-frame component's splits
        for situation, mask in situations.items():
            for flag, selected in (("in", mask), ("out", ~mask)):
                signed = values[selected].astype(np.float64)
                signed = signed[np.isfinite(signed)]
                part = np.abs(signed)
                entry = dict(share=float(selected.mean()), n=int(part.size))
                if part.size:
                    levels = np.percentile(part, (50, 99, 99.9))
                    entry.update(
                        zero=float((part == 0).mean()),
                        mean=float(signed.mean()),
                        **{f"p{q:g}": float(v) for q, v in zip((50, 99, 99.9), levels)},
                        max=float(part.max()),
                    )
                splits.setdefault(quantity, {}).setdefault(base, {}).setdefault(situation, {})[
                    flag
                ] = entry
    out = SCRATCH / "stats" / player
    out.mkdir(parents=True, exist_ok=True)
    n_one = next(len(v) for k, v in collected.items() if "/1/" in k)
    games = dict(
        segments=len(paths),
        environments=meta["environments"],
        ticks_per_segment=meta["ticks"],
        episode_ends=resets,
        games_touched=meta["environments"] + resets,
        ship_ticks_1=int(n_one),
        ship_ticks_8=int(len(collected["position/persistence/8/x"])),
        ship_ticks_32=int(len(collected["position/persistence/32/x"])),
    )
    (out / f"{preset}.json").write_text(
        json.dumps(dict(meta=games, stats=stats, splits=splits), indent=1)
    )
    histogram_dir = SCRATCH / "histograms" / player / preset
    histogram_dir.mkdir(parents=True, exist_ok=True)
    for quantity, entry in hists.items():
        (histogram_dir / f"{quantity}.json").write_text(
            json.dumps(dict(edges=EDGES.tolist(), units=UNITS[quantity], series=entry))
        )
    keep = rng.choice(n_one, size=min(SUBSAMPLE, n_one), replace=False)
    keep.sort()
    sample = {
        name.replace("/", "."): values[keep]
        for name, values in collected.items()
        if (name.startswith("_") and "/" not in name) or "/1/" in name
    }
    sample.update({f"_situation.{name}": mask[keep] for name, mask in situations.items()})
    # the 8- and 32-tick horizon samples for position, velocity, heading
    for horizon in (8, 32):
        names = [n for n in collected if f"/{horizon}/" in n]
        count = len(collected[names[0]])
        pick = np.sort(rng.choice(count, size=min(SUBSAMPLE, count), replace=False))
        sample.update({name.replace("/", "."): collected[name][pick] for name in names})
    (SCRATCH / "samples" / player).mkdir(parents=True, exist_ok=True)
    np.savez(SCRATCH / "samples" / player / f"{preset}.npz", **sample)
    print(f"{player} {preset}: {games}", flush=True)


UNITS = dict(
    position="m",
    velocity="m/s",
    heading="rad",
    nose_rate="rad/s",
    angle_of_attack="rad",
    shield_recharge_wait="s",
    shield="fraction of the type's maximum",
    energy="fraction of the type's maximum",
    gun_cooldown="ticks",
    log_refractive_index="ln n",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", nargs="+", default=["scripted", "calm-fern"])
    parser.add_argument(
        "--presets", nargs="+", default=["frontline", "deathmatch", "deathmatch_wide"]
    )
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    for player in args.players:
        for preset in args.presets:
            if (SCRATCH / "segments" / player / preset / "segment_0.npz").exists():
                measure(player, preset, rng)


if __name__ == "__main__":
    main()
