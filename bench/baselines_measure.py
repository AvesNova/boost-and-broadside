"""Measure what each one-tick baseline (P, R, E; `bench.baselines_core`) leaves, and the
axis correlation of the position and velocity residuals over dead-reckoning horizons
(docs/rewrite/sections/baselines-bench-brief.md), from the recorded segments of
`bench/state_deltas_collect.py`.

One segment in memory at a time, with the previous segment's last tick prepended (the
segments of a player and preset are one continuous play). Each segment's partial results
are written to

    <out>/<player>/<preset>/segment_<k>.npz

and merged by `bench/baselines_report.py`: per series (quantity / baseline / component)
and subset, a signed log histogram of the residual (100 bins per decade from 1e-12, the
exact zeros apart) and its max; pairwise win counts between the baselines; the
regression moments of each scalar's change on last tick's; per horizon, forecast,
quantity and frame, the streaming moments of the two components and a random subsample
for the rank correlation.

Run (CPU only, capped):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 15 \
        env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 uv run --no-sync python \
        -m bench.baselines_measure
"""

import argparse
import datetime
import json
import pathlib
import time

import numpy as np

from bench import baselines_core as core

SEGMENTS = pathlib.Path("/tmp/bnb-scratch-deltas/segments")
OUT = pathlib.Path("/tmp/bnb-scratch-baselines/measure")
PLAYERS = ("scripted", "calm-fern")
PRESETS = ("frontline", "deathmatch", "deathmatch_wide")
LOWEST = -12  # log10 of the first histogram edge
PER_DECADE = 100
DECADES = 16
BINS = PER_DECADE * DECADES  # per sign; the code is 0 (exact zero), 1..BINS, BINS+1..2 BINS
HORIZONS = (1, 3, 15, 30, 90, 150)
FORECASTS = ("P", "CV", "R", "E")
FRAMES = ("world", "nose", "track")
SUBSAMPLE = 20_000  # horizon samples kept per segment and horizon
SUBSETS = (
    "all",
    "fallback",  # t - 1 missing: E is R
    "held",  # power and stick as on t - 1
    "changed",
    "power_changed",
    "stick_changed",
    "quiet",  # no field band at t - 1, t, t + 1 and no damage on t - 1 or t
    "held_quiet",
    "changed_quiet",
    "field",  # in a field band at t or t + 1
    "no_field",
    "hit",  # damage (shot or boundary) on tick t
    "no_hit",
)
VECTOR_COMPONENTS = ("x", "y", "nose_along", "nose_across", "track_along", "track_across", "norm")
SCALARS = dict(
    heading="heading",
    nose_rate="omega",
    angle_of_attack="attack",
    energy="energy",
    shield="shield",
    shield_recharge_wait="wait",
    gun_cooldown="cooldown",
    log_refractive_index="log_index",
)
REGRESSED = ("nose_rate", "angle_of_attack", "energy", "log_refractive_index")


def valid_pairs(alive, done, k):
    """(L - k, B, N): alive at every tick t..t+k and no reset after any tick t..t+k-1."""
    dead = np.concatenate([np.zeros((1,) + alive.shape[1:], np.int32), np.cumsum(~alive, 0)])
    resets = np.concatenate([np.zeros((1, done.shape[1]), np.int32), np.cumsum(done, 0)])
    L = alive.shape[0]
    no_death = dead[k + 1 : L + 1] - dead[0 : L - k] == 0
    no_reset = resets[k:L] - resets[0 : L - k] == 0
    return no_death & no_reset[..., None]


def series_names() -> list[str]:
    """Every one-tick series, quantity/baseline/component."""
    names = []
    for quantity, baselines in (("position", ("P", "R", "E", "E-arc")), ("velocity", "PRE")):
        for baseline in baselines:
            names += [f"{quantity}/{baseline}/{c}" for c in VECTOR_COMPONENTS]
    for quantity in SCALARS:
        names += [f"{quantity}/{baseline}/value" for baseline in "PRE"]
    return names


def codes_of(values):
    """The signed histogram code of each value (0 for an exact zero)."""
    absolute = np.abs(values)
    with np.errstate(divide="ignore"):
        index = np.floor((np.log10(absolute) - LOWEST) * PER_DECADE)
    index = np.clip(np.nan_to_num(index, nan=0.0, neginf=0.0), 0, BINS - 1).astype(np.int32)
    return np.where(values == 0, 0, np.where(values > 0, 1 + index, 1 + BINS + index))


def load(path, columns, card, dt, carry):
    """The states (L', B, N) of a segment with the carried last tick prepended, the flags,
    and the new carry."""
    data = np.load(path)
    arrays = {name: data[name] for name in ("rows", "power", "stick", "hit", "boundary", "done")}
    new_carry = {name: value[-1:] for name, value in arrays.items()}
    if carry is not None:
        arrays = {name: np.concatenate([carry[name], value]) for name, value in arrays.items()}
    rows = arrays["rows"].astype(np.float64)
    C = {name: i for i, name in enumerate(columns)}
    col = lambda name: rows[..., C[name]]  # noqa: E731
    ship_type = col("ship_type").astype(int)
    max_shield = np.asarray(card["max_shield"])[ship_type]
    max_energy = np.asarray(card["max_energy"])[ship_type]
    rate = 12.0 * np.asarray(card["shield_recharge_rate_scale"])[ship_type] / max_shield
    state = dict(
        x=col("x"),
        y=col("y"),
        vx=col("velocity_x"),
        vy=col("velocity_y"),
        heading=np.arctan2(col("heading_sin"), col("heading_cos")),
        omega=col("nose_rate"),
        attack=col("angle_of_attack"),
        energy=col("energy") / max_energy,
        shield=col("shield") / max_shield,
        wait=col("shield_recharge_wait"),
        cooldown=np.round(col("gun_cooldown") / dt),
        log_index=col("log_refractive_index"),
        rate=rate,
    )
    in_field = (
        (col("log_refractive_index") != 0)
        | (col("log_index_gradient_x") != 0)
        | (col("log_index_gradient_y") != 0)
    )
    flags = dict(
        alive=col("alive") > 0.5,
        fired=col("fired") > 0.5,  # row t: the ship fired on tick t - 1
        damage=arrays["hit"] | arrays["boundary"],  # on tick t
        power=arrays["power"],
        stick=arrays["stick"],
        done=arrays["done"],
        field=in_field,
    )
    return state, flags, new_carry


def shift_back(array):
    """Row t holds array[t - 1] (row 0 its own: masked by has_previous)."""
    return np.concatenate([array[:1], array[:-1]])


def vector_components(ex, ey, heading, ux, uy):
    """{component: values} of a 2-D residual in the three frames, and its norm."""
    c, s = np.cos(heading), np.sin(heading)
    nose_along, nose_across = core.to_frame(ex, ey, c, s)
    track_along, track_across = core.to_frame(ex, ey, ux, uy)
    return dict(
        x=ex, y=ey, nose_along=nose_along, nose_across=nose_across,
        track_along=track_along, track_across=track_across, norm=np.hypot(ex, ey),
    )  # fmt: skip


def measure_segment(path, meta, carry, rng):
    """The partial results of one segment, and the new carry."""
    dt = meta["tick_length"]
    state, flags, new_carry = load(path, meta["columns"], meta["card"], dt, carry)
    L = flags["alive"].shape[0]
    one = valid_pairs(flags["alive"], flags["done"], 1)  # (L - 1, B, N), pairs (t, t + 1)
    current = {name: value[:-1] for name, value in state.items()}
    following = {name: value[1:] for name, value in state.items()}
    previous = {name: shift_back(value) for name, value in current.items()}
    has_previous = np.concatenate([np.zeros_like(one[:1]), one[:-1]]) & one
    fired = flags["fired"]
    shot_last = fired[:-1]  # row t: fired on tick t - 1
    damage = flags["damage"]
    damage_last = shift_back(damage[:-1]) & has_previous
    power, stick = flags["power"][:-1], flags["stick"][:-1]
    power_changed = (power != shift_back(power)) & has_previous
    stick_changed = (stick != shift_back(stick)) & has_previous
    changed = power_changed | stick_changed
    field = flags["field"]
    field_now = field[:-1] | field[1:]
    quiet = has_previous & ~(field_now | shift_back(field[:-1])) & ~damage[:-1] & ~damage_last
    subsets = dict(
        all=one,
        fallback=one & ~has_previous,
        held=has_previous & ~changed,
        changed=changed,
        power_changed=power_changed,
        stick_changed=stick_changed,
        quiet=quiet,
        held_quiet=quiet & ~changed,
        changed_quiet=quiet & changed,
        field=one & field_now,
        no_field=one & ~field_now,
        hit=one & damage[:-1],
        no_hit=one & ~damage[:-1],
    )
    selected = {name: mask[one] for name, mask in subsets.items()}

    rule = core.rule(current, dt)
    extrapolated, (arc_x, arc_y) = core.extrapolate(
        previous, current, dt, shot_last, damage_last, has_previous
    )
    baselines = dict(P=core.persistence(current), R=rule, E=extrapolated)
    heading = current["heading"][one]
    ux, uy, slow = core.track_frame(current["vx"], current["vy"], current["heading"])
    ux, uy = ux[one], uy[one]
    residuals = {}
    for name, base in baselines.items():
        for quantity, (a, b) in dict(position=("x", "y"), velocity=("vx", "vy")).items():
            ex = (following[a] - base[a])[one]
            ey = (following[b] - base[b])[one]
            for component, value in vector_components(ex, ey, heading, ux, uy).items():
                residuals[f"{quantity}/{name}/{component}"] = value
        for quantity, key in SCALARS.items():
            value = (following[key] - base[key])[one]
            if quantity == "heading":
                value = core.wrap(value)
            residuals[f"{quantity}/{name}/value"] = value
    ex, ey = (following["x"] - arc_x)[one], (following["y"] - arc_y)[one]
    for component, value in vector_components(ex, ey, heading, ux, uy).items():
        residuals[f"position/E-arc/{component}"] = value

    names = series_names()
    histograms = np.zeros((len(names), len(SUBSETS), 2 * BINS + 1), np.int64)
    maxima = np.zeros((len(names), len(SUBSETS)))
    masks = [selected[s] for s in SUBSETS]
    for i, name in enumerate(names):
        value = residuals[name]
        code = codes_of(value)
        absolute = np.abs(value)
        for j, mask in enumerate(masks):
            histograms[i, j] = np.bincount(code[mask], minlength=2 * BINS + 1)
            if mask.any():
                maxima[i, j] = absolute[mask].max()
    # pairwise wins: |first| < |second| and |first| > |second|, per component and subset
    comparisons = []
    for name in names:
        quantity, baseline, component = name.split("/")
        if baseline != "P":
            continue
        others = ("R", "E", "E-arc") if quantity == "position" else ("R", "E")
        for first, second in (
            [(o, "P") for o in others]
            + [(o, "R") for o in others if o != "R"]
            + ([("E-arc", "E")] if quantity == "position" else [])
        ):
            comparisons.append((quantity, component, first, second))
    wins = np.zeros((len(comparisons), len(SUBSETS), 2), np.int64)
    for i, (quantity, component, first, second) in enumerate(comparisons):
        a = np.abs(residuals[f"{quantity}/{first}/{component}"])
        b = np.abs(residuals[f"{quantity}/{second}/{component}"])
        for j, mask in enumerate(masks):
            wins[i, j] = ((a < b)[mask].sum(), (a > b)[mask].sum())
    # the change on t against last tick's: sum d_prev d, d_prev^2, d^2, n (has_previous)
    regression = np.zeros((len(REGRESSED), len(SUBSETS), 4))
    for i, quantity in enumerate(REGRESSED):
        key = SCALARS[quantity]
        change = (following[key] - current[key])[one]
        last = (current[key] - previous[key])[one]
        usable = has_previous[one]
        if quantity == "energy":  # no shot on either tick
            usable = usable & ~shot_last[one] & ~fired[1:][one]
        for j, mask in enumerate(masks):
            m = mask & usable
            regression[i, j] = (
                (last[m] * change[m]).sum(), (last[m] ** 2).sum(), (change[m] ** 2).sum(), m.sum()
            )  # fmt: skip
    counts = dict(
        pairs=int(one.sum()),
        fallback=int((one & ~has_previous).sum()),
        track_fallback=int((one & slow).sum()),
        **{f"subset_{s}": int(selected[s].sum()) for s in SUBSETS},
    )
    del residuals

    # the horizons: truth(t + k) - forecast, in the frames at t
    moments = np.zeros((len(HORIZONS), len(FORECASTS), 2, len(FRAMES), 6))
    samples = {}
    c, s = np.cos(current["heading"]), np.sin(current["heading"])
    tx, ty, _ = core.track_frame(current["vx"], current["vy"], current["heading"])
    produced = core.forecasts(previous, current, dt, has_previous, HORIZONS)
    # the mirror signs at t: the nose's turn and last tick's normal acceleration (0 without
    # t - 1), to fold the left-right symmetry out of the across components
    _, normal, _ = core.acceleration(previous, current, dt)
    turn_sign = np.sign(current["omega"]).astype(np.int8)
    curve_sign = np.where(has_previous, np.sign(normal), 0).astype(np.int8)
    for h, k in enumerate(HORIZONS):
        mask = valid_pairs(flags["alive"], flags["done"], k)  # (L - k, B, N)
        count = int(mask.sum())
        rows = L - k
        pick = np.sort(rng.choice(count, size=min(SUBSAMPLE, count), replace=False))
        hp = has_previous[:rows][mask]
        samples[f"{k}/has_previous"] = hp[pick]
        samples[f"{k}/turn_sign"] = turn_sign[:rows][mask][pick]
        samples[f"{k}/curve_sign"] = curve_sign[:rows][mask][pick]
        for f, forecast in enumerate(FORECASTS):
            fx, fy, fvx, fvy = (value[:rows] for value in produced[k][forecast])
            for q, (truth_x, truth_y, guess_x, guess_y) in enumerate(
                (
                    (state["x"][k:], state["y"][k:], fx, fy),
                    (state["vx"][k:], state["vy"][k:], fvx, fvy),
                )
            ):
                ex, ey = (truth_x - guess_x)[mask], (truth_y - guess_y)[mask]
                frames = dict(
                    world=(ex, ey),
                    nose=core.to_frame(ex, ey, c[:rows][mask], s[:rows][mask]),
                    track=core.to_frame(ex, ey, tx[:rows][mask], ty[:rows][mask]),
                )
                for r, frame in enumerate(FRAMES):
                    a, b = frames[frame]
                    moments[h, f, q, r] = (a.size, a.sum(), b.sum(), (a * a).sum(),
                                           (b * b).sum(), (a * b).sum())  # fmt: skip
                    quantity = ("position", "velocity")[q]
                    stacked = np.stack([a[pick], b[pick]], -1).astype(np.float32)
                    samples[f"{k}/{forecast}/{quantity}/{frame}"] = stacked
        produced.pop(k)
    result = dict(
        histograms=histograms,
        maxima=maxima,
        wins=wins,
        regression=regression,
        moments=moments,
        **{f"sample:{name}": value for name, value in samples.items()},
    )
    index = dict(
        names=names,
        subsets=list(SUBSETS),
        comparisons=[list(c) for c in comparisons],
        regressed=list(REGRESSED),
        horizons=list(HORIZONS),
        forecasts=list(FORECASTS),
        frames=list(FRAMES),
        counts=counts,
    )
    return result, index, new_carry


def main() -> None:
    """Measure every player and preset, one segment at a time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", nargs="+", default=list(PLAYERS))
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    jobs = [
        (player, preset, path)
        for player in args.players
        for preset in args.presets
        for path in sorted((SEGMENTS / player / preset).glob("segment_*.npz"))
    ]
    started = time.time()
    carry, last_group = None, None
    for done, (player, preset, path) in enumerate(jobs, 1):
        if (player, preset) != last_group:
            carry, last_group = None, (player, preset)
            meta = json.loads((SEGMENTS / player / preset / "meta.json").read_text())
        out = pathlib.Path(args.out) / player / preset
        out.mkdir(parents=True, exist_ok=True)
        result, index, carry = measure_segment(path, meta, carry, rng)
        np.savez_compressed(out / f"{path.stem}.npz", **result)
        (out / f"{path.stem}.json").write_text(json.dumps(index))
        elapsed = time.time() - started
        finish = datetime.datetime.now() + datetime.timedelta(
            seconds=elapsed / done * (len(jobs) - done)
        )
        print(
            f"{player} {preset} {path.stem}: {index['counts']['pairs']} pairs, "
            f"{done}/{len(jobs)}, {elapsed:.0f}s, ETA {finish:%H:%M}",
            flush=True,
        )


if __name__ == "__main__":
    main()
