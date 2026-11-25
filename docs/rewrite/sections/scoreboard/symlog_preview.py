"""What the categorical bins would look like with symlog spacing (proposal A) and with an extra
"exactly zero" class in front of symlog bins (proposal B). Measurement and pictures only.

    OMP_NUM_THREADS=4 PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/symlog_preview.py collect DATA_DIR [ENVS] [TICKS]
    PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/symlog_preview.py analyse DATA_DIR STATS_JSON
    PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/symlog_preview.py render STATS_JSON FIGURE_DIR

collect: target_histograms.py's collection (scripted yardstick on both teams, the training
game, the learner's team sizes, seed 0), keeping every raw target instead of histograms. Every
BATCH ticks the batch's next-state rows and its rewards / living / done flags go to
DATA_DIR/chunks/ (crash insurance) with a progress line and a clock-time ETA. At the end
DATA_DIR/value_{damage,kill_death,outcome}.npy (Monte Carlo returns, lambda = 1, every living
ship's tick of the episodes that ended inside the run) and DATA_DIR/next_{column}.npy, all
float32 and complete (no subsampling); the chunks are then removed.

analyse: knees, exact occupancy of the 70 bins (nearest centre, as target_histograms.py) for
the current linear support, the symlog support and the zero class + symlog support.
render: one PNG per target and a summary table PNG.
"""

import json
import math
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BATCH = 300
VALUE_LEVELS = ("damage", "kill_death", "outcome")
NEXT_COLUMNS = ("along", "across", "heading", "speed", "energy", "shield")
MEASURED_COLUMNS, BUFFER_COLUMNS = 64, 3
COLUMNS = MEASURED_COLUMNS + 2 * BUFFER_COLUMNS
MASS_SHARES = (0.5, 0.9, 0.99)
# The data rule for a knee (fixed before looking at the knees): the nonzero |x| of one sign on
# a log-spaced histogram (PER_DECADE bins a decade), density = count / bin width; the local
# log-log slope is the least-squares slope of log density against log |x| over a window of
# WINDOW_DECADES centred on the bin; scanning upward from the 0.1st percentile of nonzero |x|
# (below it too few rows), the knee is the first |x| where the slope reaches SLOPE_RULE.
PER_DECADE = 40
WINDOW_DECADES = 0.5
SLOPE_RULE = -0.5
START_QUANTILE = 0.001
# A side has a flat core when the decade below its knee (at least CORE_DECADES of it inside the
# scan) averages a slope above CORE_SLOPE; a target without one has no knee by the data rule
# and falls back (the event knee for a value level, the current knee 1 for the shield).
CORE_DECADES = 0.5
CORE_SLOPE = -0.25
ZERO_CLASS_SHARE = 0.05  # proposal B is shown for every target with at least this many exact 0


def collect(out: Path, envs: int, ticks: int) -> None:
    import torch

    from lean import game, scripted
    from lean.contract import LEVEL_NAMES
    from lean.learn.config import LearnConfig
    from lean.learn.rewards import blank_ledger, clear_ledger, reward_step, tick_events
    from lean.learn.rollout import draw_team_sizes, scripted_actions
    from lean.model.policy import ModelConfig, next_state_target

    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    learn = LearnConfig()
    model = ModelConfig()
    names = [column.name for column in model.next_state_columns]
    assert tuple(names) == NEXT_COLUMNS, names
    game_config = game.GameConfig(training_spawn=True)
    sizes = draw_team_sizes(learn, envs, generator)
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    scripted_config = scripted.ScriptedConfig()
    persona = scripted.personas(
        scripted_config, envs, slots, generator, device, learn.persona_spread
    )
    memory = scripted.fresh_memory(envs, slots, generator, device)
    ledger = blank_ledger(envs, slots, device)
    rewards = torch.zeros(ticks, envs, slots, len(LEVEL_NAMES), device=device)
    valued = torch.zeros(ticks, envs, slots, dtype=torch.bool, device=device)
    done_at = torch.zeros(ticks, envs, dtype=torch.bool, device=device)
    next_rows, batch_rows = [], []
    began = time.time()
    with torch.no_grad():
        for tick in range(ticks):
            move = scripted_actions(scripted_config, state, persona, memory, generator)
            next_state, step_out = game.tick(game_config, state, move.action, generator)
            reward, ledger = reward_step(learn, ledger, tick_events(step_out, state), state.team)
            rewards[tick] = reward
            valued[tick] = state.alive
            done_at[tick] = step_out.done
            valid = state.alive & next_state.alive & ~step_out.done[:, None]
            target = next_state_target(model, state, next_state, step_out, game_config.tick_length)
            batch_rows.append(target[valid])
            done = step_out.done
            ledger = clear_ledger(ledger, done)
            fresh_sizes = draw_team_sizes(learn, envs, generator)
            fresh = game.deal(game_config, envs, slots, fresh_sizes, generator, device)
            state = game.take_deal(next_state, done, fresh)
            new_memory = scripted.fresh_memory(envs, slots, generator, device)
            memory = {
                name: torch.where(
                    done.view(-1, *[1] * (move.memory[name].dim() - 1)),
                    new_memory[name],
                    move.memory[name],
                )
                for name in move.memory
            }
            if (tick + 1) % BATCH == 0 or tick + 1 == ticks:
                index = tick // BATCH
                rows = torch.cat(batch_rows)
                next_rows.append(rows)
                batch_rows = []
                np.save(chunks / f"next_{index:03d}.npy", rows.cpu().numpy())
                first = index * BATCH
                np.savez(
                    chunks / f"reward_{index:03d}.npz",
                    rewards=rewards[first : tick + 1].cpu().numpy(),
                    valued=valued[first : tick + 1].cpu().numpy(),
                    done=done_at[first : tick + 1].cpu().numpy(),
                )
                elapsed = time.time() - began
                remaining = elapsed / (tick + 1) * (ticks - tick - 1)
                eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
                print(
                    f"{datetime.now():%H:%M:%S} tick {tick + 1} of {ticks}, {elapsed:.0f} s, "
                    f"ETA {eta} (collection; returns and files after), "
                    f"episodes ended {int(done_at[: tick + 1].sum())}",
                    flush=True,
                )

        discounts = torch.tensor(learn.discounts, device=device)
        running = torch.zeros(envs, slots, len(LEVEL_NAMES), device=device)
        returns = torch.empty_like(rewards)
        for tick in reversed(range(ticks)):
            keep = (~done_at[tick]).float()[:, None, None]
            running = rewards[tick] + discounts * keep * running
            returns[tick] = running
        tick_index = torch.arange(ticks, device=device)[:, None]
        last_end = torch.where(done_at, tick_index, -1).amax(0)
        complete = tick_index <= last_end[None, :]
        value_rows = returns[valued & complete[..., None]]
        del rewards, returns
        for level, name in enumerate(LEVEL_NAMES):
            if name in VALUE_LEVELS:
                np.save(out / f"value_{name}.npy", value_rows[:, level].cpu().numpy())
        next_targets = torch.cat(next_rows)
        for column, name in enumerate(names):
            np.save(out / f"next_{name}.npy", next_targets[:, column].contiguous().cpu().numpy())
    meta = {
        "envs": envs,
        "ticks": ticks,
        "episodes": int(done_at.sum()),
        "value_rows": int(value_rows.shape[0]),
        "next_rows": int(next_targets.shape[0]),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "seconds": time.time() - began,
    }
    (out / "collection.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(chunks)
    print(f"wrote {out} at {datetime.now():%H:%M:%S}: {meta}", flush=True)


# ---------------------------------------------------------------- analysis (numpy, CPU)


def symlog(x: np.ndarray | float, knee: float) -> np.ndarray:
    """s(x) = sign(x) knee log1p(|x| / knee): slope 1 inside the knee, log outside."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * knee * np.log1p(np.abs(x) / knee)


def inverse_symlog(s: np.ndarray | float, knee: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return np.sign(s) * knee * np.expm1(np.abs(s) / knee)


def buffered_centres(low: float, high: float) -> np.ndarray:
    """lean.model.value.buffered_centres in float64: 64 measured columns, 3 buffer each end."""
    width = (high - low) / (MEASURED_COLUMNS - 1)
    return np.linspace(low - BUFFER_COLUMNS * width, high + BUFFER_COLUMNS * width, COLUMNS)


def occupancy(transformed: np.ndarray, centres: np.ndarray) -> dict:
    """Shares per bin (nearest centre, clamped into the edge bins), bins holding MASS_SHARES
    (largest first), effective bins (exp of the entropy) and the clamped share."""
    width = centres[1] - centres[0]
    nearest = np.clip(np.rint((transformed - centres[0]) / width), 0, len(centres) - 1)
    counts = np.bincount(nearest.astype(np.int64), minlength=len(centres)).astype(np.float64)
    share = counts / counts.sum()
    return {"shares": share.tolist(), **share_stats(share)} | {
        "clamped": float(np.mean((transformed < centres[0]) | (transformed > centres[-1])))
    }


def share_stats(share: np.ndarray) -> dict:
    ordered = np.cumsum(np.sort(share)[::-1])
    positive = share[share > 0]
    return {
        "bins_holding": {str(m): int(np.sum(ordered < m)) + 1 for m in MASS_SHARES},
        "effective_bins": float(math.exp(-np.sum(positive * np.log(positive)))),
    }


def slope_profile(magnitudes: np.ndarray) -> dict:
    """Log-spaced density of positive magnitudes and its local log-log slope (the data rule)."""
    logs = np.log10(magnitudes)
    low = math.floor(logs.min() * PER_DECADE) / PER_DECADE
    high = math.ceil(logs.max() * PER_DECADE) / PER_DECADE + 1.0 / PER_DECADE
    edges = np.arange(low, high + 1e-9, 1.0 / PER_DECADE)
    counts, _ = np.histogram(logs, edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = counts / np.diff(10.0**edges)
    half = int(round(WINDOW_DECADES * PER_DECADE / 2))
    slopes = np.full(len(centres), np.nan)
    for i in range(half, len(centres) - half):
        window = slice(i - half, i + half + 1)
        keep = counts[window] > 0
        if keep.sum() >= half + 1:
            slopes[i] = np.polyfit(centres[window][keep], np.log10(density[window][keep]), 1)[0]
    start = float(np.quantile(magnitudes, START_QUANTILE))
    knee = None
    for centre, slope in zip(centres, slopes):
        if 10.0**centre >= start and not np.isnan(slope) and slope <= SLOPE_RULE:
            knee = float(10.0**centre)
            break
    first = next(
        (float(s) for c, s in zip(centres, slopes) if 10.0**c >= start and not np.isnan(s)),
        float("nan"),
    )
    # The core check: a flat core needs the decade below the knee (within the scan) to cover at
    # least CORE_DECADES and to average a slope above CORE_SLOPE.
    core_slope, coverage = float("nan"), 0.0
    if knee is not None:
        top = math.log10(knee)
        bottom = max(math.log10(start), top - 1.0)
        coverage = top - bottom
        inside = [s for c, s in zip(centres, slopes) if bottom <= c < top and not np.isnan(s)]
        core_slope = float(np.mean(inside)) if inside else float("nan")
    return {
        "knee": knee,
        "flat_core": bool(coverage >= CORE_DECADES and core_slope > CORE_SLOPE),
        "core_slope": core_slope,
        "core_decades": coverage,
        "start": start,
        "slope_at_start": first,
        "log10_centres": centres.tolist(),
        "counts": counts.tolist(),
        "slopes": [None if np.isnan(s) else float(s) for s in slopes],
    }


def data_knee(values: np.ndarray) -> dict:
    """The data rule on each sign of the nonzero values; the knee used is the mean of the
    sides' knees (one side when the other has fewer than 1% of the nonzero rows)."""
    nonzero = values[values != 0]
    sides = {}
    for side, mask in (("negative", nonzero < 0), ("positive", nonzero > 0)):
        share = float(mask.mean()) if nonzero.size else 0.0
        if share >= 0.01:
            sides[side] = {"share_of_nonzero": share, **slope_profile(np.abs(nonzero[mask]))}
    knees = [side["knee"] for side in sides.values() if side["flat_core"]]
    every = [side["knee"] for side in sides.values() if side["knee"] is not None]
    return {
        "sides": sides,
        "flat_core": bool(knees),
        "knee": float(np.mean(knees)) if knees else None,
        "knee_every_side": float(np.mean(every)) if every else None,
    }


def bin_widths(centres_s: np.ndarray, knee: float) -> dict:
    """Widths in target units of the bins either side of 0 (or of the bin nearest 0) and of the
    outermost bins."""
    width = centres_s[1] - centres_s[0]
    edges = np.append(centres_s - width / 2, centres_s[-1] + width / 2)
    widths = np.diff(inverse_symlog(edges, knee))
    nearest_zero = int(np.argmin(np.abs(centres_s)))
    return {
        "innermost": float(widths[nearest_zero]),
        "outermost": float(max(widths[0], widths[-1])),
        "outermost_low": float(widths[0]),
        "outermost_high": float(widths[-1]),
    }


def event_knees() -> dict:
    """One reward event's size times e^-3, from the code: a hit is the card's hit_damage (at
    the bullet's reference speed, full closeness and aspect) per damage_unit times the damage
    weight; a kill (or a death) is the kill_death weight."""
    from lean.game import card
    from lean.learn.config import LearnConfig

    learn = LearnConfig()
    (hit,) = (float(h) for h in card.CARD["hit_damage"])  # one ship type
    damage_event = hit / learn.damage_unit * learn.reward_weights[0]
    kill_event = learn.reward_weights[1]
    return {
        "hit_damage_points": hit,
        "damage_unit": learn.damage_unit,
        "damage_weight": learn.reward_weights[0],
        "kill_death_weight": kill_event,
        "damage_event": damage_event,
        "kill_death_event": kill_event,
        "damage": damage_event * math.exp(-3.0),
        "kill_death": kill_event * math.exp(-3.0),
    }


def analyse(data: Path, stats_path: Path) -> None:
    from lean.learn.config import LearnConfig
    from lean.model.policy import ModelConfig

    learn = LearnConfig()
    model = ModelConfig()
    level_index = {"damage": 0, "kill_death": 1, "outcome": 4}
    events = event_knees()
    stats = {
        "collection": json.loads((data / "collection.json").read_text()),
        "events": events,
        "rule": {
            "per_decade": PER_DECADE,
            "window_decades": WINDOW_DECADES,
            "slope": SLOPE_RULE,
            "start_quantile": START_QUANTILE,
        },
        "targets": {},
    }
    targets = [(f"value_{n}", n) for n in VALUE_LEVELS] + [(f"next_{n}", n) for n in NEXT_COLUMNS]
    columns = {column.name: column for column in model.next_state_columns}
    for key, name in targets:
        began = time.time()
        values = np.load(data / f"{key}.npy").astype(np.float64)
        if key.startswith("value_"):
            low, high = learn.value_low[level_index[name]], learn.value_high[level_index[name]]
            current_symlog_knee = None
        else:
            column = columns[name]
            low, high = column.low, column.high
            current_symlog_knee = 1.0 if column.name == "shield" else None
        zero = float(np.mean(values == 0))
        entry = {
            "rows": int(values.size),
            "measured_low": low,
            "measured_high": high,
            "share_exactly_zero": zero,
            "share_tiny_nonzero": {
                f"{t:g}": float(np.mean((values != 0) & (np.abs(values) < t))) for t in (1e-6, 1e-4)
            },
        }
        # Current support: linear, or the shield's symlog (knee 1) in symlog space.
        if current_symlog_knee is None:
            centres = buffered_centres(low, high)
            entry["linear"] = {
                **occupancy(values, centres),
                "centres": centres.tolist(),
                "width": float(centres[1] - centres[0]),
            }
        else:
            k = current_symlog_knee
            centres = buffered_centres(float(symlog(low, k)), float(symlog(high, k)))
            entry["linear"] = {
                **occupancy(symlog(values, k), centres),
                "centres": inverse_symlog(centres, k).tolist(),
                "current_symlog_knee": k,
                **bin_widths(centres, k),
            }
        rule = data_knee(values)
        entry["data_rule"] = rule
        if name == "outcome":
            stats["targets"][key] = entry
            print(f"{key}: {time.time() - began:.0f} s", flush=True)
            continue
        if key.startswith("value_"):
            knee, source = events[name], "event"
        elif rule["flat_core"]:
            knee, source = rule["knee"], "data rule"
        else:
            knee, source = 1.0, "fallback: current knee 1 (no flat core)"
        entry["knee"] = knee
        entry["knee_source"] = source
        centres = buffered_centres(float(symlog(low, knee)), float(symlog(high, knee)))
        entry["symlog"] = {
            **occupancy(symlog(values, knee), centres),
            "centres": inverse_symlog(centres, knee).tolist(),
            **bin_widths(centres, knee),
        }
        if zero >= ZERO_CLASS_SHARE:
            b_knee = rule["knee"] if rule["flat_core"] else knee
            entry["zero_class_knee_source"] = "data rule" if rule["flat_core"] else source
            nonzero = values[values != 0]
            centres = buffered_centres(float(symlog(low, b_knee)), float(symlog(high, b_knee)))
            conditional = occupancy(symlog(nonzero, b_knee), centres)
            joint = np.concatenate([[zero], (1.0 - zero) * np.array(conditional["shares"])])
            entry["zero_class"] = {
                "knee": b_knee,
                "p_zero": zero,
                **conditional,
                "centres": inverse_symlog(centres, b_knee).tolist(),
                **bin_widths(centres, b_knee),
                "joint_71": share_stats(joint),
            }
        stats["targets"][key] = entry
        print(f"{key}: knee {knee}, {time.time() - began:.0f} s", flush=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=1))
    print(f"wrote {stats_path}", flush=True)


# ---------------------------------------------------------------- charts

INK, MUTED, BAR, ZERO_BAR = "#1f2328", "#6b7280", "#2f6db5", "#8a5a00"
UNITS = {
    "value_damage": "return",
    "value_kill_death": "return",
    "value_outcome": "return",
    "next_along": "m",
    "next_across": "m",
    "next_heading": "rad",
    "next_speed": "m/s",
    "next_energy": "store",
    "next_shield": "points",
}
LABEL_BINS = (3, 17, 27, 34, 42, 52, 66)


def number(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.3g}"
    return f"{x:.1e}".replace("e-0", "e-")


def stats_line(support: dict) -> str:
    held = support["bins_holding"]
    return (
        f"effective bins {support['effective_bins']:.1f}  |  50 / 90 / 99% in "
        f"{held['0.5']} / {held['0.9']} / {held['0.99']} bins"
    )


def occupancy_row(axis, support: dict, title: str, note: str) -> None:
    shares = np.array(support["shares"])
    centres = np.array(support["centres"])
    axis.bar(np.arange(COLUMNS), np.maximum(shares, 1e-12), width=0.8, color=BAR, linewidth=0)
    axis.set_yscale("log")
    axis.set_ylim(1e-7, 2.0)
    axis.set_xlim(-1, COLUMNS)
    for edge in (BUFFER_COLUMNS - 0.5, COLUMNS - BUFFER_COLUMNS - 0.5):
        axis.axvline(edge, color=MUTED, linewidth=1.0, linestyle=":")
    axis.set_xticks(list(LABEL_BINS))
    axis.set_xticklabels([f"{b}\n{number(centres[b])}" for b in LABEL_BINS])
    axis.tick_params(labelsize=14, colors=INK)
    axis.set_ylabel("share", fontsize=15, color=INK)
    axis.set_title(title, fontsize=19, color=INK, loc="left", pad=48)
    axis.text(0.0, 1.03, note, transform=axis.transAxes, fontsize=14, color=INK, va="bottom")
    for side in ("right",):
        axis.spines[side].set_visible(False)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)


def width_note(support: dict, unit: str) -> str:
    return (
        f"innermost bin {number(support['innermost'])} {unit}, outermost "
        f"{number(support['outermost'])} {unit}"
    )


def render(stats_path: Path, figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    stats = json.loads(stats_path.read_text())
    for key, entry in stats["targets"].items():
        unit = UNITS[key]
        rows = [("current", entry["linear"])]
        if "symlog" in entry:
            rows.append(("symlog", entry["symlog"]))
        if "zero_class" in entry:
            rows.append(("zero", entry["zero_class"]))
        figure = plt.figure(figsize=(10.8, 5.2 * len(rows) + 1.0), dpi=100)
        grid = figure.add_gridspec(len(rows), 2, width_ratios=(1, 10), wspace=0.3, hspace=0.75)
        figure.suptitle(
            f"{key}  ({entry['rows'] / 1e6:.1f}M rows, {entry['share_exactly_zero']:.1%} "
            f"exactly 0)",
            fontsize=22,
            color=INK,
            x=0.02,
            ha="left",
            y=0.995,
        )
        for row, (kind, support) in enumerate(rows):
            axis = figure.add_subplot(grid[row, 1])
            if kind == "current":
                if "current_symlog_knee" in support:
                    title = "Current: symlog, knee 1 (contract.symlog)"
                    note = stats_line(support) + "\n" + width_note(support, unit)
                else:
                    title = "Current: linear 70 bins"
                    note = (
                        stats_line(support) + f"\nevery bin {number(support['width'])} {unit} wide"
                    )
            elif kind == "symlog":
                title = f"A: symlog, knee {number(entry['knee'])} {unit} ({entry['knee_source']})"
                note = stats_line(support) + "\n" + width_note(support, unit)
            else:
                title = (
                    f"B: zero class + symlog of the nonzero, knee {number(support['knee'])} {unit}"
                )
                joint = support["joint_71"]
                note = (
                    "nonzero only: "
                    + stats_line(support)
                    + f"\nall 71 classes: effective {joint['effective_bins']:.1f}; "
                    + width_note(support, unit)
                )
            occupancy_row(axis, support, title, note)
            axis.set_xlabel(
                "bin index / bin centre in " + unit + " (dotted: buffer columns)",
                fontsize=14,
                color=INK,
            )
            side = figure.add_subplot(grid[row, 0])
            side.set_yscale("log")
            side.set_ylim(1e-7, 2.0)
            side.set_xticks([0])
            side.tick_params(labelleft=False, labelsize=13)
            for spine in ("right", "top"):
                side.spines[spine].set_visible(False)
            if kind == "zero":
                side.bar([0], [support["p_zero"]], width=0.8, color=ZERO_BAR)
                side.set_xticklabels(["P(0)"])
                side.text(
                    0,
                    support["p_zero"] * 1.3,
                    f"{support['p_zero']:.3f}",
                    ha="center",
                    fontsize=12,
                    color=INK,
                )
                axis.set_ylabel("share of nonzero", fontsize=15, color=INK)
            else:
                side.set_visible(False)
        figure.subplots_adjust(
            left=0.03,
            right=0.98,
            top=1 - 1.6 / (5.2 * len(rows) + 1.0),
            bottom=0.9 / (5.2 * len(rows) + 1.0),
        )
        figure.savefig(figures / f"{key}.png")
        plt.close(figure)
        print(f"wrote {figures / f'{key}.png'}")
    summary_figure(stats, figures, plt)


def summary_figure(stats: dict, figures: Path, plt) -> None:
    header = ["target", "data rule", "knee A", "knee B", "current", "A", "B nonzero", "B 71"]
    cells = []
    for key, entry in stats["targets"].items():
        unit = UNITS[key]
        current = entry["linear"]["effective_bins"]
        a = entry.get("symlog", {}).get("effective_bins")
        b = entry.get("zero_class", {})
        cells.append(
            [
                key.replace("value_", "").replace("next_", ""),
                number(entry["data_rule"]["knee"])
                if entry["data_rule"]["flat_core"]
                else "no core",
                f"{number(entry['knee'])} {unit}" if "knee" in entry else "(linear)",
                f"{number(b['knee'])}" if b else "-",
                f"{current:.1f}"
                + (" (symlog 1)" if "current_symlog_knee" in entry["linear"] else ""),
                f"{a:.1f}" if a is not None else "-",
                f"{b['effective_bins']:.1f}" if b else "-",
                f"{b['joint_71']['effective_bins']:.1f}" if b else "-",
            ]
        )
    figure, axis = plt.subplots(figsize=(10.8, 1.0 + 0.62 * (len(cells) + 1)), dpi=100)
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.0, 2.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d4da")
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#eef2f7")
    table.auto_set_column_width(list(range(len(header))))
    axis.set_title(
        "Effective bins (exp entropy of the 70-bin occupancy)",
        fontsize=19,
        color=INK,
        loc="left",
    )
    figure.text(
        0.01,
        0.01,
        "Knee A: event rule (value), data rule or fallback (next state). B: P(exactly 0) + "
        "symlog of the nonzero;\n'B nonzero' the conditional 70 bins, 'B 71' all 71 classes. "
        "No core: the density has no flat region (see the page).",
        fontsize=12,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(figures / "summary.png")
    plt.close(figure)
    print(f"wrote {figures / 'summary.png'}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(
            Path(sys.argv[2]),
            int(sys.argv[3]) if len(sys.argv) > 3 else 2048,
            int(sys.argv[4]) if len(sys.argv) > 4 else 3600,
        )
    elif sys.argv[1] == "analyse":
        analyse(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        render(Path(sys.argv[2]), Path(sys.argv[3]))
