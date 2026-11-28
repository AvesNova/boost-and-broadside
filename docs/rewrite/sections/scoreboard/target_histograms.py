"""Distributions of every learning target: fine histograms, summary statistics and how the mass
sits on the current categorical supports (and on symlog-spaced alternatives for the critic).

    OMP_NUM_THREADS=4 PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/target_histograms.py collect OUT_DIR [ENVS] [TICKS]
    PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scoreboard/target_histograms.py render OUT_DIR FIGURE_DIR

The collection is target_ranges.py's (scripted yardstick on both teams, the training game,
the learner's team sizes, E environments x T ticks, finished environments re-dealt). Per tick:
next_state_target of every ship alive before and after the step in an unfinished game, every
bullet outcome that ended (the hit head's labels), the entropy of the scripted teacher's
42-way joint distribution for every living ship (the BC policy target), and the rewards.
After the run: each living ship's discounted return per level, Monte Carlo to its episode's
end (lambda = 1), for every tick of the episodes that ended inside the run. The learner's
own critic target is the lambda-return on its critic (LearnConfig.gae_lambdas), which needs a
trained critic; the Monte Carlo return is its lambda = 1 end and the widest of the family.

Every PROGRESS_EVERY ticks the next-state histograms so far (over PRESET_RANGES, with under-
and overflow counts) and the hit counts are written to OUT_DIR/partial.npz, and a progress line
with a clock-time ETA to stdout. At the end OUT_DIR/histograms.npz (fine histograms over each
target's measured extremes) and OUT_DIR/stats.json.
"""

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from lean import game, scripted
from lean.contract import LEVEL_NAMES
from lean.learn.config import LearnConfig
from lean.learn.rewards import blank_ledger, clear_ledger, reward_step, tick_events
from lean.learn.rollout import draw_team_sizes, scripted_actions, teacher_joint
from lean.model.policy import ModelConfig, next_state_target
from lean.model.value import BUFFER_COLUMNS, buffered_centres, symlog

FINE_BINS = 2000
PERCENTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
PROGRESS_EVERY = 300
MASS_SHARES = (0.5, 0.9, 0.99, 0.999)
# Ranges of the partial (crash-insurance) histograms: 1.25 x the 2026-11-25 extremes.
PRESET_RANGES = {
    "along": (-2.4, 2.4),
    "across": (-2.3, 2.3),
    "heading": (-0.125, 0.125),
    "speed": (-6.8, 6.8),
    "energy": (-1220.0, 1220.0),
    "shield": (-77.0, 1.0),
}
VALUE_LEVELS = ("damage", "kill_death", "outcome")
FIXED_KNEES = (0.1, 0.25, 0.5, 1.0, 2.0)


def symlog_knee(x: torch.Tensor, knee: float) -> torch.Tensor:
    """s(x) = sign(x) knee log1p(|x| / knee): linear (slope 1) inside the knee, log outside."""
    return torch.sign(x) * knee * torch.log1p(x.abs() / knee)


def summary(values: torch.Tensor) -> dict:
    """Count, moments (float64), extremes and PERCENTILES of a 1-d tensor."""
    x = values.double()
    count = x.numel()
    mean = x.mean()
    centred = x - mean
    variance = (centred**2).mean()
    kurtosis = (centred**4).mean() / variance.clamp(min=1e-300) ** 2 - 3.0
    ordered = torch.sort(values.float()).values
    last = count - 1
    absolute = torch.sort(values.float().abs()).values
    return {
        "count": count,
        "mean": float(mean),
        "std": float(variance.sqrt()),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "percentiles": {str(p): float(ordered[round(p * last)]) for p in PERCENTILES},
        "abs_percentiles": {str(p): float(absolute[round(p * last)]) for p in (0.5, 0.9, 0.99)},
        "excess_kurtosis": float(kurtosis),
        "share_exactly_zero": float((values == 0).double().mean()),
    }


def occupancy(transformed: torch.Tensor, centres: torch.Tensor) -> dict:
    """How the mass of targets (already in the support's space) sits on uniform `centres`
    (nearest centre, as lean.learn.returns.usage_counts; beyond the outermost centres counts
    in the edge bins and as clamped): per-bin shares, bins holding MASS_SHARES of the mass
    (largest bins first), occupied bins (>= 0.1%, the range monitor's rule), the entropy of
    the occupancy as its perplexity (effective bins)."""
    bins = centres.numel()
    width = centres[1] - centres[0]
    nearest = ((transformed - centres[0]) / width).round().clamp(0, bins - 1).long()
    counts = torch.bincount(nearest, minlength=bins).double()
    share = counts / counts.sum()
    ordered = torch.sort(share, descending=True).values.cumsum(0)
    holding = {str(m): int((ordered < m).sum()) + 1 for m in MASS_SHARES}
    positive = share[share > 0]
    entropy = float(-(positive * positive.log()).sum())
    clamped = float(((transformed < centres[0]) | (transformed > centres[-1])).double().mean())
    return {
        "shares": share.tolist(),
        "bins_holding": holding,
        "occupied_bins": int((share >= 0.001).sum()),
        "perplexity": math.exp(entropy),
        "clamped": clamped,
        "edge_bins_share": float(share[:BUFFER_COLUMNS].sum() + share[-BUFFER_COLUMNS:].sum()),
    }


def fine_histogram(values: torch.Tensor, low: float, high: float) -> np.ndarray:
    if high <= low:
        high = low + 1e-6
    return torch.histc(values.float(), FINE_BINS, low, high).cpu().numpy().astype(np.int64)


def collect(out: Path, envs: int, ticks: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    learn = LearnConfig(objective_levels=True)  # all five levels, LEVEL_NAMES order
    model = ModelConfig()
    game_config = game.GameConfig(training_spawn=True)
    sizes = draw_team_sizes(learn, envs, generator)
    state = game.reset(game_config, envs, sizes, generator, device)
    slots = state.exists.shape[1]
    scripted_config = scripted.ScriptedConfig()
    persona = scripted.personas(
        scripted_config, envs, slots, generator, device, learn.persona_spread
    )
    ledger = blank_ledger(envs, slots, device)
    rewards = torch.zeros(ticks, envs, slots, len(LEVEL_NAMES), device=device)
    valued = torch.zeros(ticks, envs, slots, dtype=torch.bool, device=device)
    done_at = torch.zeros(ticks, envs, dtype=torch.bool, device=device)
    next_rows, entropies = [], []
    outcomes = torch.zeros(3, dtype=torch.long, device=device)
    names = [column.name for column in model.next_state_columns]
    partial = {name: torch.zeros(FINE_BINS + 2, dtype=torch.long, device=device) for name in names}
    began = time.time()
    with torch.no_grad():
        for tick in range(ticks):
            move = scripted_actions(
                scripted_config, state, persona, generator, learn.team_size_maximum
            )
            record = move.teacher.flatten(0, 2)
            joint = teacher_joint(record[record.float().sum(-1) > 0])
            entropies.append(-(joint * joint.clamp(min=1e-30).log()).sum(-1))
            next_state, step_out = game.tick(game_config, state, move.action, generator)
            reward, ledger = reward_step(learn, ledger, tick_events(step_out, state), state.team)
            rewards[tick] = reward
            valued[tick] = state.alive
            done_at[tick] = step_out.done
            valid = state.alive & next_state.alive & ~step_out.done[:, None]
            target = next_state_target(model, state, next_state, step_out, game_config.tick_length)[
                valid
            ]
            next_rows.append(target)
            for column, name in enumerate(names):
                low, high = PRESET_RANGES[name]
                index = ((target[:, column] - low) / (high - low) * FINE_BINS).floor()
                index = index.clamp(-1, FINE_BINS).long() + 1
                partial[name] += torch.bincount(index, minlength=FINE_BINS + 2)
            ended = step_out.shot_outcome[step_out.shot_outcome >= 0]
            outcomes += torch.bincount(ended.flatten(), minlength=3)
            done = step_out.done
            ledger = clear_ledger(ledger, done)
            fresh_sizes = draw_team_sizes(learn, envs, generator)
            fresh = game.deal(game_config, envs, slots, fresh_sizes, generator, device)
            state = game.take_deal(next_state, done, fresh)
            if (tick + 1) % PROGRESS_EVERY == 0 or tick + 1 == ticks:
                np.savez(
                    out / "partial.npz",
                    ticks_done=tick + 1,
                    hit_counts=outcomes.cpu().numpy(),
                    **{f"next_{n}": partial[n].cpu().numpy() for n in names},
                    **{f"range_{n}": np.array(PRESET_RANGES[n]) for n in names},
                )
                elapsed = time.time() - began
                remaining = elapsed / (tick + 1) * (ticks - tick - 1)
                eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
                print(
                    f"tick {tick + 1} of {ticks}, {elapsed:.0f} s, ETA {eta} (collection; "
                    f"stats after), episodes ended {int(done_at[: tick + 1].sum())}",
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
        next_targets = torch.cat(next_rows)
        del next_rows
        teacher_entropy = torch.cat(entropies)
        print(f"collection done {datetime.now():%H:%M:%S}; statistics", flush=True)

        stats: dict = {
            "envs": envs,
            "ticks": ticks,
            "episodes": int(done_at.sum()),
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "seconds": time.time() - began,
            "targets": {},
        }
        arrays: dict = {}

        def add(key: str, values: torch.Tensor, family: str, support=None, knee=0.0):
            entry = {"family": family, **summary(values)}
            arrays[f"hist_{key}"] = fine_histogram(values, entry["min"], entry["max"])
            arrays[f"range_{key}"] = np.array([entry["min"], entry["max"]])
            if support is not None:
                low, high = support
                centres = buffered_centres(low, high, knee).to(values.device)
                transformed = symlog(values.float(), knee)
                entry["support"] = {
                    "measured_low": low,
                    "measured_high": high,
                    "space": "linear" if not knee else f"symlog knee {knee:g}",
                    "outermost_centres": [float(centres[0]), float(centres[-1])],
                    "column_width": float(centres[1] - centres[0]),
                    "occupancy": occupancy(transformed, centres),
                }
                arrays[f"centres_{key}"] = centres.cpu().numpy()
            stats["targets"][key] = entry
            print(f"  {key}: {entry['count']} rows", flush=True)

        for level, name in enumerate(LEVEL_NAMES):
            if name not in VALUE_LEVELS:
                continue
            values = value_rows[:, level].contiguous()
            low, high = learn.value_low[level], learn.value_high[level]
            add(f"value_{name}", values, "value", (low, high), learn.value_knee[level])
            entry = stats["targets"][f"value_{name}"]
            knees = {f"{k:g}": k for k in FIXED_KNEES}
            for share, quantile in entry["abs_percentiles"].items():
                if quantile > 0:
                    knees[f"|x| p{float(share) * 100:g} = {quantile:.3g}"] = quantile
            alternatives = {}
            for label, knee in knees.items():
                s_low = float(symlog_knee(torch.tensor(low), knee))
                s_high = float(symlog_knee(torch.tensor(high), knee))
                centres = buffered_centres(s_low, s_high).to(values.device)
                result = occupancy(symlog_knee(values.float(), knee), centres)
                width = float(centres[1] - centres[0])
                # 70 bins: 0 lies on the edge between the two innermost bins, each [0, width]
                # in s, so knee * expm1(width / knee) wide in return units.
                result["innermost_bin_width"] = knee * math.expm1(width / knee)
                result["outermost_bin_width"] = knee * (
                    math.expm1(abs(float(centres[-1])) / knee)
                    - math.expm1((abs(float(centres[-1])) - width) / knee)
                )
                result["knee"] = knee
                alternatives[label] = result
            entry["symlog_alternatives"] = alternatives

        for column, spec in enumerate(model.next_state_columns):
            values = next_targets[:, column].contiguous()
            add(
                f"next_{spec.name}",
                values,
                "next_state",
                (spec.low, spec.high),
                spec.knee,
            )
        add("teacher_entropy", teacher_entropy, "policy")
        total = int(outcomes.sum())
        stats["hit"] = {
            "bullets": total,
            "shares": {
                name: float(count) / total
                for name, count in zip(("missed", "enemy", "team-mate"), outcomes.tolist())
            },
        }
        stats["seconds_total"] = time.time() - began

    np.savez(out / "histograms.npz", **arrays)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out / 'stats.json'} at {datetime.now():%H:%M:%S}", flush=True)


def render(out: Path, figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    stats = json.loads((out / "stats.json").read_text())
    arrays = np.load(out / "histograms.npz")
    bar_colour, ink, muted = "#2f6db5", "#1f2328", "#8a9099"
    for key, entry in stats["targets"].items():
        counts = arrays[f"hist_{key}"]
        low, high = arrays[f"range_{key}"]
        edges = np.linspace(low, high, FINE_BINS + 1)
        figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=150)
        axis.stairs(np.maximum(counts, 0.8), edges, fill=True, color=bar_colour, linewidth=0)
        axis.set_yscale("log")
        axis.set_ylim(0.8, counts.max() * 3)
        percentiles = entry["percentiles"]
        for share, style in (("0.001", "--"), ("0.999", "--")):
            axis.axvline(percentiles[share], color=ink, linestyle=style, linewidth=1.0)
        axis.axvline(np.nan, color=ink, linestyle="--", linewidth=1.0, label="p0.1 / p99.9")
        support = entry.get("support")
        if support is not None and support["space"] == "linear":
            centres = arrays[f"centres_{key}"]
            width = centres[1] - centres[0]
            bin_edges = np.append(centres - width / 2, centres[-1] + width / 2)
            for edge in bin_edges:
                axis.axvline(edge, ymin=0, ymax=0.04, color=muted, linewidth=0.6)
            for value in (support["measured_low"], support["measured_high"]):
                axis.axvline(value, color="#c0392b", linewidth=1.2)
            for value in (bin_edges[0], bin_edges[-1]):
                axis.axvline(value, color="#c0392b", linewidth=1.2, linestyle=":")
            axis.axvline(np.nan, color="#c0392b", linewidth=1.2, label="measured range")
            axis.axvline(
                np.nan, color="#c0392b", linewidth=1.2, linestyle=":", label="support edge"
            )
            axis.set_xlim(min(low, bin_edges[0]) - width, max(high, bin_edges[-1]) + width)
            occ = support["occupancy"]["bins_holding"]
            note = (
                f"70 linear bins (ticks at bottom): 50% in {occ['0.5']}, 90% in {occ['0.9']}, "
                f"99% in {occ['0.99']}, 99.9% in {occ['0.999']}"
            )
        elif support is not None:
            occ = support["occupancy"]["bins_holding"]
            note = (
                f"70 symlog bins: 50% in {occ['0.5']}, 90% in {occ['0.9']}, 99% in "
                f"{occ['0.99']} (edges not drawn: symlog space)"
            )
        else:
            note = "no categorical support"
        if entry["share_exactly_zero"] > 0.01:
            note += f"\n{entry['share_exactly_zero']:.1%} of rows exactly 0 (one bar at 0)"
        axis.set_title(
            f"{key}  ({entry['count'] / 1e6:.1f}M rows, excess kurtosis "
            f"{entry['excess_kurtosis']:.1f})",
            fontsize=11,
            color=ink,
            loc="left",
        )
        axis.text(0.0, -0.26, note, transform=axis.transAxes, fontsize=8, color=ink)
        axis.set_ylabel("count (log)", fontsize=9)
        axis.tick_params(labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(fontsize=7, frameon=False, loc="upper right")
        figure.tight_layout()
        figure.savefig(figures / f"{key}.png")
        plt.close(figure)
        print(f"wrote {figures / f'{key}.png'}")
    occupancy_figure(stats, figures, plt)


def occupancy_figure(stats: dict, figures: Path, plt) -> None:
    """Small multiples: each critic level's share of targets per bin (70 bins, log y) on the
    current linear support and on symlog supports at knees 0.25 and 1."""
    levels = [key for key in stats["targets"] if key.startswith("value_")]
    figure, axes = plt.subplots(len(levels), 1, figsize=(6.4, 2.6 * len(levels)), dpi=150)
    styles = (("linear", "#2f6db5", "-"), ("0.25", "#c0392b", "--"), ("1", "#1f2328", ":"))
    for axis, key in zip(axes, levels):
        entry = stats["targets"][key]
        for label, colour, style in styles:
            if label == "linear":
                occ = entry["support"]["occupancy"]
                name = "linear (current)"
            else:
                occ = entry["symlog_alternatives"][label]
                name = f"symlog knee {label}"
            shares = np.maximum(np.array(occ["shares"]), 1e-9)
            axis.step(
                np.arange(len(shares)),
                shares,
                where="mid",
                color=colour,
                linestyle=style,
                linewidth=1.4,
                label=f"{name}: {occ['perplexity']:.1f} effective bins",
            )
        axis.set_yscale("log")
        axis.set_ylim(1e-7, 1.5)
        axis.set_title(key, fontsize=10, loc="left")
        axis.set_ylabel("share of targets", fontsize=8)
        axis.tick_params(labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(fontsize=7, frameon=False, loc="lower center")
    axes[-1].set_xlabel("bin (0 to 69; 3 buffer bins at each end)", fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "value_occupancy.png")
    plt.close(figure)
    print(f"wrote {figures / 'value_occupancy.png'}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(
            Path(sys.argv[2]),
            int(sys.argv[3]) if len(sys.argv) > 3 else 2048,
            int(sys.argv[4]) if len(sys.argv) > 4 else 3600,
        )
    else:
        render(Path(sys.argv[2]), Path(sys.argv[3]))
