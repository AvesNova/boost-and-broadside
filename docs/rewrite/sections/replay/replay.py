"""Replay videos of scripted-vs-scripted games: simulate on the CPU, draw with PIL, encode
with ffmpeg (H.264, yuv420p, faststart). No game or agent code is changed: the game is
lean.game (deal, tick) at batch size 1 and every ship is the default scripted agent
(ScriptedConfig(), the yardstick), both teams in one call as in evaluation. The labels are the
agent's own reasons for the tick's decision (lean.scripted.opponent.explain).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:src uv run --no-sync python \
        docs/rewrite/sections/replay/replay.py --seeds 0 1 2 --out DIR

See NOTES.md beside this file.
"""

import argparse
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lean import game, scripted
from lean.game import core
from lean.scripted.opponent import TACTICS, explain

SIZE = 1080  # output frame, square
SUPER = 2  # drawn at SUPER x and reduced (antialiasing)
FPS = 30
TRAIL_TICKS = 45  # 1.5 s
HOLD_SECONDS = 3.0  # the final frame held, with the result
DEATH_MARK_TICKS = 45
TEAM_COLOURS = ((80, 160, 255), (255, 95, 80))
TEAM_NAMES = ("Blue", "Red")
BACKGROUND = (14, 17, 24)
SHORT = {
    "disengaged": "SEEK",
    "merge": "MERGE",
    "offensive": "OFF",
    "turning": "TURN",
    "defensive": "BREAK",
}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ---------------------------------------------------------------------- simulation


@torch.no_grad()
def simulate(config: core.GameConfig, seed: int, sizes=(4, 4)) -> dict:
    """One whole game; per frame (the deal, then after each tick) every ship's pose, state and
    the label of the decision taken from that frame. Arrays (T, N) unless noted."""
    generator = torch.Generator().manual_seed(seed)
    slots = sum(sizes)
    state = game.deal(config, 1, slots, torch.tensor([list(sizes)]), generator, "cpu")
    memory = scripted.fresh_memory(1, slots, generator, "cpu")
    agent = scripted.ScriptedConfig()
    frames = []
    deaths = []  # (tick, x, y, team)
    result = -1
    while True:
        reasons = explain(agent, state, None, None, memory)
        weights = torch.stack([reasons.weights[name] for name in TACTICS], -1)[0]  # (N, 5)
        frame = {
            "x": state.x[0].clone(),
            "y": state.y[0].clone(),
            "heading": torch.atan2(state.heading_sin[0], state.heading_cos[0]),
            "alive": state.alive[0].clone(),
            "shield": (state.shield / state.constants["max_shield"])[0],
            "store": (state.energy * state.constants["inverse_max_energy"])[0],
            "speed": (state.speed / state.constants["corner_speed"])[0],
            "seen": (state.ticks_since_seen[0] == 0) if state.seen is not None else None,
            "weights": weights,
            "plan": torch.sign(reasons.last_pass[0]),
            "target": reasons.marginals.target[0].clone(),
            "bullets": _real_bullets(state),
        }
        frames.append(frame)
        if result >= 0:
            break
        uniforms = scripted.draw_uniforms(1, slots, generator, "cpu")
        action, memory, _ = scripted.decide(agent, state, None, None, uniforms, memory)
        previous = state
        state, out = game.tick(config, state, action, generator)
        for slot in torch.nonzero(out.died[0]).flatten().tolist():
            deaths.append(
                (
                    int(state.tick[0]),
                    float(previous.x[0, slot]),
                    float(previous.y[0, slot]),
                    int(state.team[0, slot]),
                )
            )
        if bool(out.done[0]):
            result = int(out.result[0])
    team = state.team[0].numpy()
    exists = state.exists[0].numpy()
    stacked = {
        name: torch.stack([f[name] for f in frames]).numpy()
        for name in (
            "x",
            "y",
            "heading",
            "alive",
            "shield",
            "store",
            "speed",
            "weights",
            "plan",
            "target",
        )
    }
    stacked["seen"] = (
        torch.stack([f["seen"] for f in frames]).numpy() if frames[0]["seen"] is not None else None
    )
    stacked["bullets"] = [f["bullets"] for f in frames]
    alive = stacked["alive"][-1] & exists
    left = [int((alive & (team == k)).sum()) for k in (0, 1)]
    wiped = min(left) == 0
    return {
        **stacked,
        "team": team,
        "exists": exists,
        "fields": state.fields[0].numpy(),  # rows x, y, radius, band, target ln n
        "deaths": deaths,
        "result": result,
        "wiped": wiped,
        "ticks": int(state.tick[0]),
        "left": left,
        "seed": seed,
        "arena": config.arena_radius,
        "ramp": config.boundary_ramp_width,
        "clock": config.clock_ticks,
        "tick_length": config.tick_length,
    }


def _real_bullets(state) -> np.ndarray:
    """(B, 3) x, y, owner team of the real bullets in flight (ghosts are not drawn)."""
    active = (state.bullet_moves_left[0] > 0) & state.bullet_real[0]  # (N, K)
    owner = state.team[0][:, None].expand_as(active)
    return torch.stack(
        [state.bullet_x[0][active], state.bullet_y[0][active], owner[active].float()], -1
    ).numpy()


# ---------------------------------------------------------------------- camera


def camera(record: dict) -> np.ndarray:
    """(T, 3) centre x, y and half width in metres: the living ships' bounding box with a
    margin, at least 1,100 m and at most the whole arena, smoothed forward and backward."""
    xs, ys, alive = record["x"], record["y"], record["alive"] & record["exists"]
    whole = record["arena"] + record["ramp"]
    raw = np.zeros((len(xs), 3))
    for t in range(len(xs)):
        live = alive[t] if alive[t].any() else record["exists"]
        x, y = xs[t][live], ys[t][live]
        low_x, high_x, low_y, high_y = x.min(), x.max(), y.min(), y.max()
        half = max(high_x - low_x, high_y - low_y) / 2 + 450.0
        raw[t] = ((low_x + high_x) / 2, (low_y + high_y) / 2, min(max(half, 1100.0), whole))
    smooth = raw.copy()
    keep = math.exp(-1.0 / (1.0 * FPS))  # 1 s time constant
    for t in range(1, len(raw)):
        smooth[t] = keep * smooth[t - 1] + (1 - keep) * smooth[t]
    for t in range(len(raw) - 2, -1, -1):
        smooth[t] = keep * smooth[t + 1] + (1 - keep) * smooth[t]
    # Never cut a ship off: the half width covers every living ship about the smoothed
    # centre with a margin, held for a short window and lightly smoothed.
    need = np.zeros(len(raw))
    for t in range(len(raw)):
        live = alive[t] if alive[t].any() else record["exists"]
        reach = np.maximum(np.abs(xs[t][live] - smooth[t, 0]), np.abs(ys[t][live] - smooth[t, 1]))
        need[t] = min(reach.max() + 350.0, whole)
    window = FPS // 2
    held = np.array([need[max(0, t - window) : t + window + 1].max() for t in range(len(need))])
    half = np.maximum(smooth[:, 2], held)
    keep = math.exp(-1.0 / (0.3 * FPS))
    for t in range(1, len(half)):
        half[t] = keep * half[t - 1] + (1 - keep) * half[t]
    for t in range(len(half) - 2, -1, -1):
        half[t] = keep * half[t + 1] + (1 - keep) * half[t]
    smooth[:, 2] = np.maximum(half, need)
    return smooth


# ---------------------------------------------------------------------- drawing


def _fade(colour, amount):
    """colour blended toward the background by amount (0 the colour, 1 the background)."""
    return tuple(int(c + (b - c) * amount) for c, b in zip(colour, BACKGROUND))


class Painter:
    def __init__(self, record: dict, title: str):
        self.r = record
        self.title = title
        self.cam = camera(record)
        s = SUPER
        self.font = ImageFont.truetype(BOLD_PATH, 15 * s)
        self.small = ImageFont.truetype(FONT_PATH, 13 * s)
        self.hud = ImageFont.truetype(BOLD_PATH, 26 * s)
        self.big = ImageFont.truetype(BOLD_PATH, 44 * s)
        self.death_marks = record["deaths"]

    def frame(self, t: int, final: bool = False) -> Image.Image:
        s, r = SUPER, self.r
        width = SIZE * s
        image = Image.new("RGB", (width, width), BACKGROUND)
        draw = ImageDraw.Draw(image)
        cx, cy, half = self.cam[t]
        top = 70 * s  # HUD band
        view = width - top
        scale = view / (2 * half)

        def to_pixels(x, y):
            return ((x - cx) * scale + width / 2, top + view / 2 - (y - cy) * scale)

        # Fields: high index (slow) warm, low index cool, faint.
        for fx, fy, radius, band, level in r["fields"].T:
            px, py = to_pixels(fx, fy)
            outer = (radius + band / 2) * scale
            colour = (60, 48, 30) if level > 0 else (26, 44, 58)
            draw.ellipse((px - outer, py - outer, px + outer, py + outer), fill=colour)
        # Arena edge and the end of the damage ramp; a grid every 500 m.
        grid = 500.0
        g0 = math.floor((cx - half) / grid) * grid
        for k in range(int(2 * half / grid) + 3):
            gx, _ = to_pixels(g0 + k * grid, 0)
            draw.line((gx, top, gx, width), fill=(26, 30, 38), width=s)
        g0 = math.floor((cy - half) / grid) * grid
        for k in range(int(2 * half / grid) + 3):
            _, gy = to_pixels(0, g0 + k * grid)
            draw.line((0, gy, width, gy), fill=(26, 30, 38), width=s)
        for radius, colour in (
            (r["arena"], (210, 190, 60)),
            (r["arena"] + r["ramp"], (170, 50, 50)),
        ):
            ox, oy = to_pixels(-radius, radius)
            ex, ey = to_pixels(radius, -radius)
            draw.ellipse((ox, oy, ex, ey), outline=colour, width=2 * s)

        team, alive = r["team"], r["alive"][t] & r["exists"]
        xs, ys = r["x"], r["y"]
        # Trails.
        for i in np.nonzero(r["exists"])[0]:
            start = max(0, t - TRAIL_TICKS)
            points = [
                to_pixels(xs[k, i], ys[k, i]) for k in range(start, t + 1) if r["alive"][k, i]
            ]
            colour = TEAM_COLOURS[team[i]]
            for k in range(1, len(points)):
                age = (len(points) - k) / TRAIL_TICKS
                draw.line(
                    (*points[k - 1], *points[k]), fill=_fade(colour, 0.25 + 0.7 * age), width=2 * s
                )
        # Target lines (from each living ship that is not seeking to its target).
        for i in np.nonzero(alive)[0]:
            tactic = int(np.argmax(r["weights"][t, i]))
            j = int(r["target"][t, i])
            if TACTICS[tactic] != "disengaged" and alive[j] and team[j] != team[i]:
                draw.line(
                    (*to_pixels(xs[t, i], ys[t, i]), *to_pixels(xs[t, j], ys[t, j])),
                    fill=_fade(TEAM_COLOURS[team[i]], 0.8),
                    width=s,
                )
        # Bullets.
        for bx, by, owner in r["bullets"][t]:
            px, py = to_pixels(bx, by)
            d = 2.5 * s
            colour = (255, 235, 140) if owner == 0 else (255, 170, 230)
            draw.ellipse((px - d, py - d, px + d, py + d), fill=colour)
        # Death marks.
        for tick, dx, dy, dteam in self.death_marks:
            if tick <= t < tick + DEATH_MARK_TICKS or (final and tick <= t):
                px, py = to_pixels(dx, dy)
                d = 12 * s
                colour = _fade(TEAM_COLOURS[dteam], 0.3)
                draw.line((px - d, py - d, px + d, py + d), fill=colour, width=3 * s)
                draw.line((px - d, py + d, px + d, py - d), fill=colour, width=3 * s)
        # Ships.
        for i in np.nonzero(alive)[0]:
            px, py = to_pixels(xs[t, i], ys[t, i])
            h = r["heading"][t, i]
            colour = TEAM_COLOURS[team[i]]
            length, back = 15 * s, 9 * s
            nose = (px + length * math.cos(h), py - length * math.sin(h))
            left = (
                px - back * math.cos(h) - 8 * s * math.sin(h),
                py + back * math.sin(h) - 8 * s * math.cos(h),
            )
            right = (
                px - back * math.cos(h) + 8 * s * math.sin(h),
                py + back * math.sin(h) + 8 * s * math.cos(h),
            )
            seen = True if r["seen"] is None else bool(r["seen"][t, i])
            draw.polygon(
                (nose, left, right), fill=colour if seen else None, outline=colour, width=2 * s
            )
            # Shield bar (white/grey) and store bar (cyan) under the ship.
            bar_w, bar_h = 30 * s, 4 * s
            bx, by = px - bar_w / 2, py + 16 * s
            for row, (value, colour_bar) in enumerate(
                ((r["shield"][t, i], (235, 235, 235)), (r["store"][t, i], (90, 210, 210)))
            ):
                y0 = by + row * (bar_h + s)
                draw.rectangle((bx, y0, bx + bar_w, y0 + bar_h), fill=(50, 54, 62))
                draw.rectangle(
                    (bx, y0, bx + bar_w * float(np.clip(value, 0, 1)), y0 + bar_h), fill=colour_bar
                )
            # Label: slot, dominant tactic (and its plan when turning), a runner-up when close.
            w = r["weights"][t, i]
            order = np.argsort(-w)
            name = TACTICS[order[0]]
            label = SHORT[name]
            if name == "turning":
                label += " 2c" if r["plan"][t, i] > 0 else " 1c" if r["plan"][t, i] < 0 else ""
            if w[order[0]] < 0.6:
                label += "/" + SHORT[TACTICS[order[1]]]
            draw.text(
                (px + 14 * s, py - 30 * s),
                f"{i} {label}",
                font=self.font,
                fill=colour,
                stroke_width=2 * s,
                stroke_fill=BACKGROUND,
            )
            draw.text(
                (px + 14 * s, py - 12 * s),
                f"{r['speed'][t, i]:.2f}vc",
                font=self.small,
                fill=(170, 175, 185),
                stroke_width=2 * s,
                stroke_fill=BACKGROUND,
            )

        # Minimap (whole arena) bottom right.
        mm = 150 * s
        mx0, my0 = width - mm - 12 * s, width - mm - 12 * s
        whole = r["arena"] + r["ramp"]
        draw.rectangle((mx0, my0, mx0 + mm, my0 + mm), fill=(10, 12, 16), outline=(60, 64, 72))

        def mini(x, y):
            return (mx0 + mm / 2 + x / whole * mm / 2, my0 + mm / 2 - y / whole * mm / 2)

        a0, a1 = mini(-r["arena"], r["arena"]), mini(r["arena"], -r["arena"])
        draw.ellipse((*a0, *a1), outline=(210, 190, 60), width=s)
        c0, c1 = mini(cx - half, cy + half), mini(cx + half, cy - half)
        draw.rectangle((*c0, *c1), outline=(120, 120, 130), width=s)
        for i in np.nonzero(alive)[0]:
            px, py = mini(xs[t, i], ys[t, i])
            draw.ellipse(
                (px - 3 * s, py - 3 * s, px + 3 * s, py + 3 * s), fill=TEAM_COLOURS[team[i]]
            )

        # HUD.
        draw.rectangle((0, 0, width, top), fill=(8, 10, 14))
        seconds = t * r["tick_length"]
        counts = [int((alive & (team == k)).sum()) for k in (0, 1)]
        total = r["clock"] * r["tick_length"]
        clock = f"{int(seconds // 60)}:{seconds % 60:04.1f} / "
        clock += f"{int(total // 60)}:{round(total % 60):02d}"
        draw.text((16 * s, 18 * s), clock, font=self.hud, fill=(230, 230, 230))
        draw.text((260 * s, 18 * s), f"Blue {counts[0]}", font=self.hud, fill=TEAM_COLOURS[0])
        draw.text((400 * s, 18 * s), f"Red {counts[1]}", font=self.hud, fill=TEAM_COLOURS[1])
        draw.text(
            (width - 16 * s, 24 * s), self.title, font=self.small, fill=(160, 165, 175), anchor="ra"
        )
        legend = (
            "labels: OFF chase | TURN 1c/2c | BREAK defend | MERGE | SEEK no fight;  "
            "bars: shield, store;  vc: speed / corner speed;  hollow: hidden from enemy"
        )
        draw.text((16 * s, top + 8 * s), legend, font=self.small, fill=(120, 125, 135))
        if final:
            text = self.result_text()
            box = draw.textbbox((width / 2, width / 2), text, font=self.big, anchor="mm")
            pad = 18 * s
            draw.rectangle(
                (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                fill=(8, 10, 14),
                outline=(200, 200, 200),
                width=2 * s,
            )
            draw.text(
                (width / 2, width / 2),
                text,
                font=self.big,
                fill=(240, 240, 240),
                anchor="mm",
                align="center",
            )
        return image.reduce(SUPER)

    def result_text(self) -> str:
        r = self.r
        seconds = r["ticks"] * r["tick_length"]
        when = f"{int(seconds // 60)}:{seconds % 60:04.1f}"
        if r["result"] == 2:
            head = "Draw"
        else:
            head = f"{TEAM_NAMES[r['result']]} wins"
        if r["wiped"]:
            how = f"wipe at {when}"
        else:
            left = r["left"]
            how = (
                f"clock: {left[0]}-{left[1]} ships alive"
                if left[0] != left[1]
                else f"clock: {left[0]}-{left[1]}, shield lead"
            )
        return f"{head}\n{how}"


def encode(record: dict, title: str, path: Path) -> None:
    painter = Painter(record, title)
    frames = len(record["x"])
    command = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{SIZE}x{SIZE}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for t in range(frames):
        process.stdin.write(painter.frame(t).tobytes())
    final = painter.frame(frames - 1, final=True).tobytes()
    for _ in range(int(HOLD_SECONDS * FPS)):
        process.stdin.write(final)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--still",
        type=int,
        nargs="*",
        default=[],
        help="also save these frames (ticks) as PNG, for checking",
    )
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    arguments.out.mkdir(parents=True, exist_ok=True)
    config = core.GameConfig()  # the core game: map, fields, fog, 1,800-tick clock, P-51D
    for number, seed in enumerate(arguments.seeds, 1):
        started = time.perf_counter()
        record = simulate(config, seed)
        simulated = time.perf_counter() - started
        title = f"game {number} of {len(arguments.seeds)}, seed {seed}, scripted 4v4"
        painter = Painter(record, title)
        print(
            f"seed {seed}: {record['ticks']} ticks, {painter.result_text()!r}, "
            f"simulated in {simulated:.0f} s",
            flush=True,
        )
        for t in arguments.still:
            if t < len(record["x"]):
                painter.frame(t).save(arguments.out / f"seed{seed}_t{t:04d}.png")
        path = arguments.out / f"scripted4v4_seed{seed}.mp4"
        encode(record, title, path)
        print(f"  wrote {path} in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
