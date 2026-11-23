"""Fly one ship by keyboard in a top-down view: python -m lean.sandbox (PYTHONPATH=.:src).

Tkinter (standard library) draws it, so nothing extra needs installing. Physics ticks at the
game's 30 Hz; the view redraws as fast as Tk allows and interpolates the ships between ticks.
"""

import argparse
import ast
import dataclasses
import math
import time
import tkinter as tk

import torch

from lean.game import card, core, tuning

from .world import World, field_list, stick_index

LEVEL_NAMES = {1: "fine", 2: "sustained", 3: "full"}
POWER_NAMES = {0: "coast", 1: "dive (boost)", 2: "climb (reverse)"}
GRID_SPACINGS = (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0)
BACKGROUND, GRID, GRID_MAJOR, TEXT = "#0d1117", "#1b2330", "#2a3546", "#d8dee9"
PLAYER, OPPONENT, TRAIL, VELOCITY = "#58a6ff", "#ff6b6b", "#3b6ea8", "#7ee787"
HELP = (
    "arrows/A D turn (shift sustained, ctrl fine)   up/W dive   down/S climb   space fire   "
    "tab ship   C opponent   P pause   +/- wheel zoom   R reset   esc quit"
)


def parse_config(settings: list[str]) -> core.GameConfig:
    """The game's GameConfig (no clock) with --set name=value overrides (Python literals where
    they parse)."""
    values = {"clock_ticks": 10**9}
    for item in settings:
        name, _, text = item.partition("=")
        try:
            values[name] = ast.literal_eval(text)
        except ValueError, SyntaxError:
            values[name] = text
    return core.GameConfig(**values)


def physics_label(config: core.GameConfig) -> str:
    """The non-default GameConfig fields (any physics switch shows here), or 'default'."""
    default = core.GameConfig()
    changed = [
        f"{f.name}={getattr(config, f.name)!r}"
        for f in dataclasses.fields(config)
        if f.name != "clock_ticks" and getattr(config, f.name) != getattr(default, f.name)
    ]
    return ", ".join(changed) or "default"


class App:
    def __init__(self, args):
        self.config = parse_config(args.set)
        self.time_scale = args.time_scale
        self.world = World(
            self.config, player_type=card.TYPE_NAMES.index(args.ship), opponent_on=not args.no_opponent,
            opponent=args.opponent,
        )
        self.zoom = args.zoom  # pixels per metre
        self.seconds, self.demo = args.seconds, args.demo
        self.root = tk.Tk()
        self.root.title("B&B flight sandbox")
        if args.headless:
            self.root.withdraw()
        self.canvas = tk.Canvas(
            self.root, width=1280, height=820, background=BACKGROUND, highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.held: set[str] = set()
        self.presses: dict[str, int] = {}
        self.paused = False
        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease>", self.on_release)
        self.root.bind("<Button-4>", lambda e: self.zoom_by(1.25))
        self.root.bind("<Button-5>", lambda e: self.zoom_by(0.8))
        self.root.bind("<MouseWheel>", lambda e: self.zoom_by(1.25 if e.delta > 0 else 0.8))
        self.start = self.last = time.perf_counter()
        self.accumulator = 0.0
        self.frames = 0
        self.control = (0, 0, 0, False)  # side, level, power, fire

    # ------------------------------------------------------------------ input

    @staticmethod
    def key(event) -> str:
        return (
            "Tab"
            if event.keysym == "ISO_Left_Tab"
            else (event.keysym.lower() if len(event.keysym) == 1 else event.keysym)
        )

    def on_press(self, event):
        k = self.key(event)
        first = k not in self.held  # autorepeat sends presses while held
        self.held.add(k)
        self.presses[k] = self.presses.get(k, 0) + 1
        if first:
            if k == "Escape":
                self.root.destroy()
            elif k == "Tab":
                self.world.set_player_type((self.world.player_type + 1) % len(card.TYPE_NAMES))
            elif k == "r":
                self.world.reset()
            elif k == "c":
                self.world.toggle_opponent()
            elif k == "p":
                self.paused = not self.paused
        if k in ("plus", "equal", "KP_Add"):
            self.zoom_by(1.1)
        elif k in ("minus", "KP_Subtract"):
            self.zoom_by(1 / 1.1)
        return "break"

    def on_release(self, event):
        # X11 autorepeat sends release + press pairs; drop the key only if no press follows soon.
        k = self.key(event)
        count = self.presses.get(k, 0)
        self.root.after(
            30, lambda: self.held.discard(k) if self.presses.get(k, 0) == count else None
        )

    def zoom_by(self, factor: float):
        self.zoom = min(8.0, max(0.02, self.zoom * factor))

    def read_controls(self) -> tuple[int, int, int, bool]:
        h = self.held
        if self.demo:  # a scripted input cycle for the headless test
            phase = int((time.perf_counter() - self.start) * 2) % 6
            return (1, -1, 0, -1, 1, 0)[phase], 1 + phase % 3, phase % 3, phase % 2 == 0
        left = "Left" in h or "a" in h
        right = "Right" in h or "d" in h
        side = int(left) - int(right)
        level = (
            1
            if ("Control_L" in h or "Control_R" in h)
            else 2
            if ("Shift_L" in h or "Shift_R" in h)
            else 3
        )
        dive, climb = "Up" in h or "w" in h, "Down" in h or "s" in h
        power = 1 if dive and not climb else 2 if climb and not dive else 0
        return side, level, power, "space" in h

    # ------------------------------------------------------------------ loop

    def frame(self):
        now = time.perf_counter()
        elapsed, self.last = now - self.last, now
        h = self.world.h
        if not self.paused:
            self.accumulator = min(self.accumulator + elapsed, 5 * h)
            while self.accumulator >= h:
                self.control = self.read_controls()
                side, level, power, fire = self.control
                self.world.advance(stick_index(side, level), power, fire)
                self.accumulator -= h
        self.draw(self.accumulator / h)
        self.frames += 1
        if self.seconds and now - self.start > self.seconds:
            print(
                f"ran {now - self.start:.1f} s at {self.time_scale:g}x real time: {self.world.tick} ticks, {self.frames} frames, "
                f"readout {self.world.readout()}"
            )
            self.root.destroy()
            return
        self.root.after(1, self.frame)

    def run(self):
        self.root.after(1, self.frame)
        self.root.mainloop()

    # ------------------------------------------------------------------ drawing

    def draw(self, alpha: float):
        c, w = self.canvas, self.world
        c.delete("all")
        width, height = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        poses = []
        for (x0, y0, a0), (x1, y1, a1) in zip(w.previous, w.poses()):
            turn = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            poses.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0), a0 + alpha * turn))
        camera_x, camera_y = poses[0][0], poses[0][1]
        z = self.zoom

        def screen(x, y):
            return width / 2 + (x - camera_x) * z, height / 2 - (y - camera_y) * z

        # Grid: the finest spacing at least 40 px apart, every fourth line brighter.
        spacing = next((s for s in GRID_SPACINGS if s * z >= 40), GRID_SPACINGS[-1])
        half_w, half_h = width / 2 / z, height / 2 / z
        for axis, centre, half in ((0, camera_x, half_w), (1, camera_y, half_h)):
            first = math.floor((centre - half) / spacing)
            for i in range(first, math.ceil((centre + half) / spacing) + 1):
                colour = GRID_MAJOR if i % 4 == 0 else GRID
                if axis == 0:
                    sx, _ = screen(i * spacing, 0)
                    c.create_line(sx, 0, sx, height, fill=colour)
                else:
                    _, sy = screen(0, i * spacing)
                    c.create_line(0, sy, width, sy, fill=colour)

        # Arena boundary (damage starts) and the end of the ramp, as polylines.
        for radius, colour, dash in (
            (self.config.arena_radius, "#d29922", None),
            (self.config.arena_radius + self.config.boundary_ramp_width, "#8b3a3a", (4, 4)),
        ):
            points = [
                p
                for i in range(361)
                for p in screen(
                    radius * math.cos(math.radians(i)), radius * math.sin(math.radians(i))
                )
            ]
            c.create_line(*points, fill=colour, width=2, dash=dash)

        # Fields: the outer and inner edges of each band, blue raises the index, purple lowers it.
        for fx, fy, radius, band, level in field_list(w.state):
            colour = "#4d7cfe" if level > 0 else "#b26bff"
            for r in (radius - band / 2, radius + band / 2):
                if r > 0:
                    x0, y0 = screen(fx - r, fy + r)
                    x1, y1 = screen(fx + r, fy - r)
                    c.create_oval(
                        x0, y0, x1, y1, outline=colour, dash=(3, 3) if r > radius else None
                    )

        if len(w.trail) > 1:
            c.create_line(*[p for x, y in w.trail for p in screen(x, y)], fill=TRAIL, width=2)

        s = w.state
        for ship in range(2):  # bullets as one-tick streaks
            colour = "#ffd166" if ship == 0 else "#ff9e64"
            for k in range(s.bullet_x.shape[2]):
                if int(s.bullet_moves_left[0, ship, k]) > 0 and bool(s.bullet_real[0, ship, k]):
                    bx, by = float(s.bullet_x[0, ship, k]), float(s.bullet_y[0, ship, k])
                    step = float(s.bullet_speed[0, ship, k]) * w.h
                    tail = (
                        bx - step * float(s.bullet_dir_x[0, ship, k]),
                        by - step * float(s.bullet_dir_y[0, ship, k]),
                    )
                    c.create_line(*screen(*tail), *screen(bx, by), fill=colour, width=2)

        for ship, colour in ((1, OPPONENT), (0, PLAYER)):
            if not bool(s.alive[0, ship]):
                continue
            x, y, heading = poses[ship]
            self.draw_ship(screen, x, y, heading, colour)
            track = heading - float(s.angle[0, ship])  # velocity vector, half a second long
            reach = 0.5 * float(s.speed[0, ship])
            c.create_line(
                *screen(x, y),
                *screen(x + reach * math.cos(track), y + reach * math.sin(track)),
                fill=VELOCITY,
            )
            if ship == 1:
                self.opponent_marker(
                    screen(x, y), width, height, math.hypot(x - camera_x, y - camera_y)
                )

        self.draw_readout()
        self.draw_strip(width, height)
        c.create_text(
            12, height - 12, text=HELP, anchor="sw", fill="#8b949e", font=("TkFixedFont", 9)
        )

    def draw_ship(self, screen, x, y, heading, colour):
        size = max(12.0, 10.0 / self.zoom)  # metres; never under 10 px
        cos, sin = math.cos(heading), math.sin(heading)
        body = [(1.6, 0.0), (-1.0, 0.8), (-0.5, 0.0), (-1.0, -0.8)]
        points = [
            p
            for u, v in body
            for p in screen(x + size * (u * cos - v * sin), y + size * (u * sin + v * cos))
        ]
        self.canvas.create_polygon(*points, fill=colour, outline="white")
        r = 12.0 * self.zoom  # hit radius
        cx, cy = screen(x, y)
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#30363d")

    def opponent_marker(self, point, width, height, distance):
        px, py = point
        if 0 <= px <= width and 0 <= py <= height:
            return
        dx, dy = px - width / 2, py - height / 2
        scale = min((width / 2 - 30) / max(abs(dx), 1e-6), (height / 2 - 30) / max(abs(dy), 1e-6))
        ex, ey = width / 2 + dx * scale, height / 2 + dy * scale
        self.canvas.create_oval(ex - 6, ey - 6, ex + 6, ey + 6, fill=OPPONENT)
        self.canvas.create_text(
            ex, ey - 14, text=f"{distance:.0f} m", fill=OPPONENT, font=("TkFixedFont", 9)
        )

    def draw_readout(self):
        w, r = self.world, self.world.readout()
        side, level, power, fire = self.control
        stick = (
            "straight"
            if side == 0
            else f"{'left' if side > 0 else 'right'} {LEVEL_NAMES[level]}"
            f" [index {stick_index(side, level)}]"
        )
        radius = "-" if math.isinf(r["turn_radius"]) else f"{r['turn_radius']:6.0f} m"
        lines = [
            f"ship   {card.TYPE_NAMES[w.player_type]}  (tab)",
            f"physics {physics_label(self.config)}",
            f"time   {self.time_scale:g}x real (game {tuning.TIME_FACTOR:g}x)",
            f"stick  {stick}",
            f"power  {POWER_NAMES[power]}{'   FIRE' if fire else ''}",
            "",
            f"speed  {r['speed']:6.1f} m/s",
            f"turn   {r['turn_rate_degrees']:6.1f} deg/s   radius {radius}",
            f"g      {r['lateral_g']:6.2f} lateral (g = {card.GRAVITY} m/s2)",
            f"AoA    {r['angle_of_attack_degrees']:6.2f} deg  (stall {r['stall_degrees']:.1f})",
            f"store  {100 * r['store']:6.1f} %",
            "",
            f"shield {100 * r['shield']:5.0f} %   deaths {w.deaths}"
            f"   opponent kills {w.opponent_kills}",
            f"opponent {'on' if w.opponent_on else 'off'}  (c)" + ("   PAUSED" if self.paused else ""),
        ]
        self.canvas.create_rectangle(
            6, 6, 430, 20 + 17 * len(lines), fill="#161b22", outline="#30363d"
        )
        for row, line in enumerate(lines):
            self.canvas.create_text(
                14, 12 + 17 * row, text=line, anchor="nw", fill=TEXT, font=("TkFixedFont", 10)
            )
        top = 14 + 17 * 10  # the store bar beside the store line
        self.canvas.create_rectangle(250, top, 420, top + 12, outline="#8b949e")
        self.canvas.create_rectangle(
            250, top, 250 + 170 * min(r["store"], 1.0), top + 12, fill="#e3b341", outline=""
        )

    def draw_strip(self, width, height):
        """Speed (green, 0 to the top label) and store (yellow, 0 to 100 %) over the last 10 s."""
        history = self.world.history
        box_w, box_h = 360, 120
        left, top = width - box_w - 12, height - box_h - 30
        self.canvas.create_rectangle(
            left, top, left + box_w, top + box_h, fill="#161b22", outline="#30363d"
        )
        if len(history) < 2:
            return
        top_speed = max(400.0, max(v for v, _ in history))
        steps = history.maxlen - 1
        offset = history.maxlen - len(history)
        for index, scale, colour in ((0, top_speed, VELOCITY), (1, 1.0, "#e3b341")):
            points = [
                p
                for i, sample in enumerate(history)
                for p in (
                    left + box_w * (offset + i) / steps,
                    top + box_h * (1 - min(sample[index] / scale, 1.0)),
                )
            ]
            self.canvas.create_line(*points, fill=colour, width=2)
        self.canvas.create_text(
            left + 4,
            top + 2,
            anchor="nw",
            fill=VELOCITY,
            font=("TkFixedFont", 8),
            text=f"speed (top {top_speed:.0f} m/s)",
        )
        self.canvas.create_text(
            left + box_w - 4,
            top + 2,
            anchor="ne",
            fill="#e3b341",
            font=("TkFixedFont", 8),
            text="store (top 100 %)   last 10 s",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ship", choices=card.TYPE_NAMES, default=card.TYPE_NAMES[0])
    parser.add_argument(
        "--no-opponent", action="store_true", help="start without the opponent (C toggles it)"
    )
    parser.add_argument(
        "--opponent",
        choices=("scripted", "chaser"),
        default="scripted",
        help="the scripted opponent (the yardstick, as in training and evaluation) or the chaser",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=tuning.TIME_FACTOR,
        metavar="K",
        help="fly at K times real time (lengths real; speeds and turn rates xK, accelerations and "
        "gravity xK^2); default the game's 2",
    )
    parser.add_argument("--zoom", type=float, default=0.6, help="pixels per metre")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a GameConfig field, e.g. --set field_count=0 (no fields) or --set fog=False",
    )
    parser.add_argument("--seconds", type=float, default=0.0, help="quit after this long (testing)")
    parser.add_argument("--headless", action="store_true", help="hide the window (testing)")
    parser.add_argument(
        "--demo", action="store_true", help="scripted inputs instead of the keyboard (testing)"
    )
    torch.set_num_threads(1)
    arguments = parser.parse_args()
    if arguments.time_scale <= 0.0:
        parser.error("--time-scale must be positive")
    with tuning.time_scale(arguments.time_scale):
        App(arguments).run()


if __name__ == "__main__":
    main()
