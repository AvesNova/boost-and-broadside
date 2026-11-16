"""The README's hero clip: a policy winning a game against Imugi while outnumbered.

`record` films `games` games (`bnb.figures.video.render`, each to its game's end with the
death marks on) at one lineup (`HeroConfig.ships`: the policy's ships and Imugi's, the
break-even lineup of the crossover), seeds `first_seed` on, and writes each game's MP4 with
a JSON of its outcome beside it (`game_seed<s>.mp4`, `game_seed<s>.json`: the tally of
`bnb.figures.video.Tally`, the lineup, the frames and the file's bytes). A game whose JSON
exists is not played again, so a stopped recording continues. `pick` chooses the clip: a
win for the policy, decisive but reasonable (`decisive`); `gif` makes the README's GIF and
its poster frame from it with ffmpeg (a palette made from the clip, under `gif_bytes`).
"""

import dataclasses
import datetime
import json
import pathlib
import shutil
import subprocess
import time

from .video import FFMPEG, VideoConfig, render

CHOSEN = "chosen.json"  # the pick, beside the games


@dataclasses.dataclass(frozen=True)
class HeroConfig:
    """games: how many games to film; first_seed: the first game's seed (each game its own, on
    from it); ships: the policy's ships and Imugi's; video: the videos' settings (no
    interface, which a GIF's width leaves unreadable; the camera on the live ships; to the
    game's end, `seconds` the bound in video seconds: 70 at 4x is the real-scale elimination
    clock and the held end); the pick: a win with the game's end between `shortest` and
    `longest` game seconds, at least `spare` of the policy's ships left, at least one of
    them lost (`contested`), and its survivors' mean shield at least `shield_left` (no
    last-hit finish); the GIF: at most `gif_seconds` long (a longer clip plays faster),
    `gif_width` pixels wide at `gif_rate` frames per second, under `gif_bytes` (narrower
    until it is)."""

    games: int = 8
    first_seed: int = 0
    ships: tuple[int, int] = (5, 6)
    video: VideoConfig = VideoConfig(
        seconds=70.0, interface=False, until_end=True, hold=2.0, follow=True
    )
    shortest: float = 30.0
    longest: float = 120.0
    spare: int = 2
    contested: bool = True
    shield_left: float = 0.25
    gif_seconds: float = 24.0
    gif_width: int = 640
    gif_rate: int = 12
    gif_bytes: int = 10_000_000


def game_name(seed: int) -> str:
    """A game's file name, without its suffix (.mp4, .json): its seed."""
    return f"game_seed{seed}"


def record(policy: pathlib.Path, preset: str, config: HeroConfig, out: pathlib.Path) -> list:
    """Film the games (those not filmed before) into `out`; returns every game's record.
    Prints a line per game with the ETA as a clock time."""
    out.mkdir(parents=True, exist_ok=True)
    records, started, played = [], time.time(), 0
    seeds = range(config.first_seed, config.first_seed + config.games)
    for index, seed in enumerate(seeds):
        name = game_name(seed)
        path = out / f"{name}.json"
        if path.is_file():
            records.append(json.loads(path.read_text()))
            continue
        video = dataclasses.replace(config.video, seed=seed, ships=config.ships)
        numbers = render(policy, preset, video, out / f"{name}.mp4")
        one = dict(
            name=name,
            video=f"{name}.mp4",
            seed=seed,
            preset=preset,
            policy=str(policy),
            ships=list(config.ships),
            time_scale=video.time_scale,
            frame_rate=video.frame_rate,
            **numbers,
        )
        path.write_text(json.dumps(one, indent=1) + "\n")
        records.append(one)
        played += 1
        left = len(seeds) - index - 1
        eta = datetime.datetime.now() + datetime.timedelta(
            seconds=(time.time() - started) / played * left
        )
        print(
            f"{name}: {one['first_result']} at {one['ended_at']} s, ships left "
            f"{one['policy_left']} v {one['scripted_left']}, kills {one['policy_kills']} v "
            f"{one['scripted_kills']} ({one['seconds_taken']} s to film); "
            f"{left} to go, ETA {eta:%H:%M}",
            flush=True,
        )
    return records


def decisive(one: dict, config: HeroConfig) -> list[str]:
    """Why a game is not the clip (none: it may be): not a win, its end outside the
    duration window, too few ships to spare, uncontested, or a last-hit finish."""
    reasons = []
    if one.get("first_result") != "win":
        reasons.append(f"a {one.get('first_result') or 'game unfinished'}")
        return reasons
    if not config.shortest <= one["ended_at"] <= config.longest:
        reasons.append(f"ended at {one['ended_at']} s")
    if one["policy_left"] < config.spare:
        reasons.append(f"{one['policy_left']} ships left")
    if config.contested and one["policy_losses"] < 1:
        reasons.append("no ship lost")
    if one["policy_shield_left"] < config.shield_left:
        reasons.append(f"survivors' shield {one['policy_shield_left']}")
    return reasons


def pick(records: list, config: HeroConfig) -> tuple[dict | None, str]:
    """The clip and why: among the decisive wins, the one nearest a middle game (half the
    policy's ships left, the end at the middle of the duration window); with none, the
    win that misses the fewest rules, said so; None when no game was won."""
    wins = [one for one in records if one.get("first_result") == "win"]
    if not wins:
        return None, f"no win in {len(records)} games"
    middle = (config.shortest + config.longest) / 2

    def distance(one: dict) -> float:
        share = one["policy_left"] / config.ships[0]
        return abs(share - 0.5) + abs(one["ended_at"] - middle) / (config.longest - middle)

    passing = [one for one in wins if not decisive(one, config)]
    if passing:
        chosen = min(passing, key=distance)
        return chosen, f"decisive win, one of {len(passing)} of {len(records)} games"
    chosen = min(wins, key=lambda one: (len(decisive(one, config)), distance(one)))
    return chosen, f"no decisive win; the nearest misses: {', '.join(decisive(chosen, config))}"


def _ffmpeg() -> str:
    found = shutil.which(FFMPEG)
    if found is None:
        raise SystemExit(f"the hero clip needs ffmpeg on the PATH ({FFMPEG!r} not found)")
    return found


def _duration(path: pathlib.Path) -> float:
    probe = shutil.which("ffprobe") or "ffprobe"
    output = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(output.strip())


def gif(clip: pathlib.Path, out: pathlib.Path, config: HeroConfig) -> dict:
    """The README GIF of `clip` at `out` (.gif) and its poster frame beside it (.png, the
    clip's last frame: the end with every death mark); a clip longer than `gif_seconds`
    plays faster to fit. A palette made from the clip (only what changes between frames
    counts), Bayer dithered; narrower by an eighth until it is under `gif_bytes`. Returns
    the GIF's numbers."""
    ffmpeg = _ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = _duration(clip)
    speed = max(1.0, seconds / config.gif_seconds)
    width = config.gif_width
    while True:
        filters = (
            f"setpts=PTS/{speed:.4f},fps={config.gif_rate},scale={width}:-2:flags=lanczos,"
            "split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip)]
        subprocess.run([*command, "-filter_complex", filters, str(out)], check=True)
        size = out.stat().st_size
        if size <= config.gif_bytes or width <= 240:
            break
        width = int(width * 7 / 8) // 2 * 2
    poster = out.with_suffix(".png")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", str(clip)]
        + ["-frames:v", "1", "-update", "1", str(poster)],
        check=True,
    )
    return dict(
        gif=str(out),
        poster=str(poster),
        bytes=size,
        width=width,
        seconds=round(seconds / speed, 2),
        speed=round(speed, 3),
    )


def make(
    policy: pathlib.Path, preset: str, config: HeroConfig, out: pathlib.Path, gif_path=None
) -> dict:
    """Record the games, pick the clip and make its GIF (`gif_path`, default
    `out`/hero.gif); writes and returns the choice (`out`/chosen.json)."""
    records = record(policy, preset, config, out)
    chosen, reason = pick(records, config)
    result = dict(
        reason=reason,
        chosen=chosen,
        games=[
            dict(name=one["name"], result=one["first_result"], misses=decisive(one, config))
            for one in records
        ],
    )
    if chosen is not None:
        target = pathlib.Path(gif_path) if gif_path is not None else out / "hero.gif"
        result["gif"] = gif(out / chosen["video"], target, config)
    (out / CHOSEN).write_text(json.dumps(result, indent=1) + "\n")
    return result
