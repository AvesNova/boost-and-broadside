"""Hero-clip search for a v1 checkpoint: headless batches, then deterministic re-render.

A batch is (seed, num_envs, n0, n1). The same batch replayed on the same GPU reproduces
every env's game (checked by `render`, which compares each rendered game to the log), so a
game is identified by (batch seed, env index) and filmed by replaying its batch.

  hero.py batch  CKPT --n0 8 --n1 14 --envs 512 --seeds 0-3 --out games.jsonl
  hero.py render CKPT --n0 8 --n1 14 --envs 512 --seed 2 --pick 17,90 --outdir D --stem 8v14

Unlike evaluate_matchup, finished envs are never reset (the capture convention): a game
ends at elimination or at max_steps, and its terminal state is held.
"""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import torch  # noqa: E402

from boost_and_broadside.config import EnvConfig, ShipConfig  # noqa: E402
from boost_and_broadside.config.defaults import MODEL_CONFIG, SHIP_CONFIG  # noqa: E402
from boost_and_broadside.evaluation.agents import ResolvedAgent, resolve_agent_spec  # noqa: E402
from boost_and_broadside.evaluation.environment import create_evaluation_env  # noqa: E402
from boost_and_broadside.evaluation.match import MatchRunner  # noqa: E402
from boost_and_broadside.evaluation.tournament import recorded_field_map  # noqa: E402
from boost_and_broadside.modes.capture import _open_encoder, _write_gif  # noqa: E402
from boost_and_broadside.train.rl.checkpoint_schema import load_checkpoint_payload  # noqa: E402
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig  # noqa: E402

TIMELINE_EVERY = 8


class EnvSlice:
    """A view of one env of a batched TensorState, as env 0 (what the renderer draws)."""

    def __init__(self, state, index: int, batch: int):
        self._state, self._i, self._b = state, index, batch

    def __getattr__(self, name):
        value = getattr(self._state, name)
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == self._b:
            return value[self._i : self._i + 1]
        return value


def setup(ckpt: str, n0: int, n1: int, envs: int, seed: int, max_steps: int, scenario: str):
    payload = load_checkpoint_payload(ckpt, map_location="cpu")
    recorded_ship = ShipConfig(**payload["ship_config"])
    if recorded_ship != SHIP_CONFIG:
        raise SystemExit("checkpoint ship_config differs from main's SHIP_CONFIG")
    base_env = EnvConfig(**payload["env_config"])
    field_map = recorded_field_map(payload, base_env, Path(ckpt).parent.name)
    env_config = replace(base_env, num_ships=n0 + n1, max_episode_steps=max_steps)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy = resolve_agent_spec(ckpt, SHIP_CONFIG, MODEL_CONFIG, "cuda", num_ships=n0 + n1)
    if scenario == "self":
        other = ResolvedAgent("policy", policy.agent, bundle=policy.bundle)
    else:
        other = resolve_agent_spec("scripted", SHIP_CONFIG, MODEL_CONFIG, "cuda")
    env = create_evaluation_env(
        envs, SHIP_CONFIG, env_config, device, field_map_config=field_map, field_map_seed=seed
    )
    runner = MatchRunner(
        env,
        [policy, other],
        team0_index=torch.zeros(envs, dtype=torch.long, device=device),
        team1_index=torch.ones(envs, dtype=torch.long, device=device),
        ship_config=SHIP_CONFIG,
        num_ships=n0 + n1,
    )
    env.reset(options={"team_sizes": (n0, n1)}, seed=seed)
    runner.init_hidden()
    return env, runner


def counts(state):
    alive, team = state.ship_alive, state.ship_team_id
    return (alive & (team == 0)).sum(1), (alive & (team == 1)).sum(1)


def health_sums(state):
    alive, team = state.ship_alive, state.ship_team_id
    h = state.ship_health * alive
    return (h * (team == 0)).sum(1), (h * (team == 1)).sum(1)


def play(env, runner, envs, max_steps, on_frame=None, stop_when=None):
    """Step until every env has ended (or `stop_when` says stop). Returns per-env records."""
    ended = torch.full((envs,), -1, dtype=torch.long, device="cuda")
    final0 = torch.zeros(envs, dtype=torch.long, device="cuda")
    final1 = torch.zeros(envs, dtype=torch.long, device="cuda")
    timeline = []
    step = 0
    with torch.no_grad():
        while True:
            dones, truncated = runner.step()
            if on_frame is not None:
                on_frame(step, env.state, ended)
            c0, c1 = counts(env.state)
            if step % TIMELINE_EVERY == 0:
                timeline.append(torch.stack([c0, c1], 1).to(torch.int8))
            newly = (dones | truncated) & (ended < 0)
            ended[newly] = step
            final0[newly] = c0[newly]
            final1[newly] = c1[newly]
            step += 1
            if stop_when is not None:
                if stop_when(step, ended):
                    break
            elif bool((ended >= 0).all()) or step >= max_steps + 4:
                break
    tl = torch.stack(timeline, 1).cpu()  # (B, T, 2)
    return ended.cpu(), final0.cpu(), final1.cpu(), tl


def records(seed, n0, n1, envs, ended, f0, f1, tl, scenario):
    out = []
    for i in range(envs):
        s0, s1, L = int(f0[i]), int(f1[i]), int(ended[i])
        winner = "blue" if s0 > 0 and s1 == 0 else "red" if s1 > 0 and s0 == 0 else "tie"
        t = tl[i, : L // TIMELINE_EVERY + 1]
        ours, theirs = t[:, 0].tolist(), t[:, 1].tolist()
        out.append(
            dict(
                scenario=scenario,
                n0=n0,
                n1=n1,
                envs=envs,
                seed=seed,
                env=i,
                winner=winner,
                blue_left=s0,
                red_left=s1,
                steps=L + 1,
                ours=ours,
                theirs=theirs,
            )
        )
    return out


def cmd_batch(a):
    for seed in a.seeds:
        t0 = time.time()
        env, runner = setup(a.ckpt, a.n0, a.n1, a.envs, seed, a.max_steps, a.scenario)
        ended, f0, f1, tl = play(env, runner, a.envs, a.max_steps)
        recs = records(seed, a.n0, a.n1, a.envs, ended, f0, f1, tl, a.scenario)
        with open(a.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        wins = sum(r["winner"] == "blue" for r in recs)
        print(
            f"{a.scenario} {a.n0}v{a.n1} seed {seed}: {wins}/{a.envs} blue wins, "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        del env, runner
        torch.cuda.empty_cache()


def cmd_render(a):
    picks = [int(p) for p in a.pick.split(",")]
    log = {}
    if a.log:
        for line in open(a.log):
            r = json.loads(line)
            if (r["seed"], r["n0"], r["n1"], r["envs"], r.get("scenario", "vs_scripted")) == (
                a.seed,
                a.n0,
                a.n1,
                a.envs,
                a.scenario,
            ):
                log[r["env"]] = r
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    renderer = GameRenderer(SHIP_CONFIG, RenderConfig(window_size=a.window, show_ui=False))
    env, runner = setup(a.ckpt, a.n0, a.n1, a.envs, a.seed, a.max_steps, a.scenario)
    hold = round(a.hold_ms / 1000 * a.fps)
    paths = {i: outdir / f"{a.stem}_seed{a.seed:03d}_env{i:03d}.mp4" for i in picks}
    encoders = {i: _open_encoder(paths[i], a.window, a.fps) for i in picks}
    open_set = set(picks)

    def on_frame(step, state, ended):
        for i in list(open_set):
            e = int(ended[i])
            if e >= 0 and step - e > hold:
                encoders[i].stdin.close()
                encoders[i].wait()
                open_set.discard(i)
                continue
            renderer._draw_frame(EnvSlice(state, i, a.envs))
            encoders[i].stdin.write(pygame.image.tostring(renderer._screen, "RGB"))

    def stop_when(step, ended):
        return not open_set or step >= a.max_steps + hold + 8

    ended, f0, f1, tl = play(env, runner, a.envs, a.max_steps, on_frame, stop_when)
    for i in list(open_set):
        encoders[i].stdin.close()
        encoders[i].wait()
    renderer.close()
    recs = records(a.seed, a.n0, a.n1, a.envs, ended, f0, f1, tl, a.scenario)
    for i in picks:
        r = recs[i]
        match = None
        if i in log:
            g = log[i]
            match = (g["winner"], g["blue_left"], g["red_left"], g["steps"]) == (
                r["winner"],
                r["blue_left"],
                r["red_left"],
                r["steps"],
            )
        gif = _write_gif(paths[i], a.gif_fps, a.gif_width) if a.gif else None
        print(
            json.dumps(
                dict(
                    mp4=str(paths[i]),
                    gif=str(gif) if gif else None,
                    winner=r["winner"],
                    blue_left=r["blue_left"],
                    red_left=r["red_left"],
                    steps=r["steps"],
                    seconds=round(r["steps"] / a.fps, 1),
                    matches_log=match,
                )
            ),
            flush=True,
        )


def parse_seeds(spec):
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("batch", "render"):
        s = sub.add_parser(name)
        s.add_argument("ckpt")
        s.add_argument("--n0", type=int, required=True)
        s.add_argument("--n1", type=int, required=True)
        s.add_argument("--envs", type=int, default=512)
        s.add_argument("--max-steps", type=int, default=1024)
        s.add_argument("--scenario", default="vs_scripted", choices=["vs_scripted", "self"])
    sub.choices["batch"].add_argument("--seeds", type=parse_seeds, required=True)
    sub.choices["batch"].add_argument("--out", required=True)
    r = sub.choices["render"]
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--pick", required=True)
    r.add_argument("--outdir", required=True)
    r.add_argument("--stem", required=True)
    r.add_argument("--log")
    r.add_argument("--fps", type=int, default=60)
    r.add_argument("--window", type=int, default=720)
    r.add_argument("--hold-ms", type=int, default=1500)
    r.add_argument("--gif", action="store_true")
    r.add_argument("--gif-fps", type=int, default=24)
    r.add_argument("--gif-width", type=int, default=480)
    a = p.parse_args()
    {"batch": cmd_batch, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
