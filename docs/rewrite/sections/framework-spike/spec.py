"""Shared specification for the PyTorch-versus-JAX spike. Needs only numpy and the stdlib.

Both torch_bench.py and jax_bench.py take every constant, every initial weight, every initial
game state and every stored rollout from this file, so the two implement the same maths on the
same numbers. The prose version of the workloads is in README.md ("The three workloads").
"""

import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

# ---- policy (workload A) -------------------------------------------------------------------
WIDTH = 128          # token width
HEADS = 2            # attention heads (head width 64)
BLOCKS = 2           # each block: 1 temporal sublayer + SPATIAL spatial sublayers
SPATIAL = 2          # a spatial sublayer = masked self-attention + gated MLP
SHIP_FEATURES = 20
GLOBAL_FEATURES = 8
ACTIONS = 42         # 3 turn x 7 thrust x 2 fire
VALUE_BINS = 77
RGLRU_C = 8.0        # Griffin's constant c in a_t = a^(c r_t)
NORM_EPS = 1e-6      # RMSNorm: x / sqrt(mean(x^2) + eps) * gain
ROPE_BANDS = 6       # per axis; rotates the first 4 * ROPE_BANDS = 24 dims of each 64-wide head
ROPE_FREQS = (2.0 * math.pi / np.geomspace(200.0, 16000.0, ROPE_BANDS)).astype(np.float32)  # rad/m

# ---- toy game (workload B) -----------------------------------------------------------------
SLOTS = 8            # bullet slots per ship
DT = 0.1             # seconds per tick
ARENA_RADIUS = 4000.0
BOUNDARY_PULL = 2.0  # 1/s, pulls a ship back in proportion to how far it is outside the arena
TURN_RATE = 1.5      # rad/s at full turn
THRUST_ACC = 60.0    # m/s^2 at full thrust
DRAG = 0.3           # 1/s (terminal speed 200 m/s)
MIN_SPEED, MAX_SPEED = 50.0, 250.0
START_SPEED = 150.0
MUZZLE_SPEED = 900.0
BULLET_LIFE = 11     # ticks
COOLDOWN = 5         # ticks
HIT_RADIUS = 60.0
DAMAGE = 25.0
MAX_HP = 100.0
RESPAWN_DELAY = 30   # ticks

# ---- PPO (workload C) ----------------------------------------------------------------------
CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 3e-4
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 0.5


def init_params(seed: int = 0) -> dict:
    """All weights as float32 numpy arrays. A Linear's weight is stored [in, out] (y = x @ w + b)."""
    rng = np.random.default_rng(seed)
    p = {}

    def linear(name, n_in, n_out, bias=True):
        bound = 1.0 / math.sqrt(n_in)
        p[name + ".w"] = rng.uniform(-bound, bound, (n_in, n_out)).astype(np.float32)
        if bias:
            p[name + ".b"] = rng.uniform(-bound, bound, (n_out,)).astype(np.float32)

    def gain(name):  # RMSNorm gain, perturbed from 1 so the parity check exercises it
        p[name] = (1.0 + 0.1 * rng.standard_normal(WIDTH)).astype(np.float32)

    for kind, n_in in (("ship", SHIP_FEATURES), ("global", GLOBAL_FEATURES)):
        linear(f"embed.{kind}.in", n_in, WIDTH)
        gain(f"embed.{kind}.norm")
        linear(f"embed.{kind}.out", WIDTH, WIDTH)
    for b in range(BLOCKS):
        gain(f"block{b}.temporal.norm")
        linear(f"block{b}.temporal.gate_a", WIDTH, WIDTH)   # recurrence gate r_t
        linear(f"block{b}.temporal.gate_x", WIDTH, WIDTH)   # input gate i_t
        a_to_c = rng.uniform(0.9, 0.999, WIDTH)             # a^c uniform in [0.9, 0.999]
        a = a_to_c ** (1.0 / RGLRU_C)
        p[f"block{b}.temporal.lam"] = np.log(a / (1.0 - a)).astype(np.float32)
        for s in range(SPATIAL):
            gain(f"block{b}.spatial{s}.attn_norm")
            linear(f"block{b}.spatial{s}.qkv", WIDTH, 3 * WIDTH, bias=False)
            linear(f"block{b}.spatial{s}.attn_out", WIDTH, WIDTH)
            gain(f"block{b}.spatial{s}.mlp_norm")
            linear(f"block{b}.spatial{s}.mlp_in", WIDTH, 2 * WIDTH)
            linear(f"block{b}.spatial{s}.mlp_out", WIDTH, WIDTH)
    gain("final_norm")
    linear("policy_head", WIDTH, ACTIONS)
    linear("value_head", WIDTH, VALUE_BINS)
    return p


def init_state(envs: int, ships: int, pad_to: int | None = None, seed: int = 0) -> dict:
    """Initial game state for `ships` real ships (two teams of ships/2), optionally padded with
    absent slots to `pad_to` ship slots. Real ships come first; team 0 is the first half of them."""
    rng = np.random.default_rng(seed)
    n = pad_to or ships
    half = ships // 2
    index = np.arange(n)
    exists = index < ships
    team = ((index >= half) & exists).astype(np.int32)
    slot_in_team = np.where(team == 1, index - half, index)
    x = np.where(team == 0, -1500.0, 1500.0)
    y = (slot_in_team - (half - 1) / 2.0) * 150.0
    jitter = np.zeros((envs, n, 2))
    jitter[:, :ships] = rng.uniform(-50.0, 50.0, (envs, ships, 2))   # same draws padded or not
    spawn = np.where(exists[None, :, None], np.stack([x, y], -1)[None] + jitter, 0.0).astype(np.float32)
    spawn_heading = np.broadcast_to(np.where(team == 0, 0.0, math.pi), (envs, n)).astype(np.float32)
    full = lambda value, dtype: np.full((envs, n), value, dtype)
    return {
        "exists": np.broadcast_to(exists, (envs, n)).copy(),
        "team": np.broadcast_to(team, (envs, n)).copy(),
        "spawn_pos": spawn,
        "spawn_heading": spawn_heading.copy(),
        "pos": spawn.copy(),
        "heading": spawn_heading.copy(),
        "speed": full(START_SPEED, np.float32),
        "hp": full(MAX_HP, np.float32),
        "alive": np.broadcast_to(exists, (envs, n)).copy(),
        "cooldown": full(0, np.int32),
        "respawn": full(0, np.int32),
        "bullet_pos": np.zeros((envs, n, SLOTS, 2), np.float32),
        "bullet_vel": np.zeros((envs, n, SLOTS, 2), np.float32),
        "bullet_life": np.zeros((envs, n, SLOTS), np.int32),
        "tick": np.zeros((envs,), np.int32),
    }


def init_hidden(envs: int, slots: int) -> np.ndarray:
    """Recurrent state of the two policy forwards of a rollout tick:
    [perspective(2), block, env, token(1 global + slots), width]."""
    return np.zeros((2, BLOCKS, envs, slots + 1, WIDTH), np.float32)


def random_actions(seed: int, steps: int, envs: int, slots: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, ACTIONS, (steps, envs, slots)).astype(np.int32)


def make_policy_inputs(envs: int, ships: int, seed: int = 1, time: int | None = None) -> dict:
    """Random policy inputs (not from the game) for the parity check and for timing A alone.
    With `time`, every array gets a time axis after the env axis (the learner form)."""
    rng = np.random.default_rng(seed)
    lead = (envs,) if time is None else (envs, time)
    alive = rng.random(lead + (ships,)) < 0.8
    return {
        "ship": rng.standard_normal(lead + (ships, SHIP_FEATURES)).astype(np.float32),
        "glob": rng.standard_normal(lead + (GLOBAL_FEATURES,)).astype(np.float32),
        "pos": rng.uniform(-ARENA_RADIUS, ARENA_RADIUS, lead + (ships, 2)).astype(np.float32),
        "alive": alive,
    }


def make_rollout(envs: int, time: int, ships: int, seed: int = 2) -> dict:
    """A stored rollout for one PPO minibatch: [env, time, ship, ...]. Synthetic, the right shapes
    and value ranges; the benchmark measures the cost of the update, not learning."""
    rng = np.random.default_rng(seed)
    batch = make_policy_inputs(envs, ships, seed, time)
    shape = (envs, time, ships)
    batch["action"] = rng.integers(0, ACTIONS, shape).astype(np.int32)
    batch["old_logp"] = (-math.log(ACTIONS) + 0.1 * rng.standard_normal(shape)).astype(np.float32)
    batch["advantage"] = rng.standard_normal(shape).astype(np.float32)
    # Two-hot value target: weight split between two neighbouring bins.
    position = rng.uniform(0, VALUE_BINS - 1, shape)
    low = np.floor(position).astype(np.int64)
    high_weight = (position - low).astype(np.float32)
    target = np.zeros(shape + (VALUE_BINS,), np.float32)
    np.put_along_axis(target, low[..., None], (1.0 - high_weight)[..., None], -1)
    np.put_along_axis(target, (low + 1)[..., None], high_weight[..., None], -1)  # low + 1 <= 76
    batch["value_target"] = target
    batch["h0"] = np.zeros((BLOCKS, envs, ships + 1, WIDTH), np.float32)
    return batch


# ---- measurement helpers shared by both benchmark scripts ----------------------------------

def timed_repeats(step, sync, repeats: int, iterations: int) -> dict:
    """Run `step()` `iterations` times, `repeats` times over; `sync()` is called before each
    timer stop (GPU synchronise / block_until_ready). Returns per-call milliseconds."""
    per_call = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        sync()
        per_call.append((time.perf_counter() - start) * 1000.0 / iterations)
    q1, median, q3 = np.percentile(per_call, [25, 50, 75])
    return {"median_ms": float(median), "q1_ms": float(q1), "q3_ms": float(q3),
            "iqr_ms": float(q3 - q1), "repeats": repeats, "iterations": iterations}


def pick_iterations(step, sync, budget_s: float = 1.0, cap: int = 2000) -> int:
    """How many calls fit in about `budget_s` seconds (at least 3)."""
    sync()
    start = time.perf_counter()
    step()
    sync()
    one = max(time.perf_counter() - start, 1e-6)
    return int(min(cap, max(3, budget_s / one)))


def latency_stats(samples_ms) -> dict:
    samples = np.asarray(samples_ms)
    return {"p50_ms": float(np.percentile(samples, 50)), "p99_ms": float(np.percentile(samples, 99)),
            "max_ms": float(samples.max()), "mean_ms": float(samples.mean()), "ticks": int(samples.size)}


def machine() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        text = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
                               "Hypervisor vendor", "L2 cache", "L3 cache"):
                info["lscpu " + key.strip()] = value.strip()
    except (OSError, subprocess.SubprocessError):
        info["lscpu"] = "not available (not Linux?)"
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        pass
    return info


GPU_QUERY = "pstate,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,utilization.gpu,memory.used"


def load_snapshot() -> dict:
    """Machine load outside the timed region: 1/5/15-minute load average and, if `nvidia-smi`
    exists, the GPU's state (temperature, SM clock, power, utilisation, memory in use)."""
    snapshot = {"loadavg": [round(x, 2) for x in os.getloadavg()]}
    try:
        text = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        if text:
            snapshot["gpu"] = dict(zip(GPU_QUERY.split(","), (v.strip() for v in text.split(","))))
    except (OSError, subprocess.SubprocessError):
        pass
    return snapshot


def run_child(script: str, argv: list, env: dict | None = None) -> list:
    """Run `script` in a fresh process; collect the records it prints as 'RECORD <json>' lines.
    Each record gets the machine load just before and just after the child ran."""
    full_env = dict(os.environ)
    full_env.update(env or {})
    before = load_snapshot()
    done = subprocess.run([sys.executable, script] + [str(a) for a in argv],
                          capture_output=True, text=True, env=full_env)
    after = load_snapshot()
    records = [json.loads(line[7:]) for line in done.stdout.splitlines() if line.startswith("RECORD ")]
    for record in records:
        record["load_before"], record["load_after"] = before, after
    if done.returncode != 0:
        records.append({"suite": "error", "argv": [str(a) for a in argv], "returncode": done.returncode,
                        "stderr_tail": done.stderr[-3000:]})
        print(f"  CHILD FAILED ({done.returncode}): {' '.join(map(str, argv))}\n{done.stderr[-1500:]}", flush=True)
    return records


def emit(record: dict) -> None:
    print("RECORD " + json.dumps(record), flush=True)


def write_results(path: str, meta: dict, records: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"meta": meta, "records": records}, handle, indent=1)
    print(f"wrote {len(records)} records to {path}")
