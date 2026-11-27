"""JAX side of the framework spike: policy forward (A), toy game step (B), PPO update (C),
and the measurements described in README.md.

    python jax_bench.py --device cpu|cuda --suite all|compile|steady|sizes|loop|latency|ops
                        [--envs 1,2048] [--out results/<name>.json]
"""
# ruff: noqa: E402, E731  (a design-record script, kept as written)

import os
import sys
import time

PROCESS_START = time.perf_counter()


def _argument(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# These must be decided before jax is imported.
if _argument("--device", "cpu") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
if _argument("--threads") == "pin":
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})  # Linux only
if _argument("--threads") == "1":
    # Not an official JAX setting (see README "Thread settings"): asks XLA's CPU backend not to
    # use its Eigen thread pool.
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
        " intra_op_parallelism_threads=1"
    ).strip()

import argparse
import math
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
import spec
from jax import lax

if os.environ.get("SPIKE_JAX_CACHE"):  # persistent compilation cache, for the warm compile time
    jax.config.update("jax_compilation_cache_dir", os.environ["SPIKE_JAX_CACHE"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

IMPORT_SECONDS = time.perf_counter() - PROCESS_START
HERE = os.path.abspath(__file__)
ROPE_FREQS = jnp.asarray(spec.ROPE_FREQS)


# ============================================================================================
# A. Policy (params: the dict from spec.init_params, as jax arrays)
# ============================================================================================


def linear(p, name, x):
    y = x @ p[name + ".w"]
    return y + p[name + ".b"] if name + ".b" in p else y


def rms_norm(gain, x):
    return x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + spec.NORM_EPS) * gain


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def embed(p, kind, x):
    x = rms_norm(p[f"embed.{kind}.norm"], linear(p, f"embed.{kind}.in", x))
    return linear(p, f"embed.{kind}.out", gelu(x))


def rglru_gates(p, name, x):
    """h_t = a_t * h_{t-1} + b_t. The recurrence maths is float32 whatever the dtype of x
    (1 - a_t^2 with a_t near 1 has no precision left in bf16)."""
    x = rms_norm(p[name + ".norm"], x)
    r = jax.nn.sigmoid(linear(p, name + ".gate_a", x)).astype(jnp.float32)
    i = jax.nn.sigmoid(linear(p, name + ".gate_x", x)).astype(jnp.float32)
    a = jnp.exp(
        spec.RGLRU_C * r * jax.nn.log_sigmoid(p[name + ".lam"])
    )  # a^(c r_t), a = sigmoid(lam)
    return a, jnp.sqrt(1.0 - a * a) * (i * x.astype(jnp.float32))


def rglru_step(p, name, x, h):
    a, b = rglru_gates(p, name, x)
    h = a * h + b
    return x + h.astype(x.dtype), h


def rglru_scan(p, name, x, h):  # x [env, time, token, width]
    a, b = rglru_gates(p, name, x)

    def body(h, ab):
        h = ab[0] * h + ab[1]
        return h, h

    h, states = lax.scan(body, h, (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)))
    return x + jnp.swapaxes(states, 0, 1).astype(x.dtype), h


def rotate(x, cos, sin):
    """Rotary position rotation of the first 4 * ROPE_BANDS dims of each head."""
    n = 2 * spec.ROPE_BANDS
    x1, x2, rest = x[..., :n], x[..., n : 2 * n], x[..., 2 * n :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], -1)


def spatial(p, name, x, mask, cos, sin):  # x [batch, token, width]; mask [batch, token]
    """Masked self-attention over the tokens of one environment, then a gated MLP."""
    batch, tokens, _ = x.shape
    qkv = linear(p, name + ".qkv", rms_norm(p[name + ".attn_norm"], x))
    q, k, v = jnp.transpose(qkv.reshape(batch, tokens, 3, spec.HEADS, -1), (2, 0, 3, 1, 4))
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)  # each [batch, head, token, 64]
    scores = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    scores = jnp.where(mask[:, None, None, :], scores, -jnp.inf)
    attended = jax.nn.softmax(scores, -1) @ v
    x = x + linear(p, name + ".attn_out", jnp.swapaxes(attended, 1, 2).reshape(batch, tokens, -1))
    value, gate = jnp.split(linear(p, name + ".mlp_in", rms_norm(p[name + ".mlp_norm"], x)), 2, -1)
    return x + linear(p, name + ".mlp_out", gelu(value) * gate)


def tokens(p, ship, glob, pos, alive):
    """Token 0 is the global token (always present, at the origin); then one per ship slot."""
    x = jnp.concatenate([embed(p, "global", glob)[..., None, :], embed(p, "ship", ship)], -2)
    pos = jnp.concatenate([jnp.zeros_like(pos[..., :1, :]), pos], -2)
    mask = jnp.concatenate([jnp.ones_like(alive[..., :1]), alive], -1)
    angle = jnp.concatenate([pos[..., 0:1] * ROPE_FREQS, pos[..., 1:2] * ROPE_FREQS], -1)
    return x, mask, jnp.cos(angle).astype(x.dtype), jnp.sin(angle).astype(x.dtype)


def heads(p, x):
    y = rms_norm(p["final_norm"], x[..., 1:, :])
    return linear(p, "policy_head", y), linear(p, "value_head", y)


def policy_step(p, ship, glob, pos, alive, h):
    """One tick. ship [env, slot, 20], glob [env, 8], pos [env, slot, 2], alive [env, slot],
    h [block, env, token, width] -> policy logits, value logits, new h."""
    x, mask, cos, sin = tokens(p, ship, glob, pos, alive)
    cos, sin = cos[:, None], sin[:, None]
    new_h = []
    for b in range(spec.BLOCKS):
        x, state = rglru_step(p, f"block{b}.temporal", x, h[b])
        new_h.append(state)
        for s in range(spec.SPATIAL):
            x = spatial(p, f"block{b}.spatial{s}", x, mask, cos, sin)
    return heads(p, x) + (jnp.stack(new_h),)


def policy_sequence(p, ship, glob, pos, alive, h0):
    """Learner form: every input has [env, time] leading axes; the recurrence is a lax.scan over
    time, attention runs with time folded into the batch."""
    x, mask, cos, sin = tokens(p, ship, glob, pos, alive)
    envs, steps, n_tokens, _ = x.shape
    fold = lambda t: t.reshape((envs * steps,) + t.shape[2:])
    mask, cos, sin = fold(mask), fold(cos)[:, None], fold(sin)[:, None]
    new_h = []
    for b in range(spec.BLOCKS):
        x, state = rglru_scan(p, f"block{b}.temporal", x, h0[b])
        new_h.append(state)
        x = fold(x)
        for s in range(spec.SPATIAL):
            x = spatial(p, f"block{b}.spatial{s}", x, mask, cos, sin)
        x = x.reshape(envs, steps, n_tokens, -1)
    return heads(p, x) + (jnp.stack(new_h),)


# ============================================================================================
# B. Toy game (a stand-in for the real game's operation structure, not a port)
# ============================================================================================


def game_step(s: dict, action):
    """One tick for every environment. action [env, slot] in 0..41 = turn + 3 * thrust + 21 * fire."""
    turn = (action % 3 - 1).astype(jnp.float32)
    thrust = (action // 3 % 7).astype(jnp.float32) / 6.0
    fire = action >= 21
    alive, team = s["alive"], s["team"]
    envs, slots = alive.shape

    # Flight (dead ships stay frozen).
    heading = s["heading"] + turn * (spec.TURN_RATE * spec.DT)
    heading = jnp.remainder(heading + math.pi, 2.0 * math.pi) - math.pi
    speed = s["speed"] + spec.DT * (thrust * spec.THRUST_ACC - spec.DRAG * s["speed"])
    speed = jnp.clip(speed, spec.MIN_SPEED, spec.MAX_SPEED)
    heading = jnp.where(alive, heading, s["heading"])
    speed = jnp.where(alive, speed, s["speed"])
    direction = jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1)
    velocity = speed[..., None] * direction
    pos = s["pos"] + spec.DT * velocity
    radius = jnp.sqrt((pos * pos).sum(-1, keepdims=True))
    outside = jnp.maximum(radius - spec.ARENA_RADIUS, 0.0)
    pos = pos - (spec.BOUNDARY_PULL * spec.DT) * outside * pos / jnp.maximum(radius, 1.0)
    pos = jnp.where(alive[..., None], pos, s["pos"])

    # Bullets move and age.
    bullet_pos = s["bullet_pos"] + spec.DT * s["bullet_vel"]
    bullet_life = jnp.maximum(s["bullet_life"] - 1, 0)

    # Hits: every bullet against every enemy ship, [env, bullet, ship].
    flat_pos = bullet_pos.reshape(envs, slots * spec.SLOTS, 2)
    dx = flat_pos[:, :, None, 0] - pos[:, None, :, 0]
    dy = flat_pos[:, :, None, 1] - pos[:, None, :, 1]
    bullet_team = jnp.repeat(team, spec.SLOTS, 1)
    hit = (
        (bullet_life.reshape(envs, -1) > 0)[:, :, None]
        & alive[:, None, :]
        & (bullet_team[:, :, None] != team[:, None, :])
        & (dx * dx + dy * dy < spec.HIT_RADIUS**2)
    )
    hp = s["hp"] - spec.DAMAGE * hit.sum(1)
    bullet_life = jnp.where(hit.any(2).reshape(envs, slots, spec.SLOTS), 0, bullet_life)

    # Fire into the first free slot.
    free = bullet_life == 0
    first_free = jnp.argmax(free, -1)
    shoot = alive & fire & (s["cooldown"] == 0) & free.any(-1)
    new = shoot[..., None] & (jnp.arange(spec.SLOTS) == first_free[..., None])
    bullet_pos = jnp.where(new[..., None], pos[:, :, None, :], bullet_pos)
    muzzle = velocity + spec.MUZZLE_SPEED * direction
    bullet_vel = jnp.where(new[..., None], muzzle[:, :, None, :], s["bullet_vel"])
    bullet_life = jnp.where(new, spec.BULLET_LIFE, bullet_life)
    cooldown = jnp.where(shoot, spec.COOLDOWN, jnp.maximum(s["cooldown"] - 1, 0))

    # Death, then respawn after a delay.
    died = alive & (hp <= 0.0)
    alive = alive & ~died
    respawn = jnp.where(died, spec.RESPAWN_DELAY, jnp.maximum(s["respawn"] - 1, 0))
    back = s["exists"] & ~alive & (respawn == 0)
    return {
        "exists": s["exists"],
        "team": team,
        "spawn_pos": s["spawn_pos"],
        "spawn_heading": s["spawn_heading"],
        "pos": jnp.where(back[..., None], s["spawn_pos"], pos),
        "heading": jnp.where(back, s["spawn_heading"], heading),
        "speed": jnp.where(back, spec.START_SPEED, speed),
        "hp": jnp.where(back, spec.MAX_HP, hp),
        "alive": alive | back,
        "cooldown": cooldown,
        "respawn": respawn,
        "bullet_pos": bullet_pos,
        "bullet_vel": bullet_vel,
        "bullet_life": bullet_life,
        "tick": s["tick"] + 1,
    }


def observe(s: dict):
    """Game state -> policy inputs (ship features [env, slot, 20] from team 0's side, global
    features [env, 8], positions, alive mask). Team 1's view negates feature 10 (see other_side)."""
    f32 = lambda t: t.astype(jnp.float32)
    pos, alive, team = s["pos"], s["alive"], s["team"]
    alive_f = f32(alive)
    in_1 = f32(team) * f32(s["exists"])
    in_0 = f32(s["exists"]) - in_1
    alive_0, alive_1 = alive_f * in_0, alive_f * in_1
    count_0 = jnp.maximum(in_0.sum(1, keepdims=True), 1.0)
    count_1 = jnp.maximum(in_1.sum(1, keepdims=True), 1.0)
    n_alive_0 = alive_0.sum(1, keepdims=True)
    n_alive_1 = alive_1.sum(1, keepdims=True)
    frac_0, frac_1 = n_alive_0 / count_0, n_alive_1 / count_1
    hp_0 = (s["hp"] * alive_0).sum(1, keepdims=True) / (count_0 * spec.MAX_HP)
    hp_1 = (s["hp"] * alive_1).sum(1, keepdims=True) / (count_1 * spec.MAX_HP)
    centre_0 = (pos * alive_0[..., None]).sum(1, keepdims=True) / jnp.maximum(n_alive_0, 1.0)[
        ..., None
    ]
    centre_1 = (pos * alive_1[..., None]).sum(1, keepdims=True) / jnp.maximum(n_alive_1, 1.0)[
        ..., None
    ]
    bullets = f32(s["bullet_life"] > 0).mean(-1)
    scaled = pos / spec.ARENA_RADIUS
    from_0 = (pos - centre_0) / spec.ARENA_RADIUS
    from_1 = (pos - centre_1) / spec.ARENA_RADIUS
    speed = s["speed"] / spec.MAX_SPEED
    cos, sin = jnp.cos(s["heading"]), jnp.sin(s["heading"])
    ship = jnp.stack(
        [
            scaled[..., 0],
            scaled[..., 1],
            speed * cos,
            speed * sin,
            cos,
            sin,
            speed,
            s["hp"] / spec.MAX_HP,
            alive_f,
            f32(s["cooldown"]) / spec.COOLDOWN,
            1.0 - 2.0 * f32(team),
            f32(s["respawn"]) / spec.RESPAWN_DELAY,
            jnp.sqrt((scaled * scaled).sum(-1)),
            bullets,
            from_0[..., 0],
            from_0[..., 1],
            from_1[..., 0],
            from_1[..., 1],
            jnp.broadcast_to(frac_0, alive_f.shape),
            jnp.broadcast_to(frac_1, alive_f.shape),
        ],
        -1,
    )
    gap = centre_0[:, 0] - centre_1[:, 0]
    glob = jnp.concatenate(
        [
            f32(s["tick"])[:, None] / 1000.0,
            frac_0,
            frac_1,
            hp_0,
            hp_1,
            jnp.sqrt((gap * gap).sum(-1, keepdims=True)) / spec.ARENA_RADIUS,
            bullets.sum(1, keepdims=True) / (count_0 + count_1),
            jnp.ones_like(frac_0),
        ],
        -1,
    )
    return ship, glob, pos, alive


def other_side(ship):
    return ship.at[..., 10].multiply(-1.0)


def choose(p, s: dict, h):
    """The two policy forwards of a tick (one per team's view; each sees every ship) and the
    merged per-ship logits: a ship takes the output of its own team's forward."""
    ship, glob, pos, alive = observe(s)
    logits_0, value_0, h_0 = policy_step(p, ship, glob, pos, alive, h[0])
    logits_1, value_1, h_1 = policy_step(p, other_side(ship), glob, pos, alive, h[1])
    own = (s["team"] == 0)[..., None]
    return (
        jnp.where(own, logits_0, logits_1),
        jnp.where(own, value_0, value_1),
        jnp.stack([h_0, h_1]),
    )


def rollout_tick(p, s: dict, h, key):
    """Observe, two policy forwards, sample an action per ship, step the game."""
    logits, value, h = choose(p, s, h)
    key, sample_key = jax.random.split(key)
    action = jax.random.categorical(sample_key, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], -1)[..., 0]
    return game_step(s, action), h, key, action, logp, value


def rollout_scan(p, s: dict, h, key, ticks: int):
    """The whole rollout as one lax.scan: no Python between ticks. Keeps actions and log-probs."""

    def body(carry, _):
        s, h, key, action, logp, _ = rollout_tick(p, *carry)
        return (s, h, key), (action, logp)

    return lax.scan(body, (s, h, key), None, length=ticks)


# ============================================================================================
# C. PPO update
# ============================================================================================


def to_bf16(p: dict) -> dict:
    """JAX has no autocast. The plain equivalent: keep float32 master weights and cast them (and
    the inputs) to bfloat16 inside the differentiated function; gradients come back float32."""
    return {k: v if k.endswith(".lam") else v.astype(jnp.bfloat16) for k, v in p.items()}


def ppo_loss(p, b: dict, bf16: bool = False):
    ship, glob = b["ship"], b["glob"]
    if bf16:
        p, ship, glob = to_bf16(p), ship.astype(jnp.bfloat16), glob.astype(jnp.bfloat16)
    logits, value_logits, _ = policy_sequence(p, ship, glob, b["pos"], b["alive"], b["h0"])
    logp_all = jax.nn.log_softmax(logits.astype(jnp.float32))
    logp = jnp.take_along_axis(logp_all, b["action"][..., None], -1)[..., 0]
    ratio = jnp.exp(logp - b["old_logp"])
    surrogate = jnp.minimum(
        ratio * b["advantage"], jnp.clip(ratio, 1.0 - spec.CLIP, 1.0 + spec.CLIP) * b["advantage"]
    )
    weight = b["alive"].astype(jnp.float32)
    count = jnp.maximum(weight.sum(), 1.0)
    policy_loss = -(surrogate * weight).sum() / count
    value_loss = (
        -(
            (b["value_target"] * jax.nn.log_softmax(value_logits.astype(jnp.float32))).sum(-1)
            * weight
        ).sum()
        / count
    )
    entropy = (-(jnp.exp(logp_all) * logp_all).sum(-1) * weight).sum() / count
    return policy_loss + spec.VALUE_COEF * value_loss - spec.ENTROPY_COEF * entropy


def make_optimizer():
    return optax.chain(
        optax.clip_by_global_norm(spec.MAX_GRAD_NORM),
        optax.adam(spec.LEARNING_RATE, eps=spec.ADAM_EPS),
    )


def make_ppo_update(optimizer, bf16: bool = False):
    def ppo_update(p, opt_state, batch):
        """One minibatch gradient step."""
        loss, grads = jax.value_and_grad(ppo_loss)(p, batch, bf16)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss, optax.global_norm(grads)

    return ppo_update


# ============================================================================================
# Benchmark plumbing
# ============================================================================================


def to_jax(tree: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in tree.items()}


class Units:
    """The four timed units at one size, each a zero-argument callable that carries its state.
    `sync()` blocks until the last result is ready."""

    def __init__(self, envs, ships, pad_to=None, bf16=False, micro=(16, 128)):
        self.envs, self.ships, self.pad_to, self.bf16, self.micro = envs, ships, pad_to, bf16, micro
        self.slots = pad_to or ships
        self.params = to_jax(spec.init_params())
        self.state = to_jax(spec.init_state(envs, ships, pad_to))
        self.h = jnp.asarray(spec.init_hidden(envs, self.slots))
        self.inputs = to_jax(spec.make_policy_inputs(envs, self.slots))
        self.h_single = self.h[0]
        self.actions = jnp.asarray(spec.random_actions(3, 64, envs, self.slots))
        self.key = jax.random.key(0)
        self.i = 0
        self.last = None

    def sync(self):
        jax.block_until_ready(self.last)

    def build(self, unit: str, jitted=None):
        """`jitted`: reuse an already-jitted function (the size test passes the same one)."""
        if unit == "A":
            step = jitted or jax.jit(policy_step)

            def run():
                i = self.inputs
                self.last = step(
                    self.params, i["ship"], i["glob"], i["pos"], i["alive"], self.h_single
                )
                self.h_single = self.last[2]
        elif unit == "B":
            step = jitted or jax.jit(game_step)

            def run():
                self.state = self.last = step(self.state, self.actions[self.i % 64])
                self.i += 1
        elif unit == "tick":
            step = jitted or jax.jit(rollout_tick)

            def run():
                self.last = step(self.params, self.state, self.h, self.key)
                self.state, self.h, self.key = self.last[:3]
        elif unit == "C":
            self.batch = to_jax(spec.make_rollout(self.micro[0], self.micro[1], self.ships))
            optimizer = make_optimizer()
            self.opt_state = optimizer.init(self.params)
            step = jax.jit(make_ppo_update(optimizer, self.bf16))

            def run():
                self.last = step(self.params, self.opt_state, self.batch)
                self.params, self.opt_state = self.last[:2]
        else:
            raise ValueError(unit)
        self.step = step
        return run


def first_calls(run, sync) -> dict:
    out = {}
    for name in ("first_call_s", "second_call_s", "third_call_s"):
        start = time.perf_counter()
        run()
        sync()
        out[name] = time.perf_counter() - start
    return out


def steady(run, sync, repeats, budget_s=1.0) -> dict:
    for _ in range(3):
        run()
    return spec.timed_repeats(run, sync, repeats, spec.pick_iterations(run, sync, budget_s))


def cache_size(jitted) -> int:
    """Number of compiled specialisations a jitted function holds (private helper `_cache_size`,
    used for the recompile count only)."""
    return int(jitted._cache_size())


# ---- children (each runs in a fresh process) -----------------------------------------------


def child_compile(a):
    units = Units(a.child_envs, a.ships, bf16=a.precision == "bf16", micro=a.micro)
    run = units.build(a.unit)
    record = {
        "suite": "compile",
        "unit": a.unit,
        "variant": "jit",
        "envs": a.child_envs,
        "ships": a.ships,
        "precision": a.precision,
        "import_s": IMPORT_SECONDS,
    }
    if a.unit == "C":
        record.update(envs=a.micro[0], time=a.micro[1])
    record.update(first_calls(run, units.sync))
    spec.emit(record)


def child_steady(a):
    for unit in a.unit.split(","):
        units = Units(a.child_envs, a.ships, bf16=a.precision == "bf16", micro=a.micro)
        run = units.build(unit)
        record = {
            "suite": "steady",
            "unit": unit,
            "variant": "jit",
            "envs": a.child_envs,
            "ships": a.ships,
            "precision": a.precision,
        }
        if unit == "C":
            record.update(envs=a.micro[0], time=a.micro[1])
        record.update(first_calls(run, units.sync))
        record.update(steady(run, units.sync, a.repeats, a.budget))
        spec.emit(record)


def child_sizes(a):
    """Jit the rollout tick at N = 10, then meet N = 20, 40, 100 (and N = 10 again)."""
    padded = a.method == "padded"
    step = jax.jit(rollout_tick)
    for ships in [10] + a.sizes + [10]:
        units = Units(a.child_envs, ships, pad_to=a.pad_to if padded else None)
        run = units.build("tick", jitted=step)
        before = cache_size(step)
        record = {
            "suite": "sizes",
            "method": a.method,
            "envs": a.child_envs,
            "ships": ships,
            "slots": units.slots,
        }
        try:
            record.update(first_calls(run, units.sync))
            record.update(steady(run, units.sync, a.repeats, a.budget))
        except Exception as error:  # a failure here is a finding, not something to hide
            record["error"] = f"{type(error).__name__}: {str(error)[:1500]}"
        record["graphs_compiled_here"] = cache_size(step) - before
        record["graphs_compiled_total"] = cache_size(step)
        spec.emit(record)


def child_loop(a):
    """T ticks: as one jitted lax.scan, or as a Python loop over the jitted tick."""
    units = Units(a.child_envs, a.ships)
    if a.variant == "scan":
        scan = jax.jit(rollout_scan, static_argnames="ticks")

        def rollout():
            units.last = scan(units.params, units.state, units.h, units.key, ticks=a.ticks)
            units.state, units.h, units.key = units.last[0]

        label = "lax.scan, one jit"
    else:
        tick = units.build("tick")

        def rollout():
            kept = []
            for _ in range(a.ticks):
                tick()
                kept.append(units.last[3:5])
            units.last = (
                units.last,
                jnp.stack([k[0] for k in kept]),
                jnp.stack([k[1] for k in kept]),
            )

        label = "jit tick, Python loop"

    start = time.perf_counter()
    rollout()
    units.sync()
    first = time.perf_counter() - start
    rollout()
    stats = spec.timed_repeats(rollout, units.sync, a.repeats, 1)
    spec.emit(
        {
            "suite": "loop",
            "variant": label,
            "envs": a.child_envs,
            "ships": a.ships,
            "ticks": a.ticks,
            "first_rollout_s": first,
            "rollout": stats,
            "ticks_per_s": a.ticks / (stats["median_ms"] / 1000.0),
            "env_ticks_per_s": a.ticks * a.child_envs / (stats["median_ms"] / 1000.0),
        }
    )


def child_latency(a):
    """Batch size 1: one game tick plus two policy forwards, timed per tick."""
    units = Units(1, a.ships)
    tick = units.build("tick")
    for _ in range(a.warmup):
        tick()
    units.sync()
    samples = []
    for _ in range(a.latency_ticks):
        start = time.perf_counter()
        tick()
        units.sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    record = {
        "suite": "latency",
        "variant": "jit",
        "ships": a.ships,
        "threads": a.threads,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "warmup": a.warmup,
    }
    try:
        record["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        pass
    record.update(spec.latency_stats(samples))
    spec.emit(record)


def count_equations(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        inner = [v for v in eqn.params.values() if hasattr(v, "eqns") or hasattr(v, "jaxpr")]
        total += 1 if not inner else sum(count_equations(getattr(v, "jaxpr", v)) for v in inner)
    return total


def child_ops(a):
    """Context only: primitives in the traced program (before XLA fuses them). JAX does not
    dispatch them one at a time at run time; the whole jitted function is one executable."""
    units = Units(1, a.ships)
    i = units.inputs
    parts = {
        "rollout tick (observe + 2 forwards + sample + game step)": jax.make_jaxpr(rollout_tick)(
            units.params, units.state, units.h, units.key
        ),
        "observe": jax.make_jaxpr(observe)(units.state),
        "one policy forward (A)": jax.make_jaxpr(policy_step)(
            units.params, i["ship"], i["glob"], i["pos"], i["alive"], units.h[0]
        ),
        "game step (B)": jax.make_jaxpr(game_step)(units.state, units.actions[0]),
    }
    for label, closed in parts.items():
        spec.emit(
            {
                "suite": "ops",
                "what": label,
                "ships": a.ships,
                "jaxpr_primitives": count_equations(closed.jaxpr),
            }
        )


CHILDREN = {
    "compile": child_compile,
    "steady": child_steady,
    "sizes": child_sizes,
    "loop": child_loop,
    "latency": child_latency,
    "ops": child_ops,
}


# ---- parent: one fresh process per measurement ---------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "compile", "steady", "sizes", "loop", "latency", "ops"],
    )
    parser.add_argument(
        "--envs",
        default=None,
        help="comma list for steady/loop; default 1,2048 (cuda) or 1,256 (cpu)",
    )
    parser.add_argument("--ships", type=int, default=10)
    parser.add_argument("--micro", default="16x128", help="PPO micro-batch: environments x ticks")
    parser.add_argument(
        "--compile-envs",
        type=int,
        default=None,
        help="E for the compile suite (default: largest --envs)",
    )
    parser.add_argument(
        "--size-envs",
        type=int,
        default=None,
        help="E for the size test; default 256 (cuda) or 32 (cpu)",
    )
    parser.add_argument("--sizes", default="20,40,100")
    parser.add_argument("--pad-to", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=128, help="rollout length for the loop suite")
    parser.add_argument("--latency-ticks", type=int, default=3000)
    parser.add_argument("--latency-ships", default="10,40")
    parser.add_argument("--latency-threads", default="default,1,pin")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per timed repeat")
    parser.add_argument(
        "--precisions", default=None, help="for C: f32 or f32,bf16 (default f32,bf16 on cuda)"
    )
    parser.add_argument("--out", default=None)
    # Internal: what a child process measures.
    parser.add_argument("--child", default=None, choices=list(CHILDREN))
    parser.add_argument("--unit", default="tick")
    parser.add_argument("--variant", default="jit")
    parser.add_argument("--method", default="rejit")
    parser.add_argument("--precision", default="f32")
    parser.add_argument("--threads", default="default")
    parser.add_argument("--child-envs", type=int, default=1)
    a = parser.parse_args()
    cuda = a.device == "cuda"
    if cuda and jax.default_backend() != "gpu":
        sys.exit(
            f"--device cuda, but jax.default_backend() is {jax.default_backend()!r} "
            "(is the CUDA build of jax installed? see README)"
        )
    a.micro = tuple(int(x) for x in str(a.micro).split("x"))
    a.sizes = [int(x) for x in str(a.sizes).split(",")]

    if a.child:
        CHILDREN[a.child](a)
        return

    envs = [int(x) for x in (a.envs or ("1,2048" if cuda else "1,256")).split(",")]
    compile_envs = a.compile_envs or max(envs)
    size_envs = a.size_envs or (256 if cuda else 32)
    precisions = (a.precisions or ("f32,bf16" if cuda else "f32")).split(",")
    micro = f"{a.micro[0]}x{a.micro[1]}"
    common = [
        "--device",
        a.device,
        "--ships",
        a.ships,
        "--micro",
        micro,
        "--repeats",
        a.repeats,
        "--budget",
        a.budget,
    ]
    suites = (
        ["compile", "steady", "sizes", "loop", "latency", "ops"] if a.suite == "all" else [a.suite]
    )
    records = []

    def child(name, *extra, env=None, note=None):
        print(f"[{name}] {' '.join(map(str, extra))}", flush=True)
        got = spec.run_child(HERE, ["--child", name, *common, *extra], env)
        for record in got:
            if note:
                record.update(note)
            print(
                "   ",
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in record.items()
                    if k != "stderr_tail"
                },
                flush=True,
            )
        records.extend(got)

    if "compile" in suites:
        # Cold: fresh process, empty persistent-cache directory (JAX caches nothing on disk unless
        # asked). Warm: a second fresh process reading that directory.
        for unit, precision in [("A", "f32"), ("B", "f32"), ("tick", "f32")] + [
            ("C", p) for p in precisions
        ]:
            cache = tempfile.mkdtemp(prefix="spike-jax-cache-")
            for temperature in ("cold", "warm"):
                child(
                    "compile",
                    "--unit",
                    unit,
                    "--precision",
                    precision,
                    "--child-envs",
                    compile_envs,
                    env={"SPIKE_JAX_CACHE": cache},
                    note={"cache": temperature},
                )
            shutil.rmtree(cache, ignore_errors=True)

    if "steady" in suites:
        for e in envs:
            child("steady", "--unit", "A,B,tick", "--child-envs", e)
        for precision in precisions:
            child("steady", "--unit", "C", "--precision", precision)

    if "sizes" in suites:
        for method in ("rejit", "padded"):
            child(
                "sizes",
                "--method",
                method,
                "--child-envs",
                size_envs,
                "--sizes",
                ",".join(map(str, a.sizes)),
                "--pad-to",
                a.pad_to,
            )

    if "loop" in suites:
        for variant in ("scan", "python"):
            for e in envs:
                child("loop", "--variant", variant, "--child-envs", e, "--ticks", a.ticks)

    if "latency" in suites:
        if cuda:
            print("note: the latency suite is meant for --device cpu; running it on cuda as asked")
        for ships in a.latency_ships.split(","):
            for threads in a.latency_threads.split(","):
                child(
                    "latency",
                    "--ships",
                    ships,
                    "--threads",
                    threads,
                    "--latency-ticks",
                    a.latency_ticks,
                    "--warmup",
                    a.warmup,
                )

    if "ops" in suites:
        child("ops", "--ships", a.ships)

    meta = {
        "framework": "jax",
        "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__,
        "optax": optax.__version__,
        "numpy": np.__version__,
        "device": a.device,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "compile_options": "jax.jit(fn) with defaults; float32 (jax_enable_x64 off)",
        "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "argv": sys.argv[1:],
        "envs": envs,
        "compile_envs": compile_envs,
        "size_envs": size_envs,
        "micro_batch": micro,
        "machine": spec.machine(),
    }
    out = a.out or os.path.join(os.path.dirname(HERE), "results", f"jax_{a.device}_{a.suite}.json")
    spec.write_results(out, meta, records)


if __name__ == "__main__":
    main()
