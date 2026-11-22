"""PyTorch side of the framework spike: policy forward (A), toy game step (B), PPO update (C),
and the measurements described in README.md.

    python torch_bench.py --device cpu|cuda --suite all|compile|steady|sizes|loop|latency|ops
                          [--envs 1,2048] [--out results/<name>.json]
"""

import os
import sys
import time

PROCESS_START = time.perf_counter()
if "--threads" in sys.argv and sys.argv[sys.argv.index("--threads") + 1] == "pin":
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})  # Linux only; before the import

import argparse
import collections
import math
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import spec

IMPORT_SECONDS = time.perf_counter() - PROCESS_START
HERE = os.path.abspath(__file__)


# ============================================================================================
# A. Policy
# ============================================================================================

class Embed(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.inp = nn.Linear(features, spec.WIDTH)
        self.norm = nn.RMSNorm(spec.WIDTH, eps=spec.NORM_EPS)
        self.out = nn.Linear(spec.WIDTH, spec.WIDTH)

    def forward(self, x):
        return self.out(F.gelu(self.norm(self.inp(x))))


class RGLRU(nn.Module):
    """Temporal sublayer: x + RG-LRU(RMSNorm(x)), one independent recurrence per token."""

    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(spec.WIDTH, eps=spec.NORM_EPS)
        self.gate_a = nn.Linear(spec.WIDTH, spec.WIDTH)
        self.gate_x = nn.Linear(spec.WIDTH, spec.WIDTH)
        self.lam = nn.Parameter(torch.zeros(spec.WIDTH))

    def gates(self, x):
        """h_t = a_t * h_{t-1} + b_t. The recurrence maths is float32 even under bf16 autocast
        (1 - a_t^2 with a_t near 1 has no precision left in bf16)."""
        x = self.norm(x)
        r = torch.sigmoid(self.gate_a(x)).float()
        i = torch.sigmoid(self.gate_x(x)).float()
        a = torch.exp(spec.RGLRU_C * r * F.logsigmoid(self.lam))   # a^(c r_t), a = sigmoid(lam)
        return a, torch.sqrt(1.0 - a * a) * (i * x.float())

    def step(self, x, h):
        a, b = self.gates(x)
        h = a * h + b
        return x + h, h

    def scan(self, x, h):                    # x [env, time, token, width]
        a, b = self.gates(x)
        states = []
        for t in range(x.shape[1]):
            h = a[:, t] * h + b[:, t]
            states.append(h)
        return x + torch.stack(states, 1), h


def rotate(x, cos, sin):
    """Rotary position rotation of the first 4 * ROPE_BANDS dims of each head."""
    n = 2 * spec.ROPE_BANDS
    x1, x2, rest = x[..., :n], x[..., n:2 * n], x[..., 2 * n:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], -1)


class Spatial(nn.Module):
    """Masked self-attention over the tokens of one environment, then a gated MLP."""

    def __init__(self):
        super().__init__()
        self.attn_norm = nn.RMSNorm(spec.WIDTH, eps=spec.NORM_EPS)
        self.qkv = nn.Linear(spec.WIDTH, 3 * spec.WIDTH, bias=False)
        self.attn_out = nn.Linear(spec.WIDTH, spec.WIDTH)
        self.mlp_norm = nn.RMSNorm(spec.WIDTH, eps=spec.NORM_EPS)
        self.mlp_in = nn.Linear(spec.WIDTH, 2 * spec.WIDTH)
        self.mlp_out = nn.Linear(spec.WIDTH, spec.WIDTH)

    def forward(self, x, mask, cos, sin):    # x [batch, token, width]; mask [batch, token]
        batch, tokens, _ = x.shape
        qkv = self.qkv(self.attn_norm(x)).view(batch, tokens, 3, spec.HEADS, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                       # each [batch, head, token, 64]
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.attn_out(attended.transpose(1, 2).reshape(batch, tokens, -1))
        value, gate = self.mlp_in(self.mlp_norm(x)).chunk(2, -1)
        return x + self.mlp_out(F.gelu(value) * gate)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = RGLRU()
        self.spatial = nn.ModuleList(Spatial() for _ in range(spec.SPATIAL))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_ship = Embed(spec.SHIP_FEATURES)
        self.embed_global = Embed(spec.GLOBAL_FEATURES)
        self.blocks = nn.ModuleList(Block() for _ in range(spec.BLOCKS))
        self.final_norm = nn.RMSNorm(spec.WIDTH, eps=spec.NORM_EPS)
        self.policy_head = nn.Linear(spec.WIDTH, spec.ACTIONS)
        self.value_head = nn.Linear(spec.WIDTH, spec.VALUE_BINS)
        self.register_buffer("rope_freqs", torch.from_numpy(spec.ROPE_FREQS), persistent=False)

    def tokens(self, ship, glob, pos, alive):
        """Token 0 is the global token (always present, at the origin); then one per ship slot."""
        x = torch.cat([self.embed_global(glob).unsqueeze(-2), self.embed_ship(ship)], -2)
        pos = torch.cat([torch.zeros_like(pos[..., :1, :]), pos], -2)
        mask = torch.cat([torch.ones_like(alive[..., :1]), alive], -1)
        angle = torch.cat([pos[..., 0:1] * self.rope_freqs, pos[..., 1:2] * self.rope_freqs], -1)
        return x, mask, angle.cos(), angle.sin()

    def heads(self, x):
        y = self.final_norm(x[..., 1:, :])
        return self.policy_head(y), self.value_head(y)

    def step(self, ship, glob, pos, alive, h):
        """One tick. ship [env, slot, 20], glob [env, 8], pos [env, slot, 2], alive [env, slot],
        h [block, env, token, width] -> policy logits, value logits, new h."""
        x, mask, cos, sin = self.tokens(ship, glob, pos, alive)
        cos, sin = cos[:, None], sin[:, None]
        new_h = []
        for b, block in enumerate(self.blocks):
            x, state = block.temporal.step(x, h[b])
            new_h.append(state)
            for spatial in block.spatial:
                x = spatial(x, mask, cos, sin)
        return self.heads(x) + (torch.stack(new_h),)

    def sequence(self, ship, glob, pos, alive, h0):
        """Learner form: every input has [env, time] leading axes; the recurrence runs over time,
        attention runs with time folded into the batch."""
        x, mask, cos, sin = self.tokens(ship, glob, pos, alive)
        envs, steps, tokens, _ = x.shape
        mask = mask.flatten(0, 1)
        cos, sin = cos.flatten(0, 1)[:, None], sin.flatten(0, 1)[:, None]
        new_h = []
        for b, block in enumerate(self.blocks):
            x, state = block.temporal.scan(x, h0[b])
            new_h.append(state)
            x = x.flatten(0, 1)
            for spatial in block.spatial:
                x = spatial(x, mask, cos, sin)
            x = x.view(envs, steps, tokens, -1)
        return self.heads(x) + (torch.stack(new_h),)


def load_params(policy: Policy, params: dict) -> Policy:
    """Copy spec.init_params() into the module (a spec Linear weight is [in, out])."""
    def linear(module, name):
        module.weight.data.copy_(torch.from_numpy(params[name + ".w"].T))
        if module.bias is not None:
            module.bias.data.copy_(torch.from_numpy(params[name + ".b"]))

    def gain(module, name):
        module.weight.data.copy_(torch.from_numpy(params[name]))

    for kind, embed in (("ship", policy.embed_ship), ("global", policy.embed_global)):
        linear(embed.inp, f"embed.{kind}.in")
        gain(embed.norm, f"embed.{kind}.norm")
        linear(embed.out, f"embed.{kind}.out")
    for b, block in enumerate(policy.blocks):
        gain(block.temporal.norm, f"block{b}.temporal.norm")
        linear(block.temporal.gate_a, f"block{b}.temporal.gate_a")
        linear(block.temporal.gate_x, f"block{b}.temporal.gate_x")
        block.temporal.lam.data.copy_(torch.from_numpy(params[f"block{b}.temporal.lam"]))
        for s, spatial in enumerate(block.spatial):
            gain(spatial.attn_norm, f"block{b}.spatial{s}.attn_norm")
            linear(spatial.qkv, f"block{b}.spatial{s}.qkv")
            linear(spatial.attn_out, f"block{b}.spatial{s}.attn_out")
            gain(spatial.mlp_norm, f"block{b}.spatial{s}.mlp_norm")
            linear(spatial.mlp_in, f"block{b}.spatial{s}.mlp_in")
            linear(spatial.mlp_out, f"block{b}.spatial{s}.mlp_out")
    gain(policy.final_norm, "final_norm")
    linear(policy.policy_head, "policy_head")
    linear(policy.value_head, "value_head")
    return policy


# ============================================================================================
# B. Toy game (a stand-in for the real game's operation structure, not a port)
# ============================================================================================

def game_step(s: dict, action):
    """One tick for every environment. action [env, slot] in 0..41 = turn + 3 * thrust + 21 * fire."""
    turn = (action % 3 - 1).float()
    thrust = (action // 3 % 7).float() / 6.0
    fire = action >= 21
    alive, team = s["alive"], s["team"]
    envs, slots = alive.shape

    # Flight (dead ships stay frozen).
    heading = s["heading"] + turn * (spec.TURN_RATE * spec.DT)
    heading = torch.remainder(heading + math.pi, 2.0 * math.pi) - math.pi
    speed = s["speed"] + spec.DT * (thrust * spec.THRUST_ACC - spec.DRAG * s["speed"])
    speed = speed.clamp(spec.MIN_SPEED, spec.MAX_SPEED)
    heading = torch.where(alive, heading, s["heading"])
    speed = torch.where(alive, speed, s["speed"])
    direction = torch.stack([heading.cos(), heading.sin()], -1)
    velocity = speed[..., None] * direction
    pos = s["pos"] + spec.DT * velocity
    radius = torch.sqrt((pos * pos).sum(-1, keepdim=True))
    outside = (radius - spec.ARENA_RADIUS).clamp(min=0.0)
    pos = pos - (spec.BOUNDARY_PULL * spec.DT) * outside * pos / radius.clamp(min=1.0)
    pos = torch.where(alive[..., None], pos, s["pos"])

    # Bullets move and age.
    bullet_pos = s["bullet_pos"] + spec.DT * s["bullet_vel"]
    bullet_life = (s["bullet_life"] - 1).clamp(min=0)

    # Hits: every bullet against every enemy ship, [env, bullet, ship].
    flat_pos = bullet_pos.flatten(1, 2)
    dx = flat_pos[:, :, None, 0] - pos[:, None, :, 0]
    dy = flat_pos[:, :, None, 1] - pos[:, None, :, 1]
    bullet_team = team[:, :, None].expand(-1, -1, spec.SLOTS).flatten(1, 2)
    hit = ((bullet_life.flatten(1, 2) > 0)[:, :, None] & alive[:, None, :]
           & (bullet_team[:, :, None] != team[:, None, :])
           & (dx * dx + dy * dy < spec.HIT_RADIUS ** 2))
    hp = s["hp"] - spec.DAMAGE * hit.sum(1)
    bullet_life = torch.where(hit.any(2).view(envs, slots, spec.SLOTS), 0, bullet_life)

    # Fire into the first free slot.
    free = bullet_life == 0
    first_free = free.int().argmax(-1)
    shoot = alive & fire & (s["cooldown"] == 0) & free.any(-1)
    new = shoot[..., None] & (torch.arange(spec.SLOTS, device=action.device) == first_free[..., None])
    bullet_pos = torch.where(new[..., None], pos[:, :, None, :], bullet_pos)
    muzzle = velocity + spec.MUZZLE_SPEED * direction
    bullet_vel = torch.where(new[..., None], muzzle[:, :, None, :], s["bullet_vel"])
    bullet_life = torch.where(new, spec.BULLET_LIFE, bullet_life)
    cooldown = torch.where(shoot, spec.COOLDOWN, (s["cooldown"] - 1).clamp(min=0))

    # Death, then respawn after a delay.
    died = alive & (hp <= 0.0)
    alive = alive & ~died
    respawn = torch.where(died, spec.RESPAWN_DELAY, (s["respawn"] - 1).clamp(min=0))
    back = s["exists"] & ~alive & (respawn == 0)
    return {
        "exists": s["exists"], "team": team,
        "spawn_pos": s["spawn_pos"], "spawn_heading": s["spawn_heading"],
        "pos": torch.where(back[..., None], s["spawn_pos"], pos),
        "heading": torch.where(back, s["spawn_heading"], heading),
        "speed": torch.where(back, spec.START_SPEED, speed),
        "hp": torch.where(back, spec.MAX_HP, hp),
        "alive": alive | back,
        "cooldown": cooldown, "respawn": respawn,
        "bullet_pos": bullet_pos, "bullet_vel": bullet_vel, "bullet_life": bullet_life,
        "tick": s["tick"] + 1,
    }


def observe(s: dict):
    """Game state -> policy inputs (ship features [env, slot, 20] from team 0's side, global
    features [env, 8], positions, alive mask). Team 1's view negates feature 10 (see other_side)."""
    pos, alive, team = s["pos"], s["alive"], s["team"]
    alive_f = alive.float()
    in_1 = team.float() * s["exists"].float()
    in_0 = s["exists"].float() - in_1
    alive_0, alive_1 = alive_f * in_0, alive_f * in_1
    count_0 = in_0.sum(1, keepdim=True).clamp(min=1.0)
    count_1 = in_1.sum(1, keepdim=True).clamp(min=1.0)
    n_alive_0 = alive_0.sum(1, keepdim=True)
    n_alive_1 = alive_1.sum(1, keepdim=True)
    frac_0, frac_1 = n_alive_0 / count_0, n_alive_1 / count_1
    hp_0 = (s["hp"] * alive_0).sum(1, keepdim=True) / (count_0 * spec.MAX_HP)
    hp_1 = (s["hp"] * alive_1).sum(1, keepdim=True) / (count_1 * spec.MAX_HP)
    centre_0 = (pos * alive_0[..., None]).sum(1, keepdim=True) / n_alive_0.clamp(min=1.0)[..., None]
    centre_1 = (pos * alive_1[..., None]).sum(1, keepdim=True) / n_alive_1.clamp(min=1.0)[..., None]
    bullets = (s["bullet_life"] > 0).float().mean(-1)
    scaled = pos / spec.ARENA_RADIUS
    from_0 = (pos - centre_0) / spec.ARENA_RADIUS
    from_1 = (pos - centre_1) / spec.ARENA_RADIUS
    speed = s["speed"] / spec.MAX_SPEED
    cos, sin = s["heading"].cos(), s["heading"].sin()
    ship = torch.stack([
        scaled[..., 0], scaled[..., 1], speed * cos, speed * sin, cos, sin, speed,
        s["hp"] / spec.MAX_HP, alive_f, s["cooldown"].float() / spec.COOLDOWN,
        1.0 - 2.0 * team.float(), s["respawn"].float() / spec.RESPAWN_DELAY,
        torch.sqrt((scaled * scaled).sum(-1)), bullets,
        from_0[..., 0], from_0[..., 1], from_1[..., 0], from_1[..., 1],
        frac_0.expand_as(alive_f), frac_1.expand_as(alive_f)], -1)
    gap = centre_0[:, 0] - centre_1[:, 0]
    glob = torch.cat([
        s["tick"].float()[:, None] / 1000.0, frac_0, frac_1, hp_0, hp_1,
        torch.sqrt((gap * gap).sum(-1, keepdim=True)) / spec.ARENA_RADIUS,
        bullets.sum(1, keepdim=True) / (count_0 + count_1), torch.ones_like(frac_0)], -1)
    return ship, glob, pos, alive


def other_side(ship):
    return torch.cat([ship[..., :10], -ship[..., 10:11], ship[..., 11:]], -1)


def choose(policy: Policy, s: dict, h):
    """The two policy forwards of a tick (one per team's view; each sees every ship) and the
    merged per-ship logits: a ship takes the output of its own team's forward."""
    ship, glob, pos, alive = observe(s)
    logits_0, value_0, h_0 = policy.step(ship, glob, pos, alive, h[0])
    logits_1, value_1, h_1 = policy.step(other_side(ship), glob, pos, alive, h[1])
    own = (s["team"] == 0)[..., None]
    return torch.where(own, logits_0, logits_1), torch.where(own, value_0, value_1), torch.stack([h_0, h_1])


def rollout_tick(policy: Policy, s: dict, h):
    """Observe, two policy forwards, sample an action per ship, step the game."""
    logits, value, h = choose(policy, s, h)
    action = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
    logp = torch.log_softmax(logits, -1).gather(-1, action[..., None]).squeeze(-1)
    return game_step(s, action), h, action, logp, value


# ============================================================================================
# C. PPO update
# ============================================================================================

def ppo_loss(policy: Policy, b: dict):
    logits, value_logits, _ = policy.sequence(b["ship"], b["glob"], b["pos"], b["alive"], b["h0"])
    logp_all = torch.log_softmax(logits.float(), -1)
    logp = logp_all.gather(-1, b["action"][..., None]).squeeze(-1)
    ratio = torch.exp(logp - b["old_logp"])
    surrogate = torch.minimum(ratio * b["advantage"],
                              ratio.clamp(1.0 - spec.CLIP, 1.0 + spec.CLIP) * b["advantage"])
    weight = b["alive"].float()
    count = weight.sum().clamp(min=1.0)
    policy_loss = -(surrogate * weight).sum() / count
    value_loss = -((b["value_target"] * torch.log_softmax(value_logits.float(), -1)).sum(-1) * weight).sum() / count
    entropy = (-(logp_all.exp() * logp_all).sum(-1) * weight).sum() / count
    return policy_loss + spec.VALUE_COEF * value_loss - spec.ENTROPY_COEF * entropy


def ppo_update(loss_fn, policy, optimizer, batch, bf16: bool):
    """One minibatch gradient step. `loss_fn` is ppo_loss or torch.compile(ppo_loss)."""
    with torch.autocast(batch["ship"].device.type, dtype=torch.bfloat16, enabled=bf16):
        loss = loss_fn(policy, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), spec.MAX_GRAD_NORM)
    optimizer.step()
    return loss, grad_norm


def make_optimizer(policy):
    return torch.optim.Adam(policy.parameters(), lr=spec.LEARNING_RATE, eps=spec.ADAM_EPS)


# ============================================================================================
# Benchmark plumbing
# ============================================================================================

def to_device(tree: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in tree.items()}


def make_sync(device):
    return torch.cuda.synchronize if device.type == "cuda" else (lambda: None)


COMPILE_KWARGS = {"default": {}, "dynamic": {"dynamic": True}, "mark_dynamic": {}}


def compile_fn(fn, variant: str):
    """variant: eager | compiled (torch.compile defaults: mode='default', backend='inductor',
    dynamic=None, fullgraph=False) | dynamic (dynamic=True) | mark_dynamic (defaults + marks)."""
    if variant == "eager":
        return fn
    kind = "default" if variant == "compiled" else variant
    return torch.compile(fn, **COMPILE_KWARGS[kind])


def mark_ship_dim(state: dict, h):
    """torch._dynamo.mark_dynamic on the ship axis of every input of the tick."""
    for name, value in state.items():
        if name != "tick":
            torch._dynamo.mark_dynamic(value, 1)
    torch._dynamo.mark_dynamic(h, 3)


class Units:
    """The four timed units at one size, each a zero-argument callable that carries its state."""

    def __init__(self, device, variant, envs, ships, pad_to=None, bf16=False, micro=(16, 128)):
        self.device, self.variant, self.envs, self.ships, self.pad_to = device, variant, envs, ships, pad_to
        self.bf16, self.micro = bf16, micro
        self.slots = pad_to or ships
        self.policy = load_params(Policy(), spec.init_params()).to(device)
        self.fresh()

    def fresh(self):
        d = self.device
        self.state = to_device(spec.init_state(self.envs, self.ships, self.pad_to), d)
        self.h = torch.from_numpy(spec.init_hidden(self.envs, self.slots)).to(d)
        self.inputs = to_device(spec.make_policy_inputs(self.envs, self.slots), d)
        self.h_single = self.h[0].clone()
        self.actions = torch.from_numpy(spec.random_actions(3, 64, self.envs, self.slots)).to(d).long()
        self.i = 0

    def build(self, unit: str):
        variant = self.variant
        if unit == "A":
            step = compile_fn(self.policy.step, variant)

            @torch.no_grad()
            def run():
                i = self.inputs
                _, _, self.h_single = step(i["ship"], i["glob"], i["pos"], i["alive"], self.h_single)
        elif unit == "B":
            step = compile_fn(game_step, variant)

            @torch.no_grad()
            def run():
                self.state = step(self.state, self.actions[self.i % 64])
                self.i += 1
        elif unit == "tick":
            step = compile_fn(rollout_tick, variant)

            @torch.no_grad()
            def run():
                if variant == "mark_dynamic":
                    mark_ship_dim(self.state, self.h)
                self.state, self.h, *self.last = step(self.policy, self.state, self.h)
        elif unit == "C":
            self.batch = to_device(spec.make_rollout(self.micro[0], self.micro[1], self.ships), self.device)
            self.batch["action"] = self.batch["action"].long()
            optimizer = make_optimizer(self.policy)
            loss_fn = compile_fn(ppo_loss, variant)

            def run():
                self.last_loss, _ = ppo_update(loss_fn, self.policy, optimizer, self.batch, self.bf16)
        else:
            raise ValueError(unit)
        return run


def compiled_graphs() -> int:
    """Graphs TorchDynamo has compiled so far in this process (private counter, used for the
    recompile count only)."""
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


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


# ---- children (each runs in a fresh process) -----------------------------------------------

def child_compile(a, device):
    units = Units(device, a.variant, a.child_envs, a.ships, bf16=a.precision == "bf16", micro=a.micro)
    run = units.build(a.unit)
    record = {"suite": "compile", "unit": a.unit, "variant": a.variant, "envs": a.child_envs,
              "ships": a.ships, "precision": a.precision, "import_s": IMPORT_SECONDS}
    if a.unit == "C":
        record.update(envs=a.micro[0], time=a.micro[1])
    record.update(first_calls(run, make_sync(device)))
    record["graphs_compiled"] = compiled_graphs()
    spec.emit(record)


def child_steady(a, device):
    sync = make_sync(device)
    for unit in a.unit.split(","):
        units = Units(device, a.variant, a.child_envs, a.ships, bf16=a.precision == "bf16", micro=a.micro)
        run = units.build(unit)
        record = {"suite": "steady", "unit": unit, "variant": a.variant, "envs": a.child_envs,
                  "ships": a.ships, "precision": a.precision}
        if unit == "C":
            record.update(envs=a.micro[0], time=a.micro[1])
        record.update(first_calls(run, sync))
        record.update(steady(run, sync, a.repeats, a.budget))
        record["graphs_compiled"] = compiled_graphs()
        spec.emit(record)


def child_sizes(a, device):
    """Compile the rollout tick at N = 10, then meet N = 20, 40, 100 (and N = 10 again)."""
    sync = make_sync(device)
    padded = a.method.endswith("padded")
    variant = {"default": "compiled", "padded": "compiled", "dynamic": "dynamic",
               "mark_dynamic": "mark_dynamic", "eager": "eager", "eager_padded": "eager"}[a.method]
    step = compile_fn(rollout_tick, variant)
    policy = None
    for ships in [10] + a.sizes + [10]:
        units = Units(device, variant, a.child_envs, ships, pad_to=a.pad_to if padded else None)
        policy = units.policy = policy or units.policy      # one set of weights for every size

        @torch.no_grad()
        def run():
            if variant == "mark_dynamic":
                mark_ship_dim(units.state, units.h)
            units.state, units.h, _, _, _ = step(units.policy, units.state, units.h)

        before = compiled_graphs()
        record = {"suite": "sizes", "method": a.method, "envs": a.child_envs, "ships": ships,
                  "slots": units.slots}
        try:
            record.update(first_calls(run, sync))
            record.update(steady(run, sync, a.repeats, a.budget))
        except Exception as error:  # a failure here is a finding, not something to hide
            record["error"] = f"{type(error).__name__}: {str(error)[:1500]}"
        record["graphs_compiled_here"] = compiled_graphs() - before
        record["graphs_compiled_total"] = compiled_graphs()
        spec.emit(record)


def child_loop(a, device):
    """T ticks as a Python loop over the tick (compiled or eager)."""
    sync = make_sync(device)
    units = Units(device, a.variant, a.child_envs, a.ships)
    tick = units.build("tick")

    def rollout():   # keeps actions and log-probs, as the JAX scan does
        kept = []
        for _ in range(a.ticks):
            tick()
            kept.append(units.last[:2])
        units.kept = torch.stack([k[0] for k in kept]), torch.stack([k[1] for k in kept])

    start = time.perf_counter()
    rollout()
    sync()
    first = time.perf_counter() - start
    rollout()
    stats = spec.timed_repeats(rollout, sync, a.repeats, 1)
    spec.emit({"suite": "loop", "variant": a.variant + " tick, Python loop", "envs": a.child_envs,
               "ships": a.ships, "ticks": a.ticks, "first_rollout_s": first, "rollout": stats,
               "ticks_per_s": a.ticks / (stats["median_ms"] / 1000.0),
               "env_ticks_per_s": a.ticks * a.child_envs / (stats["median_ms"] / 1000.0)})


def child_latency(a, device):
    """Batch size 1: one game tick plus two policy forwards, timed per tick."""
    if a.threads == "1":
        torch.set_num_threads(1)
    sync = make_sync(device)
    units = Units(device, a.variant, 1, a.ships)
    tick = units.build("tick")
    for _ in range(a.warmup):
        tick()
    samples = []
    for _ in range(a.latency_ticks):
        start = time.perf_counter()
        tick()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    record = {"suite": "latency", "variant": a.variant, "ships": a.ships, "threads": a.threads,
              "torch_num_threads": torch.get_num_threads(), "warmup": a.warmup}
    try:
        record["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        pass
    record.update(spec.latency_stats(samples))
    spec.emit(record)


class OpCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts dispatched ops, split into compute and view ops, as in simplification/regions.py."""

    METADATA = {"aten.sym_size.int", "aten.sym_stride.int", "aten.sym_numel.default",
                "aten.sym_storage_offset.default", "aten.is_contiguous.default",
                "aten.lift_fresh.default", "aten._local_scalar_dense.default"}

    def __init__(self):
        super().__init__()
        self.compute, self.view = collections.Counter(), collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name not in self.METADATA:
            (self.view if getattr(func, "is_view", False) else self.compute)[name] += 1
        return func(*args, **(kwargs or {}))


def child_ops(a, device):
    units = Units(device, "eager", 1, a.ships)
    s, h, inputs = units.state, units.h, units.inputs

    def count(fn):
        with torch.no_grad(), OpCounter() as counter:
            fn()
        return counter

    parts = {
        "rollout tick (observe + 2 forwards + sample + game step)": lambda: rollout_tick(units.policy, s, h),
        "observe": lambda: observe(s),
        "one policy forward (A)": lambda: units.policy.step(inputs["ship"], inputs["glob"], inputs["pos"], inputs["alive"], h[0]),
        "game step (B)": lambda: game_step(s, units.actions[0]),
    }
    for label, fn in parts.items():
        counter = count(fn)
        spec.emit({"suite": "ops", "what": label, "ships": a.ships,
                   "compute_ops": sum(counter.compute.values()), "view_ops": sum(counter.view.values()),
                   "top_compute_ops": counter.compute.most_common(12)})


CHILDREN = {"compile": child_compile, "steady": child_steady, "sizes": child_sizes,
            "loop": child_loop, "latency": child_latency, "ops": child_ops}


# ---- parent: one fresh process per measurement ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--suite", default="all",
                        choices=["all", "compile", "steady", "sizes", "loop", "latency", "ops"])
    parser.add_argument("--envs", default=None, help="comma list for steady/loop; default 1,2048 (cuda) or 1,256 (cpu)")
    parser.add_argument("--ships", type=int, default=10)
    parser.add_argument("--micro", default="16x128", help="PPO micro-batch: environments x ticks")
    parser.add_argument("--compile-envs", type=int, default=None, help="E for the compile suite (default: largest --envs)")
    parser.add_argument("--size-envs", type=int, default=None, help="E for the size test; default 256 (cuda) or 32 (cpu)")
    parser.add_argument("--sizes", default="20,40,100")
    parser.add_argument("--pad-to", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=128, help="rollout length for the loop suite")
    parser.add_argument("--latency-ticks", type=int, default=3000)
    parser.add_argument("--latency-ships", default="10,40")
    parser.add_argument("--latency-threads", default="default,1,pin")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per timed repeat")
    parser.add_argument("--precisions", default=None, help="for C: f32 or f32,bf16 (default f32,bf16 on cuda)")
    parser.add_argument("--skip-compiled-c", action="store_true", help="skip torch.compile of the PPO update")
    parser.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"],
                        help="torch.set_float32_matmul_precision (PyTorch default: highest)")
    parser.add_argument("--out", default=None)
    # Internal: what a child process measures.
    parser.add_argument("--child", default=None, choices=list(CHILDREN))
    parser.add_argument("--unit", default="tick")
    parser.add_argument("--variant", default="compiled")
    parser.add_argument("--method", default="default")
    parser.add_argument("--precision", default="f32")
    parser.add_argument("--threads", default="default")
    parser.add_argument("--child-envs", type=int, default=1)
    a = parser.parse_args()
    cuda = a.device == "cuda"
    if cuda and not torch.cuda.is_available():
        sys.exit("--device cuda, but torch.cuda.is_available() is False")
    device = torch.device(a.device)
    a.micro = tuple(int(x) for x in str(a.micro).split("x"))
    a.sizes = [int(x) for x in str(a.sizes).split(",")]
    torch.set_float32_matmul_precision(a.matmul_precision)

    if a.child:
        CHILDREN[a.child](a, device)
        return

    envs = [int(x) for x in (a.envs or ("1,2048" if cuda else "1,256")).split(",")]
    compile_envs = a.compile_envs or max(envs)
    size_envs = a.size_envs or (256 if cuda else 32)
    precisions = (a.precisions or ("f32,bf16" if cuda else "f32")).split(",")
    micro = f"{a.micro[0]}x{a.micro[1]}"
    common = ["--device", a.device, "--ships", a.ships, "--micro", micro, "--repeats", a.repeats,
              "--budget", a.budget, "--matmul-precision", a.matmul_precision]
    suites = ["compile", "steady", "sizes", "loop", "latency", "ops"] if a.suite == "all" else [a.suite]
    records = []

    def child(name, *extra, env=None, note=None):
        print(f"[{name}] {' '.join(map(str, extra))}", flush=True)
        got = spec.run_child(HERE, ["--child", name, *common, *extra], env)
        for record in got:
            if note:
                record.update(note)
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
                          if k not in ("top_compute_ops", "stderr_tail")}, flush=True)
        records.extend(got)

    def c_variants():
        return [(p, v) for p in precisions for v in ("eager", "compiled")
                if not (v == "compiled" and a.skip_compiled_c)]

    if "compile" in suites:
        # Cold: an empty TORCHINDUCTOR_CACHE_DIR, fresh process. Warm: a second fresh process with
        # the same directory (inductor's on-disk FX-graph / AOTAutograd / compiled-kernel caches).
        jobs = [(unit, "f32", variant) for unit in ("A", "B", "tick") for variant in ("eager", "compiled")]
        jobs += [("C", p, v) for p, v in c_variants()]
        for unit, precision, variant in jobs:
            cache = tempfile.mkdtemp(prefix="spike-inductor-")
            for temperature in (("cold", "warm") if variant == "compiled" else ("cold",)):
                child("compile", "--unit", unit, "--variant", variant, "--precision", precision,
                      "--child-envs", compile_envs, env={"TORCHINDUCTOR_CACHE_DIR": cache},
                      note={"cache": temperature})
            shutil.rmtree(cache, ignore_errors=True)

    if "steady" in suites:
        for variant in ("eager", "compiled"):
            for e in envs:
                child("steady", "--unit", "A,B,tick", "--variant", variant, "--child-envs", e)
        for precision, variant in c_variants():
            child("steady", "--unit", "C", "--variant", variant, "--precision", precision)

    if "sizes" in suites:
        for method in ("default", "dynamic", "mark_dynamic", "padded", "eager", "eager_padded"):
            cache = tempfile.mkdtemp(prefix="spike-inductor-")
            child("sizes", "--method", method, "--child-envs", size_envs, "--sizes", ",".join(map(str, a.sizes)),
                  "--pad-to", a.pad_to, env={"TORCHINDUCTOR_CACHE_DIR": cache})
            shutil.rmtree(cache, ignore_errors=True)

    if "loop" in suites:
        for variant in ("eager", "compiled"):
            for e in envs:
                child("loop", "--variant", variant, "--child-envs", e, "--ticks", a.ticks)

    if "latency" in suites:
        if cuda:
            print("note: the latency suite is meant for --device cpu; running it on cuda as asked")
        for ships in a.latency_ships.split(","):
            for threads in a.latency_threads.split(","):
                for variant in ("eager", "compiled"):
                    child("latency", "--variant", variant, "--ships", ships, "--threads", threads,
                          "--latency-ticks", a.latency_ticks, "--warmup", a.warmup)

    if "ops" in suites:
        child("ops", "--ships", a.ships)

    meta = {
        "framework": "pytorch", "torch": torch.__version__, "numpy": np.__version__, "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_runtime": torch.version.cuda if cuda else None,
        "torch_num_threads_default": torch.get_num_threads(),
        "compile_options": "torch.compile(fn) with defaults: backend='inductor', mode='default', "
                           "dynamic=None, fullgraph=False; 'dynamic' = dynamic=True; "
                           "'mark_dynamic' = defaults + torch._dynamo.mark_dynamic on the ship axis",
        "float32_matmul_precision": a.matmul_precision,
        "cxx": shutil.which("g++") or shutil.which("c++") or shutil.which("cl"),
        "argv": sys.argv[1:], "envs": envs, "compile_envs": compile_envs, "size_envs": size_envs,
        "micro_batch": micro, "machine": spec.machine(),
    }
    out = a.out or os.path.join(os.path.dirname(HERE), "results", f"torch_{a.device}_{a.suite}.json")
    spec.write_results(out, meta, records)


if __name__ == "__main__":
    main()
