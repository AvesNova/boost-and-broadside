"""Check that torch_bench.py and jax_bench.py compute the same thing.

The two frameworks live in different virtualenvs, so this runs in three steps:

    <torch python> parity.py dump torch /tmp/parity_torch.npz
    <jax python>   parity.py dump jax   /tmp/parity_jax.npz
    <any python>   parity.py compare /tmp/parity_torch.npz /tmp/parity_jax.npz

Both dumps start from the same numpy weights, inputs, game state and actions (spec.py), on the
CPU in float32. Compared: the single-step policy forward, the learner (sequence) forward, 300
game ticks under the same random actions, the observation, the PPO loss, its gradient norm, and
the loss after each of 5 Adam steps. Not compared: sampled actions (different random generators).
Each dump also checks two things inside its own framework: the sequence forward equals repeated
single steps, and padding 10 ships to 100 slots leaves the 10 real ships' outputs unchanged.
"""

import sys

import numpy as np

import spec

GAME_TICKS = 300


def dump(framework: str, path: str) -> None:
    rng = np.random.default_rng(7)
    params = spec.init_params()
    single = spec.make_policy_inputs(3, 10)
    h0 = (0.5 * rng.standard_normal((spec.BLOCKS, 3, 11, spec.WIDTH))).astype(np.float32)
    seq = spec.make_policy_inputs(2, 10, seed=5, time=12)
    seq_h0 = (0.5 * rng.standard_normal((spec.BLOCKS, 2, 11, spec.WIDTH))).astype(np.float32)
    state = spec.init_state(4, 10)
    padded_state = spec.init_state(4, 10, pad_to=100)
    actions = spec.random_actions(11, GAME_TICKS, 4, 10)
    padded_actions = np.zeros((GAME_TICKS, 4, 100), np.int32)
    padded_actions[:, :, :10] = actions
    rollout = spec.make_rollout(2, 16, 10)
    out = {}

    if framework == "torch":
        import torch
        import torch_bench as tb
        torch.manual_seed(0)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        tree = lambda d: {k: t(v) for k, v in d.items()}
        n = lambda x: x.detach().numpy()
        policy = tb.load_params(tb.Policy(), params)
        forward = lambda i, h: [n(x) for x in policy.step(t(i["ship"]), t(i["glob"]), t(i["pos"]), t(i["alive"]), t(h))]
        sequence = lambda i, h: [n(x) for x in policy.sequence(t(i["ship"]), t(i["glob"]), t(i["pos"]), t(i["alive"]), t(h))]

        def play(s, acts):
            s = tree(s)
            for a in acts:
                s = tb.game_step(s, t(a).long())
            return {k: n(v) for k, v in s.items()}, [n(x) for x in tb.observe(s)], [n(x) for x in tb.choose(policy, s, t(spec.init_hidden(4, acts.shape[2])))]

        def ppo():
            batch = tree(rollout)
            batch["action"] = batch["action"].long()
            loss = tb.ppo_loss(policy, batch)
            grads = torch.autograd.grad(loss, list(policy.parameters()))
            norm = torch.sqrt(sum((g * g).sum() for g in grads))
            optimizer = tb.make_optimizer(policy)
            losses = [float(tb.ppo_update(tb.ppo_loss, policy, optimizer, batch, False)[0]) for _ in range(5)]
            return float(loss), float(norm), losses
        with torch.no_grad():
            results = forward(single, h0), sequence(seq, seq_h0), play(state, actions), play(padded_state, padded_actions)
            by_steps = seq_by_steps(forward, seq, seq_h0)
        loss, grad_norm, losses = ppo()
    else:
        import jax
        import jax.numpy as jnp
        import jax_bench as jb
        p = jb.to_jax(params)
        n = np.asarray
        forward = lambda i, h: [n(x) for x in jb.policy_step(p, i["ship"], i["glob"], i["pos"], i["alive"], h)]
        sequence = lambda i, h: [n(x) for x in jb.policy_sequence(p, i["ship"], i["glob"], i["pos"], i["alive"], h)]
        step = jax.jit(jb.game_step)

        def play(s, acts):
            s = jb.to_jax(s)
            for a in acts:
                s = step(s, jnp.asarray(a))
            return {k: n(v) for k, v in s.items()}, [n(x) for x in jb.observe(s)], [n(x) for x in jb.choose(p, s, jnp.asarray(spec.init_hidden(4, acts.shape[2])))]

        def ppo():
            batch = jb.to_jax(rollout)
            loss, grads = jax.value_and_grad(jb.ppo_loss)(p, batch)
            optimizer = jb.make_optimizer()
            update = jax.jit(jb.make_ppo_update(optimizer))
            q, opt_state, losses = p, optimizer.init(p), []
            for _ in range(5):
                q, opt_state, value, _ = update(q, opt_state, batch)
                losses.append(float(value))
            return float(loss), float(jb.optax.global_norm(grads)), losses
        results = forward(single, h0), sequence(seq, seq_h0), play(state, actions), play(padded_state, padded_actions)
        by_steps = seq_by_steps(forward, seq, seq_h0)
        loss, grad_norm, losses = ppo()

    (logits, values, h1), (seq_logits, seq_values, seq_h), (final, obs, chosen), (final_pad, obs_pad, chosen_pad) = results
    out.update({"step.logits": logits, "step.values": values, "step.h": h1,
                "sequence.logits": seq_logits, "sequence.values": seq_values, "sequence.h": seq_h})
    out.update({f"game.{k}": v for k, v in final.items()})
    out.update({"observe.ship": obs[0], "observe.glob": obs[1],
                "tick.logits": chosen[0], "tick.values": chosen[1], "tick.h": chosen[2]})
    out.update({"ppo.loss": np.float32(loss), "ppo.grad_norm": np.float32(grad_norm),
                "ppo.losses_after_updates": np.asarray(losses, np.float32)})
    np.savez(path, **out)

    print(f"[{framework}] dumped {len(out)} arrays to {path}")
    print(f"[{framework}] game after {GAME_TICKS} ticks: {int((final['hp'] < spec.MAX_HP).sum())} damaged ships, "
          f"{int((~final['alive']).sum())} dead, {int((final['bullet_life'] > 0).sum())} live bullets, "
          f"max |pos| {float(np.abs(final['pos']).max()):.0f} m")
    print(f"[{framework}] sequence forward vs repeated single steps: max |diff| logits "
          f"{np.abs(by_steps[0] - seq_logits).max():.2e}, values {np.abs(by_steps[1] - seq_values).max():.2e}")
    real = lambda x: x[:, :10]
    print(f"[{framework}] padded to 100 slots vs exact (the 10 real ships): max |diff| "
          f"pos {np.abs(real(final_pad['pos']) - final['pos']).max():.2e}, hp {np.abs(real(final_pad['hp']) - final['hp']).max():.2e}, "
          f"ship features {np.abs(real(obs_pad[0]) - obs[0]).max():.2e}, global features {np.abs(obs_pad[1] - obs[1]).max():.2e}, "
          f"policy logits {np.abs(real(chosen_pad[0]) - chosen[0]).max():.2e}")


def seq_by_steps(forward, seq, h):
    """The learner form must equal the single-step form applied tick by tick."""
    logits, values = [], []
    for t in range(seq["ship"].shape[1]):
        l, v, h = forward({k: x[:, t] for k, x in seq.items()}, h)
        logits.append(l)
        values.append(v)
    return np.stack(logits, 1), np.stack(values, 1)


def compare(path_a: str, path_b: str, tolerance: float = 1e-4) -> int:
    a, b = np.load(path_a), np.load(path_b)
    worst, failed = 0.0, []
    print(f"{'array':<32}{'shape':<22}{'max |diff|':>12}{'max |diff| / (1 + |value|)':>30}")
    for name in a.files:
        x, y = a[name], b[name]
        if x.dtype.kind in "bi":
            different = int((x != y).sum())
            print(f"{name:<32}{str(x.shape):<22}{different:>12} elements differ")
            if different:
                failed.append(name)
            continue
        diff = np.abs(x.astype(np.float64) - y)
        scaled = float((diff / (1.0 + np.abs(y))).max())
        print(f"{name:<32}{str(x.shape):<22}{float(diff.max()):>12.2e}{scaled:>30.2e}")
        worst = max(worst, scaled)
        if scaled > tolerance:
            failed.append(name)
    print(f"\nworst scaled difference {worst:.2e} (tolerance {tolerance:g}); "
          + ("ALL AGREE" if not failed else f"DISAGREE: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "dump" and sys.argv[2] in ("torch", "jax"):
        dump(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
