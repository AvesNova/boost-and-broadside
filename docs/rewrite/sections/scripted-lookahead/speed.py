"""GPU time per scripted call (both teams, one call) for the look-ahead agent, against the
policy's forward on the training seats (E + E / 2, self-play share 0.5), all
compiled (lean.contract.compile_dynamic), on a 4v4 state with fields and fog after a short
scripted warm-up; and each call's peak memory above the state's.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/speed.py \
        --envs 2048 --size 4
"""

import argparse
import time

import torch

from lean import game, model, scripted
from lean.contract import compile_dynamic
from lean.learn import LearnConfig
from lean.learn.rollout import seats
from lean.model import draw_position_offset
from lean.model.agent import agent_step, seat_view


def timed(function, repeats: int) -> tuple[float, float]:
    """(milliseconds per call, peak MiB above the start) over `repeats` calls after 3 warm."""
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / repeats
    return 1000.0 * seconds, (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--plain", action="store_true", help="the plain PyTorch pair stage")
    parser.add_argument("--look-ahead", default=None, help="points, e.g. 0.25,0.5,1,2")
    parser.add_argument("--retarget", type=int, default=None, help="1 / 0: paths.RETARGET")
    arguments = parser.parse_args()
    if arguments.plain:
        from lean.scripted import kernels

        kernels.USE_KERNEL = False
    if arguments.look_ahead:
        from lean.scripted import paths

        paths.LOOK_AHEAD = tuple(float(t) for t in arguments.look_ahead.split(","))
    if arguments.retarget is not None:
        from lean.scripted import paths

        paths.RETARGET = bool(arguments.retarget)
    device = torch.device("cuda")
    envs = arguments.envs
    config = game.GameConfig()
    generator = torch.Generator(device).manual_seed(0)
    sizes = torch.full((envs, 2), arguments.size, dtype=torch.long, device=device)
    state = game.reset(config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    new_config = scripted.ScriptedConfig()

    def new_call(state, uniforms):
        return scripted.decide(new_config, state, None, None, uniforms)

    compiled_new = compile_dynamic(new_call)
    with torch.no_grad():
        for _ in range(30):  # into the fight a little, with the agent itself (compiled)
            uniforms = scripted.draw_uniforms(envs, slots, generator, device)
            action, _ = compiled_new(state, uniforms)
            state, _ = game.step(
                config, state, action, game.draw_noise(config, envs, slots, generator, device)
            )
    new_uniforms = scripted.draw_uniforms(envs, slots, generator, device)
    with torch.no_grad():
        for name, function in (("look-ahead", lambda: compiled_new(state, new_uniforms)),):
            begun = time.perf_counter()
            function()
            torch.cuda.synchronize()
            first = time.perf_counter() - begun
            milliseconds, peak = timed(function, arguments.repeats)
            print(
                f"{name}: {milliseconds:.2f} ms per call, peak {peak:.0f} MiB, first call "
                f"(compile) {first:.1f} s",
                flush=True,
            )
        if arguments.skip_policy:
            return
        learn = LearnConfig()
        policy = model.Policy(model.ModelConfig()).to(device)
        self_play = round(learn.self_play_share * envs)
        learner_team = torch.randint(0, 2, (envs,), generator=generator, device=device)
        seat_environment, seat_team = seats(learner_team, self_play)
        views = (game.observe(config, state, 0), game.observe(config, state, 1))
        observation = seat_view(views, seat_environment, seat_team)
        carry = policy.initial_carry(envs + self_play, slots, device)
        offset = draw_position_offset(policy.config, envs + self_play, generator, device)

        def policy_call(observation, carry, offset):
            return agent_step(policy.step, learn.bf16, observation, carry, offset, None)

        compiled_policy = compile_dynamic(policy_call)
        begun = time.perf_counter()
        compiled_policy(observation, carry, offset)
        torch.cuda.synchronize()
        first = time.perf_counter() - begun
        milliseconds, peak = timed(
            lambda: compiled_policy(observation, carry, offset), arguments.repeats
        )
        print(
            f"policy forward ({envs + self_play} seats): {milliseconds:.2f} ms per call, peak "
            f"{peak:.0f} MiB, first call (compile) {first:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
