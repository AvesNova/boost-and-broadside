"""Digests of the round-2 arms' step loops and unrolls on the spread-fit golden's inputs
(offset seed 11, one thread), recorded before arms 3 and 4 existed (`ARMS_GOLDEN`), so the
switches they add leave arms 1 and 2 bit for bit as they were. Run on one thread in a fresh
process (a CPU matmul's summation order follows the process's thread pool):
`test_yemong_error_head.test_arms_1_and_2_step_and_unroll_bit_for_bit_as_recorded`.

`uv run --no-sync python tests/sections/arm_goldens.py --record` (one thread) rewrites it."""

import hashlib
import json
import pathlib
import sys

import torch
import yemong_cases as cases

HERE = pathlib.Path(__file__).parent
SPREAD_GOLDEN = HERE / "golden" / "yemong_v1_spread_steps.pt"
ARMS_GOLDEN = HERE / "golden" / "yemong_arms12_steps.json"
OWN = 5
ARMS = {
    "arm1": dict(order="temporal_first", spread_fit="coarsest", acceleration_inputs=True),
    "arm2": dict(
        order="temporal_first",
        spread_fit="coarsest",
        acceleration_inputs=True,
        next_state="delta",
        delta_baseline="persistence",
        spread_inputs="ellipse",
    ),
}


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def digests(settings: dict) -> dict[str, str]:
    """Each output's SHA-256 of the arm's step loop and unroll over the golden's inputs."""
    golden = torch.load(SPREAD_GOLDEN, weights_only=True)
    obs = cases.golden_obs(golden)
    net = cases.model(**settings)
    torch.manual_seed(11)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(net, net.step, carry, obs, golden["joint"])
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    outputs = dict(
        step_policy_logits=decisions.policy_logits,
        step_next_state_logits=decisions.next_state_logits,
        step_values=decisions.values,
        step_carry_state=final.state,
        step_belief_means=final.belief.means,
        step_belief_spreads=final.belief.spreads,
        step_replay_spreads=decisions.replay.spreads,
        unroll_policy_logits=unrolled.policy_logits,
        unroll_next_state_logits=unrolled.next_state_logits,
        unroll_values=unrolled.values,
    )
    return {name: _digest(value) for name, value in outputs.items()}


def reproduces() -> None:
    """Raises unless every arm's digests equal the recorded ones."""
    expected = json.loads(ARMS_GOLDEN.read_text())
    for arm, settings in ARMS.items():
        assert digests(settings) == expected[arm], arm


if __name__ == "__main__":
    torch.set_num_threads(1)
    if "--record" in sys.argv:
        recorded = {arm: digests(settings) for arm, settings in ARMS.items()}
        ARMS_GOLDEN.write_text(json.dumps(recorded, indent=1) + "\n")
    else:
        reproduces()
