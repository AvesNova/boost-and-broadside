"""The arms' step loops and unrolls on the spread-fit golden's inputs (offset seed 11),
recorded as float64 tensors (`ARMS_GOLDEN`): the switches later arms add leave arms 1 and 2, and
C-fixed (the same without the acceleration inputs), as they were, to float64 rounding
(`yemong_cases.assert_matches_golden`). The spread-fit golden's own outputs (`v1`) are
recorded here too.

`uv run --no-sync python tests/sections/arm_goldens.py --record` rewrites both."""

import pathlib
import sys

import torch
import yemong_cases as cases

HERE = pathlib.Path(__file__).parent
SPREAD_GOLDEN = HERE / "golden" / "yemong_v1_spread_steps.pt"
ARMS_GOLDEN = HERE / "golden" / "yemong_arms_steps.pt"
OWN = 5
F64 = torch.float64
FIRST = 8  # the per-tick outputs are wide: the file keeps their first eight ticks
WIDE_TICKS = 1  # and the next-state logits, wider still, their first
C_FIXED = dict(order="temporal_first", spread_fit="coarsest")
ARMS = {
    "c_fixed": dict(C_FIXED, acceleration_inputs=False),
    "arm1": dict(C_FIXED, acceleration_inputs=True),
    "arm2": dict(
        C_FIXED,
        acceleration_inputs=True,
        next_state="delta",
        delta_baseline="persistence",
        spread_inputs="ellipse",
    ),
}


def outputs(settings: dict) -> dict[str, torch.Tensor]:
    """The arm's step loop and unroll over the golden's inputs, by output name: the network in
    float64 (the weights as initialised in float32), so that the rounding of a thread count or
    a library stays small (`yemong_cases.assert_matches_golden`)."""
    golden = torch.load(SPREAD_GOLDEN, weights_only=True)
    obs = cases.cast(cases.golden_obs(golden), F64)
    net = cases.double(cases.model(**settings))
    torch.manual_seed(11)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(net, net.step, carry, obs, golden["joint"])
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    return dict(
        step_policy_logits=decisions.policy_logits[:FIRST],
        step_next_state_logits=decisions.next_state_logits[:WIDE_TICKS],
        step_values=decisions.values[:FIRST],
        step_carry_state=final.state,
        step_belief_means=final.belief.means,
        step_belief_spreads=final.belief.spreads,
        step_replay_spreads=decisions.replay.spreads[:FIRST],
        unroll_policy_logits=unrolled.policy_logits[:FIRST],
        unroll_next_state_logits=unrolled.next_state_logits[:WIDE_TICKS],
        unroll_values=unrolled.values[:FIRST],
    )


def reproduces(arm: str) -> None:
    """Raises unless the arm's outputs match the recorded ones."""
    expected = torch.load(ARMS_GOLDEN, weights_only=True)[arm]
    cases.assert_matches_golden(outputs(ARMS[arm]), expected)


def record() -> None:
    recorded = {
        arm: {
            name: value.clone() for name, value in outputs(settings).items()
        }  # a slice keeps its storage
        for arm, settings in ARMS.items()
    }
    torch.save(recorded, ARMS_GOLDEN)
    import test_yemong_spread_fit as spread

    golden = torch.load(SPREAD_GOLDEN, weights_only=True)
    stepped, _ = spread._steps(cases.double(cases.model()), golden)
    golden["outputs"] = {name: value.clone() for name, value in stepped.items()}
    torch.save(golden, SPREAD_GOLDEN)


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    else:
        for name in ARMS:
            reproduces(name)
