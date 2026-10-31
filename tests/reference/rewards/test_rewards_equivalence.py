"""v1's reward levels, damage ledger and Frontline rules, through the interface."""

import pytest
from rewards_helpers import TIGHT, case, check

# --------------------------------------------------------------------------- ledger


def test_accumulate_ledger(arm):
    golden = case("ledger_accumulate")
    inputs = golden.inputs
    out = arm.accumulate_ledger(inputs["ledger"], inputs["raw"], inputs["combat_applied"])
    check(out, golden.outputs["ledger"], "ledger", **TIGHT)


def test_attribute_recharge(arm):
    golden = case("ledger_recharge")
    inputs = golden.inputs
    attributed, unattributed, ledger = arm.attribute_recharge(
        inputs["ledger"], inputs["recharge"], inputs["deficit"])
    check(attributed, golden.outputs["attributed"], "attributed", **TIGHT)
    check(unattributed, golden.outputs["unattributed"], "unattributed", **TIGHT)
    check(ledger, golden.outputs["ledger"], "ledger", **TIGHT)


# --------------------------------------------------------------------------- levels


def test_enemy_team_share(arm):
    golden = case("levels_enemy_share")
    share = arm.enemy_team_share(golden.inputs["value"], golden.inputs["team"])
    check(share, golden.outputs["share"], "share", **TIGHT)


def test_outcome_events(arm):
    golden = case("levels_outcome")
    inputs = golden.inputs
    events = arm.outcome_events(inputs["team"], inputs["result"], inputs["done"])
    check(events, golden.outputs["events"], "events", **TIGHT)


def test_kill_death_events(arm):
    golden = case("levels_kill_death")
    inputs = golden.inputs
    events = arm.kill_death_events(inputs["team"], inputs["combat_death"],
                                   inputs["boundary_death"], inputs["raw"], inputs["ledger"])
    check(events, golden.outputs["events"], "events", **TIGHT)


def test_damage_events(arm):
    golden = case("levels_damage")
    inputs = golden.inputs
    events = arm.damage_events(
        inputs["team"], inputs["combat_applied"], inputs["boundary_applied"], inputs["raw"],
        inputs["recharge"], inputs["recharge_attributed"], inputs["unattributed"],
        golden.config["unit"])
    check(events, golden.outputs["events"], "events", **TIGHT)


# --------------------------------------------------------------------------- Frontline


def test_roles_from_front(arm):
    golden = case("frontline_roles")
    check(arm.roles_from_front(golden.inputs["front"]), golden.outputs["roles"], "roles")


@pytest.mark.parametrize("threshold", [3, 7])
def test_terminal_distances(arm, threshold):
    golden = case(f"frontline_terminal_{threshold}")
    front = golden.inputs["front"]
    offensive, defensive = arm.terminal_distances(front, front.new_full(front.shape, threshold))
    check(offensive, golden.outputs["offensive"], "offensive")
    check(defensive, golden.outputs["defensive"], "defensive")


def test_harmonic(arm):
    golden = case("frontline_harmonic")
    counts = golden.inputs["counts"]
    check(arm.harmonic(counts), golden.outputs["harmonic"], "harmonic", **TIGHT)


def test_zone_membership(arm):
    golden = case("frontline_membership")
    inputs = golden.inputs
    inside = arm.zone_membership(inputs["ship_position"], inputs["zone_position"],
                                 inputs["zone_radius"])
    check(inside, golden.outputs["inside"], "inside")


CAPTURE_INPUTS = ("team", "alive", "inside", "roles", "progress", "attack_ledger",
                  "defense_ledger", "front")


def test_capture_steps(arm):
    """Every tick of a chained capture rollout, each from its own stored inputs: meters
    rising and reversing, completions on both sides, the front and the roles rotating."""
    golden = case("frontline_capture")
    steps = golden.inputs["team"].shape[0]
    for t in range(steps):
        inputs = {name: golden.inputs[name][t] for name in CAPTURE_INPUTS}
        inside = inputs["inside"] & inputs["alive"].unsqueeze(-1)
        out = arm.capture_step(inputs["team"], inputs["alive"], inside, inputs["roles"],
                               inputs["progress"], inputs["attack_ledger"],
                               inputs["defense_ledger"], inputs["front"],
                               golden.config["tick_seconds"], golden.config["capture_seconds"])
        for name in ("progress", "direction", "attack_ledger", "defense_ledger", "gain", "loss",
                     "team0_captured", "team1_captured", "front", "roles"):
            check(out[name], golden.outputs[name][t], f"tick {t} {name}", **TIGHT)
        events = arm.zone_progress_events(inputs["team"], inside, out["roles"],
                                          inputs["progress"], out["progress"],
                                          out["team0_captured"], out["team1_captured"])
        check(events[..., 0], golden.outputs["held"][t], f"tick {t} held", **TIGHT)
        check(events[..., 1], golden.outputs["absent"][t], f"tick {t} absent", **TIGHT)


def test_timeout_result(arm):
    golden = case("frontline_timeout")
    check(arm.timeout_result(golden.inputs["front"]), golden.outputs["result"], "result")
