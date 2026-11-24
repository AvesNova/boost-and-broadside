# Reach of the scripted agent's mechanisms (2026-11-25)

The author's question (author-review-2.md S4 to S6): the screening found many rows flat in
skill and in behaviour; are their mechanisms ever reached, and do they change the decision
when they are? This page measures, for each mechanism of the BFM agent (lean/scripted, core at
c3c30eb2, after the lead turn and the yo-yos were removed), how often its gate is open or its
term is non-negligible (reach), and how much the decision changes on those states when the
mechanism is switched off (effect). No games are played with a mechanism removed. Scripts and
results: docs/rewrite/sections/scripted-reach/ (`probe.py`, `reach.py`, `reach.json`,
`reach_crossing2.json`, `reach_crossing3.json`). States:
data/scripted-screening/reach-2026-11-25/.

## Table

Reach: share of the 105,250 ship-states where the mechanism is reached (rule below). JS|r:
the mean Jensen-Shannon divergence (nats; maximum ln 2 = 0.693) between the joint action
distributions with the mechanism on and off, on the reached states. Change|r: the share of
reached states whose most likely joint action changes. JS all: the mean over all states.
n: reached states.

| mechanism | reach % | JS\|r | change\|r | JS all | n | class | recommendation |
|---|---|---|---|---|---|---|---|
| overshoot rule, fires (gate 1 rad) | 0.006 | 0.066 | 0.17 | 0.000003 | 6 | never reached | delete |
| overshoot rule, held reversed sense | 0.062 | 0.431 | 0.74 | 0.00027 | 65 | never reached | (same rule) |
| overshoot, gate 2x (2 rad), fires | 0.104 | 0.046 | 0.08 | 0.00005 | 109 | reached, small effect | delete |
| overshoot, gate 3x (3 rad), fires | 0.278 | 0.053 | 0.10 | 0.00015 | 293 | reached, small effect | delete |
| spend term (all three gates) | 1.20 | 0.014 | 0.10 | 0.00017 | 1,262 | reached, small effect | delete as a unit |
| spend_need gate | 5.05 | 0.021 | 0.08 | 0.00104 | 5,310 | reached, small effect | (with the unit) |
| spend_range gate | 2.96 | 0.011 | 0.07 | 0.00034 | 3,120 | reached, no effect | (with the unit) |
| spend_share gate | 27.5 | 0.073 | 0.19 | 0.0202 | 28,964 | reached, effect | (with the unit) |
| weakest_weight (personas only) | 0.20 | 0.458 | 0.83 | 0.00090 | 207 | reached, effect | persona knob |
| switch_gain, at the commit | 0.051 | 0.583 | 1.00 | 0.00030 | 54 | (see held sense) | keep |
| switch_gain, held switched sense | 6.35 | 0.524 | 0.90 | 0.0333 | 6,686 | reached, effect | keep |
| scissors_range | 4.34 | 0.187 | 0.44 | 0.0081 | 4,571 | reached, effect | keep |
| cash_angle | 0.82 | 0.017 | 0.16 | 0.00017 | 860 | reached, small effect | delete |
| relax_angle | 1.44 | 0.010 | 0.04 | 0.00020 | 1,513 | reached, no effect | delete |
| defend_range | 1.17 | 0.033 | 0.03 | 0.00041 | 1,233 | reached, small effect | delete |
| boundary steer (boundary_margin) | 0.55 | 0.338 | 0.58 | 0.0019 | 582 | reached, effect | keep steer; freeze row |
| speed_floor (the clamp) | 0.000 | - | - | 0 | 0 | never reached | delete the clamp |
| turning_floor (tanh vs sign) | 39.5 | 0.008 | 0.02 | 0.0032 | 41,521 | reached, no effect | freeze |

Classes: never reached, reach under 0.1%; reached with an effect, JS|r at least 0.02 nats or
change|r at least 0.10; "small effect" when only one of the two passes, or both barely, and JS
all is under 0.002 nats (the divergence page's "a little" line for a whole row); else no
effect. These cut-offs are a choice, stated here.

## Notes

1. **Overshoot rule.** Of 28,372 states where a sense is held with the target aft, 792 pass
   the side, rate and range conditions, 14 the crossing gate (under 1 rad) and 6 the speed
   condition. Over every tick of the 900 games, 30 reversals in 6.38 M ship-ticks, and 0.063%
   of ship-ticks flown under a reversed sense. When the reversed sense is held it matters (JS
   0.43, the stick head), but the footprint is too small to show anywhere. **Widening does not
   change the picture.** Games flown with the gate at 2x (300 games): 30 reversals in 2.11 M
   ship-ticks, 0.083% of ticks under a reversed sense; at 3x (3 rad, nearly no gate): 40 in
   2.14 M, 0.122%. The crossing gate is the tightest condition, but with it removed the speed
   condition (I am no faster than him) halves the rest and the rule still fires on 0.3% of
   states. 2x was the variant S5 asked for; 3x is the gate fully open. Delete the rule and its
   three rows.
2. **Spending.** The whole spend term (corner speed when a shot is near) is reached on 1.2% of
   states and moves the power head a little there (JS 0.014). Its three gates mostly shut each
   other: the store gate (spend_share) alone holds it shut on 27.5% of states, which is why
   opening that one gate shows an effect while removing the term does not. Delete the term with
   its three gates and rows as a unit, not one gate alone.
3. **Plan switch on gain.** It acts through memory: the switched plan sets the committed turn
   sense, which is then held. 3,758 of 21,723 commits (17%) were switched; 6.2% of all
   ship-ticks fly a sense a switch chose (games), and on those states the other sense changes
   the most likely action 90% of the time. The commit tick alone (0.05%) understates it. Keep.
4. **Defence.** In the yardstick defend_range equals engage_range (3 gun ranges), so the
   defend gate repeats the engaged gate; relax_angle (pi) is reached on 1.4% of states and
   changes almost nothing. Removing both makes the defensive tactic always break. Delete both.
5. **speed_floor.** The clamp never binds on these states: the lowest target, the
   scissors' one, is the floor itself. The row stays as the scissors' target; delete the clamp only.
6. **turning_floor** sets how soft the turn sense is (tanh); a hard sign instead changes the
   decision by 0.008 nats on the 39% of states where they differ. Freeze it (S4).
7. **weakest_weight** is 0 in the yardstick (never reached there); for personas it changes the
   target on 0.38% of their states, and then nearly everything. A persona knob.
8. **Boundary steer.** Reached on 0.55% of states, with a large effect there (the stick).
   Keep the steer; its margin row was not switched separately and stays frozen (S4).

## Method

**States** (`reach.py DATA collect --games 900`, GPU, 53 s). 900 games of the core game at
`GameConfig()` defaults (arena 4,200 m, mixed spawn, fields, standard spawn), 4v4. Each team
is flown by the yardstick (probability 1/2) or one of four personas drawn at spread 1
(seeded). A snapshot every 40 to 80 ticks (the first at 20 to 80); every living ship of a
snapshot is a ship-state, with the memory it decided from: 105,250 ship-states in 19,581
snapshots, 51,059 flown by the yardstick. Over every tick the run also counted, from memory
alone, the ticks flown under a sense an overshoot reversed (a held sense flipped, same target)
or a switched plan committed (a commit whose last pass flips sign, same target). The widened
overshoot counts come from two further 300-game runs on the CPU with the crossing row of
every agent times 2 and 3.

**On and off** (`probe.py`). An analysis copy of `lean.scripted.opponent.explain` with
switches; with nothing off it reproduces lean's marginals exactly (largest difference 0, `reach.py
check`). Off means: the overshoot never fires; the spend product 0 (or one gate held open);
the switch never switches; the scissors and cash gates shut; the relax and defend gates open;
no boundary steer; no speed clamp; weakest_weight 0; turning_floor 0.001 (a hard sign). The
memory mechanisms are also measured by their held sense: on states whose held sense came from
the mechanism, off is the same state with that sense the other way. Each state is decided by
the agent that flew it. The distribution is the joint over 42 actions (stick x power x fire).

**Reach rules.** A gate's term counts as reached when its weight in the blend is above 0.05:
spend, (offensive + turning) x near; one spend gate, the same with that gate's (1 - gate) in
place of it; scissors, turning x one-circle x gate; cash, turning x two-circle x gate; relax,
defensive x defend x (1 - relax); defend, defensive x relax x (1 - defend); boundary, the
centre weight. The overshoot and the switch count when they fire; weakest_weight when the
target differs; speed_floor when the clamp lifts the target by over 1%; turning_floor when
either sense differs from the sign by over 0.05.

**Limits.** One decision per stored state, with the stored memory: later ticks are not
re-flown. The states are those of the current agent, so a mechanism that would lead the fight
elsewhere if removed is not seen. Small counts: the overshoot (6 and 65 states) and the commit
tick of the switch (54).

## Trim (2026-11-25)

Agent's call after the reach check (author to confirm). Branch scripted-trim, from core at
7e115a79.

**Deleted** (code, rows, gates, tests): the overshoot rule (overshoot_range, overshoot_rate,
overshoot_crossing; `Explanation.overshoot`); the spend term with its gates (spend_need,
spend_range, spend_share); the two-circle cash (cash_angle: two-circle now holds the middle
level at the fight speed); the break relaxation and the defend range (relax_angle,
defend_range: a defensive ship always breaks, full stick at corner speed); the speed-floor
clamp (speed_floor stays as the scissors' target). Nine rows and six gates fewer.
**Frozen**: turning_floor and boundary_margin are fixed rows (law "fixed"), so personas no
longer sample them. Kept: switch_gain, scissors_range, the boundary steer, weakest_weight.

**Re-check** (`sections/scripted-reach/trim_recheck.py`, `trim_recheck.json`; GPU unit
bnb-trim-recheck, 18 min). Panel skill: the recheck protocol (panel of 8, 18 conditions, 64
games a cell, label "recheck"), fresh store data/scripted-screening/trim-2026-11-25.sqlite.
The untrimmed figure is recheck.json's BFM-CMA2 (the same protocol on core before the trim;
lean/ unchanged between). Each side's panel runs on its own code, so the comparison is not
paired; the difference's error adds the two in quadrature. Head to head: the trimmed default
against a frozen copy of the untrimmed code (`untrimmed/`) in the same games, 64 a condition,
seats alternating, each team with its own code and memory.

| measure | untrimmed (BFM-CMA2) | trimmed | difference |
|---|---|---|---|
| panel skill (9,216 games each) | 0.6319 +- 0.0050 | 0.6311 +- 0.0047 | -0.0008 +- 0.0068 |
| by radius 3,000 / 4,200 / 6,000 m | 0.621 / 0.631 / 0.644 | 0.629 / 0.634 / 0.630 | |
| by size 3v3 / 4v4 / 5v5 | 0.614 / 0.636 / 0.645 | 0.610 / 0.640 / 0.643 | |
| head to head, trimmed score (1,152 games) | | 0.495 +- 0.015 | |

The trimmed skill is 0.1 standard errors below the untrimmed one, within the 2-error bar, and
the head to head is even: **the trim is accepted**; nothing was restored.

**Operations** (scoreboard/game_ops.py, uncompiled, one thread): the scripted call for both
teams 387 -> 343 compute operations (one team 394 -> 350); view operations 108 -> 104.
