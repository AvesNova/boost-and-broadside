# Training runs

A short log of the runs whose checkpoints and artifacts are kept. The current reference
run is [`{{run_name}}`](#wild-universe-731); its measurements are what
[evaluation](evaluation.md) reports and what every published figure comes from. Earlier
runs are kept as history; their checkpoints are still on disk and still load.

Ratings below are post-hoc calibrated Elo with the scripted controller at 1000, each
measured within its own run, and each is the final live checkpoint's. They are not
comparable across runs that trained under different physics — see the note at the end.

## wild-universe-731

**Current reference run.** {{run_steps_millions:.0f}}M of a planned
{{run_target_steps_millions:,.0f}}M environment steps, {{run_updates}} updates, about
{{run_days:.1f}} days on one {{run_gpu}}; the run was stopped there (status
`{{run_status}}`). W&B run `{{run_wandb_id}}`.

The first run under frontal armour, and the only one trained under this version's physics,
in which a glancing bullet does {{config_bullet_min_damage_fraction:.0%}} of full damage.
Trained 4-vs-4 with {{config_fields}} refractive fields. The final live policy calibrates
to **{{calibration_final_live_elo:,.0f}} Elo** and the best averaged checkpoint to
{{calibration_best_averaged_elo:,.0f}}. Against the scripted controller the final live
policy holds a numerical advantage from {{crossover_1_ships_ratio:.2f}}x at 1-vs-1 (by a {{crossover_1_ships_beats_up_to_win_rate:.1%}} margin) and
{{crossover_4_ships_ratio:.2f}}x at its training size down to
{{crossover_64_ships_ratio:.2f}}x at 64-vs-64 — sixteen times its training width. Learning
is front-loaded: the calibrated curve passes {{calibration_curve_elo_at_150m:,.0f}} by
150M steps and stays within 50 Elo of the final rating from about
{{calibration_curve_within_50_from_millions:.0f}}M onward.

Reward weights, in the derived form this version uses: win {{config_reward_win}}, death
{{config_reward_death}}, damage {{config_reward_damage}}, with kill and damage payout ratios
of {{config_reward_kill_payout_ratio}} and a kill-shot fraction of
{{config_reward_kill_shot_fraction}}; facing {{config_reward_facing}} and closing speed
{{config_reward_closing_speed}}. Shoot-quality shaping off. Peak learning rate
{{learning_rate_peak:.1e}}.

[Artifacts](../checkpoints/{{run_name}}/artifacts/) ·
[figures](../checkpoints/{{run_name}}/artifacts/figures/)

## good-leaf-719

The previous reference run. {{history_good_leaf_719_steps_millions:.0f}}M environment
steps, 1004 updates, about four days on one RTX 4070 Laptop, under the earlier full-damage
physics: every bullet did full damage.

Trained 4-vs-4 with four refractive fields. Final live policy calibrates to
{{history_good_leaf_719_final_live_elo:,.0f}} Elo *on its own scale*. Its own crossover
sweep has it beating {{history_good_leaf_719_crossover_4_ships_ratio:.2f}}x its number of
scripted ships at 4 ships and {{history_good_leaf_719_crossover_64_ships_ratio:.2f}}x at
64, but only an equal number at 1-vs-1
({{history_good_leaf_719_crossover_1_ships_ratio:.2f}}x).

Reward weights: win 1.5, kill/assist 1.0, deaths 1.0 (combat and field), damage dealt and
taken 0.5, facing and closing speed 0.1. Shoot-quality shaping off. Peak learning rate
4.5e-4.

[Artifacts](../checkpoints/good-leaf-719/artifacts/) ·
[figures](../checkpoints/good-leaf-719/artifacts/figures/)

## lunar-cosmos-716

500M steps, 502 updates, under the same profile and physics as 719. On the stationary
reference ladder the two agreed closely, so their ratings are comparable, and 719's extra
500M steps were worth about 114 Elo. It carried shoot-quality shaping at 0.1 and a lower
peak learning rate, and its final policy calibrated to 1634 Elo. Its checkpoints and
artifacts are not kept in this repository.

## resilient-resonance-682

{{history_resilient_resonance_682_steps_millions:.0f}}M steps. The reference run before
719. Trained without fields, under an earlier reward schema — win 4.0, a single
undifferentiated death and damage-taken term — and materially different combat physics.

Final live policy calibrates to {{history_resilient_resonance_682_final_live_elo:,.0f}}
Elo *on its own scale*; see below.

[Artifacts](../checkpoints/resilient-resonance-682/artifacts/)

## A note on comparing ratings

Elo is defined within a pool playing one game. 682 trained with glancing bullets doing a
tenth of full damage, twice the decision rate, and no refractive fields; 719 trained with
every bullet at full damage and half the decision rate; {{run_name}} trained with frontal
armour and glancing bullets at {{config_bullet_min_damage_fraction:.0%}} of full damage.
The rules change how dangerous an untrained policy is, so they stretch or compress the
whole ladder: the distance from random to the scripted controller is
{{history_resilient_resonance_682_random_to_scripted:,.0f}} Elo in 682's world,
{{history_good_leaf_719_random_to_scripted:,.0f}} in 719's and
{{calibration_random_to_scripted:,.0f}} in 731's.

Anchoring the scripted controller at 1000 pins one point without fixing that spacing. So
682's {{history_resilient_resonance_682_final_live_elo:,.0f}}, 719's
{{history_good_leaf_719_final_live_elo:,.0f}} and 731's
{{calibration_final_live_elo:,.0f}} are not three measurements of the same quantity, and
the differences between them say nothing about which policy is stronger. Only runs that
share physics — 716 and 719 — are directly comparable, and are compared above.
