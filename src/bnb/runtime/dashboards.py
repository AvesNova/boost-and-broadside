"""The dashboards (diagnostics spec 4, the audit's "Dashboard layout"), defined in code: the
headline (the spec's 12 panels, with the ratings and the games' ends split by mode) and the
curated view, as W&B workspaces built with `wandb-workspaces` (an optional dependency, the
`tools` extra). The third view is W&B's automatic workspace.

A panel is one quantity: its series are the names matching a pattern, `*` for any slice, so
each slice is a line and a new slice needs no edit here. The crowded quantities (the ratings,
the games' ends, the rating lanes) have a panel per game mode of the training mix. The
x-axis is `counters/environment_steps`. A panel names the objectives it shows in: a PPO
run's views leave out BC's panels, a BC run's leave out reinforcement learning's (the
averaged agent, the league, PPO's KL and clip fraction), and a section left empty is left
out. A `--wandb` run
saves its views once its first record is logged; `bnb runs dashboard RUN` builds or updates
them later.
"""

import dataclasses
import posixpath
import re
from urllib.parse import urlsplit, urlunsplit

from bnb.learn.head_statistics import BELIEF_QUANTITIES

from .train import TRAINING_MIX

STEP = "counters/environment_steps"
OBJECTIVES = ("ppo", "bc")  # `bnb.runtime.train.OBJECTIVES`
MODES = tuple(mode for mode, _ in TRAINING_MIX)  # the panels split by game mode


@dataclasses.dataclass(frozen=True)
class Panel:
    """One quantity: a title, the series patterns it plots (`*` matches any slice) and the
    objectives whose views show it."""

    title: str
    series: tuple[str, ...]
    objectives: tuple[str, ...] = OBJECTIVES


@dataclasses.dataclass(frozen=True)
class Section:
    """A named group of panels, open or collapsed."""

    name: str
    panels: tuple[Panel, ...]
    open: bool = False


def panel(title: str, *series: str) -> Panel:
    """A panel of `series` (the title's own name when none is given), in every view."""
    return Panel(title, series or (title,))


def rl(title: str, *series: str) -> Panel:
    """A panel of reinforcement learning only: left out of BC's views."""
    return Panel(title, series or (title,), ("ppo",))


def bc(title: str, *series: str) -> Panel:
    """A panel of behaviour cloning only: left out of PPO's views."""
    return Panel(title, series or (title,), ("bc",))


def by_mode(*make) -> tuple[Panel, ...]:
    """Each `make(mode)`'s panels, mode by mode (MODES)."""
    return tuple(panel for mode in MODES for build in make for panel in build(mode))


def ratings(mode: str) -> tuple[Panel, ...]:
    """The mode's rating panel: the live policy's and the averaged agent's lane ratings, and
    in BC the uniform average's too."""
    return (
        rl(f"rating/{mode}", f"rating/live/{mode}", f"rating/average/{mode}"),
        bc(
            f"rating/{mode}",
            f"rating/live/{mode}",
            f"rating/average/{mode}",
            f"rating/uniform_average/{mode}",
        ),
    )


def episode_ends(mode: str) -> tuple[Panel, ...]:
    """The mode's share of training games decided before the time limit, a line per
    opponent kind (PPO's self-play and league seat; BC's self-play and teacher)."""
    decided = f"episode/end/decided/{mode}"
    return (
        rl(f"episode/end/{mode}", f"{decided}/self_play", f"{decided}/league"),
        bc(f"episode/end/{mode}", f"{decided}/self_play", f"{decided}/scripted"),
    )


def lanes(mode: str) -> tuple[Panel, ...]:
    """The mode's rating lanes: games, the first player's score and each player's rating per
    matchup, and the share of lane games decided before the time limit."""
    return (
        panel(f"lanes/games/{mode}", f"lanes/games/{mode}/*"),
        panel(f"lanes/score/{mode}", f"lanes/score/{mode}/*"),
        panel(f"lanes/rating/{mode}", f"lanes/rating/{mode}/*"),
        panel(f"lanes/episode/end/{mode}", f"episode/end/decided/{mode}/lane/*"),
    )


HEADLINE = (
    Section(
        "Is it winning?",
        (
            panel("alerts/active"),
            *by_mode(ratings),
            rl("pool/score", "pool/score/*"),
            bc("bc/score", "bc/score/*", "bc/stop/window_score"),
            bc("bc/agreement", "bc/agreement/*"),
            *by_mode(episode_ends),
        ),
        open=True,
    ),
    Section(
        "Is it learning?",
        (
            panel("policy/entropy/joint"),
            rl("policy/kl and clip fraction", "policy/kl", "policy/clip_fraction"),
            bc("bc/kl", "bc/kl", "bc/stop/kl_average"),
            panel("critic/explained_variance", "critic/explained_variance/*"),
            panel(
                "gradient norm and clip fire rate", "train/gradient_norm", "train/clip_fire_rate"
            ),
        ),
        open=True,
    ),
    Section(
        "Is it healthy?",
        (
            panel("train/environment_steps_per_second"),
            panel("next_state/kl", "next_state/kl/*"),
            panel("belief/position_metres", "belief/position_metres/*"),
            panel("representation/effective_rank", "representation/effective_rank/*"),
        ),
        open=True,
    ),
)

CURATED = (
    Section(
        "Alerts",
        (panel("alerts/active"), panel("alerts/flags", "alerts/*")),
        open=True,
    ),
    Section(
        "Strength",
        (
            *by_mode(ratings),
            panel("rating/live_stderr", "rating/live_stderr/*"),
            panel("rating/average_stderr", "rating/average_stderr/*"),
            bc("rating/uniform_average_stderr", "rating/uniform_average_stderr/*"),
            rl("pool/score", "pool/score/*"),
            rl("rating/window_games", "rating/window_games/*"),
        ),
        open=True,
    ),
    Section("Rating lanes", by_mode(lanes), open=True),
    Section(
        "bc/",
        (
            bc("bc/kl", "bc/kl", "bc/stop/kl_average"),
            bc("bc/agreement", "bc/agreement/*"),
            bc("bc/entropy", "bc/entropy/*"),
            bc("bc/score", "bc/score/*", "bc/stop/window_score", "bc/stop/window_stderr"),
            bc("bc/stop/window_score", "bc/stop/window_score/*", "bc/stop/window_stderr/*"),
            bc("bc/stop/reads", "bc/stop/reads/*"),
            bc("bc/games", "bc/games/*"),
            bc("bc/kl_by_pool", "bc/kl_by_pool/*"),
            bc("bc/entropy_by_pool", "bc/entropy_by_pool/*"),
        ),
        open=True,
    ),
    Section(
        "Health",
        (
            panel("train/environment_steps_per_second"),
            panel("train/time", "train/time/*"),
            panel("train/skipped_steps", "train/skipped_steps", "counters/skipped_steps"),
            panel("counters/recompiles"),
            panel("memory", "train/memory_peak_gigabytes", "train/memory_budget_gigabytes"),
            panel("train/ship_tokens_per_second"),
        ),
        open=True,
    ),
    Section(
        "Policy",
        (
            panel("policy/entropy/joint"),
            rl("policy/kl", "policy/kl", "policy/kl_first"),
            rl("policy/clip_fraction"),
            panel("policy/ratio_drift", "policy/ratio_drift/*"),
            panel(
                "policy/entropy",
                "policy/entropy/power",
                "policy/entropy/stick",
                "policy/entropy/shoot",
            ),
            panel("policy/epochs_completed"),
        ),
        open=True,
    ),
    Section(
        "Value",
        (
            panel("critic/explained_variance", "critic/explained_variance/*"),
            panel("critic/value_bias", "critic/value_bias/*"),
            panel("critic/support_clip", "critic/support_clip/*"),
            panel("critic/value_kl", "critic/value_kl/*"),
        ),
        open=True,
    ),
    Section(
        "Learning signal",
        (
            panel("gradient/share/trunk", "gradient/share/trunk/*"),
            panel("gradient/cosine/trunk/policy", "gradient/cosine/trunk/policy/*"),
            panel("gradient/cosine/trunk/value", "gradient/cosine/trunk/*/value"),
            panel("gradient/coherent_fraction", "gradient/coherent_fraction/*"),
            panel("gradient/actor_share"),
        ),
        open=True,
    ),
    Section(
        "Policy detail",
        (
            panel("policy/entropy_quantile", "policy/entropy_quantile/*"),
            panel("policy/near_certain_share"),
            panel("policy/factor_coupling"),
            panel("policy/far_ratio_share"),
            panel("policy/usage/power", "policy/usage/power/*"),
            panel("policy/usage/stick", "policy/usage/stick/*"),
            panel("policy/usage/shoot", "policy/usage/shoot/*"),
            panel("policy/switch_rate", "policy/switch_rate/*"),
        ),
    ),
    Section(
        "Value detail",
        (
            panel("critic/value_loss", "critic/value_loss/*"),
            panel("critic/return_mean", "critic/return_mean/*"),
            panel("returns/p5", "returns/p5/*"),
            panel("returns/p50", "returns/p50/*"),
            panel("returns/p95", "returns/p95/*"),
            panel("returns/scale", "returns/scale", "returns/scale_p5", "returns/scale_p95"),
            panel("returns/scale_floored"),
            panel("returns/advantage_rms"),
            panel("returns/advantage_mean"),
        ),
    ),
    Section(
        "Auxiliary heads",
        (
            panel("next_state/loss"),
            panel("enemy_action/loss"),
            panel("density/loss"),
            panel("shot/loss", "shot/loss/*"),
            panel("rotary_readout/loss"),
            panel("rotary_readout/error", "rotary_readout/error/*"),
            panel("rotary_readout/hidden_error", "rotary_readout/hidden_error/*"),
            panel("rotary_readout/hidden_length", "rotary_readout/hidden_length/*"),
            panel("next_state/kl", "next_state/kl/*"),
            panel("next_state/gain/ally", "next_state/gain/ally/*"),
            panel("next_state/gain/enemy_visible", "next_state/gain/enemy_visible/*"),
            panel("next_state/token_share", "next_state/token_share/*"),
            panel("next_state/cross_entropy", "next_state/cross_entropy/*"),
            panel("next_state/error/position", "next_state/error/position_metres/*"),
            panel("next_state/error/velocity", "next_state/error/velocity_metres_per_second/*"),
            panel("next_state/error/heading", "next_state/error/heading_radians/*"),
            panel("next_state/support_clip", "next_state/support_clip/*"),
            panel("next_state/support_padding", "next_state/support_padding/*"),
            panel("next_state/beats_bench_median", "next_state/beats_bench_median/*"),
            panel("next_state/slow_frame_share"),
            panel("next_state/hidden/cross_entropy", "next_state/hidden/cross_entropy/*"),
            panel("next_state/hidden/error", "next_state/hidden/error/*"),
            panel("correction/loss"),
            panel("correction/realised_error", "correction/realised_error/*"),
            panel("correction/predicted_error", "correction/predicted_error/*"),
            panel("correction/realised_square_error", "correction/realised_square_error/*"),
            panel("correction/predicted_variance", "correction/predicted_variance/*"),
            panel("correction/within_one_sigma", "correction/within_one_sigma/*"),
            panel("correction/within_two_sigma", "correction/within_two_sigma/*"),
            panel("correction/step_over_limit", "correction/step_over_limit/*"),
            panel("correction/capped_share", "correction/capped_share/*"),
            panel("correction/support_clip", "correction/support_clip/*"),
            panel("enemy_action/gain", "enemy_action/gain/*"),
            panel("enemy_action/cross_entropy", "enemy_action/cross_entropy/*"),
            panel("enemy_action/accuracy", "enemy_action/accuracy/*"),
            panel("enemy_action/entropy", "enemy_action/entropy/*"),
            panel("enemy_action/factor_cross_entropy", "enemy_action/factor_cross_entropy/*"),
            panel("density/deviance", "density/deviance/*"),
            panel("density/explained_deviance", "density/explained_deviance/*"),
            panel("shot/trajectory_kl", "shot/trajectory_kl/*"),
            panel("shot/enemy_hit_rate", "shot/enemy_hit_rate/*"),
            panel("shot/ally_hit_rate", "shot/ally_hit_rate/*"),
            panel("shot/clear_miss_rate", "shot/clear_miss_rate/*"),
            panel("supervised_share", "*/supervised_share*"),
            panel("outcome/brier_skill", "outcome/brier_skill/*"),
            panel("outcome/calibration", "outcome/expected_calibration_error", "outcome/bias"),
            panel("critic/explained_variance_fit", "critic/explained_variance_fit/*"),
        ),
    ),
    Section(
        "Belief",
        tuple(
            panel(f"belief/{quantity}", f"belief/{quantity}/*") for quantity, _ in BELIEF_QUANTITIES
        )
        + (
            panel("belief/normalised_square_error", "belief/normalised_square_error/*"),
            panel("belief/within_one_sigma", "belief/within_one_sigma/*"),
            panel("belief/within_two_sigma", "belief/within_two_sigma/*"),
            panel("belief/anisotropy", "belief/anisotropy/*"),
        ),
    ),
    Section(
        "Representation",
        (
            panel("representation/effective_rank", "representation/effective_rank/*"),
            panel("representation/dormant_share", "representation/dormant_share/*"),
            panel("representation/activation_rms", "representation/activation_rms/*"),
        ),
    ),
    Section(
        "Inputs and batch",
        (
            panel("input/envelope_violation_share", "input/envelope_violation_share/*"),
            panel("input/nonfinite_count"),
            panel("input/acceleration/mean_abs", "input/acceleration/mean_abs/*"),
            panel(
                "input/acceleration/validity",
                "input/acceleration/invalid_share",
                "input/acceleration/slow_frame",
            ),
            panel("input/acceleration/padding_share", "input/acceleration/padding_share/*"),
            panel("batch/token_share/mode", "batch/token_share/mode/*"),
            panel("batch/token_share/opponent", "batch/token_share/opponent/*"),
        ),
    ),
    Section(
        "Episode and reward",
        (
            panel("episode/count", "episode/count/*"),
            panel("episode/length_seconds", "episode/length_seconds/*"),
            panel("episode/end/timeout", "episode/end/timeout/*"),
            panel("episode/end/decided", "episode/end/decided/*"),
            panel("episode/wins", "episode/wins/*"),
            panel("episode/draws", "episode/draws/*"),
            panel("episode/losses", "episode/losses/*"),
            panel("episode/alive_share"),
            panel("episode/result", "episode/result/*"),
            panel("episode/lifespan_seconds", "episode/lifespan_seconds/*"),
            panel("episode/reward_mean", "episode/reward_mean/*"),
            panel("episode/reward_min", "episode/reward_min/*"),
            panel("episode/reward_max", "episode/reward_max/*"),
            panel("episode/reward", "episode/reward/*"),
            panel("episode/event", "episode/event/*"),
            panel("reward/rate", "reward/rate/*"),
            panel("reward/level_share", "reward/level_share/*"),
            panel("reward/event_share", "reward/event_share/*"),
            panel("reward/zero_sum_residual", "reward/zero_sum_residual/*"),
        ),
    ),
    Section(
        "Fog",
        (
            panel("fog/visible_share", "fog/visible_share/*"),
            panel("fog/mean_hidden_age_seconds"),
            panel("fog/reacquisitions_per_slot_minute"),
            panel("fog/occlusion_share", "fog/occlusion_share/*"),
            panel("fog/never_seen_share"),
            panel("fog/fogged_environment_share"),
        ),
    ),
    Section(
        "Physics, frontline and map scale",
        (
            panel("physics", "physics/*"),
            panel("frontline", "frontline/*"),
            panel("map_scale", "map_scale/*"),
        ),
    ),
    Section(
        "Gradient detail",
        (
            panel("train/parameter_norm", "train/parameter_norm/*"),
            panel("train/update_ratio", "train/update_ratio/*"),
            panel("train/nonfinite_steps", "train/nonfinite_steps/*"),
            panel("gradient/norm/model", "gradient/norm/model/*"),
            panel("gradient/norm/trunk", "gradient/norm/trunk/*"),
            panel("gradient/share/model", "gradient/share/model/*"),
            panel("gradient/agreement", "gradient/agreement/*"),
            panel("gradient/coherent_share", "gradient/coherent_share/*"),
            panel("gradient/cosine/trunk", "gradient/cosine/trunk/*"),
            panel(
                "gradient/actor_norm and critic_norm", "gradient/actor_norm", "gradient/critic_norm"
            ),
            panel("train/time/cadence_seconds"),
        ),
    ),
    Section(
        "League",
        (
            rl("pool/game_share", "pool/game_share/*"),
            rl("pool/sampling_entropy", "pool/sampling_entropy/*"),
            rl("pool", "pool/size", "pool/snapshots", "pool/cut/cuts", "pool/cut/uncredited_games"),
        ),
    ),
    Section("Schedules", (panel("schedule", "schedule/*"),)),
    Section(
        "Counters",
        (
            panel("counters/updates"),
            panel("counters/hours"),
            panel("counters/skipped_steps"),
            panel("counters/recompiles"),
            panel("counters/nonfinite_beliefs"),
        ),
    ),
)
VIEWS = {"headline": HEADLINE, "curated": CURATED}


def regex(series: tuple[str, ...]) -> str:
    """One regular expression for a panel's patterns."""
    return "|".join(
        "^" + ".*".join(re.escape(part) for part in pattern.split("*")) + "$" for pattern in series
    )


def sections_for(sections: tuple[Section, ...], objective: str) -> tuple[Section, ...]:
    """The sections of a view as `objective`'s run shows them: its panels only, and no
    section left empty."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r} is not one of {OBJECTIVES}")
    out = []
    for section in sections:
        panels = tuple(item for item in section.panels if objective in item.objectives)
        if panels:
            out.append(dataclasses.replace(section, panels=panels))
    return tuple(out)


def workspace(sections: tuple[Section, ...], entity: str, project: str, name: str):
    """The W&B workspace of `sections` (needs the `tools` extra)."""
    import wandb_workspaces.reports.v2 as reports
    import wandb_workspaces.workspaces as workspaces

    built = []
    for section in sections:
        plots = []
        for item in section.panels:
            literal = all("*" not in pattern for pattern in item.series)
            if literal:
                plots.append(reports.LinePlot(title=item.title, x=STEP, y=list(item.series)))
            else:
                plots.append(
                    reports.LinePlot(title=item.title, x=STEP, metric_regex=regex(item.series))
                )
        built.append(workspaces.Section(name=section.name, panels=plots, is_open=section.open))
    return workspaces.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=built,
        settings=workspaces.WorkspaceSettings(x_axis=STEP),
    )


def build(run_name: str, entity: str, project: str, objective: str = "ppo") -> dict:
    """The run's two views by name ("headline", "curated"), for its objective, unsaved."""
    return {
        view: workspace(sections_for(sections, objective), entity, project, f"{run_name} {view}")
        for view, sections in VIEWS.items()
    }


def app_link(url: str, app_url: str) -> str:
    """A view's link in the W&B app: `url` (`wandb-workspaces`' view URL, which keeps only
    the host of the app's address) under the app's own address `app_url`, path prefix
    included (`https://forge.coreweave.com/wandb` gives
    `https://forge.coreweave.com/wandb/<entity>/<project>?nw=<id>`)."""
    view, app = urlsplit(url), urlsplit(app_url)
    path = posixpath.join(app.path.rstrip("/") or "/", view.path.lstrip("/"))
    return urlunsplit((app.scheme, app.netloc, path, view.query, view.fragment))


def app_url() -> str:
    """The W&B app's address from W&B's configured settings (`WANDB_BASE_URL`, its settings
    files, `WANDB_APP_URL`)."""
    import wandb

    return wandb.setup().settings.app_url


def save(run_name: str, entity: str, project: str, objective: str = "ppo") -> dict[str, str]:
    """Build and save the run's two views on the W&B server; their links by view name.
    `wandb-workspaces` logs each saved view's link without the app's path prefix, so that
    line is held back and the working link returned instead."""
    from unittest import mock

    import wandb

    views = build(run_name, entity, project, objective)
    base = app_url()
    links = {}
    for name, view in views.items():
        with mock.patch.object(wandb, "termlog", lambda *arguments, **options: None):
            view.save()
        links[name] = app_link(view.url, base)
    return links


def panel_count(sections: tuple[Section, ...]) -> int:
    """The panels of a view."""
    return sum(len(section.panels) for section in sections)
