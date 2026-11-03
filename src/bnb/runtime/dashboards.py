"""The dashboards (diagnostics spec 4, the audit's "Dashboard layout"), defined in code: the
12-panel headline and the curated view, as W&B workspaces built with `wandb-workspaces` (an
optional dependency, the `tools` extra). The third view is W&B's automatic workspace.

A panel is one quantity: its series are the names matching a pattern, `*` for any slice, so
each slice is a line and a new slice needs no edit here. The x-axis is
`counters/environment_steps`. `bnb runs dashboard RUN` builds or updates a run's views.
"""

import dataclasses
import re

STEP = "counters/environment_steps"


@dataclasses.dataclass(frozen=True)
class Panel:
    """One quantity: a title and the series patterns it plots (`*` matches any slice)."""

    title: str
    series: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Section:
    """A named group of panels, open or collapsed; `bc`: shown open in BC runs only."""

    name: str
    panels: tuple[Panel, ...]
    open: bool = False


def panel(title: str, *series: str) -> Panel:
    """A panel of `series` (the title's own name when none is given)."""
    return Panel(title, series or (title,))


HEADLINE = (
    Section(
        "Is it winning?",
        (
            panel("alerts/active"),
            panel("rating/live", "rating/live/*", "rating/average/*"),
            panel("pool/score", "pool/score/*"),
            panel("episode/end", "episode/end/*"),
        ),
        open=True,
    ),
    Section(
        "Is it learning?",
        (
            panel("policy/entropy/joint"),
            panel("policy/kl and clip fraction", "policy/kl", "policy/clip_fraction"),
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
            panel("belief/position_error", "belief/position_error/*"),
            panel("repr/effective_rank", "repr/effective_rank/*"),
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
            panel("rating/live", "rating/live/*"),
            panel("rating/live_stderr", "rating/live_stderr/*"),
            panel("rating/average", "rating/average/*"),
            panel("pool/score", "pool/score/*"),
            panel("rating/window_games", "rating/window_games/*"),
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
            panel("policy/kl", "policy/kl", "policy/kl_first"),
            panel("policy/clip_fraction"),
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
            panel("grad/share/trunk", "grad/share/trunk/*"),
            panel("grad/cos/trunk", "grad/cos/trunk/*"),
            panel("grad/coherent_fraction", "grad/coherent_fraction/*"),
        ),
        open=True,
    ),
    Section(
        "bc/",
        (
            panel("bc/kl", "bc/kl", "bc/stop/kl_average"),
            panel("bc/agreement", "bc/agreement/*"),
            panel("bc/entropy", "bc/entropy/*"),
            panel("bc/score", "bc/score/*", "bc/stop/window_score", "bc/stop/window_lower_bound"),
            panel("bc/kl_by_pool", "bc/kl_by_pool/*"),
        ),
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
            panel("returns/mixed", "returns/mixed/*"),
        ),
    ),
    Section(
        "Auxiliary heads",
        (
            panel("next_state/loss"),
            panel("enemy_action/loss"),
            panel("density/loss"),
            panel("shot/loss", "shot/loss/*"),
            panel("next_state/kl", "next_state/kl/*"),
            panel("next_state/gain", "next_state/gain/*"),
            panel("enemy_action/gain", "enemy_action/gain/*"),
            panel("density/deviance_explained", "density/deviance_explained/*"),
            panel("belief/position_error", "belief/position_error/*"),
            panel("belief/normalised_square_error", "belief/normalised_square_error*"),
            panel("supervised_share", "*/supervised_share*"),
            panel("outcome/brier_skill", "outcome/brier_skill/*"),
            panel("outcome/ece", "outcome/ece", "outcome/bias"),
        ),
    ),
    Section(
        "Representation",
        (
            panel("repr/effective_rank", "repr/effective_rank/*"),
            panel("repr/dormant_share", "repr/dormant_share/*"),
            panel("repr/activation_rms", "repr/activation_rms/*"),
        ),
    ),
    Section(
        "Inputs and batch",
        (
            panel("input/envelope_violation_share", "input/envelope_violation_share/*"),
            panel("input/nonfinite_count"),
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
            panel("reward/rate", "reward/rate/*"),
            panel("reward/level_share", "reward/level_share/*"),
            panel("reward/event_share", "reward/event_share/*"),
            panel("reward/zero_sum_residual", "reward/zero_sum_residual/*"),
        ),
    ),
    Section(
        "Fog",
        (panel("fog", "fog/*"),),
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
            panel("grad/norm", "grad/norm/*"),
            panel("grad/agreement", "grad/agreement/*"),
            panel("grad/actor_share"),
        ),
    ),
    Section(
        "League",
        (
            panel("pool/game_share", "pool/game_share/*"),
            panel("pool/sampling_entropy", "pool/sampling_entropy/*"),
            panel("pool", "pool/size", "pool/snapshots", "pool/draining"),
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


def workspace(sections: tuple[Section, ...], entity: str, project: str, name: str, bc: bool):
    """The W&B workspace of `sections` (needs the `tools` extra); `bc`: the `bc/` section
    opens."""
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
        is_open = section.open or (bc and section.name == "bc/")
        built.append(workspaces.Section(name=section.name, panels=plots, is_open=is_open))
    return workspaces.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=built,
        settings=workspaces.WorkspaceSettings(x_axis=STEP),
    )


def build(run_name: str, entity: str, project: str, bc: bool = False) -> dict:
    """The run's two views by name ("headline", "curated"), unsaved."""
    return {
        view: workspace(sections, entity, project, f"{run_name} {view}", bc)
        for view, sections in VIEWS.items()
    }


def panel_count(sections: tuple[Section, ...]) -> int:
    """The panels of a view."""
    return sum(len(section.panels) for section in sections)
