"""Draw real trajectories against what the policy's heads predict, on the field map.

Two figures from one real rollout of a checkpoint:

* **Shots** -- for counterfactual shots that hit an enemy, the true flight (the
  environment's own launch and transport at zero spread, plus a fan of spread
  samples) against the shot head's decoded displacement at every age 0..29,
  with its one-sigma radius, and the hit ship's path during the flight.
* **Ships** -- one ship's real path under the actions it actually took, with the
  next-state head's one-step prediction at every decision (teacher-forced: each
  prediction reads the true observation of its own decision), and the
  prediction error over time with the spans the ship was hidden from the
  observer.

The rollout is collected exactly as training collects it -- a small CPU trainer
built from the profile, the checkpoint loaded as a warm start -- so the heads
see the observations they were trained on. Nothing is trained or written to the
run.

    uv run --no-sync python scripts/visualize_predictions.py \\
        --checkpoint checkpoints/charmed-lion-762/step_000019660800.pt \\
        --out docs/results/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

# Reference categorical slots (dataviz palette): real = blue, predicted = orange,
# the hit ship = violet; fields as light washes of a diverging pair.
REAL = "#2a78d6"
PREDICTED = "#eb6834"
TARGET = "#4a3aa7"
THICK_AIR = "#4a3aa7"
THIN_AIR = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _build_trainer(checkpoint: str, profile: str, num_envs: int, seed: int):
    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.config.resolve import LaunchOverrides
    from boost_and_broadside.profiles import resolve_named_profile
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    torch.manual_seed(seed)
    # A small evaluator: rated games are not what this script is for.
    resolved = resolve_named_profile(
        profile,
        LaunchOverrides(num_envs=num_envs),
        overrides={"elo_eval.envs_per_matchup": "4"},
    )
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=None,
    )
    trainer.load_pretrained_weights(checkpoint)
    trainer._policy_module.eval()
    return trainer


def _collect(trainer, rollouts: int) -> None:
    """Play ``rollouts`` rollouts without training; the buffer keeps the last."""
    import time

    runtime = trainer._initialize_rollout_runtime()
    trainer._train_start_time = time.time()
    with torch.no_grad():
        for index in range(rollouts):
            terminated = trainer._collect_rollout(runtime, False)
            print(f"rollout {index + 1}/{rollouts}")
        trainer._compute_rollout_gae(runtime, terminated)


def _evaluate(trainer, envs: list[int], ages: torch.Tensor):
    """Heads' outputs on the buffer's rollout for ``envs``, every age queried."""
    from boost_and_broadside.env.observation import ObsKey, YemongObservation

    buffer = trainer.buffer
    T = buffer.num_steps
    index = torch.tensor(envs)
    obs = YemongObservation(
        data={k: v[:, index] for k, v in buffer.obs.items()},
        bullets=(
            None
            if buffer.bullet_obs is None
            else {k: v[:, index] for k, v in buffer.bullet_obs.items()}
        ),
    ).slice_time(0, T)
    n_layers, tokens, width = buffer.initial_hidden.shape
    recurrent = tokens // buffer.num_envs
    hidden = (
        buffer.initial_hidden.reshape(n_layers, buffer.num_envs, recurrent, width)
        .index_select(1, index)
        .reshape(n_layers, len(envs) * recurrent, width)
    )
    actions = buffer.actions.index_select(1, index).long()
    N = buffer.num_ships
    shot_ages = ages.view(1, 1, 1, -1).expand(T, len(envs), N, -1).float()
    with torch.no_grad():
        outputs = trainer._policy_module.evaluate_actions(
            obs=obs,
            actions=actions,
            initial_hidden=hidden,
            alive_mask=obs[ObsKey.BELIEF_VALID].bool(),
            done_mask=buffer.terminated.index_select(1, index),
            return_enemy_action=True,
            return_density=True,
            shot_ages=shot_ages,
        )
    pred_next, trajectory, outcome = outputs[5], outputs[8], outputs[9]
    visible = obs[ObsKey.VISIBLE][:, :, :N].bool()
    return pred_next, trajectory, outcome, visible


def _decode_displacement(codec, logits: torch.Tensor):
    """Launch-frame mean displacement and sigma from ``(..., TRAJECTORY_DIM)`` logits."""
    from boost_and_broadside.train.rl.categorical_codes import POSITION_COLOURS
    from boost_and_broadside.train.rl.shot_codes import DISPLACEMENT_LEVELS

    width = DISPLACEMENT_LEVELS * POSITION_COLOURS**2
    groups = logits[..., :width].unflatten(-1, (DISPLACEMENT_LEVELS, POSITION_COLOURS**2))
    mean, sigma = codec.displacement.decode(torch.softmax(groups.float(), dim=-1))
    # The code is a torus decoded into [0, period); a displacement is signed.
    period = torch.tensor(codec.displacement.world_size, dtype=mean.dtype)
    return (mean + period / 2) % period - period / 2, sigma


def _fly(trainer, t: int, b: int, i: int, noise: torch.Tensor) -> torch.Tensor:
    """True flights of shot (t, b, i) for each spread draw in ``noise``: (S, D+1) world points."""
    from boost_and_broadside.env import physics, shot_labels

    launch = trainer.buffer.shot_launch
    fields = trainer.buffer.shot_fields
    config = trainer.ship_config
    S = noise.shape[0]

    def rep(x: torch.Tensor) -> torch.Tensor:
        return x[t, b, i].expand(S).reshape(S, 1)

    layout = shot_labels.FieldLayout(*(f[t, b].expand(S, -1) for f in fields))
    position = rep(launch.position)
    velocity = physics.bullet_launch_velocity(
        rep(launch.velocity),
        rep(launch.attitude),
        rep(launch.local_index),
        noise.view(S, 1),
        config,
    )
    index, gradient = rep(launch.local_index), rep(launch.field_gradient)
    points = [position]
    for _ in range(shot_labels.shot_label_delay(config)):
        flight = physics.fly_bullets(position, velocity, index, gradient, layout, config)
        position, velocity = flight.position, flight.velocity
        index, gradient = flight.local_index, flight.field_gradient
        points.append(position)
    return torch.cat(points, dim=1)


def _unwrap(points: torch.Tensor, origin: complex, world: tuple[float, float]) -> torch.Tensor:
    """Complex points re-centred on ``origin`` on the torus, so paths never jump."""
    w, h = world
    d = points - origin
    return origin + torch.complex((d.real + w / 2) % w - w / 2, (d.imag + h / 2) % h - h / 2)


def _draw_fields(ax, trainer, t: int, b: int, origin: complex) -> None:
    fields = trainer.buffer.shot_fields
    world = trainer.ship_config.world_size
    for m in range(fields.field_pos.shape[-1]):
        centre = _unwrap(fields.field_pos[t, b, m].view(1), origin, world)[0]
        radius = float(fields.field_radius[t, b, m])
        band = float(fields.field_transition_width[t, b, m])
        n = float(fields.field_index[t, b, m])
        if abs(n - 1.0) < 1e-6 or radius <= 0:
            continue
        colour = THICK_AIR if n > 1.0 else THIN_AIR
        ax.add_patch(
            Circle((centre.real, centre.imag), radius + band, color=colour, alpha=0.08, lw=0)
        )
        ax.add_patch(Circle((centre.real, centre.imag), radius, color=colour, alpha=0.18, lw=0))


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def _pick_shots(trainer, envs: list[int], count: int) -> list[tuple[int, int, int]]:
    """Valid observer shots that hit an enemy, field-crossing ones first."""
    from boost_and_broadside.env import shot_labels as sl

    rows = trainer.buffer.shot_labels
    picks = []
    for b in envs:
        valid = rows[:, b, :, 0] > 0.5
        observer = trainer.buffer.shot_launch.team_id[:, b] == 0
        example = rows[:, b, :, 1:]
        # A hit within the first few ages is a ship touching the muzzle.
        hit = (example[..., sl.EX_HIT] == sl.ENEMY) & (example[..., sl.EX_EVENT_TAU] >= 5)
        bent = (example[..., sl.EX_SNAP_EVENT][..., sl.SNAP_INDEX].log().abs() > 0.02) | (
            example[..., sl.EX_SNAP1][..., sl.SNAP_INDEX].log().abs() > 0.02
        )
        for t, i in (valid & observer & hit & bent).nonzero().tolist():
            picks.append((2, t, b, i))
        for t, i in (valid & observer & hit & ~bent).nonzero().tolist():
            picks.append((1, t, b, i))
    picks.sort(key=lambda p: -p[0])
    chosen, seen = [], set()
    for _, t, b, i in picks:
        if (b, i, t // 20) in seen:
            continue  # spread the picks over ships and time
        seen.add((b, i, t // 20))
        chosen.append((t, b, i))
        if len(chosen) == count:
            break
    return chosen


def shot_figure(trainer, envs, trajectory, outcome, path: Path, count: int = 4) -> None:
    from boost_and_broadside.env import shot_labels as sl
    from boost_and_broadside.train.rl.shot_codes import CLEAR_MISS, ENEMY_HIT, ENEMY_NEAR

    picks = _pick_shots(trainer, envs, count)
    if not picks:
        raise SystemExit("no enemy hit in this rollout; collect more rollouts")
    config = trainer.ship_config
    world = config.world_size
    D = sl.shot_label_delay(config)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 4.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    gen = torch.Generator().manual_seed(0)
    for ax, (t, b, i) in zip(axes[0], picks, strict=True):
        slot = envs.index(b)
        launch = trainer.buffer.shot_launch
        origin = complex(launch.position[t, b, i])
        attitude = complex(launch.attitude[t, b, i])
        _draw_fields(ax, trainer, t, b, origin)

        samples = torch.randn(24, 2, generator=gen) * config.bullet_spread
        noise = torch.cat(
            [torch.zeros(1, dtype=torch.complex64), torch.complex(samples[:, 0], samples[:, 1])]
        )
        flights = _unwrap(_fly(trainer, t, b, i, noise), origin, world)
        for k in range(1, flights.shape[0]):
            ax.plot(flights[k].real, flights[k].imag, color=REAL, alpha=0.12, lw=1)
        ax.plot(flights[0].real, flights[0].imag, color=REAL, lw=2, label="real flight (no spread)")

        mean, sigma = _decode_displacement(trainer.shot_codec, trajectory[t, slot, i])  # (Q, 2)
        predicted = origin + torch.complex(mean[:, 0], mean[:, 1]) * attitude
        ax.plot(predicted.real, predicted.imag, color=PREDICTED, lw=2, ls="--", label="head's mean")
        for q in range(0, D + 1, 4):
            ax.add_patch(
                Circle(
                    (float(predicted[q].real), float(predicted[q].imag)),
                    float(sigma[q]),
                    fill=False,
                    color=PREDICTED,
                    lw=1,
                    alpha=0.7,
                )
            )

        # The ship it hit, over the flight.
        target_side_hit = None
        example = trainer.buffer.shot_labels[t, b, i, 1:]
        age = int(example[sl.EX_EVENT_TAU])
        ships = _unwrap(launch.position[t : t + D, b], origin, world)  # (D, N)
        hit_row = min(t + age - 1, t + D - 1) - t
        hit_point = complex(
            flights[0, age].real, flights[0, age].imag
        )  # where the zero-spread flight is at the hit age
        enemies = launch.team_id[t, b] != launch.team_id[t, b, i]
        distances = (ships[hit_row] - hit_point).abs().masked_fill(~enemies, float("inf"))
        target_side_hit = int(distances.argmin())
        track = ships[: hit_row + 1, target_side_hit]
        ax.plot(track.real, track.imag, color=TARGET, lw=1.5, label="ship it hit")
        ax.plot([track[-1].real], [track[-1].imag], "o", color=TARGET, ms=6)
        ax.plot([origin.real], [origin.imag], "o", color=INK, ms=5)

        probabilities = torch.softmax(outcome[t, slot, i].float(), dim=-1)
        near = float(probabilities[ENEMY_NEAR].sum())
        _style(
            ax,
            f"P(enemy hit) {float(probabilities[ENEMY_HIT]):.2f}  near {near:.2f}  "
            f"clear {float(probabilities[CLEAR_MISS]):.2f}\nreal: enemy hit at age {age}",
        )
        pad = 60.0
        xs = torch.cat([flights.real.flatten(), track.real, predicted.real])
        ys = torch.cat([flights.imag.flatten(), track.imag, predicted.imag])
        ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
        ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    handles = [
        plt.Line2D([], [], color=REAL, lw=2, label="real flight, no spread"),
        plt.Line2D([], [], color=REAL, lw=1, alpha=0.3, label="real flights, spread draws"),
        plt.Line2D([], [], color=PREDICTED, lw=2, ls="--", label="shot head mean, 1σ circles"),
        plt.Line2D([], [], color=TARGET, lw=1.5, label="ship it hit"),
        plt.Rectangle((0, 0), 1, 1, color=THICK_AIR, alpha=0.25, label="thick air (n > 1)"),
        plt.Rectangle((0, 0), 1, 1, color=THIN_AIR, alpha=0.25, label="thin air (n < 1)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(
        "Counterfactual shot head: predicted vs real bullet flight",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def ship_figure(trainer, envs, pred_next, visible, path: Path) -> None:
    from boost_and_broadside.train.rl.physical_belief import POSITION_SIGMA, POSITION_X
    from boost_and_broadside.train.rl.ship_codes import PHYSICAL_MEAN_DIM

    buffer = trainer.buffer
    world = trainer.ship_config.world_size
    T = buffer.num_steps
    moments = trainer.ship_codec.decode_logits(pred_next.float())  # (T, b, N, 25+)
    predicted = torch.complex(moments[..., POSITION_X], moments[..., POSITION_X + 1])
    sigma = moments[..., PHYSICAL_MEAN_DIM + POSITION_SIGMA]
    truth = buffer.privileged_means  # (T+1, B, N, 11)
    real = torch.complex(truth[..., POSITION_X], truth[..., POSITION_X + 1])

    # One ally and the enemy that spent the most decisions hidden, alive throughout.
    slot, b = 0, envs[0]
    team = buffer.shot_launch.team_id[0, b]
    alive = buffer.alive_mask[:, b].all(0)
    contiguous = buffer.transition_contiguous[:, b].all(0)
    ok = alive & contiguous
    hidden_count = (~visible[:, slot]).sum(0)
    allies = ((team == 0) & ok).nonzero().flatten().tolist()
    enemies = ((team != 0) & ok).nonzero().flatten().tolist()
    chosen = []
    if allies:
        chosen.append(("ally", allies[0]))
    if enemies:
        chosen.append(("enemy", max(enemies, key=lambda n: int(hidden_count[n]))))
    if not chosen:
        raise SystemExit("no ship alive for the whole rollout")

    fig, axes = plt.subplots(
        2,
        len(chosen),
        figsize=(6 * len(chosen), 9),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    fig.patch.set_facecolor(SURFACE)
    for column, (kind, n) in enumerate(chosen):
        origin = complex(real[0, b, n])
        path_real = _unwrap(real[: T + 1, b, n], origin, world)
        path_pred = _unwrap(predicted[:, slot, n], origin, world)  # prediction for t+1
        ax = axes[0][column]
        _draw_fields(ax, trainer, 0, b, origin)
        ax.plot(path_real.real, path_real.imag, color=REAL, lw=2, label="real path")
        hidden = ~visible[:, slot, n]
        ax.scatter(
            path_pred.real[~hidden],
            path_pred.imag[~hidden],
            s=10,
            color=PREDICTED,
            label="one-step prediction (visible)",
            zorder=3,
        )
        ax.scatter(
            path_pred.real[hidden],
            path_pred.imag[hidden],
            s=14,
            facecolors="none",
            edgecolors=PREDICTED,
            label="one-step prediction (hidden)",
            zorder=3,
        )
        for t in range(0, T, 8):
            ax.add_patch(
                Circle(
                    (float(path_pred[t].real), float(path_pred[t].imag)),
                    float(sigma[t, slot, n]),
                    fill=False,
                    color=PREDICTED,
                    lw=0.8,
                    alpha=0.6,
                )
            )
        pad = 80.0
        xs, ys = path_real.real, path_real.imag
        ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
        ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
        _style(ax, f"{kind} ship {n}: {T} decisions ({T / 30:.1f} s), same actions as played")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False
        )

        err = (path_pred - path_real[1 : T + 1]).abs()
        ax2 = axes[1][column]
        seconds = torch.arange(T) / 30.0
        ax2.plot(seconds, err, color=PREDICTED, lw=1.5, label="position error")
        ax2.plot(seconds, sigma[:, slot, n], color=MUTED, lw=1, ls=":", label="predicted σ")
        for t in hidden.nonzero().flatten().tolist():
            ax2.axvspan(t / 30.0, (t + 1) / 30.0, color=GRID, lw=0)
        ax2.set_yscale("log")
        ax2.set_xlabel(
            "seconds into the rollout (shaded: hidden from the observer)", fontsize=8, color=MUTED
        )
        ax2.set_ylabel("px", fontsize=8, color=MUTED)
        ax2.set_facecolor(SURFACE)
        ax2.grid(color=GRID, lw=0.6)
        ax2.tick_params(colors=MUTED, labelsize=8)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)
        for spine in ax2.spines.values():
            spine.set_color(GRID)
    fig.suptitle(
        "Next-state head: one-step predicted vs real ship position",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", default="rl")
    parser.add_argument("--rollouts", type=int, default=6)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--envs", type=int, default=3, help="Environments evaluated by the heads.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/results/predictions"))
    args = parser.parse_args()

    trainer = _build_trainer(args.checkpoint, args.profile, args.num_envs, args.seed)
    _collect(trainer, args.rollouts)
    # The envs with the most valid observer enemy hits.
    from boost_and_broadside.env import shot_labels as sl

    rows = trainer.buffer.shot_labels
    hits = ((rows[..., 0] > 0.5) & (rows[..., 1 + sl.EX_HIT] == sl.ENEMY)).sum((0, 2))
    envs = hits.argsort(descending=True)[: args.envs].tolist()
    ages = torch.arange(sl.shot_label_delay(trainer.ship_config) + 1)
    pred_next, trajectory, outcome, visible = _evaluate(trainer, envs, ages)
    args.out.mkdir(parents=True, exist_ok=True)
    shot_figure(trainer, envs, trajectory, outcome, args.out / "shot_trajectories.png")
    ship_figure(trainer, envs, pred_next, visible, args.out / "ship_trajectories.png")


if __name__ == "__main__":
    main()
