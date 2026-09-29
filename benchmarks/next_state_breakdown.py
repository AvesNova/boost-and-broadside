"""The next-state head, per physical channel and per visibility class.

``loss/next_state`` is one number over eleven channels whose residuals span four
orders of magnitude, and the production series splits it only by
visible/hidden -- a split that puts allies, which an observer always sees, in
the same bucket as the enemies it happens to have in sight. This reports the
three classes separately (ally, visible enemy, hidden enemy) and per channel:

* **nats** -- the channel's mean Gaussian NLL, the objective's own currency.
* **accuracy** -- RMS residual in physical units, beside the RMS of the label
  itself. The label is the correction that carries the *believed* state onto the
  true next one, so predicting zero is the null model and ``skill`` is how much
  of that the head removes. Negative skill means the head is worse than saying
  nothing.
* **calibration** -- ``z2 = E[(r/sigma)^2]``, one for a Gaussian telling the
  truth about its own spread, above one overconfident, below one underconfident.
  Reported with the mean sigma and the fraction of tokens sitting on the
  ``LOG_SIGMA_MIN``/``MAX`` clamp, where the head has stopped paying for being
  wrong.
* **gradient** -- the trunk-parameter gradient norm each (channel, class) term
  contributes, measured the way ``benchmarks/gradient_decomposition.py``
  measures the top-level terms. This is what says which target is actually
  moving the shared trunk.

Read-only, like the decomposition tool: no checkpoint is written, no run status
recorded, no W&B, run directory untouched.

    uv run --no-sync python benchmarks/next_state_breakdown.py --run pious-butterfly-748
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run as a script, sys.path[0] is benchmarks/, so the repo root has to go on the
# path before the sibling module resolves as `benchmarks.*` -- the spelling the
# tests import it by.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.gradient_decomposition import _resume_path, _run_profile  # noqa: E402

# Observer of every stored primary observation. The live policy is always Team 0.
OBSERVER_TEAM = 0


def _bucket_masks(observation, ns_mask, num_ships):
    """``ally`` / ``visible enemy`` / ``hidden enemy`` masks over ship tokens.

    ``ObsKey.TEAM_ID`` is ``where(known, truth, 0)`` and every token here is
    already gated on ``BELIEF_VALID``, which is that same ``known``, so the team
    id is authoritative for exactly the tokens being bucketed.
    """
    from boost_and_broadside.env.observation import ObsKey

    team = observation[ObsKey.TEAM_ID][:, :, :num_ships]
    if team.dim() == 4:
        team = team[..., 0]
    ally = ns_mask & (team.round().long() == OBSERVER_TEAM)
    enemy = ns_mask & ~ally
    visible = observation[ObsKey.VISIBLE][:, :, :num_ships].bool()
    if visible.dim() == 4:
        visible = visible[..., 0]
    return {
        "ally": ally,
        "visible_enemy": enemy & visible,
        "hidden_enemy": enemy & ~visible,
        # Kept to prove the class is empty rather than assumed to be.
        "hidden_ally": ally & ~visible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram", default="auto")
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument(
        "--rollouts",
        type=int,
        default=2,
        help="Rollouts to accumulate the no-grad statistics over.",
    )
    parser.add_argument(
        "--warmup-rollouts",
        type=int,
        default=24,
        help=(
            "Rollouts to discard first. _initialize_rollout_runtime resets every "
            "env together, so a cold probe measures the opening phase -- teams "
            "spawned apart and not yet in contact, where almost no enemy is "
            "visible -- which a run 84M steps in never sees. Warm up until the "
            "reported visible-enemy fraction settles near fog/visible_fraction."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--microbatch-tokens",
        type=int,
        default=None,
        help="Narrower than the run used, if the eager graph will not fit where the compiled one did.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    import torch

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.config.diagnostics import GradientDiagnosticsConfig
    from boost_and_broadside.env.observation import ObsKey
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.profiles import resolve_named_profile
    from boost_and_broadside.train.rl.physical_belief import (
        LOG_SIGMA_MAX,
        LOG_SIGMA_MIN,
        PHYSICAL_MEAN_DIM,
        PHYSICAL_MEAN_NAMES,
        POSITION_X,
        POSITION_Y,
        SCALAR_UNCERTAINTY,
        VELOCITY_X,
        VELOCITY_Y,
    )
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    # Mean channel -> its marginal log-sigma column. Position and velocity carry
    # a full 2D covariance; the marginal spread is the per-channel statement,
    # and the correlation is reported separately rather than folded in.
    sigma_column = {POSITION_X: 0, POSITION_Y: 1, VELOCITY_X: 3, VELOCITY_Y: 4}
    sigma_column.update(dict(SCALAR_UNCERTAINTY))

    profile = args.profile or _run_profile(args.run, args.checkpoint_dir)
    if profile is None:
        parser.error(f"no profile recorded for {args.run!r}; pass --profile")
    checkpoint = _resume_path(args.run, args.checkpoint_dir)

    # top_level populates _grad_diag_params / _grad_diag_trunk, which is the
    # trunk definition the per-channel gradients are taken over.
    diagnostics = GradientDiagnosticsConfig(level="top_level", interval=1)
    launch = resolve_training_launch(
        profile=profile,
        vram=args.vram,
        device=args.device,
        seed=None,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
        wandb=False,
        allow_config_drift=False,
        gradient_diagnostics=diagnostics,
        num_envs=args.num_envs,
        microbatch_tokens=args.microbatch_tokens,
        report=print,
        resolve=resolve_named_profile,
        overrides={},
    )
    resolved = launch.resolved
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=args.device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=launch.execution.compile_mode,
        gradient_diagnostics=diagnostics,
    )
    trainer.load_checkpoint(str(checkpoint))
    scales = trainer.next_state.scales
    num_ships = trainer.buffer.num_ships
    classes = ("ally", "visible_enemy", "hidden_enemy", "hidden_ally")

    # name -> channel -> running sums, so one pass over the rollout answers
    # every statistic and nothing is held per token.
    totals = {
        name: {
            channel: dict.fromkeys(
                ("n", "nll", "sq", "label_sq", "sigma_sq", "z2", "at_floor", "at_ceiling"), 0.0
            )
            for channel in PHYSICAL_MEAN_NAMES
        }
        for name in classes
    }
    gradient_norms: dict[str, dict[str, float]] = {name: {} for name in classes}
    # A Gaussian NLL over a heavy-tailed label is described badly by RMS alone:
    # one respawn teleport outweighs ten thousand ordinary steps. Keep a bounded
    # sample per (class, channel) so the median and the tail are reportable too.
    samples: dict[str, dict[str, list]] = {
        name: {channel: [] for channel in PHYSICAL_MEAN_NAMES} for name in classes
    }
    sample_cap = 40_000

    runtime = trainer._initialize_rollout_runtime()
    for warmup_index in range(args.warmup_rollouts):
        trainer._collect_rollout(runtime, False)
        if (warmup_index + 1) % 4 == 0:
            visibility = trainer.wrapper.last_visibility.ship
            state = trainer.wrapper.env.state
            enemy = (state.ship_team_id[:, :num_ships] != OBSERVER_TEAM) & state.ship_alive[
                :, :num_ships
            ]
            seen = float((visibility[:, OBSERVER_TEAM, :num_ships] & enemy).sum() / enemy.sum())
            print(f"warmup {warmup_index + 1}/{args.warmup_rollouts}  visible enemy {seen:.3f}")

    for rollout_index in range(args.rollouts):
        terminated = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, terminated)
        trainer._precompute_ns_labels(trainer.buffer)

        iterator = trainer.buffer.get_minibatch_iterator(
            trainer.cfg.num_minibatches, trainer.cfg.microbatch_tokens
        )
        for minibatch_index, chunks in enumerate(iterator):
            for chunk in chunks:
                batch = chunk.to(trainer.device)
                steps = batch.alive.shape[0]
                observation = batch.obs.slice_time(0, steps)
                # The training mask, channel for channel with _compute_loss.
                ns_mask = (
                    observation[ObsKey.BELIEF_VALID][:, :, :num_ships].bool()
                    & ~batch.terminated.unsqueeze(-1)
                    & batch.transition_contiguous
                )
                buckets = _bucket_masks(observation, ns_mask, num_ships)
                take_gradients = (
                    rollout_index == 0 and minibatch_index == 0 and not gradient_norms["ally"]
                )

                def evaluate():
                    """One eager forward, the path a diagnosed update takes."""
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return trainer._eager_evaluate_actions(
                            obs=observation,
                            actions=batch.actions.long(),
                            initial_hidden=batch.hidden,
                            alive_mask=observation[ObsKey.BELIEF_VALID].bool(),
                            done_mask=batch.terminated,
                            return_encoder_output=False,
                            return_enemy_action=True,
                            return_density=True,
                        )

                # ---- Statistics: no graph, so the whole rollout fits ---------
                with torch.no_grad():
                    prediction = evaluate()[5].float()  # (T, B, N, 24)
                    labels = batch.ns_labels
                    per_dim = trainer.next_state.loss(prediction, labels)
                    residual = trainer.next_state.residual(prediction, labels)
                    uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
                    log_sigma = torch.stack(
                        [uncertainty[..., sigma_column[i]] for i in range(PHYSICAL_MEAN_DIM)],
                        dim=-1,
                    )  # (T, B, N, 11) each channel's marginal spread
                    sigma = log_sigma.exp()
                    standardized = (residual / sigma.clamp(min=1e-12)).pow(2)
                    for name, mask in buckets.items():
                        count = float(mask.sum())
                        if count == 0.0:
                            continue
                        weight = mask.float().unsqueeze(-1)
                        sums = {
                            "nll": (per_dim * weight).sum((0, 1, 2)),
                            "sq": (residual.pow(2) * weight).sum((0, 1, 2)),
                            "label_sq": (labels.pow(2) * weight).sum((0, 1, 2)),
                            "sigma_sq": (sigma.pow(2) * weight).sum((0, 1, 2)),
                            "z2": (standardized * weight).sum((0, 1, 2)),
                            "at_floor": ((log_sigma <= LOG_SIGMA_MIN + 1e-4) * weight).sum(
                                (0, 1, 2)
                            ),
                            "at_ceiling": ((log_sigma >= LOG_SIGMA_MAX - 1e-4) * weight).sum(
                                (0, 1, 2)
                            ),
                        }
                        for index, channel in enumerate(PHYSICAL_MEAN_NAMES):
                            totals[name][channel]["n"] += count
                            for key, value in sums.items():
                                totals[name][channel][key] += float(value[index])
                            held = samples[name][channel]
                            if sum(len(part) for part in held) < sample_cap:
                                held.append(
                                    torch.stack(
                                        (
                                            labels[..., index][mask].abs(),
                                            residual[..., index][mask].abs(),
                                            sigma[..., index][mask],
                                        )
                                    )
                                    .cpu()
                                    .T
                                )
                    del prediction, per_dim, residual, sigma, log_sigma, standardized

                # ---- Trunk gradient per (channel, class) --------------------
                # One graph, 11 x 3 traversals off it. Taken on a single
                # micro-batch: the relative sizes are what the report is for,
                # and retaining the graph is what costs memory.
                if take_gradients:
                    per_dim = trainer.next_state.loss(evaluate()[5].float(), batch.ns_labels)
                    denominator = ns_mask.float().sum().clamp(min=1.0) * PHYSICAL_MEAN_DIM
                    for name, mask in buckets.items():
                        if mask.sum() == 0:
                            continue
                        mask_f = mask.float()
                        for index, channel in enumerate(PHYSICAL_MEAN_NAMES):
                            term = (
                                trainer.cfg.next_state_coef
                                * (per_dim[..., index] * mask_f).sum()
                                / denominator
                            )
                            grads = torch.autograd.grad(
                                term,
                                trainer._grad_diag_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            total = torch.zeros((), device=trainer.device)
                            for grad, is_trunk in zip(
                                grads, trainer._grad_diag_trunk, strict=True
                            ):
                                if grad is not None and is_trunk:
                                    total = total + grad.float().pow(2).sum()
                            gradient_norms[name][channel] = float(total.sqrt())
                            del grads
                    del per_dim
                    torch.cuda.empty_cache()
        seen = totals["visible_enemy"]["position_x"]["n"]
        hidden = totals["hidden_enemy"]["position_x"]["n"]
        print(
            f"rollout {rollout_index + 1}/{args.rollouts} done  "
            f"cumulative visible-enemy fraction {seen / max(seen + hidden, 1.0):.3f}"
        )

    report: dict[str, dict] = {}
    for name in classes:
        entries = {}
        for index, channel in enumerate(PHYSICAL_MEAN_NAMES):
            row = totals[name][channel]
            if row["n"] == 0.0:
                continue
            n, scale = row["n"], scales[index]
            rms = (row["sq"] / n) ** 0.5
            label_rms = (row["label_sq"] / n) ** 0.5
            held = samples[name][channel]
            drawn = torch.cat(held) if held else torch.zeros(0, 3)
            quantiles = (
                drawn.quantile(
                    torch.tensor([0.5, 0.999]), dim=0
                )  # (2, 3) label, residual, sigma
                if drawn.numel()
                else torch.full((2, 3), float("nan"))
            )
            median_label = float(quantiles[0, 0])
            median_residual = float(quantiles[0, 1])
            entries[channel] = {
                "tokens": n,
                "nats": row["nll"] / n,
                "rms_residual_physical": rms * scale,
                "rms_label_physical": label_rms * scale,
                "skill_vs_zero": 1.0 - rms / label_rms if label_rms > 0 else float("nan"),
                "median_residual_physical": median_residual * scale,
                "median_label_physical": median_label * scale,
                "median_sigma_physical": float(quantiles[0, 2]) * scale,
                "p999_label_physical": float(quantiles[1, 0]) * scale,
                "median_skill_vs_zero": (
                    1.0 - median_residual / median_label if median_label > 0 else float("nan")
                ),
                "sampled": int(drawn.shape[0]),
                "rms_sigma_physical": (row["sigma_sq"] / n) ** 0.5 * scale,
                "z2": row["z2"] / n,
                "sigma_at_floor": row["at_floor"] / n,
                "sigma_at_ceiling": row["at_ceiling"] / n,
                "trunk_grad_norm": gradient_norms[name].get(channel),
            }
        if entries:
            report[name] = entries

    result = {
        "run": args.run,
        "profile": profile,
        "checkpoint": str(checkpoint),
        "resumed_global_step": trainer._global_step,
        "device": torch.cuda.get_device_name(args.device) if args.device == "cuda" else "cpu",
        "next_state_coef": trainer.cfg.next_state_coef,
        "rollouts": args.rollouts,
        "observer_team": OBSERVER_TEAM,
        "warmup_rollouts": args.warmup_rollouts,
        "visible_enemy_fraction": (
            report["visible_enemy"]["position_x"]["tokens"]
            / (
                report["visible_enemy"]["position_x"]["tokens"]
                + report["hidden_enemy"]["position_x"]["tokens"]
            )
            if "visible_enemy" in report and "hidden_enemy" in report
            else None
        ),
        "delta_scales": dict(zip(PHYSICAL_MEAN_NAMES, scales, strict=True)),
        "classes": report,
    }
    for name, entries in report.items():
        print()
        print(f"=== {name} ===")
        header = (
            f"{'channel':<17}{'nats':>8}"
            f"{'med err':>10}{'med lbl':>10}{'med skl':>9}{'med sig':>10}"
            f"{'rms err':>11}{'rms lbl':>11}{'p99.9 lbl':>11}"
            f"{'z2':>8}{'floor':>7}{'|g|':>8}"
        )
        print(header)
        print("-" * len(header))
        for channel, entry in entries.items():
            grad = entry["trunk_grad_norm"]
            print(
                f"{channel:<17}{entry['nats']:>8.2f}"
                f"{entry['median_residual_physical']:>10.4g}"
                f"{entry['median_label_physical']:>10.4g}"
                f"{entry['median_skill_vs_zero']:>9.2f}"
                f"{entry['median_sigma_physical']:>10.4g}"
                f"{entry['rms_residual_physical']:>11.4g}"
                f"{entry['rms_label_physical']:>11.4g}"
                f"{entry['p999_label_physical']:>11.4g}"
                f"{entry['z2']:>8.3g}{entry['sigma_at_floor']:>7.2f}"
                f"{'--' if grad is None else f'{grad:.3g}':>8}"
            )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    trainer.shutdown()


if __name__ == "__main__":
    main()
