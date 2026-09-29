"""Is the hex density target actually learned, or is it easy for cheap reasons?

``loss/global_density`` fell from 0.222 to 0.009 against a target power of
0.208 -- apparently 96% of the target explained, including the *enemy* field,
most of whose ships the observer cannot see. Three things could produce that
without the head knowing anything about beliefs:

* the fit is quoted against ``E[target^2]``, not against the target's variance,
  and a field with a large constant component scores well from its mean alone;
* the ally and enemy halves share one mean, so the half the observer can see
  outright can carry the half it cannot;
* the frontline concentrates both fleets near the front, and the front position
  is an input on the very token the head reads, so "mirror of the front" is a
  strong predictor that needs no belief at all.

This scores the head against that ladder, each half separately, on held-out
environments:

    zero        what the head scores before training (the target's power).
    mean        the per-cell average heat map. A constant. Beating this is the
                minimum bar, and the residual here is the target's *variance*,
                which is the honest denominator for an R^2.
    front       per-cell linear in the signed front position.
    ally        enemy cells from the front and the *visible* ally field. What is
                inferable with no belief whatsoever -- the bar that matters.
    head        the trained global density head.

If ``head`` does not clear ``ally`` on the enemy half, the head is not using
its beliefs and the aggregate series was flattering it.

    uv run --no-sync python benchmarks/density_baselines.py --run pious-butterfly-748
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.gradient_decomposition import _resume_path, _run_profile  # noqa: E402

OBSERVER_TEAM = 0


def _fit_linear(features, targets, ridge: float):
    """Least-squares map from ``features`` to ``targets`` with a ridge term.

    Args:
        features: (N, F) design matrix, intercept column included by the caller.
        targets:  (N, C) values to predict.
        ridge:    Tikhonov term on the normal equations, so a constant or
                  duplicated feature cannot make the solve singular.

    Returns:
        (F, C) coefficients.
    """
    import torch

    gram = features.T @ features  # (F, F)
    gram += ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    return torch.linalg.solve(gram, features.T @ targets)  # (F, C)


def _mse(prediction, truth) -> float:
    """Mean square error over every cell and sample."""

    return float((prediction - truth).pow(2).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram", default="auto")
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument("--warmup-rollouts", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--step-stride", type=int, default=2, help="Keep every Nth stored step.")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--microbatch-tokens", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples-output", type=Path, default=None,
                        help="Also save the raw targets/predictions for the PNG renderer.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    import torch

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.config.diagnostics import GRADIENT_DIAGNOSTICS_OFF
    from boost_and_broadside.env.observation import ObsKey
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.profiles import resolve_named_profile
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    profile = args.profile or _run_profile(args.run, args.checkpoint_dir)
    if profile is None:
        parser.error(f"no profile recorded for {args.run!r}; pass --profile")
    checkpoint = _resume_path(args.run, args.checkpoint_dir)

    launch = resolve_training_launch(
        profile=profile,
        vram=args.vram,
        device=args.device,
        seed=None,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
        wandb=False,
        allow_config_drift=False,
        gradient_diagnostics=GRADIENT_DIAGNOSTICS_OFF,
        num_envs=None,
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
        gradient_diagnostics=GRADIENT_DIAGNOSTICS_OFF,
    )
    trainer.load_checkpoint(str(checkpoint))

    # The counter advances while the environment warms up, so the step this
    # measurement belongs to is the one the checkpoint was saved at.
    checkpoint_step = trainer._global_step
    runtime = trainer._initialize_rollout_runtime()
    for _ in range(args.warmup_rollouts):
        trainer._collect_rollout(runtime, False)

    targets, predictions, fronts, env_ids, hidden_counts = [], [], [], [], []
    for rollout in range(args.rollouts):
        trainer._collect_rollout(runtime, False)
        iterator = trainer.buffer.get_minibatch_iterator(
            trainer.cfg.num_minibatches, trainer.cfg.microbatch_tokens
        )
        for chunks in iterator:
            for chunk in chunks:
                batch = chunk.to(trainer.device)
                steps = batch.alive.shape[0]
                observation = batch.obs.slice_time(0, steps)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    evaluated = trainer._eager_evaluate_actions(
                        obs=observation,
                        actions=batch.actions.long(),
                        initial_hidden=batch.hidden,
                        alive_mask=observation[ObsKey.BELIEF_VALID].bool(),
                        done_mask=batch.terminated,
                        return_encoder_output=False,
                        return_enemy_action=True,
                        return_density=True,
                    )
                prediction = evaluated[-1].float()  # (T, B, 2C)
                target = batch.density_targets.float()  # (T, B, 2C)
                # The global token's own front channel, the feature a
                # front-only baseline is allowed to use. It lives at token index
                # N -- the ship slots of this channel are zero-padding, and
                # reading index 0 silently yields a constant.
                front = observation[ObsKey.FRONT_POSITION].float()
                front = front[:, :, trainer.buffer.num_ships].reshape(steps, -1, 1)  # (T, B, 1)
                # Enemies the observer cannot currently see. The sharp test of
                # whether the head's enemy field is belief or bookkeeping: if
                # its error does not grow as this rises, it is reading visible
                # ships rather than remembering hidden ones.
                ships = trainer.buffer.num_ships
                team = observation[ObsKey.TEAM_ID][:, :, :ships]
                visible = observation[ObsKey.VISIBLE][:, :, :ships]
                alive = observation[ObsKey.ALIVE][:, :, :ships]
                if team.dim() == 4:
                    team, visible, alive = team[..., 0], visible[..., 0], alive[..., 0]
                hidden = (
                    (team.round().long() != OBSERVER_TEAM) & ~visible.bool() & alive.bool()
                ).sum(-1)  # (T, B)

                keep = slice(None, None, args.step_stride)
                width = prediction.shape[1]
                targets.append(target[keep].reshape(-1, target.shape[-1]).cpu())
                predictions.append(prediction[keep].reshape(-1, prediction.shape[-1]).cpu())
                fronts.append(front[keep].reshape(-1, 1).cpu())
                hidden_counts.append(hidden[keep].reshape(-1).cpu())
                # Environment identity, so the fit/score split never puts two
                # steps of one episode on both sides of it.
                ids = torch.arange(width).repeat(len(range(0, steps, args.step_stride)))
                env_ids.append(ids + rollout * 100000)
        print(f"rollout {rollout + 1}/{args.rollouts} collected")

    target = torch.cat(targets).double()
    head = torch.cat(predictions).double()
    front = torch.cat(fronts).double()
    env_id = torch.cat(env_ids)
    hidden_enemies = torch.cat(hidden_counts)
    cells = target.shape[1] // 2
    print(f"samples {target.shape[0]:,}  cells per half {cells}")
    # A padded or mis-indexed channel reads as a constant, and a constant
    # feature makes its baseline silently identical to the mean predictor
    # rather than failing. Front spread is the one thing here worth asserting.
    if float(front.std()) < 1e-6:
        raise RuntimeError(
            "the front feature does not vary; it is being read from the wrong "
            "token and every front baseline below would be meaningless"
        )
    print(f"front: mean {float(front.mean()):.1f}  std {float(front.std()):.1f}")

    # Split by environment, not by row: steps within an episode are strongly
    # correlated, so a random row split would leak the answer into the fit.
    score_mask = (env_id % 2) == 1
    fit_mask = ~score_mask
    fit_front, score_front = front[fit_mask], front[score_mask]
    ones_fit = torch.ones_like(fit_front)
    ones_score = torch.ones_like(score_front)

    report: dict[str, dict[str, float]] = {}
    for name, columns in (("ally", slice(0, cells)), ("enemy", slice(cells, 2 * cells))):
        truth_fit = target[fit_mask][:, columns]
        truth_score = target[score_mask][:, columns]
        scores: dict[str, float] = {}

        scores["zero"] = _mse(torch.zeros_like(truth_score), truth_score)

        cell_mean = truth_fit.mean(0, keepdim=True)
        scores["mean"] = _mse(cell_mean.expand_as(truth_score), truth_score)

        design_fit = torch.cat([ones_fit, fit_front], dim=1)
        design_score = torch.cat([ones_score, score_front], dim=1)
        coefficients = _fit_linear(design_fit, truth_fit, args.ridge)
        scores["front"] = _mse(design_score @ coefficients, truth_score)

        if name == "enemy":
            # Front, plus the ally field the observer sees outright. Anything
            # the head scores beyond this it had to get from its beliefs.
            ally_fit = target[fit_mask][:, :cells]
            ally_score = target[score_mask][:, :cells]
            design_fit = torch.cat([ones_fit, fit_front, ally_fit], dim=1)
            design_score = torch.cat([ones_score, score_front, ally_score], dim=1)
            coefficients = _fit_linear(design_fit, truth_fit, args.ridge)
            scores["ally"] = _mse(design_score @ coefficients, truth_score)

        scores["head"] = _mse(head[score_mask][:, columns], truth_score)

        variance = scores["mean"]
        report[name] = {
            **scores,
            # Against the constant heat map, which is the honest denominator:
            # the fraction of the field's actual variation the predictor
            # explains, rather than the fraction of its raw power.
            **{f"r2_vs_mean/{k}": 1.0 - v / variance for k, v in scores.items()},
        }

    result = {
        "run": args.run,
        "checkpoint": str(checkpoint),
        "resumed_global_step": checkpoint_step,
        "samples": int(target.shape[0]),
        "cells_per_half": cells,
        "warmup_rollouts": args.warmup_rollouts,
        "halves": report,
    }
    order = ("zero", "mean", "front", "ally", "head")
    for name, scores in report.items():
        print()
        print(f"=== {name} half ===")
        print(f"{'predictor':<10}{'MSE':>12}{'R2 vs mean':>13}")
        print("-" * 35)
        for key in order:
            if key not in scores:
                continue
            print(f"{key:<10}{scores[key]:>12.6f}{scores[f'r2_vs_mean/{key}']:>13.3f}")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    if args.samples_output is not None:
        torch.save(
            {
                "target": target[score_mask].float(),
                "head": head[score_mask].float(),
                "hidden_enemies": hidden_enemies[score_mask],
                "cells": cells,
                "run": args.run,
                "global_step": checkpoint_step,
                "baselines": report,
            },
            args.samples_output,
        )
        print(f"wrote {args.samples_output}")
    trainer.shutdown()


if __name__ == "__main__":
    main()
