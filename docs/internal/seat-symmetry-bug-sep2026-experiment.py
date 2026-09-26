"""Seat-asymmetry experiment: a/a, a/s, s/a, s/s run as one batch.

Four conditions occupy four contiguous blocks of one environment batch, so every
condition sees the same physics build, the same step schedule and the same
wall clock.  Two drivers are available:

  --driver eval   MatchRunner + advance_autonomous_decision (the Elo-eval path)
  --driver train  YemongEnvWrapper + the PPO rollout's dual-ego semantics

`train` is the path the BC rollout uses and the one under suspicion; `eval` is
the control that reports a healthy 44% win rate vs scripted.

The scripted controller is built exactly as the trainer builds it --
`StochasticScriptedAgent(ship_config, StochasticAgentConfig())` -- not via
`resolve_agent_spec("scripted")`, which rescales `frontline_zone_radius` by
fleet size and would make the two drivers face different teachers.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.core import MatchResult
from boost_and_broadside.env.observation import perceived_observation_from_state
from boost_and_broadside.env.outcome import outcome_masks
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.profiles import resolve_named_profile
from boost_and_broadside.runtime.actions import PendingActionState
from boost_and_broadside.train.rl.belief import DualBeliefTracker
from boost_and_broadside.train.rl.opponents import flip_team_obs
from boost_and_broadside.train.rl.policy_io import load_policy_bundle

# team0 controller, team1 controller.  "a" = the trained policy, "s" = scripted.
CONDITIONS = [("a", "a"), ("a", "s"), ("s", "a"), ("s", "s")]
LABELS = [f"{t0} vs {t1}" for t0, t1 in CONDITIONS]


class Tally:
    """Per-block win/loss/draw counts and episode lengths."""

    def __init__(self, num_blocks: int, device: torch.device) -> None:
        self.t0 = torch.zeros(num_blocks, dtype=torch.long, device=device)
        self.t1 = torch.zeros(num_blocks, dtype=torch.long, device=device)
        self.draw = torch.zeros(num_blocks, dtype=torch.long, device=device)
        self.length_sum = torch.zeros(num_blocks, dtype=torch.float64, device=device)

    def add(self, block_of_env, t0_mask, t1_mask, tied_mask, ep_len, num_blocks):
        for mask, target in ((t0_mask, self.t0), (t1_mask, self.t1), (tied_mask, self.draw)):
            if mask.any():
                target += torch.bincount(block_of_env[mask], minlength=num_blocks)
        done = t0_mask | t1_mask | tied_mask
        if done.any():
            self.length_sum += torch.bincount(
                block_of_env[done], weights=ep_len[done].double(), minlength=num_blocks
            )

    def games(self):
        return self.t0 + self.t1 + self.draw

    def report(self):
        games = self.games()
        rows = []
        for i, label in enumerate(LABELS):
            n = int(games[i])
            if n == 0:
                rows.append({"condition": label, "games": 0})
                continue
            t0, t1, dr = int(self.t0[i]), int(self.t1[i]), int(self.draw[i])
            # Score counts a draw as half a win, matching the Elo update.
            score = (t0 + 0.5 * dr) / n
            rows.append(
                {
                    "condition": label,
                    "games": n,
                    "team0_win_rate": t0 / n,
                    "team1_win_rate": t1 / n,
                    "draw_rate": dr / n,
                    "team0_score": score,
                    "stderr": (score * (1 - score) / n) ** 0.5,
                    "mean_episode_len": float(self.length_sum[i]) / n,
                }
            )
        return rows


def build_blocks(envs_per_block, device):
    num_blocks = len(CONDITIONS)
    block_of_env = torch.arange(num_blocks, device=device).repeat_interleave(envs_per_block)
    return num_blocks, block_of_env


def load_policy(path, ship_config, model_config, num_ships, device):
    bundle = load_policy_bundle(
        path,
        device=str(device),
        num_ships=num_ships,
        ship_config=ship_config,
        model_config=model_config,
    )
    print(f"loaded policy: step={bundle.global_step} paradigm={bundle.paradigm} path={path}")
    return bundle


# --------------------------------------------------------------------------
# eval driver
# --------------------------------------------------------------------------
def run_eval_driver(args, resolved, bundle, scripted_agent, device):
    ship_config = resolved.ship_config
    env_config = resolved.env_config
    num_ships = env_config.num_ships
    num_blocks, block_of_env = build_blocks(args.envs_per_block, device)
    num_envs = num_blocks * args.envs_per_block

    # The policy is registered twice over the same weights: MatchRunner gives an
    # agent one forward pass per step, so a policy that held both seats in the
    # a-vs-a block would play team 0 from the flipped team-1 view.  Two entries
    # with independent hidden/belief state keep each seat on its own view.
    policy_seat0 = ResolvedAgent("policy", bundle.policy, bundle=bundle)
    policy_seat1 = ResolvedAgent("policy", bundle.policy, bundle=bundle)
    scripted = ResolvedAgent("scripted", scripted_agent)
    agents = [policy_seat0, scripted, policy_seat1]
    A, S, A1 = 0, 1, 2
    team0_by_condition = {"a": A, "s": S}
    team1_by_condition = {"a": A1, "s": S}

    team0_index = torch.empty(num_envs, dtype=torch.long, device=device)
    team1_index = torch.empty(num_envs, dtype=torch.long, device=device)
    for i, (t0, t1) in enumerate(CONDITIONS):
        sl = slice(i * args.envs_per_block, (i + 1) * args.envs_per_block)
        team0_index[sl] = team0_by_condition[t0]
        team1_index[sl] = team1_by_condition[t1]

    env = create_evaluation_env(num_envs, ship_config, env_config, device)
    env.reset(seed=args.seed)
    runner = MatchRunner(env, agents, team0_index, team1_index, ship_config, num_ships)
    runner.init_hidden()

    tally = Tally(num_blocks, device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    start = time.time()
    for step in range(args.max_steps):
        dones, truncated = runner.step()
        ep_len += 1
        done_any = dones | truncated
        if done_any.any():
            t0m, t1m, tie = outcome_masks(env.state, done_any)
            tally.add(block_of_env, t0m, t1m, tie, ep_len, num_blocks)
            ep_len = torch.where(done_any, torch.zeros_like(ep_len), ep_len)
            runner.reset_finished(done_any)
        if progress(step, start, tally, args):
            break
    return tally


# --------------------------------------------------------------------------
# train driver -- replicates PPOTrainer._collect_primary_step
# --------------------------------------------------------------------------
def run_train_driver(args, resolved, bundle, scripted_agent, device):
    ship_config = resolved.ship_config
    env_config = resolved.env_config
    num_ships = env_config.num_ships
    num_blocks, block_of_env = build_blocks(args.envs_per_block, device)
    num_envs = num_blocks * args.envs_per_block
    policy = bundle.policy

    wrapper = YemongEnvWrapper(
        num_envs=num_envs,
        ship_config=ship_config,
        env_config=env_config,
        rewards=resolved.train_config.rewards,
        device=device,
        include_bullets=resolved.model_config.reads_bullets,
    )
    obs = wrapper.reset(seed=args.seed)

    num_recurrent = policy.num_recurrent_tokens
    hidden = policy.initial_hidden(num_envs, num_recurrent, device)
    hidden_t1 = policy.initial_hidden(num_envs, num_recurrent, device)
    beliefs = DualBeliefTracker(
        num_envs,
        num_ships,
        ship_config.dt * env_config.action_repeat,
        policy.coordinator,
        device,
    )
    obs = beliefs.compose(obs)
    action_state = PendingActionState.allocate(num_envs, num_ships, device)

    # Which controller owns each seat, per environment.
    agent_holds_team0 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    agent_holds_team1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for i, (t0, t1) in enumerate(CONDITIONS):
        sl = slice(i * args.envs_per_block, (i + 1) * args.envs_per_block)
        agent_holds_team0[sl] = t0 == "a"
        agent_holds_team1[sl] = t1 == "a"

    tally = Tally(num_blocks, device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    start = time.time()
    for step in range(args.max_steps):
        team_id = obs["team_id"][:, :num_ships]

        with torch.no_grad():
            scripted_action = scripted_agent.get_actions(
                wrapper.env.state, wrapper.last_visibility.ship
            ).int()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_t1 = flip_team_obs(obs.for_team(1), num_ships)
                obs_both = obs.concat_batch(obs_t1)
                hidden_both = torch.cat([hidden, hidden_t1], dim=1)
                action_both, _, _, pred_both, hidden_both = policy.get_action_and_value(
                    obs_both, hidden_both
                )
            action_t0, action_t1 = action_both[:num_envs], action_both[num_envs:]
            pred_t0, pred_t1 = pred_both[:num_envs], pred_both[num_envs:]
            hidden = hidden_both[:, : num_envs * num_recurrent]
            hidden_t1 = hidden_both[:, num_envs * num_recurrent :]

            # Each seat takes its controller's action, then ships take their seat's.
            team0_src = torch.where(
                agent_holds_team0.view(-1, 1, 1), action_t0.int(), scripted_action
            )
            team1_src = torch.where(
                agent_holds_team1.view(-1, 1, 1), action_t1.int(), scripted_action
            )
            action = torch.where((team_id == 0).unsqueeze(-1), team0_src, team1_src)

            next_obs, _, dones, truncated, info = wrapper.step(action_state.applied_action())
            done_any = dones | truncated
            action_state.commit(action, info["actuator_contiguous"], done_any)
            action_state.write_observation(
                next_obs, next_obs["team_id"][:, :num_ships], num_ships
            )

            ep_len += 1
            result = info["match_result"]
            t0m = result == int(MatchResult.TEAM0_WIN)
            t1m = result == int(MatchResult.TEAM1_WIN)
            tie = result == int(MatchResult.DRAW)
            if done_any.any():
                tally.add(block_of_env, t0m, t1m, tie, ep_len, num_blocks)
                ep_len = torch.where(done_any, torch.zeros_like(ep_len), ep_len)

            hidden = policy.reset_hidden_for_envs(hidden, done_any, num_recurrent)
            hidden_t1 = policy.reset_hidden_for_envs(hidden_t1, done_any, num_recurrent)
            beliefs.advance(obs, pred_t0, pred_t1)
            beliefs.reset(done_any)
            obs = beliefs.compose(next_obs)

        if progress(step, start, tally, args):
            break
    return tally


def progress(step, start, tally, args):
    games = tally.games()
    if (step + 1) % args.log_every == 0:
        elapsed = time.time() - start
        counts = " ".join(
            f"{label}={int(games[i])}" for i, label in enumerate(LABELS)
        )
        print(
            f"  step {step + 1:5d}  {elapsed:6.0f}s  "
            f"{(step + 1) / elapsed:5.1f} dec/s  games: {counts}",
            flush=True,
        )
    return bool((games >= args.target_episodes).all())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--driver", choices=["eval", "train"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--envs-per-block", type=int, default=240)
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--target-episodes", type=int, default=300)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    resolved = resolve_named_profile("bc")
    num_ships = resolved.env_config.num_ships
    bundle = load_policy(
        args.checkpoint, resolved.ship_config, resolved.model_config, num_ships, device
    )
    bundle.policy.eval()
    # Built as the trainer builds it, so the teacher matches the BC run.
    scripted_agent = StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig())

    print(f"\ndriver={args.driver}  envs/block={args.envs_per_block}  "
          f"total envs={len(CONDITIONS) * args.envs_per_block}")
    runner = run_eval_driver if args.driver == "eval" else run_train_driver
    tally = runner(args, resolved, bundle, scripted_agent, device)

    rows = tally.report()
    print(f"\n=== driver={args.driver} ===")
    header = f"{'condition':12}{'games':>7}{'t0 win':>9}{'t1 win':>9}{'draw':>8}{'t0 score':>10}{'+/-':>8}{'ep len':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if not r.get("games"):
            print(f"{r['condition']:12}{0:>7}")
            continue
        print(
            f"{r['condition']:12}{r['games']:>7}{r['team0_win_rate']:>9.3f}"
            f"{r['team1_win_rate']:>9.3f}{r['draw_rate']:>8.3f}"
            f"{r['team0_score']:>10.3f}{r['stderr']:>8.3f}{r['mean_episode_len']:>9.0f}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"driver": args.driver, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
