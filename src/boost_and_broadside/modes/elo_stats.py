"""elo_stats mode: run every agent pair simultaneously and compute Elo ratings.

Loads all checkpoints from a training run plus scripted and random agents,
distributes B parallel environments across all directed matchups, runs them
simultaneously, and reports per-agent Elo, win rates, and episode lengths.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.modes.agent_factory import ResolvedAgent, resolve_agent_spec
from boost_and_broadside.modes.match import MatchRunner
from boost_and_broadside.train.rl.elo_eval import expected_score
from boost_and_broadside.train.rl.policy_io import load_policy_bundle

# All scripted agents, in display order. "scripted" (stochastic) is kept first
# so scripted_idx == num_checkpoints regardless of list length.
SCRIPTED_SPECS = [
    "scripted",
    "scripted_team",
    "jouster",
    "team_jouster",
    "reverse_turret",
    "boom_zoom",
    "abreast",
    "run_away",
    "spiral_evader",
    "jinking",
]


def find_run_dir(run_spec: str, checkpoint_dir: str) -> Path:
    """Return the checkpoint subdirectory for a run spec."""
    root = Path(checkpoint_dir)
    if run_spec == "latest":
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if not subdirs:
            sys.exit(f"Error: no run directories found under '{checkpoint_dir}'.")

        def newest_pt_mtime(d: Path) -> float:
            pts = list(d.glob("*.pt"))
            return max(p.stat().st_mtime for p in pts) if pts else 0.0

        return max(subdirs, key=newest_pt_mtime)
    else:
        run_dir = root / run_spec
        if not run_dir.is_dir():
            sys.exit(f"Error: run directory not found: '{run_dir}'")
        return run_dir


def _load_checkpoint_agent(
    path: Path, model_config: ModelConfig, ship_config: ShipConfig, num_ships: int, device: str
) -> ResolvedAgent:
    """Load a .pt checkpoint and return a ResolvedAgent.

    The field can span runs, so each checkpoint is rebuilt from the configs it
    recorded rather than from the ones this invocation happens to be running.
    """
    bundle = load_policy_bundle(
        str(path),
        device=device,
        num_ships=num_ships,
        ship_config=ship_config,
        model_config=model_config,
    )
    return ResolvedAgent("policy", bundle.policy, bundle=bundle)


def run_elo_stats_mode(
    run_spec: str,
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    elo_k_factor: float = 32.0,
    matchups: list[str] | None = None,
    custom_agents: list[str] | None = None,
) -> None:
    """Run all-vs-all parallel matchups and report Elo ratings."""

    if not matchups:
        matchups = ["2v2"]

    for matchup in matchups:
        parts = matchup.split("v")
        if len(parts) != 2:
            print(f"Skipping invalid matchup: {matchup}")
            continue
        n0, n1 = int(parts[0]), int(parts[1])
        N = n0 + n1
        curr_env_config = replace(env_config, num_ships=N)
        dev = torch.device(device)
        B = num_envs

        print(f"\n{'=' * 60}")
        print(f"=== Elo Matchup: {matchup} (Team0: {n0}, Team1: {n1}) ===")
        print(f"{'=' * 60}")

        # ------------------------------------------------------------------ #
        # Step 1 — Discover and load agents                                   #
        # ------------------------------------------------------------------ #
        agents: list[ResolvedAgent] = []
        labels: list[str] = []
        num_checkpoints = 0
        run_dir: Path | None = None

        if custom_agents:
            print(f"Loading custom agents: {custom_agents}")
            for spec in custom_agents:
                agents.append(
                    resolve_agent_spec(
                        spec, ship_config, model_config, device, checkpoint_dir, num_ships=N
                    )
                )
                labels.append(Path(spec).stem if ".pt" in spec else spec)

            scripted_idx = labels.index("scripted") if "scripted" in labels else -1
            random_idx = labels.index("random") if "random" in labels else -1
        else:
            if run_spec != "none":
                run_dir = find_run_dir(run_spec, checkpoint_dir)
                print(f"Run directory: {run_dir}")
                ckpt_paths = sorted(run_dir.glob("*.pt"), key=lambda p: p.name)
                if not ckpt_paths:
                    sys.exit(f"Error: no .pt checkpoints found in '{run_dir}'.")
                print(f"Loading {len(ckpt_paths)} checkpoint(s)...")
                for path in ckpt_paths:
                    agents.append(
                        _load_checkpoint_agent(path, model_config, ship_config, N, device)
                    )
                    labels.append(path.stem)
                    print(f"  {path.stem}")
                num_checkpoints = len(ckpt_paths)

            # All scripted agents — "scripted" (stochastic) is always index num_checkpoints
            for spec in SCRIPTED_SPECS:
                agents.append(
                    resolve_agent_spec(spec, ship_config, model_config, device, num_ships=N)
                )
                labels.append(spec)
            scripted_idx = num_checkpoints  # index of the stochastic scripted agent

            agents.append(ResolvedAgent("random", None))
            labels.append("random")
            random_idx = len(agents) - 1

        K = len(agents)
        num_scripted = len(SCRIPTED_SPECS) if not custom_agents else 0
        num_random = 1 if not custom_agents else 0
        print(
            f"Total agents: {K}  "
            f"(checkpoints={num_checkpoints}, scripted={num_scripted}, random={num_random})"
        )

        # ------------------------------------------------------------------ #
        # Step 2 — Matchup setup                                              #
        # ------------------------------------------------------------------ #
        # Directed pairs: agent i as team-0, agent j as team-1, for all i≠j
        matchups_pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
        M = len(matchups_pairs)  # K*(K-1)
        print(f"Directed matchups: {M}  ({K}×{K - 1})")

        if B < M:
            sys.exit(f"Error: num_envs ({B}) < num_matchups ({M}). Increase --num_envs.")

        # Distribute envs evenly; first (B % M) matchups get one extra env
        base, rem = divmod(B, M)
        matchup_sizes = [base + (1 if m < rem else 0) for m in range(M)]

        # Build (B,) tensors: which agent controls team-0/team-1 in each env
        env_agent0_idx = torch.empty(B, dtype=torch.long, device=dev)
        env_agent1_idx = torch.empty(B, dtype=torch.long, device=dev)
        env_matchup_idx = torch.empty(B, dtype=torch.long, device=dev)

        offset = 0
        for m_idx, (i, j) in enumerate(matchups_pairs):
            sz = matchup_sizes[m_idx]
            env_agent0_idx[offset : offset + sz] = i
            env_agent1_idx[offset : offset + sz] = j
            env_matchup_idx[offset : offset + sz] = m_idx
            offset += sz

        # Per-agent sorted env indices (for sliced forward passes and hidden state)
        active_envs: list[torch.Tensor] = []
        for a_idx in range(K):
            mask = (env_agent0_idx == a_idx) | (env_agent1_idx == a_idx)
            active_envs.append(mask.nonzero(as_tuple=True)[0])

        # ------------------------------------------------------------------ #
        # Step 3 — Initialize hidden states and environment                   #
        # ------------------------------------------------------------------ #
        env = TensorEnv(B, ship_config, curr_env_config, dev)
        runner = MatchRunner(
            env,
            agents,
            team0_index=env_agent0_idx,
            team1_index=env_agent1_idx,
            ship_config=ship_config,
            num_ships=N,
        )
        runner.init_hidden()
        env.reset(options={"team_sizes": (n0, n1)})

        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        ep_lengths = torch.zeros(B, dtype=torch.int64, device=dev)
        matchup_a_wins = torch.zeros(M, dtype=torch.float32, device=dev)
        matchup_b_wins = torch.zeros(M, dtype=torch.float32, device=dev)
        matchup_ties = torch.zeros(M, dtype=torch.float32, device=dev)

        total_steps = 0
        t0 = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Step 4 — Main simulation loop                                       #
        # ------------------------------------------------------------------ #
        while not finished.all():
            dones, truncated = runner.step()
            done_any = dones | truncated
            total_steps += B

            new_done = done_any & ~finished
            if new_done.any():
                ep_lengths[new_done] = env.state.step_count[new_done].long()

                alive = env.state.ship_alive
                team = env.state.ship_team_id
                team0_alive = (alive & (team == 0)).any(dim=1)
                team1_alive = (alive & (team == 1)).any(dim=1)
                team0_won = new_done & team0_alive & ~team1_alive
                team1_won = new_done & team1_alive & ~team0_alive
                tied = new_done & ~team0_won & ~team1_won

                # Scatter outcomes into per-matchup accumulators
                nd_idx = env_matchup_idx[new_done]
                matchup_a_wins.scatter_add_(0, nd_idx, team0_won[new_done].float())
                matchup_b_wins.scatter_add_(0, nd_idx, team1_won[new_done].float())
                matchup_ties.scatter_add_(0, nd_idx, tied[new_done].float())

                finished |= new_done

            runner.reset_finished(done_any, options={"team_sizes": (n0, n1)})

        elapsed = time.perf_counter() - t0

        # ------------------------------------------------------------------ #
        # Step 5 — Elo computation (iterative convergence)                   #
        # ------------------------------------------------------------------ #
        elo = [0.0] * K
        a_wins_cpu = matchup_a_wins.cpu().tolist()
        b_wins_cpu = matchup_b_wins.cpu().tolist()
        ties_cpu = matchup_ties.cpu().tolist()

        # Precompute lookup: (i, j) -> matchup index
        matchup_lookup: dict[tuple[int, int], int] = {
            pair: m for m, pair in enumerate(matchups_pairs)
        }

        def _score_as_team0(i: int, j: int) -> float:
            """Win rate of i playing as team-0 against j (team-1)."""
            m = matchup_lookup[(i, j)]
            n = matchup_sizes[m]
            return (a_wins_cpu[m] + 0.5 * ties_cpu[m]) / n if n > 0 else 0.5

        def _win_rate_vs(a_idx: int, opp_idx: int) -> float | None:
            """Win rate of a_idx vs opp_idx, averaged over both role directions."""
            if opp_idx < 0 or a_idx == opp_idx or (a_idx, opp_idx) not in matchup_lookup:
                return None
            # Direction 1: a as team-0
            r0 = _score_as_team0(a_idx, opp_idx)
            # Direction 2: a as team-1 (opp as team-0); a's score = b_wins + 0.5*ties
            m2 = matchup_lookup[(opp_idx, a_idx)]
            n2 = matchup_sizes[m2]
            r1 = (b_wins_cpu[m2] + 0.5 * ties_cpu[m2]) / n2 if n2 > 0 else 0.5
            return (r0 + r1) / 2.0

        for _ in range(200):
            for m_idx, (i, j) in enumerate(matchups_pairs):
                n_games = matchup_sizes[m_idx]
                win_rate_i = (a_wins_cpu[m_idx] + 0.5 * ties_cpu[m_idx]) / n_games
                expected_i = expected_score(elo[i], elo[j])
                delta = elo_k_factor * (win_rate_i - expected_i)
                elo[i] += delta
                elo[j] -= delta

        # ------------------------------------------------------------------ #
        # Step 6 — Per-agent stats                                            #
        # ------------------------------------------------------------------ #

        # Identify special agents by label
        avg_idx = next((a for a, lb in enumerate(labels) if lb == "best_avg"), None)

        # Per-agent average episode length across all their active envs
        ep_lengths_cpu = ep_lengths.cpu()
        agent_ep_len = [
            float(ep_lengths_cpu[active_envs[a].cpu()].float().mean()) for a in range(K)
        ]

        # Role delta: avg win rate as team-0 minus avg win rate as team-1
        # (positive = better when controlling team-0 ships)
        def _role_delta(a_idx: int) -> float:
            as_t0 = [_score_as_team0(a_idx, j) for j in range(K) if j != a_idx]
            as_t1 = []
            for i in range(K):
                if i == a_idx:
                    continue
                m = matchup_lookup[(i, a_idx)]
                n = matchup_sizes[m]
                as_t1.append((b_wins_cpu[m] + 0.5 * ties_cpu[m]) / n if n > 0 else 0.5)
            if not as_t0 or not as_t1:
                return 0.0
            return (sum(as_t0) / len(as_t0)) - (sum(as_t1) / len(as_t1))

        # ------------------------------------------------------------------ #
        # Step 7 — Print report                                               #
        # ------------------------------------------------------------------ #
        sim_fps = 1.0 / ship_config.dt
        sps = total_steps / elapsed

        # Sort agents by Elo descending for display
        order = sorted(range(K), key=lambda a: elo[a], reverse=True)

        label_w = max(len(lb) for lb in labels)
        has_avg = avg_idx is not None

        # Build header columns
        cols = [
            ("Elo", 6),
        ]
        if random_idx >= 0:
            cols.append(("vs random", 10))
        if scripted_idx >= 0:
            cols.append(("vs scripted", 12))

        if has_avg:
            cols.append(("vs avg", 8))
        cols += [
            ("role Δ", 8),
            ("avg ep len", 10),
        ]

        hdr_parts = "  ".join(f"{name:>{w}}" for name, w in cols)
        row_w = label_w + 4 + sum(w + 2 for _, w in cols)
        w_total = max(72, row_w)
        sep = "─" * w_total

        title = (
            run_dir.name
            if run_dir is not None
            else "custom agents"
            if custom_agents
            else "scripted-only"
        )
        print(f"\n{sep}")
        print(f"  Elo Stats: {title}")
        envs_per_matchup = B // M if M > 0 else B
        print(
            f"  {K} agents  |  {B:,} total envs  |  {M} directed matchups  |  "
            f"~{envs_per_matchup} envs/matchup"
        )
        print(f"{sep}")
        print(f"  {'Agent':<{label_w}}  {hdr_parts}")
        print(f"  {'─' * (w_total - 4)}")

        def _pct(v: float | None) -> str:
            return f"{100 * v:.1f}%" if v is not None else "—"

        for a_idx in order:
            lb = labels[a_idx]
            vr = _win_rate_vs(a_idx, random_idx) if random_idx >= 0 else None
            vs = _win_rate_vs(a_idx, scripted_idx) if scripted_idx >= 0 else None
            va = _win_rate_vs(a_idx, avg_idx) if has_avg else None
            delta = _role_delta(a_idx)
            el = agent_ep_len[a_idx]

            row = f"  {lb:<{label_w}}  {elo[a_idx]:>6.0f}"
            if random_idx >= 0:
                row += f"  {_pct(vr):>10}"
            if scripted_idx >= 0:
                row += f"  {_pct(vs):>12}"

            if has_avg:
                row += f"  {_pct(va):>8}"
            delta_sign = "+" if delta >= 0 else ""
            row += f"  {delta_sign}{100 * delta:.1f}%{'':<4}  {el:>10.1f}"
            print(row)

        print(f"{sep}")
        print(
            f"  Wall time: {elapsed:.2f}s   |   {sps:,.0f} steps/s  "
            f"({sps / sim_fps:,.0f} sim-steps/s)"
        )
        print(f"{sep}")

        # ------------------------------------------------------------------ #
        # Step 8 — Win-rate heatmap (tab-separated, copyable as CSV)         #
        # ------------------------------------------------------------------ #
        # Rows = team-0 agent, columns = team-1 agent, cell = team-0 win rate
        print("\n  Win-rate heatmap (row=team-0, col=team-1)  —  tab-separated")
        print("  Copy into a spreadsheet for colour formatting\n")

        short = [lb[:16] for lb in labels]  # truncate for readability

        # Header row
        print("\t" + "\t".join(short[j] for j in order))
        for i in order:
            cells = []
            for j in order:
                if i == j:
                    cells.append("—")
                else:
                    cells.append(f"{100 * _score_as_team0(i, j):.1f}%")
            print(short[i] + "\t" + "\t".join(cells))
        print()
