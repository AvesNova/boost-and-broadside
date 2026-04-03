"""elo_stats mode: run every agent pair simultaneously and compute ELO ratings.

Loads all checkpoints from a training run plus scripted and random agents,
distributes B parallel environments across all directed matchups, runs them
simultaneously, and reports per-agent ELO, win rates, and episode lengths.
"""

import sys
import time
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.constants import NUM_POWER_ACTIONS, NUM_SHOOT_ACTIONS, NUM_TURN_ACTIONS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.modes.agent_factory import ResolvedAgent
from boost_and_broadside.modes.collect import _obs_from_state


def find_run_dir(run_spec: str, checkpoint_dir: str) -> Path:
    """Return the checkpoint subdirectory for a run spec.

    Args:
        run_spec:       "latest" or a specific run name like "bright-cloud-219".
        checkpoint_dir: Root checkpoint directory.
    """
    root = Path(checkpoint_dir)
    if run_spec == "latest":
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if not subdirs:
            sys.exit(f"Error: no run directories found under '{checkpoint_dir}'.")
        # Pick the subdir whose newest .pt file is most recent
        def newest_pt_mtime(d: Path) -> float:
            pts = list(d.glob("*.pt"))
            return max(p.stat().st_mtime for p in pts) if pts else 0.0
        return max(subdirs, key=newest_pt_mtime)
    else:
        run_dir = root / run_spec
        if not run_dir.is_dir():
            sys.exit(f"Error: run directory not found: '{run_dir}'")
        return run_dir


def _load_checkpoint_agent(path: Path, model_config: ModelConfig, ship_config: ShipConfig, device: str) -> ResolvedAgent:
    """Load a .pt checkpoint and return a ResolvedAgent."""
    from boost_and_broadside.models.mvp.policy import MVPPolicy
    ckpt   = torch.load(str(path), map_location=device, weights_only=False)
    K      = ckpt["policy_state_dict"]["value_head.weight"].shape[0]
    policy = MVPPolicy(model_config, ship_config, num_value_components=K).to(device)
    result = policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    if result.missing_keys:
        print(f"    [warn] missing keys in {path.name}: {result.missing_keys}")
    policy.eval()
    return ResolvedAgent("policy", policy)


def run_elo_stats_mode(
    run_spec:       str,
    num_envs:       int,
    ship_config:    ShipConfig,
    env_config:     EnvConfig,
    model_config:   ModelConfig,
    device:         str,
    checkpoint_dir: str = "checkpoints",
    elo_k_factor:   float = 32.0,
) -> None:
    """Run all-vs-all parallel matchups and report ELO ratings.

    Args:
        run_spec:       "latest" or a specific run directory name.
        num_envs:       Total parallel environments (split across matchups).
        ship_config:    Physics constants.
        env_config:     Environment sizing.
        model_config:   Policy architecture.
        device:         Torch device string.
        checkpoint_dir: Root directory containing run subdirectories.
        elo_k_factor:   ELO K-factor (same as training default).
    """
    B   = num_envs
    N   = env_config.num_ships
    dev = torch.device(device)

    # ------------------------------------------------------------------ #
    # Step 1 — Discover and load agents                                   #
    # ------------------------------------------------------------------ #
    run_dir = find_run_dir(run_spec, checkpoint_dir)
    print(f"Run directory: {run_dir}")

    ckpt_paths = sorted(run_dir.glob("*.pt"), key=lambda p: p.name)
    if not ckpt_paths:
        sys.exit(f"Error: no .pt checkpoints found in '{run_dir}'.")

    agents: list[ResolvedAgent] = []
    labels: list[str] = []

    print(f"Loading {len(ckpt_paths)} checkpoint(s)...")
    for path in ckpt_paths:
        agents.append(_load_checkpoint_agent(path, model_config, ship_config, device))
        labels.append(path.stem)  # e.g. "step_000001966080"
        print(f"  {path.stem}")

    # Scripted and random come last — indices used below for win-rate queries
    scripted_agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
    agents.append(ResolvedAgent("scripted", scripted_agent))
    labels.append("scripted")
    scripted_idx = len(agents) - 1

    agents.append(ResolvedAgent("random", None))
    labels.append("random")
    random_idx = len(agents) - 1

    K = len(agents)
    print(f"Total agents: {K}  (checkpoints={K - 2}, scripted=1, random=1)")

    # ------------------------------------------------------------------ #
    # Step 2 — Matchup setup                                              #
    # ------------------------------------------------------------------ #
    # Directed pairs: agent i as team-0, agent j as team-1, for all i≠j
    matchups = [(i, j) for i in range(K) for j in range(K) if i != j]
    M = len(matchups)  # K*(K-1)
    print(f"Directed matchups: {M}  ({K}×{K-1})")

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
    for m_idx, (i, j) in enumerate(matchups):
        sz = matchup_sizes[m_idx]
        env_agent0_idx[offset:offset + sz]  = i
        env_agent1_idx[offset:offset + sz]  = j
        env_matchup_idx[offset:offset + sz] = m_idx
        offset += sz

    # Per-agent sorted env indices (for sliced forward passes and hidden state)
    active_envs: list[torch.Tensor] = []
    for a_idx in range(K):
        mask = (env_agent0_idx == a_idx) | (env_agent1_idx == a_idx)
        active_envs.append(mask.nonzero(as_tuple=True)[0])

    # ------------------------------------------------------------------ #
    # Step 3 — Initialize hidden states and environment                   #
    # ------------------------------------------------------------------ #
    for a_idx, agent in enumerate(agents):
        if agent.kind == "policy":
            B_a = active_envs[a_idx].shape[0]
            agent.hidden = agent.agent.initial_hidden(B_a, N, dev)

    env = TensorEnv(B, ship_config, env_config, dev)
    env.reset()

    finished        = torch.zeros(B,  dtype=torch.bool,  device=dev)
    ep_lengths      = torch.zeros(B,  dtype=torch.int64, device=dev)
    matchup_a_wins  = torch.zeros(M,  dtype=torch.float32, device=dev)
    matchup_b_wins  = torch.zeros(M,  dtype=torch.float32, device=dev)
    matchup_ties    = torch.zeros(M,  dtype=torch.float32, device=dev)

    # Preallocate reusable tensors
    all_acts = torch.zeros(K, B, N, 3, dtype=torch.int32, device=dev)
    arange_B = torch.arange(B, device=dev)

    total_steps = 0
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 4 — Main simulation loop                                       #
    # ------------------------------------------------------------------ #
    while not finished.all():
        state = env.state
        obs   = _obs_from_state(state, ship_config)

        # Compute each agent's actions for its active envs
        for a_idx, agent in enumerate(agents):
            active = active_envs[a_idx]
            B_a    = active.shape[0]

            if agent.kind == "random":
                all_acts[a_idx, active] = torch.stack([
                    torch.randint(0, NUM_POWER_ACTIONS, (B_a, N), device=dev),
                    torch.randint(0, NUM_TURN_ACTIONS,  (B_a, N), device=dev),
                    torch.randint(0, NUM_SHOOT_ACTIONS, (B_a, N), device=dev),
                ], dim=-1).int()

            elif agent.kind == "scripted":
                # Scripted is a cheap vectorized op — run on full state, fill all B
                with torch.no_grad():
                    all_acts[a_idx] = agent.agent.get_actions(state)

            else:  # policy
                obs_a = {k: v[active] for k, v in obs.items()}
                with torch.no_grad():
                    acts_a, _, _, agent.hidden = agent.agent.get_action_and_value(obs_a, agent.hidden)
                all_acts[a_idx, active] = acts_a.int()

        # Assemble: team-0 ships get env_agent0's actions, team-1 ships get env_agent1's
        team0_acts = all_acts[env_agent0_idx, arange_B]   # (B, N, 3)
        team1_acts = all_acts[env_agent1_idx, arange_B]   # (B, N, 3)
        action = torch.where(
            (state.ship_team_id == 0).unsqueeze(-1),
            team0_acts,
            team1_acts,
        )

        dones, truncated = env.step(action)
        done_any     = dones | truncated
        total_steps += B

        new_done = done_any & ~finished
        if new_done.any():
            ep_lengths[new_done] = env.state.step_count[new_done].long()

            alive = env.state.ship_alive
            team  = env.state.ship_team_id
            team0_alive = (alive & (team == 0)).any(dim=1)
            team1_alive = (alive & (team == 1)).any(dim=1)
            team0_won = new_done & team0_alive & ~team1_alive
            team1_won = new_done & team1_alive & ~team0_alive
            tied      = new_done & ~team0_won & ~team1_won

            # Scatter outcomes into per-matchup accumulators
            nd_idx = env_matchup_idx[new_done]
            matchup_a_wins.scatter_add_(0, nd_idx, team0_won[new_done].float())
            matchup_b_wins.scatter_add_(0, nd_idx, team1_won[new_done].float())
            matchup_ties.scatter_add_(  0, nd_idx, tied[new_done].float())

            finished |= new_done

        if done_any.any():
            env.reset_envs(done_any)
            for a_idx, agent in enumerate(agents):
                if agent.kind == "policy":
                    active      = active_envs[a_idx]
                    active_done = done_any[active]
                    if active_done.any():
                        agent.hidden = agent.agent.reset_hidden_for_envs(
                            agent.hidden, active_done, N,
                        )

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Step 5 — ELO computation (iterative convergence)                   #
    # ------------------------------------------------------------------ #
    elo = [1000.0] * K
    a_wins_cpu = matchup_a_wins.cpu().tolist()
    b_wins_cpu = matchup_b_wins.cpu().tolist()
    ties_cpu   = matchup_ties.cpu().tolist()

    # Precompute lookup: (i, j) -> matchup index
    matchup_lookup: dict[tuple[int, int], int] = {pair: m for m, pair in enumerate(matchups)}

    def _score_as_team0(i: int, j: int) -> float:
        """Win rate of i playing as team-0 against j (team-1)."""
        m = matchup_lookup[(i, j)]
        n = matchup_sizes[m]
        return (a_wins_cpu[m] + 0.5 * ties_cpu[m]) / n if n > 0 else 0.5

    def _win_rate_vs(a_idx: int, opp_idx: int) -> float | None:
        """Win rate of a_idx vs opp_idx, averaged over both role directions."""
        if a_idx == opp_idx or (a_idx, opp_idx) not in matchup_lookup:
            return None
        # Direction 1: a as team-0
        r0 = _score_as_team0(a_idx, opp_idx)
        # Direction 2: a as team-1 (opp as team-0); a's score = b_wins + 0.5*ties
        m2 = matchup_lookup[(opp_idx, a_idx)]
        n2 = matchup_sizes[m2]
        r1 = (b_wins_cpu[m2] + 0.5 * ties_cpu[m2]) / n2 if n2 > 0 else 0.5
        return (r0 + r1) / 2.0

    for _ in range(200):
        for m_idx, (i, j) in enumerate(matchups):
            n_games    = matchup_sizes[m_idx]
            win_rate_i = (a_wins_cpu[m_idx] + 0.5 * ties_cpu[m_idx]) / n_games
            expected_i = 1.0 / (1.0 + 10.0 ** ((elo[j] - elo[i]) / 400.0))
            delta      = elo_k_factor * (win_rate_i - expected_i)
            elo[i]    += delta
            elo[j]    -= delta

    # ------------------------------------------------------------------ #
    # Step 6 — Per-agent stats                                            #
    # ------------------------------------------------------------------ #

    # Identify special agents by label
    avg_idx = next((a for a, lb in enumerate(labels) if lb == "best_avg"), None)

    # Per-agent average episode length across all their active envs
    ep_lengths_cpu = ep_lengths.cpu()
    agent_ep_len = [
        float(ep_lengths_cpu[active_envs[a].cpu()].float().mean())
        for a in range(K)
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
        return (sum(as_t0) / len(as_t0)) - (sum(as_t1) / len(as_t1))

    # ------------------------------------------------------------------ #
    # Step 7 — Print report                                               #
    # ------------------------------------------------------------------ #
    sim_fps = 1.0 / ship_config.dt
    sps     = total_steps / elapsed

    # Sort agents by ELO descending for display
    order = sorted(range(K), key=lambda a: elo[a], reverse=True)

    label_w = max(len(lb) for lb in labels)
    has_avg = avg_idx is not None

    # Build header columns
    cols = [
        ("ELO",        6),
        ("vs random",  10),
        ("vs scripted", 12),
    ]
    if has_avg:
        cols.append(("vs avg", 8))
    cols += [
        ("role Δ",    8),
        ("avg ep len", 10),
    ]

    hdr_parts = "  ".join(f"{name:>{w}}" for name, w in cols)
    row_w     = label_w + 4 + sum(w + 2 for _, w in cols)
    w_total   = max(72, row_w)
    sep       = "─" * w_total

    print(f"\n{sep}")
    print(f"  ELO Stats: {run_dir.name}")
    print(f"  {K} agents  |  {B:,} total envs  |  {M} directed matchups  |  ~{B // M} envs/matchup")
    print(f"{sep}")
    print(f"  {'Agent':<{label_w}}  {hdr_parts}")
    print(f"  {'─' * (w_total - 4)}")

    def _pct(v: float | None) -> str:
        return f"{100 * v:.1f}%" if v is not None else "—"

    for a_idx in order:
        lb    = labels[a_idx]
        vr    = _win_rate_vs(a_idx, random_idx)
        vs    = _win_rate_vs(a_idx, scripted_idx)
        va    = _win_rate_vs(a_idx, avg_idx) if has_avg else None
        delta = _role_delta(a_idx)
        el    = agent_ep_len[a_idx]

        row = (
            f"  {lb:<{label_w}}"
            f"  {elo[a_idx]:>6.0f}"
            f"  {_pct(vr):>10}"
            f"  {_pct(vs):>12}"
        )
        if has_avg:
            row += f"  {_pct(va):>8}"
        delta_sign = "+" if delta >= 0 else ""
        row += f"  {delta_sign}{100 * delta:.1f}%{'':<4}  {el:>10.1f}"
        print(row)

    print(f"{sep}")
    print(f"  Wall time: {elapsed:.2f}s   |   {sps:,.0f} steps/s  ({sps / sim_fps:,.0f} sim-steps/s)")
    print(f"{sep}")

    # ------------------------------------------------------------------ #
    # Step 8 — Win-rate heatmap (tab-separated, copyable as CSV)         #
    # ------------------------------------------------------------------ #
    # Rows = team-0 agent, columns = team-1 agent, cell = team-0 win rate
    print(f"\n  Win-rate heatmap (row=team-0, col=team-1)  —  tab-separated")
    print(f"  Copy into a spreadsheet for colour formatting\n")

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
