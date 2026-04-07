import time
import torch
import numpy as np
import wandb
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from evotorch import Problem
from evotorch.algorithms import Cosyne
from evotorch.logging import StdOutLogger, WandbLogger

from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.env2.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.env2.agents.evolvable_agent import BatchedEvolvableAgent


class BoostAndBroadsideProblem(Problem):
    def __init__(self, cfg: DictConfig, base_env: TensorEnv):
        self.cfg = cfg
        self.base_env = base_env
        self.max_episode_steps = cfg.train.evolve.get("max_episode_steps", 1024)
        self.matches_per_eval = cfg.train.evolve.get("matches_per_eval", 3)
        self.shared_parameters = cfg.train.evolve.get("shared_parameters", False)

        self.agent_logic = BatchedEvolvableAgent(base_env.config)

        solution_length = 24 if self.shared_parameters else 4 * 24

        super().__init__(
            objective_sense="max",
            solution_length=solution_length,
            bounds=(0.0, 1.0),
            device=base_env.device,
        )

        # Indices for ramps
        self.ramp_indices = [
            (0, 1),  # max_health: low, high
            (4, 5),  # base_thrust: low, high
            (6, 7),  # normal_turn_drag_coeff: low, high
            (8, 9),  # normal_turn_lift_coeff: low, high
            (10, 11),  # sharp_turn_drag_coeff: low, high
            (12, 13),  # sharp_turn_lift_coeff: low, high
            (14, 15),  # bullet_speed: low, high
            (17, 18),  # bullet_damage: low, high
            (19, 20),  # bullet_lifetime: low, high
        ]

        # Baseline agent config (untouched)
        default_v = torch.tensor(
            StochasticAgentConfig.default_vector(),
            device=self.device,
            dtype=torch.float32,
        )
        self.baseline_agent = default_v.view(1, 1, 24).repeat(1, 4, 1)[0]

    def _sort_ramps(self, pop: torch.Tensor) -> torch.Tensor:
        """Ensures that logically chained parameters (min_bound, max_bound) are sorted."""
        for idx1, idx2 in self.ramp_indices:
            pair = pop[..., [idx1, idx2]]
            sorted_pair, _ = torch.sort(pair, dim=-1)
            pop[..., idx1] = sorted_pair[..., 0]
            pop[..., idx2] = sorted_pair[..., 1]
        return pop

    def _evaluate_batch(self, solutions):
        values = solutions.values
        pop_size = values.shape[0]

        # EvoTorch tensors are ReadOnlyTensors, so we must clone to sort
        if self.shared_parameters:
            configs = values.clone().view(pop_size, 24)
        else:
            configs = values.clone().view(pop_size, 4, 24)

        configs = torch.clamp(configs, 0.0, 1.0)
        configs = self._sort_ramps(configs)

        # Override values in EvoTorch batch directly to enforce bounds in population
        solutions.access_values(keep_evals=True)[:] = configs.view(pop_size, -1)

        if self.shared_parameters:
            configs = configs.unsqueeze(1).repeat(1, 4, 1)
        # Evaluation Environment (Team 1 vs Baseline Team 0)
        total_envs = pop_size * self.matches_per_eval
        eval_env = TensorEnv(
            num_envs=total_envs,
            config=self.base_env.config,
            device=self.device,
            max_ships=self.base_env.max_ships,
            max_bullets=getattr(self.base_env, "max_bullets", 20),
            max_episode_steps=self.max_episode_steps,
        )

        env_wins_team_0 = torch.zeros(total_envs, device=self.device)

        obs = eval_env.reset()

        config_tensor = torch.zeros(
            total_envs, eval_env.max_ships, 24, device=self.device
        )
        team_ids = eval_env.state.ship_team_id

        # Enforce exactly 1v1 team setup per env (4 ships vs 4 ships)
        t0_mask = team_ids == 0  # shape (B, 8)
        t1_mask = team_ids == 1  # shape (B, 8)

        # configs shape: (pop_size, 4, 24)
        configs_repeated = configs.repeat_interleave(self.matches_per_eval, dim=0)
        config_tensor[t0_mask] = configs_repeated.reshape(-1, 24)

        baseline_repeated = self.baseline_agent.unsqueeze(0).repeat(total_envs, 1, 1)
        config_tensor[t1_mask] = baseline_repeated.reshape(-1, 24)

        step = 0
        alive = torch.ones(total_envs, device=self.device, dtype=torch.bool)

        while alive.any() and step < self.max_episode_steps:
            actions = self.agent_logic.get_actions(eval_env.state, config_tensor)
            _, _, terminated, truncated, info = eval_env.step(actions)

            dones = terminated | truncated
            just_finished = alive & dones

            if just_finished.any():
                final_alive = info["final_observation"]["alive"]
                t0_ships_alive = (final_alive & t0_mask).sum(dim=1)
                t1_ships_alive = (final_alive & t1_mask).sum(dim=1)

                # Compare surviving ships for environments that just finished
                t0_won = just_finished & (t0_ships_alive > t1_ships_alive)
                t1_won = just_finished & (t1_ships_alive > t0_ships_alive)
                draws = just_finished & (t0_ships_alive == t1_ships_alive)

                env_wins_team_0 += t0_won.float() * 1.0
                env_wins_team_0 += draws.float() * 0.5
                env_wins_team_0 += t1_won.float() * 0.0

                # Mark these environments as no longer alive so they aren't double-counted
                alive[just_finished] = False
            step += 1

        # Write fitness back as an average over matches
        fitness = env_wins_team_0.view(pop_size, self.matches_per_eval).mean(dim=1)
        solutions.set_evals(fitness)


class SPSLogger(StdOutLogger):
    def __init__(self, searcher, popsize, max_episode_steps, matches_per_eval):
        super().__init__(searcher)
        self.popsize = popsize
        self.max_episode_steps = max_episode_steps
        self.matches_per_eval = matches_per_eval
        self.start_t = time.time()

    def _log(self, status: dict):
        step = status.get("iter", 1)
        # Calculate theoretical maximum SPS if all matches went to maximum length
        elapsed = time.time() - self.start_t
        sps = (self.popsize * self.max_episode_steps * self.matches_per_eval) / (
            elapsed + 1e-8
        )

        status["sps"] = int(sps)

        # Format single-line output
        mean_eval = status.get("mean_eval", 0.0)
        median_eval = status.get("median_eval", 0.0)
        best_eval = status.get("pop_best_eval", 0.0)
        print(
            f"Gen {step:04d} | Max Fit: {best_eval:.3f} | Mean Fit: {mean_eval:.3f} | Median Fit: {median_eval:.3f} | Max Theoretical SPS: {int(sps)}"
        )

        if wandb.run is not None:
            wandb.log({"charts/sps": sps}, step=step)

        self.start_t = time.time()


class EvoTrainer:
    def __init__(self, cfg: DictConfig, env: TensorEnv):
        self.cfg = cfg
        self.env = env
        self.device = env.device

        self.popsize = cfg.train.evolve.get("pop_size", 512)
        self.num_generations = cfg.train.evolve.get("num_generations", 100)
        self.mutation_stdev = cfg.train.evolve.get("mutation_std", 0.05)
        self.num_elites = cfg.train.evolve.get("num_elites", 64)
        self.matches_per_eval = cfg.train.evolve.get("matches_per_eval", 3)

    def train(self):
        print("=== Starting EvoTorch GPU-Batched Pipeline (Cosyne) ===")

        problem = BoostAndBroadsideProblem(self.cfg, self.env)

        # If float between 0 and 1, treated as ratio. If > 1, integer elites count.
        elites = self.num_elites / self.popsize if self.num_elites else 0.1

        searcher = Cosyne(
            problem,
            popsize=self.popsize,
            tournament_size=self.popsize // 4,
            mutation_stdev=self.mutation_stdev,
            num_elites=elites,
        )

        # Seed the first configuration in the population as our baseline default agent!
        if problem.shared_parameters:
            searcher.population.access_values(keep_evals=True)[0] = (
                problem.baseline_agent[0].reshape(-1)
            )
        else:
            searcher.population.access_values(keep_evals=True)[0] = (
                problem.baseline_agent.reshape(-1)
            )

        _ = SPSLogger(
            searcher, self.popsize, problem.max_episode_steps, self.matches_per_eval
        )
        if self.cfg.wandb.enabled:
            _ = WandbLogger(searcher, project=self.cfg.wandb.project)

        searcher.run(self.num_generations)
        print("Training Complete!")

        # Save best configuration
        import os

        best_solution = searcher.status.get("pop_best")
        if best_solution is not None:
            best_values = best_solution.values.cpu().numpy()
            os.makedirs("data/evolved_agents", exist_ok=True)
            save_path = "data/evolved_agents/best_stochastic_agent.npy"
            np.save(save_path, best_values)
            print(f"Saved best configuration to {save_path}")

            if self.cfg.wandb.enabled and wandb.run is not None:
                wandb.save(save_path)
