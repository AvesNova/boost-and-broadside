import multiprocessing as mp
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.data_collector import DataCollector
from src.game_coordinator import GameCoordinator


def collect_worker(worker_id: int, cfg_dict: dict[str, Any]) -> None:
    """
    Worker function for collecting data in a separate process

    Args:
        worker_id: Unique identifier for this worker
        cfg_dict: Configuration dictionary (serialized)
    """
    cfg = OmegaConf.create(cfg_dict)
    
    print(f"Worker {worker_id}: Starting data collection...")

    collector = DataCollector(cfg, worker_id=worker_id)
    coordinator = GameCoordinator(cfg, render_mode=cfg.collect.render_mode)

    episodes_per_mode = cfg.collect.episodes_per_mode
    total_episodes = sum(episodes_per_mode.values())

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Worker {worker_id}: Collecting episodes...",
                total=total_episodes,
            )

            for game_mode, num_episodes in episodes_per_mode.items():
                for episode_idx in range(num_episodes):
                    coordinator.reset(game_mode=game_mode)
                    episode_sim_time, terminated = coordinator.step()

                    collector.add_episode(
                        tokens_team_0=coordinator.all_tokens[0],
                        tokens_team_1=coordinator.all_tokens[1],
                        actions=coordinator.all_actions,
                        rewards=coordinator.all_rewards,
                        sim_time=episode_sim_time,
                    )

                    progress.update(task, advance=1)

                    progress.console.print(
                        f"[green]Worker {worker_id}: Completed {game_mode} episode "
                        f"{episode_idx + 1}/{num_episodes} (sim_time: {episode_sim_time:.2f}s)"
                    )

    except KeyboardInterrupt:
        print(f"\nWorker {worker_id}: Data collection interrupted by user")
    except Exception as e:
        print(f"Worker {worker_id}: Error during collection: {e}")
        raise
    finally:
        collector.finalize()
        coordinator.close()
        print(f"Worker {worker_id}: Data collection complete")


def collect(cfg: DictConfig) -> None:
    """
    Collect training data from agent gameplay using parallel workers

    Args:
        cfg: Configuration dictionary
    """
    num_workers = cfg.collect.num_workers
    
    if num_workers == 1:
        print("Starting single-worker data collection...")
        collect_worker(0, OmegaConf.to_container(cfg, resolve=True))
    else:
        print(f"Starting parallel data collection with {num_workers} workers...")
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        with mp.Pool(processes=num_workers) as pool:
            try:
                pool.starmap(
                    collect_worker,
                    [(worker_id, cfg_dict) for worker_id in range(num_workers)],
                )
            except KeyboardInterrupt:
                print("\nTerminating all workers...")
                pool.terminate()
                pool.join()
                raise
            
        print(f"\nAll {num_workers} workers completed!")
