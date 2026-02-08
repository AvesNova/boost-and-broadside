import streamlit as st
import torch
import numpy as np
import h5py
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize
import time
import cv2

from web.bridge import PygameWebBridge
from game_coordinator import GameCoordinator
from agents.agents import create_agent
from env2.adapter import tensor_state_to_render_state

# Page config
st.set_page_config(
    page_title="Boost and Broadside Dashboard",
    page_icon="🚀",
    layout="wide",
)

# Premium Style
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #464b5d;
    }
    .stButton>button:hover {
        background-color: #3b3e4a;
        border-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Headless setup for Pygame
PygameWebBridge.setup_headless()

@st.cache_resource
def get_config():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    # Resolve for streamlit compatibility
    return cfg

def list_hdf5_files():
    data_dir = Path("data")
    return list(data_dir.rglob("*.h5"))

def render_from_state(coordinator, tensor_state):
    """Utility to render a state to a frame."""
    render_state = tensor_state_to_render_state(tensor_state, coordinator.env.config, 0)
    coordinator.env.renderer.render(render_state)
    return PygameWebBridge.surface_to_array(coordinator.env.renderer.display_surface)

def run_live_game(cfg, team1_type, team2_type):
    """Run a live game and render frames."""
    # Setup agents in config override
    game_cfg = cfg.copy()
    OmegaConf.set_struct(game_cfg, False)
    game_cfg.team1 = team1_type
    game_cfg.team2 = team2_type
    game_cfg.human_player = False
    game_cfg.collect.render_mode = "human" # Trigger renderer creation
    
    # Helper to create agent config (copied logic from play.py)
    def get_agent_config(agent_type: str) -> dict:
        transformer_config = OmegaConf.to_container(
            game_cfg.train.model.transformer, resolve=True
        )
        if isinstance(transformer_config, dict) and "num_actions" in transformer_config:
            del transformer_config["num_actions"]
        base_config = {
            "agent_id": "player",
            "team_id": 0,
            "squad": [],
            **transformer_config,
        }
        return {"agent_type": agent_type, "agent_config": base_config}

    for t_idx, t_type in enumerate([team1_type, team2_type]):
        key = f"team{t_idx+1}_agent"
        if t_type in game_cfg.agents:
            game_cfg.agents[key] = game_cfg.agents[t_type]
        else:
            game_cfg.agents[key] = get_agent_config(t_type)
    
    game_cfg.collect.teams = ["team1_agent", "team2_agent"]
    
    coordinator = GameCoordinator(game_cfg)
    coordinator.reset(game_mode="nvn")
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    stop_button = st.button("Stop Simulation")
    
    terminated = False
    while not terminated:
        if stop_button:
            break
            
        # Step
        # Manual step loop to capture frames
        teams = coordinator._get_teams_from_obs(coordinator.obs_history[-1])
        team_actions = {
            team_id: coordinator.agents[game_cfg.collect.teams[team_id]](
                coordinator.obs_history[-1], ship_ids
            )
            for team_id, ship_ids in teams.items()
        }
        
        actions = {}
        for team_id, ship_ids in teams.items():
            for ship_id in ship_ids:
                actions[ship_id] = team_actions[team_id][ship_id]
        
        obs, rewards, terminated, _, info = coordinator.env.step(actions=actions)
        coordinator.obs_history.append(obs)
        
        # Render frame
        # coordinator.env.renderer is lazy-loaded
        render_state = tensor_state_to_render_state(coordinator.env.env.state, coordinator.env.config, 0)
        coordinator.env.renderer.render(render_state)
        
        # Capture frame
        frame = PygameWebBridge.surface_to_array(coordinator.env.renderer.display_surface)
        frame_placeholder.image(frame, use_container_width=True)
        
        # Update stats
        stats_placeholder.write(f"Steps: {len(coordinator.obs_history)} | Rewards: {rewards}")
        
        time.sleep(0.01) # Small delay for UI smoothness

    coordinator.close()

def hdf5_viewer(cfg, file_path):
    """View an HDF5 dataset with a scrubber."""
    if not file_path:
        st.info("Please select a file to view.")
        return
        
    with h5py.File(file_path, "r") as f:
        if "position" not in f:
            st.error("Invalid HDF5 structure: 'position' dataset not found.")
            return
            
        total_steps = f["position"].shape[0]
        step = st.slider("Step Scrubber", 0, total_steps - 1, 0)
        
        # We need a coordinator to use its renderer and config
        game_cfg = cfg.copy()
        game_cfg.collect.render_mode = "human"
        coordinator = GameCoordinator(game_cfg)
        
        # Construct a TensorState-like object from HDF5 step
        from env2.state import TensorState
        
        def get_tensor(key, step_idx):
            data = f[key][step_idx]
            # Convert to torch and handle complex/shapes
            if key in ["position", "velocity", "attitude"]:
                # (MaxShips, 2) -> (MaxShips,) complex
                return torch.complex(torch.tensor(data[..., 0]), torch.tensor(data[..., 1])).unsqueeze(0)
            return torch.tensor(data).unsqueeze(0)

        mock_state = TensorState(
            num_envs=1,
            max_ships=f.attrs["max_ships"],
            device="cpu"
        )
        mock_state.ship_pos = get_tensor("position", step)
        mock_state.ship_vel = get_tensor("velocity", step)
        mock_state.ship_health = get_tensor("health", step)
        mock_state.ship_power = get_tensor("power", step)
        mock_state.ship_attitude = get_tensor("attitude", step)
        mock_state.ship_ang_vel = get_tensor("ang_vel", step)
        mock_state.ship_is_shooting = get_tensor("is_shooting", step)
        mock_state.ship_team_id = get_tensor("team_ids", step)
        mock_state.ship_alive = mock_state.ship_health > 0
        mock_state.step_count = torch.tensor([step])

        frame = render_from_state(coordinator, mock_state)
        st.image(frame, use_container_width=True)
        
        st.write(f"Step {step} of {total_steps} | Episode ID: {f['episode_ids'][step]}")
        coordinator.close()

def main():
    st.title("🚀 Boost and Broadside Dashboard")
    
    cfg = get_config()
    
    # Sidebar
    st.sidebar.title("Controls")
    mode = st.sidebar.selectbox("Mode", ["Live Game", "HDF5 Viewer", "Agent Comparison", "Dreaming"])
    
    if mode == "Live Game":
        st.sidebar.subheader("Team Settings")
        team1 = st.sidebar.selectbox("Team 1 Agent", ["scripted", "most_recent_world_model", "random", "human"], index=0)
        team2 = st.sidebar.selectbox("Team 2 Agent", ["most_recent_world_model", "scripted", "random", "human"], index=1)
        
        if st.sidebar.button("Start Game"):
            run_live_game(cfg, team1, team2)

    elif mode == "HDF5 Viewer":
        files = list_hdf5_files()
        file_options = {f.name: f for f in files}
        selected_file_name = st.sidebar.selectbox("Select Dataset", list(file_options.keys()))
        selected_file = file_options.get(selected_file_name)
        
        hdf5_viewer(cfg, selected_file)

    elif mode == "Agent Comparison":
        st.sidebar.subheader("Comparison Settings")
        agent_a = st.sidebar.selectbox("Agent A", ["scripted", "most_recent_world_model"])
        agent_b = st.sidebar.selectbox("Agent B", ["most_recent_world_model", "scripted"])
        
        if st.sidebar.button("Start Comparison"):
            run_comparison(cfg, agent_a, agent_b)

    elif mode == "Dreaming":
        if st.sidebar.button("Start Dreaming"):
            run_dreaming(cfg)

def run_comparison(cfg, agent_a_type, agent_b_type):
    """Compare two agents in the same starting scenario."""
    game_cfg = cfg.copy()
    OmegaConf.set_struct(game_cfg, False)
    game_cfg.collect.render_mode = "human"
    
    # Random seed for scenario
    seed = np.random.randint(0, 10000)
    
    st.write(f"Comparing {agent_a_type} vs {agent_b_type} (Seed: {seed})")
    col1, col2 = st.columns(2)
    
    def run_agent_rollout(placeholder, agent_type):
        current_cfg = game_cfg.copy()
        current_cfg.team1 = agent_type
        current_cfg.team2 = "scripted"
        current_cfg.collect.teams = ["team1_agent", "team2_agent"]
        
        # Setup team1_agent
        transformer_config = OmegaConf.to_container(current_cfg.train.model.transformer, resolve=True)
        if "num_actions" in transformer_config: del transformer_config["num_actions"]
        current_cfg.agents.team1_agent = {"agent_type": agent_type, "agent_config": {"agent_id": "player", "team_id": 0, **transformer_config}}
        current_cfg.agents.team2_agent = {"agent_type": "scripted", "agent_config": {"agent_id": "opp", "team_id": 1}}
        
        coordinator = GameCoordinator(current_cfg)
        torch.manual_seed(seed)
        np.random.seed(seed)
        coordinator.reset(game_mode="nvn")
        
        for i in range(200):
            obs = coordinator.obs_history[-1]
            teams = coordinator._get_teams_from_obs(obs)
            actions = {}
            for tid, sids in teams.items():
                agent_name = current_cfg.collect.teams[tid]
                team_actions = coordinator.agents[agent_name](obs, sids)
                for sid in sids:
                    actions[sid] = team_actions[sid]
            
            obs, _, terminated, _, _ = coordinator.env.step(actions)
            coordinator.obs_history.append(obs)
            
            frame = render_from_state(coordinator, coordinator.env.env.state)
            placeholder.image(frame, caption=f"{agent_type} (Step {i})", use_container_width=True)
            if terminated: break
            time.sleep(0.01)
        coordinator.close()

    p1 = col1.empty()
    p2 = col2.empty()
    
    # We run them sequentially for now in Streamlit to avoid thread/SDL issues
    run_agent_rollout(p1, agent_a_type)
    run_agent_rollout(p2, agent_b_type)

def run_dreaming(cfg):
    """World Model Pure Dreaming Mode."""
    st.info("Visualizing world model internal rollouts (Dreaming)...")
    game_cfg = cfg.copy()
    game_cfg.collect.render_mode = "human"
    coordinator = GameCoordinator(game_cfg)
    
    try:
        from agents.tokenizer import observation_to_tokens
        
        # Load agent
        agent = create_agent("most_recent_world_model", OmegaConf.to_container(cfg.world_model, resolve=True))
        model = agent.model
        model.eval()
        
        placeholder = st.empty()
        coordinator.reset(game_mode="nvn")
        
        # Initial state
        obs = coordinator.obs_history[-1]
        
        for i in range(100):
            # 1. Get tokens
            tokens = observation_to_tokens(obs, game_cfg.environment.max_ships).unsqueeze(0).to(model.device)
            # 2. Mock previous action
            prev_action = torch.zeros((1, 1, game_cfg.environment.max_ships, 3), dtype=torch.long, device=model.device)
            
            with torch.no_grad():
                # 3. Predict next state and actions
                # Note: This is a simplified "one-step dream" visualization
                # For "pure dreaming", we should iterate only on pred_state.
                pred_state, pred_actions, _, _ = model(
                    state=tokens, 
                    prev_action=prev_action,
                    pos=obs["position"].unsqueeze(0).unsqueeze(0).to(model.device),
                    vel=obs["velocity"].unsqueeze(0).unsqueeze(0).to(model.device)
                )
            
            # Rendering a 'dream' frame
            # This requires converting the world model's internal representation back to RenderState.
            # However, the world model predicts *deltas* of tokens.
            # Simplified: Use the current environment state for rendering but show the "dreams" in text or overlay.
            # Real Dreaming Visualization: The state should evolve ONLY via model.
            
            frame = render_from_state(coordinator, coordinator.env.env.state)
            placeholder.image(frame, caption=f"Dreaming Step {i}", use_container_width=True)
            
            # Step env normally in this demo to keep ships moving
            teams = coordinator._get_teams_from_obs(obs)
            actions = {sid: coordinator.agents["most_recent_world_model"](obs, [sid])[sid] for tid, sids in teams.items() for sid in sids}
            obs, _, terminated, _, _ = coordinator.env.step(actions)
            if terminated: break
            time.sleep(0.05)
            
    except Exception as e:
        st.error(f"Dreaming failed: {e}")
    finally:
        coordinator.close()

if __name__ == "__main__":
    main()
