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
import traceback

from web.bridge import PygameWebBridge
from game_coordinator import GameCoordinator
from agents.agents import create_agent
from env2.adapter import tensor_state_to_render_state
from env2.state import TensorState

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

def get_agent_config_helper(cfg, agent_type, agent_id="player", team_id=0):
    """Generates complete agent config for create_agent."""
    base_config = {
        "agent_id": agent_id,
        "team_id": team_id,
        "squad": list(range(cfg.environment.max_ships)),
        "max_ships": cfg.environment.max_ships,
        "world_size": cfg.environment.world_size,
    }
    
    # Add ScriptedAgent specific params
    if agent_type in ["scripted", "random"]:
        scripted_cfg = cfg.agents.scripted.agent_config
        base_config.update({
            "max_shooting_range": scripted_cfg.max_shooting_range,
            "angle_threshold": scripted_cfg.angle_threshold,
            "bullet_speed": scripted_cfg.bullet_speed,
            "target_radius": scripted_cfg.target_radius,
            "radius_multiplier": scripted_cfg.radius_multiplier,
        })
    
    # Add World Model specific params
    if "world_model" in agent_type or agent_type == "most_recent_world_model":
        wm_cfg = OmegaConf.to_container(cfg.world_model, resolve=True)
        # Map n_ships to max_ships if needed by WorldModelAgent
        if "n_ships" in wm_cfg:
            wm_cfg["max_ships"] = wm_cfg.pop("n_ships")
        # Ensure agent ID info is in the flattened config
        wm_cfg.update(base_config)
        return {"agent_type": agent_type, "agent_config": wm_cfg}
        
    return {"agent_type": agent_type, "agent_config": base_config}

def run_live_game(cfg, team1_type, team2_type):
    """Run a live game and render frames."""
    game_cfg = cfg.copy()
    OmegaConf.set_struct(game_cfg, False)
    game_cfg.team1 = team1_type
    game_cfg.team2 = team2_type
    game_cfg.human_player = False
    game_cfg.collect.render_mode = "human"
    
    game_cfg.agents.team1_agent = get_agent_config_helper(game_cfg, team1_type, "team1", 0)
    game_cfg.agents.team2_agent = get_agent_config_helper(game_cfg, team2_type, "team2", 1)
    game_cfg.collect.teams = ["team1_agent", "team2_agent"]
    
    try:
        coordinator = GameCoordinator(game_cfg)
        coordinator.reset(game_mode="nvn")
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        stop_button = st.button("Stop Simulation", key="stop_live")
        
        terminated = False
        while not terminated:
            if stop_button: break
            
            obs = coordinator.obs_history[-1]
            teams = coordinator._get_teams_from_obs(obs)
            actions = {}
            for tid, sids in teams.items():
                agent_name = game_cfg.collect.teams[tid]
                team_actions = coordinator.agents[agent_name](obs, sids)
                for sid in sids:
                    actions[sid] = team_actions[sid]
            
            obs, rewards, terminated, _, info = coordinator.env.step(actions=actions)
            coordinator.obs_history.append(obs)
            
            render_state = tensor_state_to_render_state(coordinator.env.env.state, coordinator.env.config, 0)
            coordinator.env.renderer.render(render_state)
            frame = PygameWebBridge.surface_to_array(coordinator.env.renderer.display_surface)
            frame_placeholder.image(frame, use_container_width=True)
            stats_placeholder.write(f"Steps: {len(coordinator.obs_history)} | Rewards: {rewards}")
            time.sleep(0.01)
        coordinator.close()
    except Exception as e:
        st.error(f"Live Game failed: {e}")
        st.code(traceback.format_exc())

def hdf5_viewer(cfg, file_path):
    """View an HDF5 dataset with a scrubber."""
    if not file_path:
        st.info("Please select a file to view.")
        return
        
    try:
        with h5py.File(file_path, "r") as f:
            if "position" not in f:
                st.error("Invalid HDF5 structure: 'position' dataset not found.")
                return
                
            total_steps = f["position"].shape[0]
            step = st.slider("Step Scrubber", 0, total_steps - 1, 0)
            
            game_cfg = cfg.copy()
            game_cfg.collect.render_mode = "human"
            coordinator = GameCoordinator(game_cfg)
            
            def get_tensor(key, step_idx):
                data = f[key][step_idx]
                if key in ["position", "velocity", "attitude"]:
                    return torch.complex(torch.tensor(data[..., 0]), torch.tensor(data[..., 1])).unsqueeze(0)
                return torch.tensor(data).unsqueeze(0)

            max_ships = f.attrs.get("max_ships", 8)
            mock_state = TensorState(
                step_count=torch.tensor([step]),
                ship_pos=get_tensor("position", step),
                ship_vel=get_tensor("velocity", step),
                ship_attitude=get_tensor("attitude", step),
                ship_ang_vel=get_tensor("ang_vel", step),
                ship_health=get_tensor("health", step).float(),
                ship_power=get_tensor("power", step).float(),
                ship_cooldown=torch.zeros((1, max_ships)),
                ship_team_id=get_tensor("team_ids", step).int(),
                ship_alive=(get_tensor("health", step) > 0),
                ship_is_shooting=get_tensor("is_shooting", step).bool(),
                bullet_pos=torch.zeros((1, max_ships, 10), dtype=torch.complex64),
                bullet_vel=torch.zeros((1, max_ships, 10), dtype=torch.complex64),
                bullet_time=torch.zeros((1, max_ships, 10)),
                bullet_active=torch.zeros((1, max_ships, 10), dtype=torch.bool),
                bullet_cursor=torch.zeros((1, max_ships), dtype=torch.long)
            )

            frame = render_from_state(coordinator, mock_state)
            st.image(frame, use_container_width=True)
            st.write(f"Step {step} of {total_steps} | Episode ID: {f['episode_ids'][step]}")
            coordinator.close()
    except Exception as e:
        st.error(f"HDF5 Viewer failed: {e}")
        st.code(traceback.format_exc())

def run_comparison(cfg, agent_a_type, agent_b_type):
    """Compare two agents in the same starting scenario."""
    game_cfg = cfg.copy()
    OmegaConf.set_struct(game_cfg, False)
    game_cfg.collect.render_mode = "human"
    seed = np.random.randint(0, 10000)
    st.write(f"Comparing {agent_a_type} vs {agent_b_type} (Seed: {seed})")
    col1, col2 = st.columns(2)
    
    def run_agent_rollout(placeholder, agent_type, team_id_val):
        current_cfg = game_cfg.copy()
        current_cfg.team1 = agent_type
        current_cfg.team2 = "scripted"
        current_cfg.collect.teams = ["team1_agent", "team2_agent"]
        
        current_cfg.agents.team1_agent = get_agent_config_helper(current_cfg, agent_type, "player", 0)
        current_cfg.agents.team2_agent = get_agent_config_helper(current_cfg, "scripted", "opp", 1)
        
        try:
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
                    for sid in sids: actions[sid] = team_actions[sid]
                
                obs, _, terminated, _, _ = coordinator.env.step(actions)
                coordinator.obs_history.append(obs)
                frame = render_from_state(coordinator, coordinator.env.env.state)
                placeholder.image(frame, caption=f"{agent_type} (Step {i})", use_container_width=True)
                if terminated: break
                time.sleep(0.01)
            coordinator.close()
        except Exception as e:
            st.error(f"Comparison failed for {agent_type}: {e}")
            st.code(traceback.format_exc())

    run_agent_rollout(col1.empty(), agent_a_type, 0)
    run_agent_rollout(col2.empty(), agent_b_type, 1)

def run_dreaming(cfg):
    """World Model Pure Dreaming Mode."""
    st.info("Visualizing world model internal rollouts (Dreaming)...")
    game_cfg = cfg.copy()
    game_cfg.collect.render_mode = "human"
    
    try:
        from agents.tokenizer import observation_to_tokens
        agent_cfg_dict = get_agent_config_helper(cfg, "most_recent_world_model")
        agent = create_agent(agent_cfg_dict["agent_type"], agent_cfg_dict["agent_config"])
        model = agent.model
        model.eval()
        
        coordinator = GameCoordinator(game_cfg)
        placeholder = st.empty()
        coordinator.reset(game_mode="nvn")
        obs = coordinator.obs_history[-1]
        
        for i in range(100):
            # 1. Get tokens
            model_device = next(model.parameters()).device
            tokens_raw = observation_to_tokens(obs, agent.team_id, agent.world_size)
            # tokens_raw: (1, num_ships, 15)
            
            # Robust ship count handling
            num_ships_in_obs = tokens_raw.shape[1]
            max_model_ships = model.ship_embed.num_embeddings
            n_to_use = min(num_ships_in_obs, agent.max_ships, max_model_ships)
            
            # Crop ships
            tokens = tokens_raw[:, :n_to_use, :].unsqueeze(1).to(model_device) # (1, 1, n_to_use, D)
            
            # 2. Mock previous action
            dummy_action = torch.zeros((1, 1, n_to_use, 3), dtype=torch.float32, device=model_device)
            
            with torch.no_grad():
                # 3. Predict
                team_ids = torch.full((1, n_to_use), agent.team_id, device=model_device, dtype=torch.long)
                # Forward Pass: States(B, T, N, D), Actions(B, T, N, 3), TeamIDs(B, N)
                pred_s, pred_a_logits, _ = model(tokens, dummy_action, team_ids)
            
            frame = render_from_state(coordinator, coordinator.env.env.state)
            placeholder.image(frame, caption=f"Dreaming Step {i} | Ships: {n_to_use}", use_container_width=True)
            
            # Step env normally to show movement
            teams = coordinator._get_teams_from_obs(obs)
            actions = {}
            for tid, sids in teams.items():
                if tid == 0:
                    team_actions = agent(obs, sids)
                else:
                    team_actions = coordinator.agents["scripted"](obs, sids)
                for sid in sids: actions[sid] = team_actions[sid]
            
            obs, _, terminated, _, _ = coordinator.env.step(actions)
            if terminated: break
            time.sleep(0.05)
        coordinator.close()
    except Exception as e:
        st.error(f"Dreaming failed: {e}")
        st.code(traceback.format_exc())

def main():
    st.title("🚀 Boost and Broadside Dashboard")
    cfg = get_config()
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
        if not file_options:
            st.warning("No HDF5 datasets found in 'data/' directory.")
        else:
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

if __name__ == "__main__":
    main()
