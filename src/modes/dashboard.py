import subprocess
import os
from pathlib import Path

def dashboard(cfg) -> None:
    """Launch the Streamlit dashboard."""
    print("Launching Streamlit dashboard...")
    
    # Path to dashboard.py
    dashboard_path = Path(__file__).parent.parent / "web" / "dashboard.py"
    
    # Run streamlit
    cmd = ["streamlit", "run", str(dashboard_path)]
    
    # Set environment variables if needed
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path(__file__).parent.parent}:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
