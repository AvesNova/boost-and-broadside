
import csv
import torch
import logging
from train.data_loader import get_latest_data_path
from utils.dataset_stats import calculate_action_stats
from env.constants import PowerActions, TurnActions, ShootActions

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def export_stats_csv(output_path="data/dataset_stats.csv"):
    data_path = get_latest_data_path()
    log.info(f"Loading stats from: {data_path}")
    
    # Get RAW weights for frequency calculation
    raw_weights = calculate_action_stats(data_path, max_weight=float('inf'))
    
    # Get Effective weights (clamped)
    # Default max_weight=100.0 from function signature
    clamped_weights = calculate_action_stats(data_path) 
    
    rows = []
    
    # Helper to format
    def add_rows(category, enum_cls, raw_w_tensor, clamped_w_tensor):
        n_classes = len(raw_w_tensor)
        for i in range(n_classes):
            raw_w = raw_w_tensor[i].item()
            clamped_w = clamped_w_tensor[i].item()
            
            # Freq = 1 / (N * Weight)
            freq = 1.0 / (n_classes * raw_w) if raw_w > 0 else 0.0
            
            # Name
            try:
                name = enum_cls(i).name
            except ValueError:
                name = f"Index {i}"
                
            rows.append({
                "Category": category,
                "Action Index": i,
                "Name": name,
                "Frequency (%)": f"{freq*100:.2f}%",
                "Raw Weight": f"{raw_w:.4f}",
                "Effective Weight": f"{clamped_w:.4f}"
            })

    add_rows("Power", PowerActions, raw_weights["power"], clamped_weights["power"])
    add_rows("Turn", TurnActions, raw_weights["turn"], clamped_weights["turn"])
    add_rows("Shoot", ShootActions, raw_weights["shoot"], clamped_weights["shoot"])
    
    # Write CSV
    with open(output_path, "w", newline='') as f:
        fieldnames = ["Category", "Action Index", "Name", "Frequency (%)", "Raw Weight", "Effective Weight"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
        
    log.info(f"Exported stats to {output_path}")
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    export_stats_csv()
