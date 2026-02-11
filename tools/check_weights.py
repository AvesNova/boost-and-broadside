import sys

from boost_and_broadside.utils.dataset_stats import calculate_action_counts, compute_class_weights, apply_turn_exceptions, normalize_weights
from boost_and_broadside.core.constants import PowerActions, TurnActions, ShootActions

def main():
    data_path = "data/bc_pretraining/aggregated_data.h5"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    print(f"Reading from {data_path}")
    counts = calculate_action_counts(data_path)
    
    # Compute and Normalize (mimic Trainer logic)
    w_power = compute_class_weights(counts["power"])
    w_power = normalize_weights(w_power, counts["power"])
    
    w_turn = apply_turn_exceptions(compute_class_weights(counts["turn"]))
    w_turn = normalize_weights(w_turn, counts["turn"])
    
    w_shoot = compute_class_weights(counts["shoot"])
    w_shoot = normalize_weights(w_shoot, counts["shoot"])
    
    print("\nPower Weights:")
    for i, w in enumerate(w_power):
        action_name = PowerActions(i).name
        count = counts["power"][i]
        print(f"  {action_name:<10} (Count: {count:>8}): {w:.4f}")
        
    print("\nTurn Weights:")
    for i, w in enumerate(w_turn):
        action_name = TurnActions(i).name
        count = counts["turn"][i]
        print(f"  {action_name:<15} (Count: {count:>8}): {w:.4f}")

    print("\nShoot Weights:")
    for i, w in enumerate(w_shoot):
        action_name = ShootActions(i).name
        count = counts["shoot"][i]
        print(f"  {action_name:<10} (Count: {count:>8}): {w:.4f}")

if __name__ == "__main__":
    main()
