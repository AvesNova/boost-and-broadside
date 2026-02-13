import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)

from boost_and_broadside.core.features import EGO_STATE_FIELDS, TARGET_FIELDS, RELATIONAL_FIELDS

class FeatureNormalizer(nn.Module):
    """
    Handles loading, storing, and applying normalization stats for Yemong models.
    Stats are registered as buffers so they are saved with the model state_dict.
    Vectorized for high performance.
    """
    def __init__(self, stats_csv: str | Path | None = None):
        super().__init__()
        if stats_csv:
            self.load_stats(stats_csv)

    def load_stats(self, stats_csv: str | Path):
        """Loads stats from a CSV file and creates vectorized normalization buffers."""
        df = pd.read_csv(stats_csv, sep='|', index_col=0)
        
        # 1. Load raw buffers (legacy support for individual access)
        for feature, row in df.iterrows():
            for stat in ['Mean', 'Std', 'RMS', 'Min', 'Max']:
                if stat in row:
                    name = f"{feature}_{stat.lower()}".replace('.', '_')
                    self.register_buffer(name, torch.tensor(float(row[stat]), dtype=torch.float32))

        # 2. Create Vectorized Group Buffers
        self._setup_group("ego", EGO_STATE_FIELDS)
        self._setup_group("target", TARGET_FIELDS)
        self._setup_group("relational", RELATIONAL_FIELDS)

    def _setup_group(self, group_name: str, fields: list[tuple[str, str]]):
        """Pre-calculates shift and scale vectors for a feature group and individual features."""
        shifts = []
        scales = []
        
        for feature, strategy in fields:
            shift = 0.0
            scale = 1.0
            
            try:
                if strategy == "Min-Max":
                    min_val = float(getattr(self, f"{feature}_min".replace('.', '_')))
                    max_val = float(getattr(self, f"{feature}_max".replace('.', '_')))
                    shift = -min_val
                    scale = 1.0 / (max_val - min_val + 1e-6)
                elif strategy == "Scale":
                    rms_val = float(getattr(self, f"{feature}_rms".replace('.', '_')))
                    shift = 0.0
                    scale = 1.0 / (rms_val + 1e-6)
                elif strategy == "Z-Score":
                    mean_val = float(getattr(self, f"{feature}_mean".replace('.', '_')))
                    std_val = float(getattr(self, f"{feature}_std".replace('.', '_')))
                    shift = -mean_val
                    scale = 1.0 / (std_val + 1e-6)
            except AttributeError:
                log.warning(f"Stat for {feature} not found, using identity for {group_name} group.")
                
            shifts.append(shift)
            scales.append(scale)
            
            # Pre-cache individual shift/scale for generic normalize()
            feat_id = feature.replace('.', '_')
            self.register_buffer(f"{feat_id}_shift_proc", torch.tensor(shift, dtype=torch.float32))
            self.register_buffer(f"{feat_id}_scale_proc", torch.tensor(scale, dtype=torch.float32))
            
        self.register_buffer(f"{group_name}_shift", torch.tensor(shifts, dtype=torch.float32))
        self.register_buffer(f"{group_name}_scale", torch.tensor(scales, dtype=torch.float32))

    def normalize(self, x: torch.Tensor, feature: str) -> torch.Tensor:
        """Generic normalization using pre-cached individual buffers."""
        feat_id = feature.replace('.', '_')
        shift = getattr(self, f"{feat_id}_shift_proc")
        scale = getattr(self, f"{feat_id}_scale_proc")
        return (x + shift.to(x.device)) * scale.to(x.device)

    def normalize_ego(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized ego state normalization."""
        return (x + self.ego_shift.to(x.device)) * self.ego_scale.to(x.device)

    def normalize_target(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized target normalization."""
        return (x + self.target_shift.to(x.device)) * self.target_scale.to(x.device)
    
    def denormalize_target(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized target de-normalization (for rollout)."""
        return (x / self.target_scale.to(x.device)) - self.target_shift.to(x.device)

    def get_stat(self, feature: str, stat: str) -> torch.Tensor:
        """Retrieves a specific stat for a feature."""
        name = f"{feature}_{stat.lower()}".replace('.', '_')
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"Stat '{stat}' for feature '{feature}' not found in normalizer.")

    @staticmethod
    def transform(x: torch.Tensor, transformation: str) -> torch.Tensor:
        """Applies a functional transformation to a tensor."""
        if transformation == "Log":
            return torch.log(x + 1.0)
        elif transformation == "Symlog":
            return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
        elif transformation == "Identity":
            return x
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
