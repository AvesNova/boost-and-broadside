import torch
import math
import unittest
from pathlib import Path
import tempfile
import os
from boost_and_broadside.models.components.normalizer import FeatureNormalizer

class TestFeatureNormalizer(unittest.TestCase):
    def setUp(self):
        # Create a dummy stats CSV
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stats_csv = Path(self.temp_dir.name) / "test_stats.csv"
        with open(self.stats_csv, "w") as f:
            f.write("Feature|Mean|Std|RMS|SEM|Min|Q1|Median|Q3|Max\n")
            f.write("State_HEALTH|50.0|10.0|51.0|0.1|0.0|40.0|50.0|60.0|100.0\n")
            f.write("State_VX|10.0|5.0|11.18|0.1|-20.0|5.0|10.0|15.0|40.0\n")
            f.write("Relational_log_dist|5.0|1.0|5.1|0.1|0.0|4.0|5.0|6.0|10.0\n")
            f.write("Target_DX|0.0|1.0|1.0|0.1|-5.0|-1.0|0.0|1.0|5.0\n")
            
        self.normalizer = FeatureNormalizer(self.stats_csv)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_min_max(self):
        x = torch.tensor([0.0, 50.0, 100.0])
        normed = self.normalizer.normalize(x, "State_HEALTH", "Min-Max")
        self.assertTrue(torch.allclose(normed, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5))

    def test_z_score(self):
        x = torch.tensor([40.0, 50.0, 60.0])
        normed = self.normalizer.normalize(x, "State_HEALTH", "Z-Score")
        self.assertTrue(torch.allclose(normed, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5))

    def test_rms_scale(self):
        x = torch.tensor([0.0, 11.18, 22.36])
        normed = self.normalizer.normalize(x, "State_VX", "Scale (RMS)")
        self.assertTrue(torch.allclose(normed, torch.tensor([0.0, 1.0, 2.0]), atol=1e-2))

    def test_log_transform(self):
        x = torch.tensor([0.0, math.exp(1.0)-1.0])
        transformed = self.normalizer.transform(x, "Log")
        self.assertTrue(torch.allclose(transformed, torch.tensor([0.0, 1.0]), atol=1e-5))

    def test_symlog_transform(self):
        x = torch.tensor([- (math.exp(1.0)-1.0), 0.0, math.exp(1.0)-1.0])
        transformed = self.normalizer.transform(x, "Symlog")
        self.assertTrue(torch.allclose(transformed, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5))

if __name__ == "__main__":
    import math
    unittest.main()
