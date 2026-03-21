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
        normed = self.normalizer.normalize(x, "State_HEALTH")
        self.assertTrue(torch.allclose(normed, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5))

    def test_z_score(self):
        x = torch.tensor([4.0, 5.0, 6.0])
        normed = self.normalizer.normalize(x, "Relational_log_dist")
        self.assertTrue(torch.allclose(normed, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5))

    def test_rms_scale(self):
        x = torch.tensor([0.0, 11.18, 22.36])
        normed = self.normalizer.normalize(x, "State_VX")
        self.assertTrue(torch.allclose(normed, torch.tensor([0.0, 1.0, 2.0]), atol=1e-2))

    def test_vectorized_ego(self):
        # [HEALTH, POWER, VX, VY, ANG_VEL]
        # In dummy: H is min-max (0, 100), VX is scale (rms=11.18)
        # Power, VY, ANG_VEL not in dummy -> Identity
        x = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0], [50.0, 20.0, 11.18, 5.0, 5.0]])
        normed = self.normalizer.normalize_ego(x)
        
        # Row 0: H=0, P=10, VX=0
        self.assertTrue(torch.allclose(normed[0, 0], torch.tensor(0.0), atol=1e-5))
        self.assertTrue(torch.allclose(normed[0, 1], torch.tensor(10.0), atol=1e-5)) 
        # Row 1: H=0.5, P=20, VX=1.0
        self.assertTrue(torch.allclose(normed[1, 0], torch.tensor(0.5), atol=1e-5))
        self.assertTrue(torch.allclose(normed[1, 1], torch.tensor(20.0), atol=1e-5))
        self.assertTrue(torch.allclose(normed[1, 2], torch.tensor(1.0), atol=1e-5))

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
