"""Validate config.yaml: required keys, ranges, and incompatibilities (Etapa 6)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestConfigYAMLStructure(unittest.TestCase):
    def setUp(self):
        self.cfg = _load_yaml()

    def test_required_sections_present(self):
        for section in ["data", "temporal", "downsampling", "dispersion",
                        "thresholds", "high_frequency", "polarization",
                        "performance", "output", "preprocessing"]:
            self.assertIn(section, self.cfg, f"Missing section: {section}")

    def test_dispersion_range_valid(self):
        d = self.cfg["dispersion"]
        self.assertLess(d["dm_min"], d["dm_max"])
        self.assertIn(str(d.get("dm_grid_mode", "legacy_uniform")).lower(),
                      {"legacy_uniform", "smear_limited", "coarse_to_fine"})

    def test_downsampling_rates_positive(self):
        ds = self.cfg["downsampling"]
        self.assertGreaterEqual(int(ds["frequency_rate"]), 1)
        self.assertGreaterEqual(int(ds["time_rate"]), 1)
        self.assertIn(str(ds.get("temporal_mode", "sum")).lower(),
                      {"sum", "phase_preserving", "snr_preserving"})

    def test_probability_thresholds_in_unit_interval(self):
        t = self.cfg["thresholds"]
        for key in ["detection_probability", "classification_probability",
                    "classification_probability_linear"]:
            val = float(t[key])
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_snr_thresholds_positive(self):
        t = self.cfg["thresholds"]
        self.assertGreater(float(t["snr_threshold"]), 0.0)
        self.assertGreater(float(t["snr_threshold_linear"]), 0.0)

    def test_prewhiten_default_false(self):
        # SPEC-PRE-001: scientific default must be false.
        self.assertFalse(bool(self.cfg["preprocessing"]["prewhiten_before_dm"]))

    def test_at_least_one_classification_enabled(self):
        hf = self.cfg["high_frequency"]
        intensity = hf.get("enable_intensity_classification", True)
        linear = hf.get("enable_linear_classification", True)
        self.assertTrue(bool(intensity) or bool(linear),
                        "At least one classification phase must be enabled")

    def test_collapse_ratio_positive(self):
        self.assertGreater(float(self.cfg["high_frequency"].get("collapse_ratio", 2.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
