"""Tests for pipeline data contracts (SDD Etapa 3)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.core.contracts import (
    ChunkPlan,
    DMGrid,
    ObservationMetadata,
    PipelineConfigSnapshot,
)
from src.preprocessing.dm_candidate_extractor import extract_candidate_dm


class TestObservationMetadata(unittest.TestCase):
    def test_from_config_roundtrip(self):
        old = (config.TIME_RESO, config.FREQ_RESO, config.FILE_LENG, config.FREQ,
               config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE)
        try:
            config.TIME_RESO = 1e-3
            config.FREQ_RESO = 4
            config.FILE_LENG = 1000
            config.FREQ = np.array([400.0, 500.0, 600.0, 700.0])
            config.DOWN_TIME_RATE = 4
            config.DOWN_FREQ_RATE = 2
            meta = ObservationMetadata.from_config(config)
            self.assertEqual(meta.freq_low, 400.0)
            self.assertEqual(meta.freq_high, 700.0)
            self.assertAlmostEqual(meta.effective_time_reso, 4e-3)
            self.assertAlmostEqual(meta.duration_s, 1.0)
        finally:
            (config.TIME_RESO, config.FREQ_RESO, config.FILE_LENG, config.FREQ,
             config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE) = old

    def test_is_immutable(self):
        meta = ObservationMetadata(time_reso=1e-3, freq_reso=2, file_leng=10, freq_mhz=(1.0, 2.0))
        with self.assertRaises(Exception):
            meta.time_reso = 2e-3  # type: ignore[misc]


class TestChunkPlan(unittest.TestCase):
    def test_valid_length_and_block_bounds(self):
        plan = ChunkPlan(chunk_idx=1, start_sample=100, end_sample=200, overlap_left=10, overlap_right=20)
        self.assertEqual(plan.valid_length, 100)
        self.assertEqual(plan.block_start, 90)
        self.assertEqual(plan.block_end, 220)

    def test_block_start_clamped_at_zero(self):
        plan = ChunkPlan(chunk_idx=0, start_sample=5, end_sample=50, overlap_left=10)
        self.assertEqual(plan.block_start, 0)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            ChunkPlan(chunk_idx=0, start_sample=200, end_sample=100)


class TestDMGrid(unittest.TestCase):
    def test_row_mapping_never_exceeds_dm_max(self):
        grid = DMGrid.from_values(np.linspace(0, 100, 101))
        self.assertEqual(grid.dm_min, 0.0)
        self.assertEqual(grid.dm_max, 100.0)
        self.assertLessEqual(grid.dm_for_row(511, height=512), grid.dm_max)
        self.assertAlmostEqual(grid.dm_for_row(0, height=512), 0.0)

    def test_row_mapping_monotonic(self):
        grid = DMGrid.from_values(np.linspace(10, 50, 41))
        prev = -np.inf
        for row in range(0, 512, 32):
            dm = grid.dm_for_row(row, height=512)
            self.assertGreaterEqual(dm, prev)
            prev = dm

    def test_from_config(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min = 0
            config.DM_max = 50
            config.DM_GRID_MODE = "legacy_uniform"
            grid = DMGrid.from_config(config)
            self.assertEqual(grid.dm_min, 0.0)
            self.assertEqual(grid.dm_max, 50.0)
            self.assertEqual(grid.size, 51)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_from_config_honours_the_config_it_is_given(self):
        """REF-10 step 1. ``from_config`` took a config object and ignored it.

        ``test_from_config`` above cannot see that, because it passes the real
        module: reading the argument and reading the global give the same answer
        there, so the test passes either way. This one passes a different config
        and pins that the answer follows the argument.

        The failure mode this guards is the worst kind for a search pipeline. A
        contributor migrating a hot path to ``DMGrid.from_config(snapshot)``
        would have got DM trials built from whatever the process global happened
        to hold -- no error, no warning, just candidates at the wrong DM.
        """
        from types import SimpleNamespace

        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min, config.DM_max = 100.0, 200.0
            config.DM_GRID_MODE = "legacy_uniform"
            other = SimpleNamespace(
                DM_min=0.0, DM_max=7.0, DM_GRID_MODE="legacy_uniform"
            )

            grid = DMGrid.from_config(other)

            self.assertEqual(grid.dm_min, 0.0)
            self.assertEqual(grid.dm_max, 7.0)
            self.assertEqual(grid.size, 8)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_from_config_honours_the_argument_in_smear_limited_mode_too(self):
        """The other grid mode reads FREQ, TIME_RESO and DOWN_TIME_RATE as well,
        so honouring only DM_min/DM_max would leave it reading the global."""
        from types import SimpleNamespace

        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE,
               config.FREQ, config.TIME_RESO, config.DOWN_TIME_RATE)
        try:
            # A global that would give a very different step.
            config.DM_min, config.DM_max = 0.0, 100.0
            config.DM_GRID_MODE = "smear_limited"
            config.FREQ = np.linspace(1000.0, 1500.0, 8)
            config.TIME_RESO = 1e-3
            config.DOWN_TIME_RATE = 1

            other = SimpleNamespace(
                DM_min=0.0, DM_max=100.0, DM_GRID_MODE="smear_limited",
                FREQ=np.linspace(100.0, 200.0, 8),   # far lower band -> coarser step
                TIME_RESO=1e-3, DOWN_TIME_RATE=1,
                MAX_DM_SMEARING_MS="auto",
            )

            from_global = DMGrid.from_config(config)
            from_other = DMGrid.from_config(other)

            self.assertNotEqual(
                from_global.size, from_other.size,
                "the two configs should give different grids; if they do not, "
                "this test cannot tell whether the argument was honoured",
            )
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE,
             config.FREQ, config.TIME_RESO, config.DOWN_TIME_RATE) = old

    def test_from_config_still_matches_calculate_dm_values_at_the_real_call_site(self):
        """Parity: the production call site passes the real module, and must
        keep getting exactly what it got before."""
        from src.config.derived import calculate_dm_values

        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min, config.DM_max = 0.0, 50.0
            config.DM_GRID_MODE = "legacy_uniform"
            self.assertTrue(
                np.array_equal(
                    DMGrid.from_config(config).values,
                    np.asarray(calculate_dm_values(), dtype=np.float64),
                )
            )
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_dm_grid_aligned_with_extractor(self):
        grid = DMGrid.from_values(np.linspace(0, 100, 101))
        old = (config.DM_min, config.DM_max)
        try:
            config.DM_min = 0.0
            config.DM_max = 100.0
            for row in (0, 128, 255, 511):
                dm_grid = grid.dm_for_row(row, height=512)
                dm_ext, _, _ = extract_candidate_dm(
                    0.0, float(row), 512, img_height=512, dm_values=grid.values
                )
                self.assertAlmostEqual(dm_grid, dm_ext, places=4)
        finally:
            config.DM_min, config.DM_max = old


class TestDmFromImageVsDMGrid(unittest.TestCase):
    """SPEC-DM-005: _dm_from_image_at_time must match DMGrid.dm_for_row (Etapa 4)."""

    def _make_image_with_peak_at_row(self, height: int, width: int, peak_row: int) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.float32)
        img[peak_row, width // 2] = 1000.0
        return img

    def test_mapping_matches_dmgrid(self):
        """_dm_from_image_at_time and DMGrid.dm_for_row must give the same DM."""
        from src.core.high_freq_pipeline import _dm_from_image_at_time
        from src.config import config

        old_min, old_max = config.DM_min, config.DM_max
        try:
            config.DM_min = 10.0
            config.DM_max = 200.0
            height, width = 64, 100

            for peak_row in (0, height // 2, height - 1):
                img = self._make_image_with_peak_at_row(height, width, peak_row)
                dm_from_fn = _dm_from_image_at_time(img, width // 2)
                grid = DMGrid.from_values(np.linspace(config.DM_min, config.DM_max, height))
                dm_from_grid = grid.dm_for_row(peak_row, height=height)
                self.assertAlmostEqual(
                    dm_from_fn, dm_from_grid, places=4,
                    msg=f"Mismatch at peak_row={peak_row}: fn={dm_from_fn}, grid={dm_from_grid}"
                )
        finally:
            config.DM_min, config.DM_max = old_min, old_max


class TestSnapshotAndRecord(unittest.TestCase):
    def test_config_snapshot_reads_prewhiten_false(self):
        snap = PipelineConfigSnapshot.from_config(config)
        self.assertFalse(snap.prewhiten_before_dm)



class TestTheSnapshotsMatchTheGlobalsTheyReplace(unittest.TestCase):
    """Parity for the REF-10 substitutions in ``_process_block``.

    Each step of that migration swaps a ``config.X`` read for a contract field.
    The golden CSV proves the swap changed nothing for one input; this proves
    the equality the swap relies on, field by field, for arbitrary values --
    including the derived ones, where a wrong formula would survive the golden
    test whenever the factor happens to be 1.
    """

    FIELDS = [
        # (contract attribute, config key, how the hot path used to write it)
        ("time_reso", "TIME_RESO", lambda c: c.TIME_RESO),
        ("down_time_rate", "DOWN_TIME_RATE", lambda c: int(c.DOWN_TIME_RATE)),
        ("down_freq_rate", "DOWN_FREQ_RATE", lambda c: int(c.DOWN_FREQ_RATE)),
        ("freq_reso", "FREQ_RESO", lambda c: int(c.FREQ_RESO)),
        ("file_leng", "FILE_LENG", lambda c: int(c.FILE_LENG)),
    ]

    def _pin(self):
        config.TIME_RESO = 8.192e-5
        config.DOWN_TIME_RATE = 6          # not 1: a dropped factor must show
        config.DOWN_FREQ_RATE = 3
        config.FREQ_RESO = 512
        config.FILE_LENG = 123_457
        config.FREQ = np.linspace(1100.0, 1500.0, 512)
        config.DM_min, config.DM_max = 12.0, 812.0
        config.DM_GRID_MODE = "legacy_uniform"

    def setUp(self):
        self._old = (config.TIME_RESO, config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE,
                     config.FREQ_RESO, config.FILE_LENG, config.FREQ,
                     config.DM_min, config.DM_max, config.DM_GRID_MODE)
        self._pin()

    def tearDown(self):
        (config.TIME_RESO, config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE,
         config.FREQ_RESO, config.FILE_LENG, config.FREQ,
         config.DM_min, config.DM_max, config.DM_GRID_MODE) = self._old

    def test_each_migrated_field_equals_the_global_it_replaced(self):
        meta = ObservationMetadata.from_config(config)
        for attr, key, old_expression in self.FIELDS:
            with self.subTest(field=attr):
                self.assertEqual(
                    getattr(meta, attr), old_expression(config),
                    f"ObservationMetadata.{attr} no longer equals config.{key}",
                )

    def test_effective_time_reso_equals_the_expression_it_replaced(self):
        """The hot path wrote ``config.TIME_RESO * config.DOWN_TIME_RATE`` in
        three places. DOWN_TIME_RATE is 6 here on purpose: at 1 the property
        would match even if it had dropped the factor entirely."""
        meta = ObservationMetadata.from_config(config)
        self.assertAlmostEqual(
            meta.effective_time_reso,
            config.TIME_RESO * config.DOWN_TIME_RATE,
            places=15,
        )
        self.assertNotAlmostEqual(meta.effective_time_reso, config.TIME_RESO)

    def test_the_dm_range_equals_the_globals_it_replaced(self):
        snap = PipelineConfigSnapshot.from_config(config)
        self.assertEqual(snap.dm_min, float(config.DM_min))
        self.assertEqual(snap.dm_max, float(config.DM_max))

    def test_a_snapshot_does_not_follow_the_global_afterwards(self):
        """Which is the point of snapshotting, and the reason
        TestOnlyTheReadersMutateConfig has to hold for the swap to be safe."""
        meta = ObservationMetadata.from_config(config)
        before = meta.time_reso
        config.TIME_RESO = 1.0
        self.assertEqual(meta.time_reso, before)
        self.assertNotEqual(meta.time_reso, config.TIME_RESO)


class TestTheHighFrequencyFieldsResolveTheSameFallbacks(unittest.TestCase):
    """REF-10 step 4 parity: the snapshot must answer exactly as the inline
    reads it replaced.

    ``snr_detect_and_classify_candidates_in_band`` spelled these out by hand,
    twenty-one times across ten keys. Five of them carried a non-obvious
    fallback -- two defaulting to their intensity counterpart, three to a
    literal -- and those fallbacks are the part a migration gets wrong. Each is
    asserted here against the expression it replaced, with the key present and
    with it absent.

    What this does NOT cover is stated in the commit: the band function's own
    body is stubbed out in the end-to-end tests, so the substitutions inside it
    are value-identical by these assertions and not by execution.
    """

    KEYS = ("SNR_THRESH", "CLASS_PROB", "SNR_THRESH_LINEAR", "CLASS_PROB_LINEAR",
            "ENABLE_LINEAR_VALIDATION", "ENABLE_INTENSITY_CLASSIFICATION",
            "ENABLE_LINEAR_CLASSIFICATION")

    def setUp(self):
        self._saved = {k: getattr(config, k) for k in self.KEYS if hasattr(config, k)}
        self._absent = [k for k in self.KEYS if not hasattr(config, k)]

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        for k in self._absent:
            if hasattr(config, k):
                delattr(config, k)

    def _clear(self, *names):
        for n in names:
            if hasattr(config, n):
                delattr(config, n)

    def test_the_linear_thresholds_fall_back_to_their_intensity_counterparts(self):
        """``getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)`` and the
        CLASS_PROB equivalent, which is what the band function wrote."""
        config.SNR_THRESH = 7.5
        config.CLASS_PROB = 0.42
        self._clear("SNR_THRESH_LINEAR", "CLASS_PROB_LINEAR")

        snap = PipelineConfigSnapshot.from_config(config)

        self.assertEqual(snap.snr_thresh_linear, 7.5)
        self.assertEqual(snap.class_prob_linear, 0.42)
        self.assertEqual(snap.snr_thresh_linear, snap.snr_thresh)
        self.assertEqual(snap.class_prob_linear, snap.class_prob)

    def test_the_linear_thresholds_are_used_when_they_are_set(self):
        """And the fallback must not swallow a value that IS configured."""
        config.SNR_THRESH = 7.5
        config.CLASS_PROB = 0.42
        config.SNR_THRESH_LINEAR = 3.25
        config.CLASS_PROB_LINEAR = 0.11

        snap = PipelineConfigSnapshot.from_config(config)

        self.assertEqual(snap.snr_thresh_linear, 3.25)
        self.assertEqual(snap.class_prob_linear, 0.11)
        self.assertNotEqual(snap.snr_thresh_linear, snap.snr_thresh)

    def test_the_phase_switches_default_the_way_the_inline_reads_did(self):
        """``X if hasattr(config, 'X') else <default>``: False for the linear
        validation phase, True for both classification phases. Getting one of
        these backwards silently turns a detection phase on or off."""
        self._clear("ENABLE_LINEAR_VALIDATION", "ENABLE_INTENSITY_CLASSIFICATION",
                    "ENABLE_LINEAR_CLASSIFICATION")

        snap = PipelineConfigSnapshot.from_config(config)

        self.assertIs(snap.enable_linear_validation, False)
        self.assertIs(snap.enable_intensity_classification, True)
        self.assertIs(snap.enable_linear_classification, True)

    def test_the_phase_switches_follow_the_config_when_present(self):
        config.ENABLE_LINEAR_VALIDATION = True
        config.ENABLE_INTENSITY_CLASSIFICATION = False
        config.ENABLE_LINEAR_CLASSIFICATION = False

        snap = PipelineConfigSnapshot.from_config(config)

        self.assertIs(snap.enable_linear_validation, True)
        self.assertIs(snap.enable_intensity_classification, False)
        self.assertIs(snap.enable_linear_classification, False)
