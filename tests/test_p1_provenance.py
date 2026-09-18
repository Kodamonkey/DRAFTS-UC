"""Regression tests for the three Phase-1 defects about catalogue provenance.

P1-07  The barycentric MJD used a hardcoded sky position and site.
P1-08  It degraded to topocentric without saying so.
P1-10  The high-frequency burst verdict ignored which phase had actually run.

What the three have in common is that none of them raised and none of them
changed the number of rows: the CSV came out the same shape, with values that
described something other than the observation.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest

from src.config import config
from src.core import mjd_utils
from src.core.high_freq_pipeline import decide_candidate
from src.output.candidate_manager import CANDIDATE_HEADER, Candidate
from tests.observatory import effelsberg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_SOURCE = PROJECT_ROOT / "src" / "core" / "high_freq_pipeline.py"
MJD_SOURCE = PROJECT_ROOT / "src" / "core" / "mjd_utils.py"

# astropy resolves this one from its bundled ephemeris, so the tests below need
# no network and no ~10 MB download of de432s.
_NO_DOWNLOAD_EPHEM = "builtin"


def _configure_source(monkeypatch, *, ra="05:31:58.70", dec="33:08:52.5"):
    monkeypatch.setattr(config, "SOURCE_RA", ra, raising=False)
    monkeypatch.setattr(config, "SOURCE_DEC", dec, raising=False)
    monkeypatch.setattr(config, "REF_FREQ_MHZ", 1400.0, raising=False)
    # A pinned EarthLocation rather than the name: see tests/observatory.py.
    # Resolving "Effelsberg" downloads astropy's site registry, so these tests
    # asserted that the correction succeeds while depending on the network to
    # let it.
    monkeypatch.setattr(config, "OBSERVATORY", effelsberg(), raising=False)
    monkeypatch.setattr(config, "EPHEMERIS", _NO_DOWNLOAD_EPHEM, raising=False)


# =============================================================================
# P1-07 -- the sky position must come from configuration
# =============================================================================
class TestBarycentricUsesTheConfiguredSource:
    def test_source_keys_reach_config_from_the_yaml(self):
        """config.yaml `source:` must land on config, not on a literal."""
        for key in ("SOURCE_RA", "SOURCE_DEC", "REF_FREQ_MHZ", "OBSERVATORY", "EPHEMERIS"):
            assert getattr(config, key, None) is not None, f"config.{key} is not wired"

    def test_source_keys_are_accepted_by_inject_config(self):
        """Without this they would be silently dropped by the allow-list."""
        for key in ("SOURCE_RA", "SOURCE_DEC", "REF_FREQ_MHZ", "OBSERVATORY", "EPHEMERIS"):
            assert key in config._KNOWN_CONFIG_KEYS

    def test_changing_the_configured_position_changes_the_arrival_time(self, monkeypatch):
        """The regression: with the position hardcoded, this difference vanished.

        Two sources 6 h and 60 deg apart give barycentric corrections that differ
        by minutes. If the correction ignored the configured position, both calls
        would return the same number.
        """
        pytest.importorskip("astropy")

        _configure_source(monkeypatch, ra="05:31:58.70", dec="33:08:52.5")
        here = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )
        _configure_source(monkeypatch, ra="12:00:00.00", dec="-30:00:00.0")
        elsewhere = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )

        assert here["mjd_bary_status"] == "ok"
        assert elsewhere["mjd_bary_status"] == "ok"
        assert here["mjd_utc"] == elsewhere["mjd_utc"], "topocentric time must not move"
        assert here["mjd_bary_utc"] != elsewhere["mjd_bary_utc"]
        # Minutes apart, not float noise.
        assert abs(here["mjd_bary_utc"] - elsewhere["mjd_bary_utc"]) > 1e-4

    def test_no_sky_position_literal_survives_in_the_module(self):
        """The FRB 121102 coordinates must not be a fallback anywhere in mjd_utils."""
        source = MJD_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "05:31:58.70" not in literals
        assert "33:08:52.5" not in literals
        # "Effelsberg" may still appear in prose; it must not be a default value.
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            for default in func.args.defaults + [d for d in func.args.kw_defaults if d]:
                assert not (
                    isinstance(default, ast.Constant) and default.value == "Effelsberg"
                ), f"{func.name} still defaults to a hardcoded observatory"

    def test_missing_source_configuration_refuses_instead_of_guessing(self, monkeypatch):
        monkeypatch.setattr(config, "SOURCE_RA", None, raising=False)
        _ = monkeypatch  # the other keys may stay set; one missing is enough
        result = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )
        assert result["mjd_bary_status"] == "unavailable:no-source-config"
        assert result.get("mjd_bary_utc") is None


# =============================================================================
# P1-08 -- a failed correction must be visible in the catalogue
# =============================================================================
class TestDegradationIsRecorded:
    def test_topocentric_is_never_returned_under_a_barycentric_name(self, monkeypatch):
        """The regression: all four columns used to be the topocentric MJD.

        Written at twelve decimals, they were indistinguishable from a real
        correction -- a consumer reading mjd_bary_tdb got an uncorrected time.
        """
        _configure_source(monkeypatch)
        monkeypatch.setattr(mjd_utils, "ASTROPY_AVAILABLE", False)
        result = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )

        assert result["mjd_bary_status"] == "unavailable:astropy-missing"
        topo = result["mjd_utc"]
        for key in ("mjd_bary_utc", "mjd_bary_tdb", "mjd_bary_utc_inf", "mjd_bary_tdb_inf"):
            assert result[key] is None, f"{key} must be empty, not the topocentric value"
            assert result[key] != topo

    def test_an_unknown_observatory_is_reported_not_swallowed(self, monkeypatch):
        pytest.importorskip("astropy")
        _configure_source(monkeypatch)
        monkeypatch.setattr(config, "OBSERVATORY", "NoSuchSiteOnEarth", raising=False)
        result = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )
        assert result["mjd_bary_status"].startswith("unavailable:")
        assert result["mjd_bary_utc"] is None

    def test_a_successful_correction_says_so(self, monkeypatch):
        pytest.importorskip("astropy")
        _configure_source(monkeypatch)
        result = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
        )
        assert result["mjd_bary_status"] == "ok"
        assert result["mjd_bary_utc"] is not None
        assert result["mjd_bary_utc"] != result["mjd_utc"]


class TestTheObservatoryIsResolvedOncePerRun:
    """``EarthLocation.of_site`` is a network call, and this is a hot path.

    Astropy stopped bundling the site registry, so resolving a site *name*
    downloads ``sites.json``. ``calculate_candidate_mjd`` runs once per
    candidate, so without a cache a 500-candidate run makes 500 lookups -- and
    when the download cannot succeed, astropy retries both of its mirrors and
    waits out both timeouts every single time, to reach the same failure.
    """

    def _count_lookups(self, monkeypatch, site):
        calls = []

        def _spy(name):
            calls.append(name)
            raise RuntimeError("no network in this test")

        from astropy.coordinates import EarthLocation

        monkeypatch.setattr(EarthLocation, "of_site", staticmethod(_spy))
        _configure_source(monkeypatch)
        monkeypatch.setattr(config, "OBSERVATORY", site, raising=False)
        for _ in range(5):
            mjd_utils.calculate_candidate_mjd(
                t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
            )
        return calls

    def test_a_failing_lookup_is_not_retried_per_candidate(self, monkeypatch):
        """The expensive case: the answer is a failure, so it is not cached by
        anything that only remembers return values."""
        pytest.importorskip("astropy")
        calls = self._count_lookups(monkeypatch, "NoSuchSiteOnEarth")
        assert len(calls) == 1, (
            f"the site was looked up {len(calls)} times for 5 candidates; a "
            "failed lookup must be remembered too, or an offline run pays two "
            "HTTP timeouts per candidate"
        )

    def test_the_failure_still_reaches_the_status_every_time(self, monkeypatch):
        """Caching must not turn a reported failure into a silent success."""
        pytest.importorskip("astropy")
        calls = []

        def _spy(name):
            calls.append(name)
            raise RuntimeError("no network in this test")

        from astropy.coordinates import EarthLocation

        monkeypatch.setattr(EarthLocation, "of_site", staticmethod(_spy))
        _configure_source(monkeypatch)
        monkeypatch.setattr(config, "OBSERVATORY", "NoSuchSiteOnEarth", raising=False)
        for _ in range(3):
            result = mjd_utils.calculate_candidate_mjd(
                t_sec=1.0, tstart_mjd=60000.0, compute_bary=True, dm=100.0
            )
            assert result["mjd_bary_status"] == "unavailable:error"
            assert result["mjd_bary_utc"] is None

    def test_a_pinned_earthlocation_is_never_looked_up(self, monkeypatch):
        """How the golden suites stay off the network: pass the position, not
        the name. ``get_barycentric_mjd`` takes either."""
        pytest.importorskip("astropy")
        calls = self._count_lookups(monkeypatch, effelsberg())
        assert calls == [], f"an EarthLocation was resolved through of_site: {calls}"

    def test_not_requesting_the_correction_is_distinct_from_failing_at_it(self):
        result = mjd_utils.calculate_candidate_mjd(
            t_sec=1.0, tstart_mjd=60000.0, compute_bary=False
        )
        assert result["mjd_bary_status"] == "not_requested"

    def test_the_status_reaches_the_csv_next_to_empty_columns(self, tmp_path):
        """The mark is worthless if it stops at the dict."""
        assert "mjd_bary_status" in CANDIDATE_HEADER

        cand = Candidate(
            file="x.fil", chunk_id=0, slice_id=0, band_id=0, prob=0.9,
            dm=100.0, t_sec_dm_time=1.0, t_sample=1000, box=(0, 0, 1, 1),
            snr_patch_dedispersed=5.0,
            mjd_utc=60000.5,
            mjd_bary_status="unavailable:astropy-missing",
        )
        row = cand.to_row()
        assert len(row) == len(CANDIDATE_HEADER)

        cells = dict(zip(CANDIDATE_HEADER, row))
        assert cells["mjd_bary_status"] == "unavailable:astropy-missing"
        for key in ("mjd_bary_utc", "mjd_bary_tdb", "mjd_bary_utc_inf", "mjd_bary_tdb_inf"):
            assert cells[key] == ""
        assert cells["mjd_utc"] == "60000.500000000000"

        out = tmp_path / "c.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CANDIDATE_HEADER)
            writer.writerow(row)
        read_back = list(csv.DictReader(out.open(encoding="utf-8")))
        assert read_back[0]["mjd_bary_status"] == "unavailable:astropy-missing"


# =============================================================================
# P1-10 -- the verdict must follow the phase that actually ran
# =============================================================================
class TestHighFrequencyVerdict:
    @staticmethod
    def _decide(**overrides):
        kwargs = dict(
            has_intensity_result=True,
            has_linear_result=True,
            is_burst_intensity=True,
            class_prob_intensity=0.9,
            is_burst_linear=True,
            class_prob_linear=0.9,
            enable_linear_class=True,
            save_only_burst=False,
        )
        kwargs.update(overrides)
        return decide_candidate(**kwargs)

    def test_with_intensity_disabled_the_linear_verdict_decides(self):
        """The regression, stated exactly.

        Phase 3a off, Linear says "not a burst". The old code aliased the final
        verdict to the Intensity flag, which the disabled branch had set to a
        hardcoded True, so the row said "burst".
        """
        should_save, is_burst, _ = self._decide(
            has_intensity_result=False,
            is_burst_intensity=None,
            class_prob_intensity=None,
            is_burst_linear=False,
            class_prob_linear=0.1,
        )
        assert is_burst is False
        assert should_save is True  # SAVE_ONLY_BURST is False, so it is still written

    def test_with_intensity_disabled_a_linear_burst_is_a_burst(self):
        _, is_burst, _ = self._decide(
            has_intensity_result=False,
            is_burst_intensity=None,
            class_prob_intensity=None,
            is_burst_linear=True,
            class_prob_linear=0.95,
        )
        assert is_burst is True

    def test_no_phase_ran_means_no_verdict_not_a_burst(self):
        should_save, is_burst, reason = self._decide(
            has_intensity_result=False,
            has_linear_result=False,
            is_burst_intensity=None,
            class_prob_intensity=None,
            is_burst_linear=None,
            class_prob_linear=None,
        )
        assert should_save is False
        assert is_burst is None, "absent classification must not be reported as no_burst"
        assert "No classification" in reason

    @pytest.mark.parametrize("save_only_burst", [True, False])
    @pytest.mark.parametrize("intensity", [True, False])
    @pytest.mark.parametrize("linear", [True, False])
    def test_the_verdict_never_contradicts_the_save_gate(
        self, save_only_burst, intensity, linear
    ):
        """A saved row labelled no_burst under SAVE_ONLY_BURST would be a
        contradiction; so would a discarded row labelled burst."""
        should_save, is_burst, _ = self._decide(
            is_burst_intensity=intensity,
            class_prob_intensity=0.9 if intensity else 0.1,
            is_burst_linear=linear,
            class_prob_linear=0.9 if linear else 0.1,
            save_only_burst=save_only_burst,
        )
        assert is_burst == (
            (intensity and linear) if save_only_burst else (intensity or linear)
        )
        # With both phases available the gate equals the verdict in both modes.
        # (With a single phase available and SAVE_ONLY_BURST False the gate is
        # unconditionally True -- a pre-existing asymmetry this test pins rather
        # than changes; see decide_candidate.)
        assert should_save == is_burst

    def test_strict_mode_needs_both_phases(self):
        _, is_burst, _ = self._decide(
            is_burst_intensity=True, is_burst_linear=False,
            class_prob_linear=0.1, save_only_burst=True,
        )
        assert is_burst is False

    def test_no_in_range_sentinel_is_assigned_to_a_verdict(self):
        """Guards the shape of the fix, not just its effect.

        A future edit re-introducing `is_burst_intensity = True` or
        `class_prob_linear = 1.0` as a stand-in for a phase that did not run
        would restore P1-10 while every behavioural test above still passed,
        because those tests drive the pure function rather than the loop.
        """
        tree = ast.parse(HF_SOURCE.read_text(encoding="utf-8"))
        guarded = {
            "is_burst_intensity", "is_burst_linear",
            "class_prob_intensity", "class_prob_linear",
        }
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant):
                continue
            if node.value.value is None:
                continue  # None is the supported "no verdict" value
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in guarded:
                    offenders.append((target.id, node.value.value, node.lineno))
        assert not offenders, f"in-range sentinels reintroduced: {offenders}"
