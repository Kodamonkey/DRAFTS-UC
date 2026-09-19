"""``compute_snr_profile``'s positional order is a contract. REF-12.

``off_regions`` was the second positional parameter of this function and was
threaded through eleven modules without anything ever reading it. Removing it
was safe, but removing it is exactly the kind of change that can go wrong
silently: eight call sites passed it positionally, as
``compute_snr_profile(block, off_regions)``. With the parameter gone,
``dt_seconds`` slides into position 2, so a call site that was missed would keep
binding cleanly -- no TypeError -- and quietly pass an off-pulse region list
where a sampling interval belongs.

It happened to be harmless this time because the value was always ``None`` and
``dt_seconds=None`` is the default. That is luck, not safety: any future
non-``None`` value would land on ``dt_seconds`` and change every boxcar width.

So the position is pinned. These tests are not about ``off_regions`` -- it is
gone -- they are about the hazard its removal created.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.analysis.snr_utils import compute_snr_profile
from src.config import config


@pytest.fixture
def waterfall() -> np.ndarray:
    rng = np.random.default_rng(3)
    block = rng.normal(10.0, 1.0, size=(400, 16)).astype(np.float32)
    block[200:204, :] += 40.0
    return block


class TestTheSecondPositionalArgumentIsDtSeconds:
    def test_by_name(self):
        """The cheap, direct statement of the contract."""
        names = [
            p.name
            for p in inspect.signature(compute_snr_profile).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert names[0] == "waterfall"
        assert names[1] == "dt_seconds", (
            f"the second positional parameter is {names[1]!r}; call sites that "
            "pass it positionally are now passing it as something else"
        )
        assert "off_regions" not in names, (
            "off_regions is back; it was removed because nothing read it (REF-12)"
        )

    def test_by_behaviour(self, waterfall, monkeypatch):
        """And the same thing without reading the signature at all.

        ``dt_seconds`` converts ``DETECTION_WIDTHS_MS`` into boxcar widths in
        samples, so it visibly changes the result -- which is what makes this
        assertion worth anything. If a second parameter were ever reinserted
        ahead of it, the positional call would stop matching the keyword call.
        """
        monkeypatch.setattr(
            config, "DETECTION_WIDTHS_MS", [1.0, 4.0, 16.0], raising=False
        )

        positional = compute_snr_profile(waterfall, 0.02)
        keyword = compute_snr_profile(waterfall, dt_seconds=0.02)

        assert np.array_equal(positional[0], keyword[0])
        assert positional[1] == keyword[1]
        assert np.array_equal(positional[2], keyword[2])

    def test_the_behavioural_check_is_not_vacuous(self, waterfall, monkeypatch):
        """Guard the guard: dt_seconds must actually change the output, or the
        test above would pass no matter what position it sat in."""
        monkeypatch.setattr(
            config, "DETECTION_WIDTHS_MS", [1.0, 4.0, 16.0], raising=False
        )

        fine = compute_snr_profile(waterfall, dt_seconds=0.001)
        coarse = compute_snr_profile(waterfall, dt_seconds=0.02)

        assert not np.array_equal(fine[0], coarse[0]), (
            "dt_seconds no longer changes the profile, so the positional check "
            "above proves nothing"
        )
