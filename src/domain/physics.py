# This module holds the physical constants and closed-form relations the
# pipeline is built on. It is the bottom of the dependency graph.

"""Pure physics (SPEC-DM-001). No pipeline, config or I/O dependencies.

Everything here is a closed-form function of its arguments: no global state, no
configuration lookups, nothing imported from the rest of the project. That is
what makes this module safe for every other layer to import, and it is why the
dispersion constant lives here rather than in ``analysis/science_metrics.py``,
which is where it used to be defined and which sits a layer above.

SPEC-DM-001 always said every pipeline module should import the constant from
here; until this was inverted, none did -- ``physics.py`` re-exported it FROM
``science_metrics``, so the documented direction was backwards from the real one.
"""
from __future__ import annotations

__all__ = ["K_DM_MS", "dispersion_delay_ms", "dispersion_delay_s"]

#: Dispersion constant. Gives SECONDS when frequency is in MHz and DM in
#: pc cm^-3, matching PRESTO's ``delay_from_dm``. Do not restate this literal
#: anywhere else; ``tests/test_spec_constants.py`` scans ``src/`` for it.
K_DM_MS = 4.148808e3


def dispersion_delay_s(dm: float, freq_low_mhz: float, freq_high_mhz: float) -> float:
    """Dispersive delay across a band, in seconds.

    The delay is always measured from the higher frequency to the lower one, so
    the result is non-negative for ``dm >= 0`` whichever order the two
    frequencies are given in (SPEC-DM-002).
    """
    if freq_low_mhz <= 0 or freq_high_mhz <= 0:
        return 0.0
    nu_lo = min(float(freq_low_mhz), float(freq_high_mhz))
    nu_hi = max(float(freq_low_mhz), float(freq_high_mhz))
    return K_DM_MS * float(dm) * (nu_lo ** -2 - nu_hi ** -2)


def dispersion_delay_ms(dm: float, freq_low_mhz: float, freq_high_mhz: float) -> float:
    """Dispersive delay across a band, in milliseconds."""
    return dispersion_delay_s(dm, freq_low_mhz, freq_high_mhz) * 1000.0
