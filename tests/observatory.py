"""The observatory position the tests pin, resolved without the network.

Why this file exists
--------------------
``EarthLocation.of_site("Effelsberg")`` is a network call. Astropy no longer
bundles the site registry, so the name is resolved by downloading
``sites.json`` from data.astropy.org, and is served from ``~/.astropy``
afterwards only because that download once succeeded.

``test_golden_csv.py`` and ``test_golden_images.py`` both say, next to the line
that sets ``EPHEMERIS = "builtin"``, that they want "no download, no network,
the same answer everywhere". That is true of the ephemeris. It was never true
of the site: both suites were relying on a warm astropy cache on the machine
that generated the baselines.

The first run on a machine without that cache -- a fresh CI runner, an
air-gapped box, anything behind a proxy that does not allow data.astropy.org --
fails thirteen tests: the barycentric columns come back empty, the candidate
annotations lose their ``MJD_bary_inf`` line, and every stored baseline
disagrees. None of it says anything about the code under test.

So the position is pinned here instead of being looked up. These are astropy's
own numbers for ``effelsberg``, copied from the registry ``of_site`` downloads:

    https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json

    "effelsberg": {
      "name": "Effelsberg 100-m Radio Telescope",
      "latitude":  50.52483588054502,    "latitude_unit":  "degree",
      "longitude":  6.8836164652709835,  "longitude_unit": "degree",
      "elevation": 416.7160563557801,    "elevation_unit": "meter"
    }

Verified, not assumed: handing this ``EarthLocation`` to ``get_barycentric_mjd``
reproduces all four barycentric columns of ``tests/golden/lf_candidates.csv``
exactly -- to the twelve decimals the CSV stores, for all eight rows. Pinning
the position therefore costs no baseline regeneration. It only removes the
download.
"""
from __future__ import annotations

#: Astropy's registry values for ``effelsberg``; see the module docstring.
EFFELSBERG_LATITUDE_DEG = 50.52483588054502
EFFELSBERG_LONGITUDE_DEG = 6.8836164652709835
EFFELSBERG_ELEVATION_M = 416.7160563557801


def effelsberg():
    """The pinned site, as something ``config.OBSERVATORY`` can hold.

    ``get_barycentric_mjd`` accepts either a site name to look up or an
    ``EarthLocation`` to use as given, so returning the latter takes the
    registry out of the picture entirely.

    Without astropy there is no barycentric correction to pin, so the name is
    returned unchanged and the status stays ``unavailable:astropy-missing`` --
    exactly what these tests saw before this helper existed.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import EarthLocation
    except ImportError:
        return "Effelsberg"

    return EarthLocation.from_geodetic(
        lon=EFFELSBERG_LONGITUDE_DEG * u.deg,
        lat=EFFELSBERG_LATITUDE_DEG * u.deg,
        height=EFFELSBERG_ELEVATION_M * u.m,
    )
