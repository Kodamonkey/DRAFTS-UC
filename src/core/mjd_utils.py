# This module provides MJD (Modified Julian Date) utilities for the pipeline.

"""MJD utilities for FRB pipeline - converts relative times to absolute MJD."""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

# Suppress ERFA warnings about dubious years (common with astronomical data)
warnings.filterwarnings('ignore', category=UserWarning, module='erfa')

try:
    from astropy.time import Time, TimeDelta
    from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logging.warning("astropy not available - barycentric MJD calculations will be disabled")

from ..domain.physics import K_DM_MS
from ..config import config

logger = logging.getLogger(__name__)


def get_topocentric_mjd(tstart_mjd: float, t_sec: float) -> float:
    """
    Calculate topocentric MJD (UTC) from file start MJD and relative time.
    
    Parameters
    ----------
    tstart_mjd : float
        MJD at the start of the file (UTC)
    t_sec : float
        Time in seconds relative to file start
    
    Returns
    -------
    float
        Topocentric MJD (UTC)
    """
    return tstart_mjd + (t_sec / 86400.0)


def get_barycentric_mjd(
    topo_mjd: float,
    ra: str,
    dec: str,
    freq_mhz: float,
    location: str,
    ephem: str,
    dm: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """
    Calculate barycentric MJD with optional dispersion correction.
    
    Parameters
    ----------
    topo_mjd : float
        Topocentric MJD (UTC)
    ra : str
        Source RA in HH:MM:SS.ss format. Required: there is deliberately no
        default, because a default sky position produces a plausible number for
        the wrong source (SPEC note: audit finding P1-07).
    dec : str
        Source DEC in DD:MM:SS.s format. Required, same reason.
    freq_mhz : float
        Reference frequency in MHz. Required, same reason.
    location : str
        Observatory site name as ``EarthLocation.of_site`` spells it. Required.
    ephem : str
        JPL ephemerides name, e.g. "de432s". Required.
    dm : float, optional
        Dispersion measure in pc cm^-3. If None, only barycentric correction is applied.

    Returns
    -------
    tuple
        (mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf, status)

        On success the four MJDs are barycentric and ``status`` is ``"ok"``.
        When the correction cannot be computed the four values are **None** --
        not the topocentric MJD -- and ``status`` says why. Returning the
        topocentric value under a barycentric name was audit finding P1-08: the
        caller could not tell a corrected time from an uncorrected one, and the
        CSV recorded the uncorrected one to twelve decimals.
    """
    if not ASTROPY_AVAILABLE:
        logger.debug("astropy not available, barycentric MJD unavailable")
        return None, None, None, None, "unavailable:astropy-missing"

    try:
        # 1) Site location
        loc = EarthLocation.of_site(location) if isinstance(location, str) else location
        
        # 2) Time with location; work explicitly in TDB
        times_utc = Time(topo_mjd, format="mjd", scale="utc", location=loc)
        times_tdb = times_utc.tdb
        
        # 3) Precise JPL ephemerides
        solar_system_ephemeris.set(ephem)
        
        # 4) Source and barycentric correction (TDB seconds)
        src = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
        ltt_bary = times_tdb.light_travel_time(src)  # TimeDelta in TDB seconds
        
        # 5) Barycentric times
        bary_tdb = times_tdb + ltt_bary
        bary_utc = bary_tdb.utc
        
        mjd_bary_utc = bary_utc.mjd
        mjd_bary_tdb = bary_tdb.mjd
        
        # 6) Dispersion correction to infinite frequency (if DM provided)
        if dm is not None:
            dmcorr = TimeDelta(K_DM_MS * dm / (freq_mhz**2), format="sec")
            mjd_bary_utc_inf = (bary_utc - dmcorr).mjd
            mjd_bary_tdb_inf = (bary_tdb - dmcorr).mjd
        else:
            mjd_bary_utc_inf = None
            mjd_bary_tdb_inf = None

        return mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf, "ok"

    except Exception as e:
        # Check if it's a jplephem-related error (most common case)
        error_msg = str(e).lower()
        if 'jplephem' in error_msg or 'ephemeris' in error_msg:
            # This is expected if jplephem is not installed - log at debug level
            logger.debug(
                "Barycentric MJD calculation requires jplephem package. "
                "Install with: pip install jplephem. "
                "Barycentric columns will be left empty."
            )
            reason = "ephemeris"
        else:
            # Other errors - log at warning level. The usual one is an
            # observatory name EarthLocation.of_site() does not know.
            logger.warning(f"Error calculating barycentric MJD: {e}, leaving barycentric columns empty")
            reason = "error"
        return None, None, None, None, f"unavailable:{reason}"


def calculate_candidate_mjd(
    t_sec: float,
    tstart_mjd: Optional[float] = None,
    compute_bary: bool = False,
    dm: Optional[float] = None,
    ra: Optional[str] = None,
    dec: Optional[str] = None,
    freq_mhz: Optional[float] = None,
    location: Optional[str] = None,
    ephem: Optional[str] = None,
) -> dict:
    """
    Calculate MJD values for a candidate.
    
    Parameters
    ----------
    t_sec : float
        Time in seconds relative to file start
    tstart_mjd : float, optional
        MJD at the start of the file. If None, tries to get from config.
    compute_bary : bool
        If True, also compute barycentric MJD
    dm : float, optional
        Dispersion measure for infinite frequency correction
    ra : str, optional
        Source RA. If None, read from ``config.SOURCE_RA`` (config.yaml
        ``source.ra``). There is no literal fallback.
    dec : str, optional
        Source DEC. If None, read from ``config.SOURCE_DEC``.
    freq_mhz : float, optional
        Reference frequency in MHz. If None, read from ``config.REF_FREQ_MHZ``.
    location : str, optional
        Observatory site. If None, read from ``config.OBSERVATORY``.
    ephem : str, optional
        JPL ephemeris. If None, read from ``config.EPHEMERIS``.

    Returns
    -------
    dict
        Dictionary with MJD values:
        - mjd_utc: Topocentric MJD (UTC) - always present
        - mjd_bary_utc: Barycentric MJD (UTC) - if compute_bary=True
        - mjd_bary_tdb: Barycentric MJD (TDB) - if compute_bary=True
        - mjd_bary_utc_inf: Barycentric MJD at infinite frequency (UTC) - if compute_bary=True and dm provided
        - mjd_bary_tdb_inf: Barycentric MJD at infinite frequency (TDB) - if compute_bary=True and dm provided
        - mjd_bary_status: why the barycentric values are what they are -
          "ok", "not_requested", or "unavailable:<reason>". Whenever it is not
          "ok" the four barycentric values are None, never a topocentric value
          wearing a barycentric name.
    """
    # Get tstart_mjd from config if not provided
    if tstart_mjd is None:
        tstart_mjd = getattr(config, 'TSTART_MJD_CORR', None)
        if tstart_mjd is None:
            tstart_mjd = getattr(config, 'TSTART_MJD', None)
        if tstart_mjd is None:
            logger.debug("TSTART_MJD not available in config, using t_sec as relative time only")
            tstart_mjd = 0.0
    
    # Calculate topocentric MJD
    mjd_utc = get_topocentric_mjd(tstart_mjd, t_sec)
    
    result = {
        'mjd_utc': mjd_utc,
        'mjd_bary_status': 'not_requested',
    }

    if not compute_bary:
        return result

    # Source and site come from config.yaml (section `source:`). They used to be
    # literals in this file describing FRB 121102 seen from Effelsberg, and no
    # caller ever overrode them, so every observation was stamped with that one
    # source (P1-07). A missing value is now a refusal to compute, not a guess.
    if ra is None:
        ra = getattr(config, 'SOURCE_RA', None)
    if dec is None:
        dec = getattr(config, 'SOURCE_DEC', None)
    if freq_mhz is None:
        freq_mhz = getattr(config, 'REF_FREQ_MHZ', None)
    if location is None:
        location = getattr(config, 'OBSERVATORY', None)
    if ephem is None:
        ephem = getattr(config, 'EPHEMERIS', None)

    missing = [
        name
        for name, value in (
            ('source.ra', ra),
            ('source.dec', dec),
            ('source.reference_freq_mhz', freq_mhz),
            ('source.observatory', location),
            ('source.ephemeris', ephem),
        )
        if value is None
    ]
    if missing:
        logger.warning(
            "Barycentric MJD requested but config.yaml is missing %s; "
            "barycentric columns will be left empty",
            ", ".join(missing),
        )
        result['mjd_bary_status'] = 'unavailable:no-source-config'
        return result

    mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf, status = get_barycentric_mjd(
        mjd_utc,
        ra=ra,
        dec=dec,
        freq_mhz=float(freq_mhz),
        location=location,
        ephem=ephem,
        dm=dm,
    )

    result['mjd_bary_status'] = status
    result['mjd_bary_utc'] = mjd_bary_utc
    result['mjd_bary_tdb'] = mjd_bary_tdb
    result['mjd_bary_utc_inf'] = mjd_bary_utc_inf
    result['mjd_bary_tdb_inf'] = mjd_bary_tdb_inf

    return result

