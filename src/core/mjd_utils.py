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

from ..config import config

logger = logging.getLogger(__name__)

# Dispersion constant
K_DM = 4.148808e3  # s MHz^2 pc^-1 cm^3


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
    ra: str = "05:31:58.70",
    dec: str = "33:08:52.5",
    dm: Optional[float] = None,
    freq_mhz: float = 1400.0,
    location: str = "Effelsberg",
    ephem: str = "de432s",
) -> Tuple[float, float, float, float]:
    """
    Calculate barycentric MJD with optional dispersion correction.
    
    Parameters
    ----------
    topo_mjd : float
        Topocentric MJD (UTC)
    ra : str
        Source RA in HH:MM:SS.ss format
    dec : str
        Source DEC in DD:MM:SS.s format
    dm : float, optional
        Dispersion measure in pc cm^-3. If None, only barycentric correction is applied.
    freq_mhz : float
        Reference frequency in MHz (default: 1400.0)
    location : str
        Observatory name (default: "Effelsberg")
    ephem : str
        JPL ephemerides name (default: "de432s")
    
    Returns
    -------
    tuple
        (mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf)
        - mjd_bary_utc: Barycentric MJD in UTC scale
        - mjd_bary_tdb: Barycentric MJD in TDB scale
        - mjd_bary_utc_inf: Barycentric MJD in UTC at infinite frequency (if DM provided)
        - mjd_bary_tdb_inf: Barycentric MJD in TDB at infinite frequency (if DM provided)
    """
    if not ASTROPY_AVAILABLE:
        logger.debug("astropy not available, returning topocentric MJD only")
        return topo_mjd, topo_mjd, topo_mjd, topo_mjd
    
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
            dmcorr = TimeDelta(K_DM * dm / (freq_mhz**2), format="sec")
            mjd_bary_utc_inf = (bary_utc - dmcorr).mjd
            mjd_bary_tdb_inf = (bary_tdb - dmcorr).mjd
        else:
            mjd_bary_utc_inf = mjd_bary_utc
            mjd_bary_tdb_inf = mjd_bary_tdb
        
        return mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf
    
    except Exception as e:
        # Check if it's a jplephem-related error (most common case)
        error_msg = str(e).lower()
        if 'jplephem' in error_msg or 'ephemeris' in error_msg:
            # This is expected if jplephem is not installed - log at debug level
            logger.debug(
                "Barycentric MJD calculation requires jplephem package. "
                "Install with: pip install jplephem. "
                "Returning topocentric MJD only."
            )
        else:
            # Other errors - log at warning level
            logger.warning(f"Error calculating barycentric MJD: {e}, returning topocentric MJD")
        return topo_mjd, topo_mjd, topo_mjd, topo_mjd


def calculate_candidate_mjd(
    t_sec: float,
    tstart_mjd: Optional[float] = None,
    compute_bary: bool = False,
    dm: Optional[float] = None,
    ra: Optional[str] = None,
    dec: Optional[str] = None,
    freq_mhz: Optional[float] = None,
    location: str = "Effelsberg",
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
        Source RA. If None, tries to get from config or uses default.
    dec : str, optional
        Source DEC. If None, tries to get from config or uses default.
    freq_mhz : float, optional
        Reference frequency. If None, uses default 1400.0 MHz.
    location : str
        Observatory name (default: "Effelsberg")
    
    Returns
    -------
    dict
        Dictionary with MJD values:
        - mjd_utc: Topocentric MJD (UTC) - always present
        - mjd_bary_utc: Barycentric MJD (UTC) - if compute_bary=True
        - mjd_bary_tdb: Barycentric MJD (TDB) - if compute_bary=True
        - mjd_bary_utc_inf: Barycentric MJD at infinite frequency (UTC) - if compute_bary=True and dm provided
        - mjd_bary_tdb_inf: Barycentric MJD at infinite frequency (TDB) - if compute_bary=True and dm provided
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
    }
    
    # Calculate barycentric MJD if requested
    if compute_bary:
        # Get RA/DEC from config or use defaults
        if ra is None:
            ra = getattr(config, 'SOURCE_RA', "05:31:58.70")
        if dec is None:
            dec = getattr(config, 'SOURCE_DEC', "33:08:52.5")
        if freq_mhz is None:
            freq_mhz = getattr(config, 'REF_FREQ_MHZ', 1400.0)
        
        mjd_bary_utc, mjd_bary_tdb, mjd_bary_utc_inf, mjd_bary_tdb_inf = get_barycentric_mjd(
            mjd_utc,
            ra=ra,
            dec=dec,
            dm=dm,
            freq_mhz=freq_mhz,
            location=location,
        )
        
        result['mjd_bary_utc'] = mjd_bary_utc
        result['mjd_bary_tdb'] = mjd_bary_tdb
        if dm is not None:
            result['mjd_bary_utc_inf'] = mjd_bary_utc_inf
            result['mjd_bary_tdb_inf'] = mjd_bary_tdb_inf
    
    return result

