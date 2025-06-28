HEADER_ALIASES = {
    'TBIN': ['TBIN', 'TSAMP', 'SAMPTIME', 'DELTAT', 'TIME_RES'],
    'DAT_FREQ': ['DAT_FREQ', 'CHAN_FREQ', 'FREQUENCY', 'DATFREQ', 'FREQS'],
    'NCHAN': ['NCHAN', 'OBSNCHAN', 'NAXIS1', 'CHAN_CNT'],
    'NAXIS2': ['NAXIS2', 'NSUBINT', 'NUMROWS'],
    'NSBLK': ['NSBLK', 'NSAMP', 'BLOCKNS', 'NSAMPLES'],
    'NPOL': ['NPOL', 'NRCVR', 'NFEEDS', 'N_STOKES'],
    'OBSFREQ': ['OBSFREQ', 'STT_CRVAL1', 'RESTFRQ', 'FREQ'],
    'OBSBW': ['OBSBW'],
    'CHAN_BW': ['CHAN_BW', 'DELTAF', 'DFREQ'],
    'CRVAL1': ['CRVAL1', 'RESTFRQ', 'STT_CRVAL1'],
    'CDELT1': ['CDELT1', 'CHAN_BW', 'DELTAF', 'DFREQ'],
    'CRPIX1': ['CRPIX1', 'REFPIX', 'CRP1'],
}

COLUMN_ALIASES = {
    'DAT_FREQ': ['DAT_FREQ', 'CHAN_FREQ', 'FREQUENCY', 'DATFREQ', 'FREQS'],
    'EDGE_CHANNEL': ['EDGE_CHANNEL', 'EDGE_CHAN'],
    # Common variants for the main data column
    'DATA': ['DATA', 'data', 'Data', 'SUBDATA'],
}


def get_header_value(header, canonical, default=None):
    for key in HEADER_ALIASES.get(canonical, [canonical]):
        if key in header:
            return header[key]
    return default


def get_column_value(hdu, canonical):
    names = []
    if hasattr(hdu, 'columns'):
        names = hdu.columns.names
    elif hasattr(hdu, 'data') and getattr(hdu.data, 'dtype', None) is not None:
        names = hdu.data.dtype.names or []
    for key in COLUMN_ALIASES.get(canonical, [canonical]):
        if key in names:
            return hdu.data[key]
    return None

