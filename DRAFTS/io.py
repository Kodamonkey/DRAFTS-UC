import warnings
import numpy as np
from astropy.io import fits

from .aliases import get_header_value, get_column_value

def load_fits_file(file_name, reverse_flag=False):
    """Load FITS data in ``(time, pol, chan)`` order."""

    try:
        import fitsio
        data, h = fitsio.read(file_name, header=True)
    except Exception:
        with fits.open(file_name) as f:
            h = f[1].header
            data = f[1].data

    nsubint = int(get_header_value(h, "NAXIS2", 1))
    nsblk = int(get_header_value(h, "NSBLK", 1))
    npol = int(get_header_value(h, "NPOL", 2))
    nchan = int(get_header_value(h, "NCHAN", 0))

    data = data["DATA"].reshape(nsubint * nsblk, npol, nchan)[:, :2, :]
    if reverse_flag:
        data = np.ascontiguousarray(data[:, :, ::-1])

    return data


### 读取fits头文件，获取观测参数，并指定为全局变量
def get_obparams(file_name):
    """Extract observation parameters from ``file_name``."""

    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    with fits.open(file_name) as f:
        hdr = f[1].header
        primary = f[0].header

        time_reso = float(get_header_value(hdr, "TBIN", 0.0))
        freq_reso = int(get_header_value(hdr, "NCHAN", 0))
        file_leng = int(get_header_value(hdr, "NAXIS2", 1)) * int(
            get_header_value(hdr, "NSBLK", 1)
        )

        col = get_column_value(f[1], "DAT_FREQ")
        if col is not None:
            freq = col[0].astype(np.float64)
        else:
            obsfreq = get_header_value(primary, "OBSFREQ", 1400.0)
            obsbw = get_header_value(primary, "OBSBW", freq_reso)
            chan_bw = get_header_value(hdr, "CHAN_BW", obsbw / max(freq_reso, 1))
            start = obsfreq - obsbw / 2.0 + float(chan_bw) / 2.0
            freq = np.linspace(start, start + (freq_reso - 1) * abs(chan_bw), freq_reso)

        # Validate consistency
        obs_nchan = get_header_value(primary, "OBSNCHAN")
        if obs_nchan and abs(int(obs_nchan) - freq_reso) > 0:
            warnings.warn(
                f"OBSNCHAN ({obs_nchan}) inconsistent with NCHAN ({freq_reso})"
            )

        chan_bw = get_header_value(hdr, "CHAN_BW")
        obsbw = get_header_value(primary, "OBSBW")
        if chan_bw is not None and obsbw is not None:
            if abs(float(chan_bw) * freq_reso - float(obsbw)) > max(1e-6, 0.01 * abs(float(obsbw))):
                warnings.warn("CHAN_BW * NCHAN inconsistent with OBSBW")

    down_freq_rate = int(freq_reso / 512) if freq_reso else 1
    if time_reso:
        down_time_rate = int((49.152 * 16 / 1e6) / time_reso)
    else:
        down_time_rate = 1
