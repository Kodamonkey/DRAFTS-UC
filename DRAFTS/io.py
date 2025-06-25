import numpy as np
from astropy.io import fits 

def load_fits_file(file_name, reverse_flag=False):

    try:
        import fitsio
        data, h  = fitsio.read(file_name, header=True)
    except:
        with fits.open(file_name) as f:
            h    = f[1].header
            data = f[1].data
    data         = data['DATA'].reshape(h['NAXIS2']*h['NSBLK'], h['NPOL'], h['NCHAN'])[:, :2, :]
    if reverse_flag: data = np.array(data[:, :, ::-1])

    return data


### 读取fits头文件，获取观测参数，并指定为全局变量
def get_obparams(file_name):

    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    with fits.open(file_name) as f:
        time_reso  = f[1].header['TBIN'] 
        freq_reso  = f[1].header['NCHAN']
        file_leng  = f[1].header['NAXIS2'] * f[1].header['NSBLK']
        freq       = f[1].data['DAT_FREQ'][0, :].astype(np.float64)
    down_freq_rate = int(freq_reso / 512)
    down_time_rate = int((49.152 * 16 / 1e6) / time_reso)
