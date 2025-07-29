import os
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy import units as apu
from astropy.time import Time
from your.formats.filwriter import make_sigproc_object
from your import Your
import argparse
import subprocess
from .config import config

PRESTO_IMAGE_PATH = config['DEFAULT']['presto_image']


class read_10gbe_data():
    """
    Class to read the data comming from the 10Gbe
    """
    def __init__(self, filename):
        """ Filename: name of the file to read from
        """
        self.f = open(filename, 'rb')
        ind = self.find_first_header()
        self.f.seek(ind * 4)
        size = os.path.getsize(filename)
        self.n_spect = (size - ind * 4) // (2052 * 4)

    def find_first_header(self):
        """ Find the first header in the file bacause after the header is the first
        FFT channel.
        """
        data = np.frombuffer(self.f.read(2052 * 4), '>I')
        ind = np.where(data == 0xaabbccdd)[0][0]
        return (ind - 2)

    def get_spectra(self, number):
        """
        number  :   requested number of spectrums
        You have to be aware that you have enough data to read in the n_spect
        """
        spect = np.frombuffer(self.f.read(2052 * 4 * number), '>I')
        spect = spect.reshape([-1, 2052])
        self.n_spect -= number
        spectra = spect[:, 4:]
        header = spect[:, :4]
        ##change even and odd channels (bug from the fpga..)
        even = spectra[:, ::2]
        odd = spectra[:, 1::2]
        spectra = np.array((odd, even))
        spectra = np.swapaxes(spectra.T, 0, 1)
        spectra = spectra.reshape((-1, 2048))
        spectra = spectra.astype(float)
        return spectra, header

    def get_complete(self):
        """
        read the complete data, be carefull on the sizes of your file
        """
        data, header = self.get_spectra(self.n_spect)
        return data, header

    def close_file(self):
        self.f.close()

from scipy.signal import savgol_filter, medfilt
import scipy.signal as signal

def identify_rfi(sample_spect):
    """
    Get the channels with RFI
    """
    #TODO: in the meanwhile we flag the DC values

    flags = np.arange(87).tolist()
    flags = flags+[1024]

    flags += (np.arange(10)+290).tolist() #1235
    flags += (np.arange(10)+225).tolist() #1270
    flags += (np.arange(10)+160).tolist() #1250

    flags += (np.arange(27)+394).tolist()
    flags += (np.arange(5)+455).tolist()
    flags += (np.arange(4)+1024).tolist()
    flags += (np.arange(25)+1135).tolist()
    flags += (np.arange(15)+1155).tolist()
    flags += (np.arange(12)+1175).tolist()
    flags += (np.arange(30)+1210).tolist()
    flags += (np.arange(16)+1275).tolist()
    flags += (np.arange(18)+1325).tolist()
    flags += (np.arange(10)+1367).tolist()
    flags += (np.arange(5)+1381).tolist()
    flags += (np.arange(40)+1420).tolist() # 1439
    flags += (np.arange(296)+1752).tolist()
    flags += (np.arange(256)+1792).tolist()

    # flags = []
    return flags


def get_baseline(sample_spect):
    """
    Obtain the base line for the receiver
    """
    flags = identify_rfi(sample_spect)
    mask = np.ones(2048, dtype=bool)
    mask[flags] = False
    base = savgol_filter(sample_spect, 9, 3) #window 9 points, local pol order 3
    base = base*mask
    return mask, base

def moving_average(data, win_size=64):
    out = np.zeros(len(data)-win_size+1)
    for i in range(len(data)-win_size+1):
        out[i] = np.mean(data[i:win_size+i])
    return out

def get_image_data_temperature(filenames,cal_time=1,spect_time=1e-2,file_time=5 ,decimation=1,
        win_size=32,tails=32, temperature=True):
    """
    filenames   :   list with the names of the plots
    cal_time    :   calibration time at the begining of each file
    spect_time  :   time between two spectra
    file_time   :   complete time of each file in minutes
    tails       :
    temperature :   Return the data in temperature relative to the hot source
    """
    sample = read_10gbe_data(filenames[0])
    sample_spect, header = sample.get_complete()

    sample.close_file()

    P = np.median(sample_spect,axis = 1)
    gradiente = np.abs(np.gradient(P))
    maxim = np.max(gradiente)
    index  = np.where(gradiente== maxim)
    index  =  index[0][0]

    hot_index = index-30
    P_hot = (np.mean(sample_spect[hot_index:hot_index+5,:],axis=0))
    P_hot = savgol_filter(P_hot, 9, 3)

    load_index = index+ 30
    P_load = (np.mean(sample_spect[load_index:load_index+5,:],axis=0))
    P_load = savgol_filter(P_load, 9, 3)

    flags, baseline_load = get_baseline(P_load)
    spect_size = int(file_time*60/spect_time-tails)
    data = np.zeros([len(filenames)*spect_size//decimation, int(flags.shape[0])])
    data_new = np.zeros([len(filenames)*spect_size//decimation, int(flags.shape[0])])
    bases = np.zeros((len(filenames), 2048))
    clip = np.zeros(len(filenames)*spect_size//decimation, dtype=bool)

    for i in range(0, len(filenames)):
        sample = read_10gbe_data(filenames[i])
        sample_spect, header = sample.get_complete()
        sample.close_file()

        P = np.median(sample_spect,axis = 1)
        gradiente = np.abs(np.gradient(P))
        maxim = np.max(gradiente)
        index  = np.where(gradiente== maxim)
        index  =  index[0][0]

        hot_index = index-30
        P_hot = (np.mean(sample_spect[hot_index:hot_index+5,:],axis=0))
        P_hot = savgol_filter(P_hot, 9, 3)

        load_index = index+ 30
        P_load = (np.mean(sample_spect[load_index:load_index+5,:],axis=0))
        P_load = savgol_filter(P_load, 9, 3)

        flags, baseline_load = get_baseline(P_load)
        bases[i,:] = baseline_load

        t_load = 290 #temp amb
        ENR_ns = (14.85+14.74)/2. #dB
        t_hot = 10**((ENR_ns-7.8/2)/10.)*t_load+t_load # revisar descuento de perdidas  #temp ns on

        t_rx = (t_hot*P_load -t_load*P_hot)/(P_hot-P_load)

        aux = sample_spect[:spect_size,:]
        aux = (aux/P_load)*(t_rx+t_load)-t_rx

        dec_size = aux.shape[0]//decimation
        aux = aux[:dec_size*decimation,:].reshape([-1, decimation, aux.shape[1]])
        aux = np.mean(aux.astype(float), axis=1)

        data[i*(spect_size//decimation):(i+1)*(spect_size//decimation),flags] = aux[:,flags]

        data = data[1000:,:]
        mediana = (np.nanmedian(data[:,:],axis=0))
        data_new = np.subtract(data,mediana)


       #now we look at the clipping
        sat = np.bitwise_and(header[:spect_size,1],2**4-1) #just take the cliping values
        sat = sat[:dec_size*decimation].reshape([-1, decimation])
        sat = np.sum(sat, axis=1)
        sat = np.invert(sat==0)

    avg_pow = np.mean(data[:,flags], axis=1)
    avg_pow = moving_average(avg_pow, win_size=win_size)

    data_new /= np.std(data_new, axis=0)  # before data_new /= np.std(data_new[1000:,:],axis=0)

    snr = np.mean(data_new[:,flags], axis=1)
    snr = moving_average(snr, win_size=win_size)

    clip = moving_average(clip, win_size=win_size)
    clip = np.invert(clip==0)

    t = np.arange(len(avg_pow))*spect_time/60.*decimation #time in minutes
    return data, avg_pow, clip, t, bases,flags, data_new,snr


def get_date_from_logfile(path_data, filename):
    """
    Get the date from the logfile

    Parameters
    ----------
    path_data : str
        Path to the data
    filename : str
        Name of the file

    Returns
    -------
    start_time.mjd : float
    
    """
    full_path = path_data / filename
    date = full_path.name.split(" ")[0]
    hms = ":".join(full_path.name.split(" ")[1].split("_"))
    start_time = Time("T".join((date, hms)), format="isot", scale="utc", precision=9)
    return full_path, start_time.mjd


def get_output_path(path_output, full_path):
    """
    Get the output path

    Parameters
    ----------
    path_output : str
        Path to the output
    filename : str
        Name of the file

    Returns
    -------
    path_save_fil : str
    """
    fullpath_without_space = Path(full_path.as_posix().replace(" ", "_"))
    path_save_fil = path_output.joinpath(fullpath_without_space.name + ".fil")
    return path_save_fil


def get_frequencies(bandpass, nchannels):
    """
    Get the frequencies for the channels

    Parameters
    ----------  
    bandpass : tuple or list with the bandpass frequencies in MHz (low, high)
    nchannels : number of channels

    Returns
    -------
    np.linspace(*bandpass, nchannels, endpoint=False)[::-1] : numpy array
    """
    return np.linspace(*bandpass, nchannels, endpoint=False)[::-1]


def write_header_for_fil(filename, telescope, nchans, foff, fch1, tsamp, tstart, nbits):
    """
    Write the header for the filterbank file

    Parameters
    ----------
    filename : str
        Name of the file
    telescope : str
        Name of the telescope
    nchans : int
        Number of frequency channels
    foff : float
        Resolution in frequency (MHz)
    fch1 : float
        Frequency of the first channel (MHz)
    tsamp : float
        Time resolution (s)
    tstart : float
        Start time (MJD)
    nbits : int
        Number of bits
    
    Returns
    -------
    sigproc_object : object
    """
    sigproc_object = make_sigproc_object(
        rawdatafile=filename,
        source_name=telescope,  # This should be the name of the source, but we use telescope name here
        nchans=nchans, 
        foff=foff,
        fch1=fch1,
        tsamp=tsamp,
        tstart=tstart,
        src_raj=230909.6,  # HHMMSS.SS
        src_dej=484201.0,  # DDMMSS.SS
        machine_id=0,
        nbeams=0,
        ibeam=0,
        nbits=nbits,
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=0,
        data_type=0,
        az_start=-1,
        za_start=-1,
    )
    sigproc_object.write_header(filename)
    return sigproc_object


def write_data_to_fil(sigpyproc_object, filename, data):
    """
    Write the data to the filterbank file

    Parameters
    ----------
    sigpyproc_object : object
    filename : str
        Name of the file
    data : numpy array
        Data to write to the file

    Returns
    -------
    sigpyproc_object : object
    """
    sigpyproc_object.append_spectra(spectra=data, filename=filename)
    return 


def flip_data(data):
    """
    Flip the data in the frequency axis
    """
    return np.flip(data, axis=1)


def add_padding(data, padding):
    """
    Add padding to the data
    """
    median_values = np.median(data, axis=0)
    median_rows = np.tile(median_values, (padding, 1))
    return np.vstack((data, median_rows))


def remove_samples_from_data(data, remove_samples):
    """
    Remove samples from the data
    """
    return data[remove_samples:,:]


def check_conversion(filename):
    """
    Check if the conversion was successful
    """
    command = (
        "singularity exec --bind $PWD " + PRESTO_IMAGE_PATH +
        " readfile " + Path(filename).name +
        " > readfile_output.txt 2>&1"
    )
    subprocess.run([command], shell=True)
    with open('readfile_output.txt', 'r') as file:
        lines = file.readlines()
        if 'ERROR' in lines[0]:
            return False
    return True


def parser_args():
    '''
    Function to parse the command line arguments
    
    Input:
        None
    Output:
        args: object containing the command line arguments
    '''
    parser = argparse.ArgumentParser(description='Convert the data from logfile to a .fil file')
    parser.add_argument('--path_data', type=str, default=os.getcwd(), help='path to the data')
    parser.add_argument('--path_output', type=str, default=os.getcwd(), help='path to the output')
    parser.add_argument('--filename', type=str, nargs='+', default=glob.glob('*'),
                        help='Specify filenames or use the default (all files in the current directory)')
    parser.add_argument('--telescope', type=str, help='name of the telescope')
    parser.add_argument('--bandpass', type=float, nargs=2, help='frequencies in MHz')
    parser.add_argument('--freq_chann0', type=int, help='reference frequency in MHz')
    parser.add_argument('--time_resolution', type=float, help='time resolution in seconds')
    parser.add_argument('--nbits', type=int, help='number of bits')
    parser.add_argument('--nchannels', type=int, help='number of channels')
    parser.add_argument('--flip', action='store_true', help='flip the data in the frequency axis')
    parser.add_argument('--padding', type=int, default=0, help='number of rows to add to the data')
    parser.add_argument('--remove_samples', type=int, default=0, help='number of samples to remove from the data')

    return parser.parse_args()


def main():
    args = parser_args()
    path_data = Path(args.path_data)
    path_save = Path(args.path_output)
    filenames = args.filename
    telescope = args.telescope
    bandpass = args.bandpass 
    freq_chann_0 = args.freq_chann0 
    time_resolution = args.time_resolution 
    nbits = args.nbits
    nchannels = args.nchannels
    flip = args.flip
    padding = args.padding
    remove_samples = args.remove_samples
    bandwidth = bandpass[0] - bandpass[1]
    for filename in filenames:
        full_path, start_time = get_date_from_logfile(path_data, filename)
        frequencies = get_frequencies(bandpass, nchannels)
        path_save_fil = get_output_path(path_save, full_path)

        sigproc_object = write_header_for_fil(
            path_save_fil.as_posix(),
            telescope,
            nchannels,
            bandwidth/nchannels,
            frequencies[freq_chann_0],
            time_resolution,
            start_time,
            nbits
        )     
        #spectra = read_10gbe_data(full_path.as_posix()).get_complete()[0]
        data, avg_pow,  clip, t, bases,flags, data_new,snr = get_image_data_temperature([full_path.as_posix()])
        #remove calibration bar
        if remove_samples > 0:
            data = remove_samples_from_data(data, remove_samples)
            print('Removing samples...')

        if padding > 0:
            data = add_padding(data, padding)
            print('Padding data...')

        if flip:
            data = flip_data(data)
            print('Flipping data...')
            
        #spectra = spectra.astype('float' + str(nbits))
        data = data.astype('float' + str(nbits))
        #sigproc_object.append_spectra(spectra=data, filename=path_save_fil.as_posix())
        write_data_to_fil(sigproc_object, path_save_fil.as_posix(), data)
        print('Final shape:', data.shape) 
        print("File:", filename, "completed!")
        #del sigpyproc_object
        # check if the conversion was successful
        if not check_conversion(path_save_fil.as_posix()):
            print("Conversion failed!")
            return

if __name__ == '__main__':
    main()