import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import sigpyproc.readers as readers
from matplotlib.backends.backend_pdf import PdfPages
from your import Your
import your.formats.psrfits as psrfits
#from memory_profiler import profile
from pathlib import Path


k_DM = 1 / 2.41e-4 * (u.second * u.MHz**2 * u.cm**3) / u.pc
def dispersion_delay(f, fref, DM):
    ''' time delay due to dispersion with respect to fref = 0
    f: frequency
    DM: dispersion measure'''
    
    return k_DM * DM * (-(1 / fref**2) + (1 / f**2))


def incoherent_dedispersion_pulse(pulse, bandpass, fref, dm, time_resolution, channel_width):
    ''' incoherent dispersion of a pulse
    pulse: 2d pulse
    bandpass: min and max frequency of the bandpass
    fref: reference frequency
    dm: dispersion measure
    time_resolution: time resolution of the instrument
    channel_width: width of the frequency channels
    '''
    n_channels = round(((bandpass[0] - bandpass[1]) / channel_width).decompose().value.astype(int))
    f = np.linspace(*bandpass, n_channels, endpoint=False)

    time_delay = dispersion_delay(f, fref, dm)
    time_bin = (time_delay / time_resolution).decompose().value.astype(int)
    
    new_pulse = np.zeros_like(pulse)
   
    for i, s in enumerate(time_bin):
        
        new_pulse[i, :] = np.roll(pulse[i, :], -s)

    return new_pulse


def bin_data(arr_data, frame_bin, freq_bin):
    """
    Bin intensity data along the time and frequency axes.
    Parameters
    ----------
    arr_data : array
        Array of intensity data.
    frame_bin : int
        Number of frames to bin together.
    freq_bin : int
        Number of frequency channels to bin together.
    Returns
    -------
    arr : array
        Binned array of intensity data.
    """
    arr = arr_data.copy()
    arr = np.nanmean(arr.reshape(freq_bin, -1, arr.shape[1]), axis=0)
    arr = np.nanmean(arr.reshape(arr.shape[0], -1, frame_bin), axis=-1)
    
    return arr


#plot the candidates in df for each DM and DM 0 
def plot_pulse(pulse, bandpass, t_p, t_start, t_end, ts=False, sed=False):
  
    """Plots the pulse with optional time series and/or spectral energy density.

    Args:
    pulse: 2D pulse data.
    bandpass: Minimum and maximum frequency of the bandpass.
    t_i: Time of the start of the observation.
    t_p: Time of the peak of the pulse.
    t_tot: Total time of the observation.
    ts: Whether to plot the time series (default: False).
    sed: Whether to plot the spectral energy density (default: False).
    axis: Axis along which to average for time series and calculate SED (default: 0).
    """

    plim = [1, 99]  # Percentiles for intensity scaling
    f = np.linspace(*bandpass, pulse.shape[0], endpoint=False).value
    t = np.linspace(t_start, t_end, pulse.shape[1]).value
    vmin, vmax = np.nanpercentile(pulse, plim[0]), np.nanpercentile(pulse, plim[1])

    if ts and not sed:
        time_series_pulse = np.nanmean(pulse, axis=0)
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.25, 1]}, sharex=True)
        fig.suptitle('Waterfall Plot at ' + str(t_p))

        # Plot Time Series
        axs[1].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1].set_ylabel('Frequency (MHz)')
        axs[1].set_xlabel('Time (s)')
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(axs[1].collections[0], cax=cbar_ax)
        cbar.set_label('Intensity')

        axs[0].plot(t, time_series_pulse)
        axs[0].set_ylabel('Intensity')

    elif sed and not ts:
        sed_pulse = np.nanmean(pulse, axis=1)
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.1]}, sharey=True)
        fig.suptitle('Waterfall Plot at ' + str(t_p))

        # Plot Spectral Energy Density
        im = axs[0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[0].set_ylabel('Frequency (MHz)')
        axs[0].set_xlabel('Time (s)')

        # Add colorbar on the right
        cbar_ax = fig.add_axes([0.95, 0.11, 0.02, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity')

        axs[1].plot(sed_pulse, f)
        axs[1].set_xlabel('Intensity')
        axs[1].tick_params(axis='x', rotation=-90)
        plt.subplots_adjust(wspace=0.1)

    elif ts and sed:
        fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.1], 'height_ratios': [0.25, 1]}, sharex='col', sharey='row')
        fig.suptitle('Waterfall Plot at ' + str(t_p))

        # Plot Time Series
        im_pulse = axs[1, 0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1, 0].set_ylabel('Frequency (MHz)')
        axs[1, 0].set_xlabel('Time (s)')
        time_series_pulse = np.nanmean(pulse, axis=0)
        axs[0, 0].plot(t, time_series_pulse)
        axs[0, 0].set_ylabel('Intensity')
        # Plot Spectral Energy Density
        sed_pulse = np.nanmean(pulse, axis=1)
        axs[1, 1].plot(sed_pulse, f)
        axs[1, 1].set_xlabel('Intensity')
        axs[1, 1].tick_params(axis='x', rotation=-90)
        axs[0, 1].axis('off')
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.585])
        cbar = fig.colorbar(im_pulse, cax=cbar_ax)
        cbar.set_label('Intensity')
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

    else:
        fig, ax = plt.subplots()
        im = ax.pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('Waterfall Plot at ' + str(t_p))
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity')

    return fig


def read_fil_block(timestamp, plot_window, filename, dm, downsample, ts=False):
    '''
    plot the waterfall plot of the data
    timestamp: start time to plot
    plot_window: time window to plot
    filename: name of the filterbank file
    dm: dispersion measure of the candidate
    ts: whether to plot the time series or not
    '''
   
    fil = readers.FilReader(filename)
    header = fil.header
    t_delay = np.abs(dispersion_delay(header.fbottom*u.MHz, header.ftop*u.MHz, dm))
    start_sample = int((timestamp) / (header.tsamp * u.s))
    delay_samples = int(t_delay / (header.tsamp * u.s))
    window_sample = int(plot_window / (header.tsamp * u.s))
    end_sample = int(header.nsamples)
    total_sample = delay_samples + window_sample
    samples_left = end_sample - start_sample
    if total_sample > samples_left:

      data = fil.read_block(start_sample, samples_left)
      data_dedisp = data.dedisperse(dm.value)
      data_ = data_dedisp[:, :window_sample]
      new_size = data_.shape[1] // downsample * downsample
      data_ = data_[:, :new_size].downsample(downsample)
    else:
      data = fil.read_block(start_sample, total_sample)
      data_dedisp = data.dedisperse(dm.value)
      data_ = data_dedisp[:, :window_sample]
      new_size = data_.shape[1] // downsample * downsample
      data_ = data_[:, :new_size].downsample(downsample)
    
    return data_, header


def read_fits_block(timestamp, plot_window, filename, dm, downsample, stokes=0):
    '''
    plot the waterfall plot of the data from fits file
    timestamp: start time to plot
    plot_window: time window to plot
    filename: name of the fits file
    dm: dispersion measure of the candidate
    downsample: downsample factor
    ts: whether to plot the time series or not
    '''
    #fits = readers.PFITSReader(filename)
    #fits = Your(filename)
    fits = psrfits.PsrfitsFile([filename])  # check compatibility with many files from frb_search
    header = fits.header
    #top_freq = (header.center_freq + (np.abs(header.bw) / 2)) * u.MHz
    fch1 = fits.fch1
    flip = False
    #bottom_freq = (header.center_freq - (np.abs(header.bw) / 2)) * u.MHz
    if fits.foff < 0:
        top_freq = fch1 * u.MHz
        bottom_freq = (fch1 + fits.nchans * fits.foff) * u.MHz
    else:
        top_freq = (fch1 + fits.nchans * fits.foff) * u.MHz
        bottom_freq = fch1 * u.MHz
        flip = True
    tsamp = fits.native_tsamp() * u.s
    foff = np.abs(fits.foff) * u.MHz
    npol = fits.npol
    total_time = fits.nspectra() * tsamp
    t_delay = np.abs(dispersion_delay(bottom_freq, top_freq, dm))
    start_sample = int((timestamp) / (tsamp))
    delay_samples = int(t_delay / (tsamp))
    window_sample = int(plot_window / (tsamp))
    end_sample = int(fits.nspectra())
    total_sample = delay_samples + window_sample
    samples_left = end_sample - start_sample
    
    if total_sample > samples_left:
            
        data = fits.get_data(start_sample, samples_left, npoln=npol)
        if flip:
            data = data[:, :, ::-1]
        #if npol == 4:
        data = data[:, stokes, :]
        data = data.swapaxes(0, 1)
        #data_dedisp = data.dedisperse(dm.value)      
        data_dedisp = incoherent_dedispersion_pulse(data, [top_freq, bottom_freq], top_freq, dm, tsamp, foff)
        data_ = data_dedisp[:, :window_sample]
        #print(data_.shape)
        new_size = data_.shape[1] // downsample * downsample
        data_ = data_[:, :new_size]
        data_ = bin_data(data_, downsample, 1)

    else:
        #data = fits.read_block(start_sample, total_sample)
        data = fits.get_data(start_sample, total_sample, npoln=npol)
        if flip:
            data = data[:, :, ::-1]
        #if npol == 4:
        data = data[:, stokes, :]
        data = data.swapaxes(0, 1)
        #data_dedisp = data.dedisperse(dm.value)      
        data_dedisp = incoherent_dedispersion_pulse(data, [top_freq, bottom_freq], top_freq, dm, tsamp, foff)
        data_ = data_dedisp[:, :window_sample]
        #print(data_.shape)
        new_size = data_.shape[1] // downsample * downsample
        data_ = data_[:, :new_size]
        data_ = bin_data(data_, downsample, 1)

    return data_, total_time, [top_freq, bottom_freq], tsamp

#@profile
def plot_pulse_0_dm(filename, pulse, pulse_0, bandpass, t_p, t_start, t_end, dm, sigma, width, ts=False, sed=False):
    """
    Plots the pulse and pulse_0 with optional time series and/or spectral energy density.

    Args:
    pulse: 2D pulse data.
    pulse_0: 2D pulse data for pulse_0.
    bandpass: Minimum and maximum frequency of the bandpass.
    t_p: Time of the peak of the pulse.
    t_start: Time of the start of the plot window
    t_end: end of plot window
    dm: DM of the candidate
    ts: Whether to plot the time series (default: False).
    sed: Whether to plot the spectral energy density (default: False).
    """
# TODO: with gridspec making each pulse with its time series, sed and colorbar then 
# joining them with plt.subplots will improve spacing between subplots to add colorbar, labels, etc. 
    plim = [1, 99]  # Percentiles for intensity scaling
    f = np.linspace(*bandpass, pulse.shape[0], endpoint=False)
    t = np.linspace(t_start, t_end, pulse.shape[1])
    vmin, vmax = np.nanpercentile(pulse, plim[0]), np.nanpercentile(pulse, plim[1])
    vmin_0, vmax_0 = np.nanpercentile(pulse_0, plim[0]), np.nanpercentile(pulse_0, plim[1])
    first_line = t_p - width / 2
    second_line = t_p + width / 2
    if ts and not sed:
        time_series_pulse = np.nanmean(pulse, axis=0)
        time_series_pulse_0 = np.nanmean(pulse_0, axis=0)
        fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [0.25, 1]}, sharex='col', figsize=(10, 6))
        fig.suptitle(f'Candidate at {t_p}, Sigma: {sigma}, Width: {width.to("ms").value:.2f} ms from file {filename}')
        # Plot Time Series for pulse
        axs[1, 0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1, 0].set_ylabel('Frequency (MHz)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].axvline(first_line.value, color='red', linestyle='--', linewidth=1)
        axs[1, 0].axvline(second_line.value, color='red', linestyle='--', linewidth=1)
        axs[0, 0].plot(t, time_series_pulse)
        axs[0, 0].set_title('Pulse at DM ' + str(dm.value))

        # Plot Time Series for pulse_0
        im_pulse_0 = axs[1, 1].pcolormesh(t, f, pulse_0, shading='auto', vmin=vmin_0, vmax=vmax_0, rasterized=True)
        axs[1, 1].set_ylabel('Frequency (MHz)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[0, 1].plot(t, time_series_pulse_0)
        #axs[0, 1].axis('off')
        axs[0, 1].set_title('Pulse at DM ' + str(0))
        plt.subplots_adjust(hspace=0.1, wspace=0.4)

    elif sed and not ts:
        sed_pulse = np.nanmean(pulse, axis=1)
        sed_pulse_0 = np.nanmean(pulse_0, axis=1)
        fig, axs = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 0.25, 1, 0.25]}, sharey=True, figsize=(10, 6))
        fig.suptitle(f'Candidate at {t_p}, Sigma: {sigma}, Width: {width.to("ms").value:.2f} ms from file {filename}')
        # Plot Spectral Energy Density for pulse
        im_pulse = axs[0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[0].set_ylabel('Frequency (MHz)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_title('Pulse at DM ' + str(dm.value))
        axs[0].axvline(first_line.value, color='red', linestyle='--', linewidth=1)
        axs[0].axvline(second_line.value, color='red', linestyle='--', linewidth=1)

        # Plot Spectral Energy Density for pulse_0
        im_pulse_0 = axs[2].pcolormesh(t, f, pulse_0, shading='auto', vmin=vmin_0, vmax=vmax_0, rasterized=True)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('Pulse at DM ' + str(0))

        axs[3].plot(sed_pulse_0, f)
        axs[3].tick_params(axis='x', rotation=-90)
        axs[1].plot(sed_pulse, f)
        axs[1].tick_params(axis='x', rotation=-90)
        plt.subplots_adjust(wspace=0.1)  

    elif ts and sed:
        fig, axs = plt.subplots(2, 4, gridspec_kw={'width_ratios': [1, 0.1, 1, 0.1], 'height_ratios': [0.25, 1]}, sharex='col', figsize=(10, 6))
        fig.suptitle(f'Candidate at {t_p}, Sigma: {sigma}, Width: {width.to("ms").value:.2f} ms from file {filename}')
        # Plot Time Series for pulse
        im_pulse = axs[1, 0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1, 0].set_ylabel('Frequency (MHz)')
        axs[1, 0].set_xlabel('Time (s)')
        time_series_pulse = np.nanmean(pulse, axis=0)
        axs[0, 0].plot(t, time_series_pulse)
        axs[0, 0].set_title('Pulse at DM ' + str(dm.value))
        axs[1, 0].axvline(first_line.value, color='red', linestyle='--', linewidth=1)
        axs[1, 0].axvline(second_line.value, color='red', linestyle='--', linewidth=1)

        # Plot Spectral Energy Density for pulse
        sed_pulse = np.nanmean(pulse, axis=1)
        axs[1, 1].plot(sed_pulse, f)
        #axs[1, 1].set_xlabel('Intensity')
        axs[1, 1].tick_params(axis='x', rotation=-90)
        axs[0, 1].axis('off')
        axs[1, 1].tick_params(labelleft=False)
        axs[1, 1].set_ylim(axs[1, 0].get_ylim())

        # Plot Time Series for pulse_0
        im_pulse_0 = axs[1, 2].pcolormesh(t, f, pulse_0, shading='auto', vmin=vmin_0, vmax=vmax_0, rasterized=True)
        axs[1, 2].set_xlabel('Time (s)')
        time_series_pulse_0 = np.nanmean(pulse_0, axis=0)
        axs[0, 2].plot(t, time_series_pulse_0)
        axs[1, 2].tick_params(labelleft=False)
        axs[0, 2].set_title('Pulse at DM ' + str(0))

        # Plot Spectral Energy Density for pulse_0
        sed_pulse_0 = np.nanmean(pulse_0, axis=1)
        axs[1, 3].plot(sed_pulse_0, f)
        axs[1, 3].tick_params(axis='x', rotation=-90)
        axs[0, 3].axis('off')
        axs[1, 3].tick_params(labelleft=False)
        axs[1, 3].set_ylim(axs[1, 0].get_ylim())
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

    else:
      fig, axs = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1]})
      fig.suptitle(f'Candidate at {t_p}, Sigma: {sigma}, Width: {width.to("ms").value:.2f} ms from file {filename}')
      # Plot Spectral Energy Density for pulse
      im_pulse = axs[0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
      axs[0].set_xlabel('Time (s)')
      axs[0].set_ylabel('Frequency (MHz)')
      axs[0].set_title('Pulse at DM ' + str(dm.value))
      axs[0].axvline(first_line.value, color='red', linestyle='--', linewidth=1)
      axs[0].axvline(second_line.value, color='red', linestyle='--', linewidth=1)

      # Plot Spectral Energy Density for pulse_0
      im_pulse_0 = axs[1].pcolormesh(t, f, pulse_0, shading='auto', vmin=vmin_0, vmax=vmax_0, rasterized=True)
      axs[1].set_xlabel('Time (s)')
      axs[1].set_title('Pulse at DM ' + str(0))

    return fig

#@profile
def plot_waterfall_from_df(filename, df, plot_window, downsample, filename_extension, num_to_plot, from_candidate,
                           stokes, ts=False, sed=False, save=False):
    '''
    plot the waterfall plots of candidates in df

    Input:
        filename: name of the filterbank file
        df: pandas dataframe containing the candidates
        plot_window: time window to plot
        downsample: downsample factor
        ts: whether to plot the time series or not
        sed: whether to plot the spectral energy density or not
        save: whether to save the plot or not
    Output:
        plots of the candidates
    '''
    filename_without_extension = os.path.splitext(filename)[0]
    pdf_filename = 'candidates' + filename_without_extension + '_' + str(from_candidate) + '.pdf'
    if from_candidate > 0:
        df = df[from_candidate:]
        df = df.reset_index(drop=True)
    #plot the candidates in df for each DM and DM 0
    with PdfPages(pdf_filename) as pdf:
        for candidate in range(min(len(df), num_to_plot)):
            timestamp_candidate = df['Time(s)'][candidate] * u.s
            timestamp = timestamp_candidate - plot_window / 2
            timestamp = max(timestamp, 0 * u.s)
            dm = df['DM'][candidate] * u.pc / u.cm**3
            sigma = df['Sigma'][candidate]
            downfact_width = df['Downfact'][candidate]

            if filename_extension == '.fits':
                data_dm , total_time, bandpass, tsamp = read_fits_block(timestamp, plot_window, filename, dm, downsample, stokes=stokes)
                bandpass = [bandpass[0].value, bandpass[1].value] 
                data_0 = read_fits_block(timestamp, plot_window, filename, 0 * u.pc / u.cm**3, downsample, stokes=stokes)[0]
                data_dm = normalize(data_dm)
                data_0 = normalize(data_0)
                width = downfact_width * tsamp * downsample               
               
            else:

                data_dm , header = read_fil_block(timestamp, plot_window, filename, dm, downsample, ts=ts)
                data_0 = read_fil_block(timestamp, plot_window, filename, 0 * u.pc / u.cm**3, downsample, ts=ts)[0]
                data_dm = normalize(data_dm)
                data_0 = normalize(data_0)
                bandpass = [header.chan_freqs[0], header.chan_freqs[-1]]
                total_time = header.nsamples * header.tsamp * u.s
                tsamp = header.tsamp * downsample * u.s
                width = downfact_width * tsamp

            t_start = timestamp.value
            if timestamp + plot_window > total_time:
                plot_window = (total_time - timestamp)
            t_end = timestamp.value + plot_window.value
            fig = plot_pulse_0_dm(filename_without_extension, data_dm, data_0, bandpass, timestamp_candidate, t_start, t_end, dm, sigma, width, ts=ts, sed=sed)
            if save:         
                    pdf.savefig(fig)
                        #plt.savefig('candidate_' + str(candidate) + '_DM' + str(dm.value) + '.png')
                    plt.close()                       
            else:
                plt.show()         
            

import warnings
def normalize(arr_data):
    arr = arr_data.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arr -= np.nanmedian(arr, axis=-1)[..., None]
        arr /= np.nanstd(arr, axis=-1)[..., None]
    return arr


def parser_args():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Plot candidates from singlepulse search')
    parser.add_argument('--filename', type=str, help='name of the filterbank file')
    parser.add_argument('--dm', type=float, default=0, help='DM of the candidate')
    parser.add_argument('--timestamp', type=float, default=0, help='timestamp of the candidate')
    parser.add_argument('--plot_window', type=float, help='time window to plot')
    parser.add_argument('--save', action='store_true', help='save the plot')
    parser.add_argument('--timeseries', action='store_true', help='plot the time series')
    parser.add_argument('--sed', action='store_true', help='plot the spectral average')
    parser.add_argument('--from_file', nargs='?', default=False, help='read from file')
    parser.add_argument('--num_to_plot', type=int, default=150, help='number of candidates to plot')
    parser.add_argument('--stokes', type=int, default=0, help='stokes parameter: 0 for I, 1 for Q, 2 for U, 3 for V')
    parser.add_argument('--downsample', type=int, default=1, help='downsample factor')
    parser.add_argument('--from_candidate', type=int, default=0, help='start from candidate number')
    args = parser.parse_args()
    return args

#@profile
def main():
    args = parser_args()
    filename = args.filename
    dm = args.dm * u.pc / u.cm**3
    timestamp = args.timestamp * u.s
    plot_window = args.plot_window * u.s
    save = args.save
    ts = args.timeseries
    sed = args.sed
    from_file = args.from_file
    num_to_plot = args.num_to_plot
    downsample = args.downsample
    stokes = args.stokes
    from_candidate = args.from_candidate
    file_extension = Path(filename).suffix
    if not from_file:
        if file_extension == '.fits':
            data, total_time, bandpass, _ = read_fits_block(timestamp, plot_window, filename, dm, downsample, stokes=stokes)
            data = normalize(data)
            if timestamp + plot_window > total_time:
                plot_window = (total_time - timestamp)
            #bandpass = [header.fbottom, header.ftop]
            fig = plot_pulse(data, bandpass, timestamp, timestamp, timestamp + plot_window, ts=ts, sed=sed)
            if save:
                plt.savefig('candidate_DM' + str(dm.value) + '.png')
                plt.close()
            else:
                plt.show()
        elif file_extension == '.fil':
            data, header = read_fil_block(timestamp, plot_window, filename, dm, downsample, ts)
            data = normalize(data)
            bandpass = [header.chan_freqs[0], header.chan_freqs[-1]] * u.MHz
            total_time = header.nsamples * header.tsamp * u.s
            if timestamp + plot_window > total_time:
                plot_window = (total_time - timestamp)
            fig = plot_pulse(data, bandpass, timestamp, timestamp, timestamp + plot_window, ts=ts, sed=sed)
            if save:
                plt.savefig('candidate_DM' + str(dm.value) + '.png')
                plt.close()
            else:
                plt.show()
        else:
            print('File extension not supported, please use .fil or .fits')
    else: 
        df = pd.read_csv(from_file)
        plot_waterfall_from_df(filename, df, plot_window, downsample, file_extension, num_to_plot, from_candidate,
                               stokes=stokes, ts=ts, sed=sed, save=save)
    
        
    
    
    
if __name__ == '__main__':
    main()

