# script to generate single pulses to inject later in real data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from scipy.signal import convolve
import argparse
from your.formats.filwriter import make_sigproc_object
from your import Your
from pathlib import Path
import sys

k_DM = 1 / 2.41e-4 * (u.second * u.MHz**2 * u.cm**3) / u.pc# s MHz^2 cm^3 pc^-1


# single gaussian pulse

def gaussian_pulse(t, t0, sigma, A):
    ''' gaussian pulse with amplitude A, centered at t0, with width sigma
    t: time array
    t0: center of the pulse
    sigma: width of the pulse
    A: amplitude of the pulse
     
    '''
    return A * np.exp(-(t - t0)**2 / (2 * sigma**2))

# singles gaussian pulse with background noise

def gaussian_noise(shape):
    ''' white noise with mean 0 and std 1
    t: time array'''

    if isinstance(shape, np.ndarray):
        shape = shape.shape
    noise = np.random.normal(0, 1, shape)
    return noise
    

def gaussian_pulse_noise(t, t0, sigma, A, noise):
    ''' gaussian pulse plus noise with amplitude A, centered at t0, with width sigma
    t: time array
    t0: center of the pulse
    sigma: width of the pulse
    A: amplitude of the pulse
    noise: noise amplitude
    '''

    return gaussian_pulse(t, t0, sigma, A) + noise

# gaussian pulses for different frequencies with background noise
# generate a data cube with 1000 time points and 1000 frequencies points for 1000 pulses
# each frequency point has a single gaussian pulse 
# the amplitude of the pulses is displayed by a color map


# gaussian pulses for different frequencies with background noise
# generate a data cube with 1000 time points and 1000 frequencies points for 1000 pulses
# each frequency point has a single gaussian pulse centered at a different time given by a frequency inversed 
# square time dependence on time
# the amplitude of the pulses is displayed by a color map

def dispersion_delay(f, fref, DM):
    ''' time delay due to dispersion with respect to fref = 0
    f: frequency
    DM: dispersion measure'''
    
    return k_DM * DM * (-(1 / fref**2) + (1 / f**2)) 


# let´s try to plot some pulses for different bandwidths
# define a function that includes the snr 

def pulse_sim(f, t, t0, DM, std, snr):
    ''' waterfall plot of gaussian pulses with background noise
    f: frequency array
    t: time array
    t0: time of the pulse
    DM: dispersion measure
    std: standard deviation of the gaussian pulse
    snr: signal to noise ratio (amplitude of the pulse divided by the standard deviation of the noise)
    '''
    l = len(f)
    data = np.zeros((l, len(t)))

    tc = dispersion_delay(f, f[-1], DM) + t0
    data = gaussian_pulse_noise(t[None, :], tc[:, None], std, snr,
                                gaussian_noise(tc)[:, None])
    return data


def single_gaussian_pulse(bandpass, t_pulse, tf, channel_width, time_resolution, fwhm, amp, n_of_events=1, subburst=False, scatt=False):
    ''' dispersion plot of pulse
    bandpass: minimum and maximum frequency
    t_pulse: times of the pulse(s)
    tf: maximum time
    channel_width: width of the frequency channels
    time_resolution: time resolution
    fwhm: full width half maximum
    amp: amplitude of the pulse(s)
    n_of_events: number of pulses to generate (default=1)
    subburst: boolean flag to add subburst structure to the pulse(s) (default=False)
    scatt: boolean flag to add scattering broadening (default=False)
    '''
    # Generate time array
    time_bins = (tf / time_resolution).decompose().value.astype(int)
    t = np.linspace(0, tf, time_bins) 
    # Generate frequency array
    n_channels = round(((bandpass[0] - bandpass[1]) / channel_width).decompose().value.astype(int))
    f = np.linspace(bandpass[0], bandpass[-1], n_channels) 
    std = fwhm / (2 * np.sqrt(2 * np.log(2))) 

    data = np.zeros((len(f), len(t)))
    if not isinstance(t_pulse.value, np.ndarray):
        t_pulse = [t_pulse]  # Convert single value to a list

    for i in range(n_of_events):
        t0 = t_pulse[i]

        pulse = gaussian_pulse(t[None, :], t0, std, amp)
        if subburst:
        
            sub_t0 = np.random.uniform((t0 - 0.1*u.s).value, (t0 + 0.1*u.s).value) * t0.unit
            sub_amp_variation = np.random.uniform(0.1, 1)
            sub_amp = sub_amp_variation * amp
            pulse_subburst = gaussian_pulse(t[None, :], sub_t0, std, sub_amp)
        pulse += pulse_subburst
        data += pulse.value
    if scatt:
            data = add_scattering_broadening(data, 0.01)
    return data


def spectral_density_model(f, f_0, beta, running):
    ''' spectral density of a signal with a given beta and zeta
    f: frequency
    f_0: reference frequency
    beta: spectral index
    running: spectral running'''
    
    return (f / f_0).decompose().value.astype(float)**(beta + running * np.log((f / f_0).decompose().value.astype(float)))


def spectral_density_pulse(pulse, spectral_amplitude):
    ''' spectral density of a pulse
    pulse: 2d pulse
    spectral_amplitude: spectral density of the pulse
    '''
    return pulse * spectral_amplitude[:, np.newaxis]
    

def gaussian_2d(x, y, x0, y0, std_x, std_y, amp):
    ''' parameters for the simulation
    x: x array
    y: y array
    x0: center of the gaussian in x
    y0: center of the gaussian in y
    std_x: standard deviation of the gaussian in x
    std_y: standard deviation of the gaussian in y
    amp: amplitude of the gaussian
    '''
    
    g = amp * np.exp(-((x - x0)**2 / (2 * std_x**2) + (y - y0)**2 / (2 * std_y**2)))
    
    return g

def scattering_kernel(t, tscatt):
    """Generate the scattering kernel for a given scattering time.

    Parameters:
        t (array): Time array.
        tscatt (float): Scattering time in seconds.

    Returns:
        array: Scattering kernel.
    """
    kernel = np.exp(-np.abs(t) / tscatt)
    return kernel


def add_scattering_broadening(pulse, tscatt, scattering_fraction=0.5):
    """Add scattering broadening to the pulse.

    Parameters:
        pulse (array): Pulse data.
        tscatt (float): Scattering time in seconds.
        scattering_fraction (float): Fraction of pulse width to which scattering is applied.

    Returns:
        array: Pulse with scattering broadening.
    """
    n_freq, n_time = pulse.shape
    t = np.linspace(0, n_time, n_time, endpoint=False) / n_time

    # Calculate scattering kernel
    kernel = scattering_kernel(t, tscatt)

    # Convolve pulse with scattering kernel in the time domain only on the right side
    right_side_mask = t >= scattering_fraction * t.max()
    convolved_pulse = np.apply_along_axis(convolve, 1, pulse, kernel * right_side_mask, mode='same')

    # Calculate the sum of the original pulse and the convolved pulse
    pulse_sum = np.sum(pulse.value)
    convolved_sum = np.sum(convolved_pulse)

    # Calculate the scaling factor to preserve the pulse amplitude
    scale = pulse_sum / convolved_sum

    # Scale the convolved pulse to maintain the same amplitude
    convolved_pulse *= scale

    return convolved_pulse


def pulse2d(bandpass, t_tot, t_p, f_m, fwhm_f, fwhm_t, channel_width, time_resolution, amp,
            scatt=False, n_of_events=1, subburst=False):
    '''
    bandpass: min and max frequency of the bandpass
    t_tot: times of start and end of the observation
    t_p: time of the pulse(s)
    f_m: frequency at which the pulse(s) is centered
    fwhm_f: full width half max of the pulse(s) in frequency
    fwhm_t: full width half max of the pulse(s) in time
    channel_width: width of the frequency channels
    time_resolution : time resolution of the instrument
    amp: amplitude of the pulse(s)
    scatt: boolean flag to add scattering broadening (default=False)
    n_of_events: number of pulses to generate (default=1)
    subburst: boolean flag to add subburst structure (default=False)
    '''
    n_channels = round(((bandpass[0] - bandpass[1]) / channel_width).decompose().value.astype(int))
    f = np.linspace(*bandpass, n_channels, endpoint=False) 
    t = np.linspace(0, t_tot, round((t_tot / time_resolution).decompose().value.astype(int))) 
    std_f = fwhm_f / (2 * np.sqrt(2 * np.log(2)))
    std_t = fwhm_t / (2 * np.sqrt(2 * np.log(2)))
    tt, ff = np.meshgrid(t, f)

    twodpulse = np.zeros((n_channels, len(t)))

    # if not isinstance(t_p.value, np.ndarray):
    #     t_p = [t_p]  # Convert single value to a list
    #     f_m = [f_m]  # Convert single value to a list
    #     fwhm_f = [fwhm_f]  # Convert single value to a list
    #     std_t = [std_t]  # Convert single value to a list
    #     std_f = [std_f]  # Convert single value to a list
    for i in range(n_of_events):
        pulse = gaussian_2d(tt, ff, t_p[i], f_m[i], std_t[i], std_f[i], amp)
        if subburst:
            sub_t_p = np.random.uniform((t_p[i] - 0.1*u.s).value, (t_p[i] + 0.1*u.s).value) * t_p[i].unit
            sub_f_m = np.random.uniform((f_m[i] - fwhm_f[i]).value, (f_m[i] + fwhm_f[i]).value) * f_m[i].unit
            sub_amp = np.random.uniform(0.1 * amp, 0.5 * amp) 
            sub_pulse = gaussian_2d(tt, ff, sub_t_p, sub_f_m, std_t[i], std_f[i], sub_amp)
            pulse += sub_pulse

        if scatt:
            pulse = add_scattering_broadening(pulse, 0.01)

        twodpulse += pulse.value

    return twodpulse


def spectral_model_pulse(bandpass, t_pulse, tf, channel_width, time_resolution, fwhm, spectral_index, spectral_running, amplitude,
                         f0=1200*u.MHz, n_of_events=1, subburst=False, scatt=False):
    """
    Generate a simulated pulse with a given spectral model
    """
    
    time_bins = (tf / time_resolution).decompose().value.astype(int)
    t = np.linspace(0, tf, time_bins) 
    # Generate frequency array
    n_channels = round(((bandpass[0] - bandpass[1]) / channel_width).decompose().value.astype(int))
    f = np.linspace(bandpass[0], bandpass[-1], n_channels) 
    std = fwhm / (2 * np.sqrt(2 * np.log(2))) 
    amp = spectral_density_model(f, f0, spectral_index, spectral_running)
    data = np.zeros((len(f), len(t)))
    print(amp[:, np.newaxis])
    if not isinstance(t_pulse.value, np.ndarray):
        t_pulse = [t_pulse]  # Convert single value to a list
    for i in range(n_of_events):
        t0 = t_pulse[i]
        pulse = amp[:, np.newaxis] * gaussian_pulse(t[None, :], t0, std, amplitude)
        if subburst:                
            sub_t0 = np.random.uniform((t0 - 0.1*u.s).value, (t0 + 0.1*u.s).value) * t0.unit
            sub_amp_variation = np.random.uniform(0.1, 1, size=1)
            sub_amp = sub_amp_variation * amp
            sub_amplitude = sub_amp_variation * amplitude
            pulse_subburst = sub_amp[:, np.newaxis] * gaussian_pulse(t[None, :], sub_t0, std, sub_amplitude)

        pulse += pulse_subburst

        data += pulse.value

    if scatt:
        data = add_scattering_broadening(data, 0.01)

    return data


def incoherent_dispersion_pulse(pulse, bandpass, fref, dm, time_resolution, channel_width):
    ''' incoherent dispersion of a pulse
    pulse: 2d pulse
    bandpass: min and max frequency of the bandpass
    fref: reference frequency
    dm: dispersion measure
    time_resolution: time resolution of the instrument
    chennel_width: width of the frequency channels
    '''
    n_channels = round(((bandpass[0] - bandpass[1]) / channel_width).decompose().value.astype(int))
    f = np.linspace(*bandpass, n_channels, endpoint=False)

    time_delay = dispersion_delay(f, fref, dm)
    time_bin = (time_delay / time_resolution).decompose().value.astype(int)
    
    new_pulse = np.zeros_like(pulse)
   
    for i, s in enumerate(time_bin):
        
        new_pulse[i, :] = np.roll(pulse[i, :], s)

    return new_pulse
        

def frequencies(fi, ff, f_sr):
    ''' parameters for the simulation
    fi: initial frequency [MHz]
    ff: final frequency [MHz]
    f_sr: sampling rate in frequency [MHz]
    '''
    ff.to(u.MHz)
    fi.to(u.MHz)
    f_sr.to(u.MHz)
    n_channels = int((ff - fi) / f_sr) 
    f = np.linspace(fi, ff, n_channels) 
    return f

def time_frame(ti, tf, t_sr):
    ''' parameters for the simulation
    ti: initial time [s]
    tf: final time [s]
    t_sr: sampling rate in time [s]
    '''
    ti.to(u.s)
    tf.to(u.s)
    t_sr.to(u.s)
    time_samples = int((tf - ti) / t_sr)
    t = np.linspace(ti, tf, time_samples) 
    return t

def dm_trials(dmi, dmf, dm_step):
    ''' dispersion measure trials
    dmi: initial dispersion measure
    dmf: final dispersion measure
    dm_step: step in dispersion measure
    '''
    dmi.to(u.pc / u.cm**3)
    dmf.to(u.pc / u.cm**3)
    dm_step.to(u.pc / u.cm**3)
    n_dms = int((dmf - dmi) / dm_step) + 1
    dm = np.linspace(dmi, dmf, n_dms) 
    return dm


def plot_pulse(pulse, bandpass, t_i, t_p, t_tot, ts=False, sed=False):
  
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

    if not isinstance(t_p.value, np.ndarray):
     t_p = [t_p]  # Convert single value to a list

    plim = [1, 99]  # Percentiles for intensity scaling
    f = np.linspace(*bandpass, pulse.shape[0], endpoint=False).value
    t = np.linspace(0, t_tot, pulse.shape[1]).value
    #t_0 = Time(t_i, precision=6)
    #t_pulse = (t_0 + t_p[0]).to_value("iso", subfmt="date_hms")
    t_pulse = t_i + t_p[0]
    vmin, vmax = np.nanpercentile(pulse, plim[0]), np.nanpercentile(pulse, plim[1])

    if ts and not sed:
        time_series_pulse = np.nanmean(pulse, axis=0)
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.25, 1]}, sharex=True)
        fig.suptitle('Pulse at ' + str(t_pulse))

        # Plot Time Series
        axs[1].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1].set_ylabel('Frequency (MHz)')
        axs[1].set_xlabel('Time (s)')
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(axs[1].collections[0], cax=cbar_ax)
        cbar.set_label('Intensity')

        axs[0].plot(t, time_series_pulse)
        #axs[0].set_ylabel('Intensity')

    elif sed and not ts:
        sed_pulse = np.nanmean(pulse, axis=1)
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.1]}, sharey=True)
        fig.suptitle('Pulse at ' + str(t_pulse))

        # Plot Spectral Energy Density
        im = axs[0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[0].set_ylabel('Frequency (MHz)')
        axs[0].set_xlabel('Time (s)')

        # Add colorbar on the right
        cbar_ax = fig.add_axes([0.95, 0.11, 0.02, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity')

        axs[1].plot(sed_pulse, f)
        #axs[1].set_xlabel('Intensity')
        axs[1].tick_params(axis='x', rotation=-90)
        plt.subplots_adjust(wspace=0.1)

    elif ts and sed:
        fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.1], 'height_ratios': [0.25, 1]}, sharex='col', sharey='row')
        fig.suptitle('Pulse at ' + str(t_pulse))

        # Plot Time Series
        im_pulse = axs[1, 0].pcolormesh(t, f, pulse, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        axs[1, 0].set_ylabel('Frequency (MHz)')
        axs[1, 0].set_xlabel('Time (s)')
        time_series_pulse = np.nanmean(pulse, axis=0)
        axs[0, 0].plot(t, time_series_pulse)
        #axs[0, 0].set_ylabel('Intensity')
        # Plot Spectral Energy Density
        sed_pulse = np.nanmean(pulse, axis=1)
        axs[1, 1].plot(sed_pulse, f)
        #axs[1, 1].set_xlabel('Intensity')
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
        ax.set_title('Pulse at ' + str(t_pulse))
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity')

    return fig

def create_dispersed_pulse(bandpass, t_tot, t_p, f_m, fwhm_f, fwhm_t, channel_width, time_resolution, amp, fref, dm,
                           scatt=False, n_of_events=1, subburst=False):
    ''' create a dispersed pulse
    bandpass: min and max frequency of the bandwidth
    t_tot: total time of the observation 
    t_p: time of the pulse after the start of the observation
    f_m: frecuency at which the pulse is centered
    fwhm_f: full width half max of the pulse in frequency
    fwhm_t: full width half max of the pulse in time
    channel_width: width of the frequency channels
    time_resolution : time resolution of the instrument
    amp: amplitude of the pulse
    fref: reference frequency
    dm: dispersion measure
    '''
    centered_pulse = pulse2d(bandpass, t_tot, t_p, f_m, fwhm_f, fwhm_t, channel_width, time_resolution, amp, scatt=scatt, 
                             n_of_events=n_of_events, subburst=subburst)
    dispersed_pulse = incoherent_dispersion_pulse(centered_pulse, bandpass, fref, dm, time_resolution, channel_width)
    #no_disp_bin = int((t_p / time_resolution).decompose().value)
    return dispersed_pulse


def injection(data, pulse):
    ''' inject a pulse into the data
    data: data to inject the pulse into
    pulse: pulse to inject
    '''
    return data + pulse


def make_injection(data_for_injection, number_of_events=1, bandpass=np.array([1800, 1200]) * u.MHz,
                   initial_time_of_observation='2023-02-14 12:08:35', time_resolution=0.01 * u.s,
                   t_p=np.array([30, 236]) * u.s, f_m=1400 * u.MHz, fwhm_f=np.array([100, 400]) * u.MHz,
                   fwhm_t=np.array([0.01, 0.09]) * u.s, amplitude=np.array([20, 300]),
                   fref=0, dms=np.array([300, 500]) * (u.pc / u.cm**3), channel_width=0.29296875 * u.MHz,
                   mask=False, seed: int = None):
    ''' create dispersed pulses and inject them into the data
    data_for_injection: data to inject the pulse into
    number_of_events: number of dispersed pulses to generate
    bandpass: min and max frequency of the observation band
    time_resolution: time resolution of the instrument
    t_p: range of times in which the pulse can occur
    f_m: frequency at which the pulse is centered
    fwhm_f: full width half max of the pulse in frequency
    fwhm_t: range of full width half max of the pulse in time
    amplitude: amplitude of the pulse
    fref: reference frequency
    dms: range of dispersion measures
    channel_width: width of the frequency channels
    '''
    if seed is not None:
        np.random.seed(seed)

    n_channels = data_for_injection.shape[0]
    t_tot = data_for_injection.shape[1] * time_resolution

    cumulative_injection = np.zeros_like(data_for_injection)

    headers = []

    for _ in range(number_of_events):
        t_pulse = np.random.uniform(t_p[0].value, t_p[1].value) * t_p.unit
        width = np.random.uniform(fwhm_t[0].value, fwhm_t[1].value) * fwhm_t.unit
        f_width = np.random.uniform(fwhm_f[0].value, fwhm_f[1].value) * fwhm_f.unit
        amp = np.random.uniform(amplitude[0], amplitude[1])
        f_reference = bandpass[fref]
        dm = np.random.uniform(dms[0].value, dms[1].value) * dms.unit
        #std_noise = np.std(data_for_injection[1360:1490, :])
        
        centered_pulse = pulse2d(bandpass, t_tot, t_pulse, f_m, f_width, width, channel_width, time_resolution, amp, subburst=True)
        dispersed_pulse = incoherent_dispersion_pulse(centered_pulse, bandpass, f_reference, dm, time_resolution, channel_width)
        if mask:

            masked_channels = np.all(data_for_injection == 0, axis=1)
            dispersed_pulse[masked_channels, :] = 0.0

        cumulative_injection += dispersed_pulse
        #sample_minus = int((t_pulse - 0.1 * width) / time_resolution)
        #sample_plus = int((t_pulse + 0.1 * width) / time_resolution)
        #timeseries_avg = np.mean(dispersed_pulse[:, sample_minus:sample_plus], axis=0)
        #snr = (np.max(timeseries_avg)) / std_noise
        dict_for_header = {}
        dict_for_header["Arrival"] = t_pulse.value
        dict_for_header["Ttime"] = t_tot.value
        dict_for_header["AmpPulse"] = amp
        #dict_for_header['SNR'] = snr
        dict_for_header["PulseW"] = width.value
        dict_for_header["DM"] = dm.value
        dict_for_header["FreqC"] = f_m.value
        dict_for_header["FwhmF"] = f_width.value
        dict_for_header["FreqR"] = f_reference.value
        dict_for_header["TimeRes"] = time_resolution.value
        dict_for_header["MinF"] = bandpass[1].value
        dict_for_header["MaxF"] = bandpass[0].value
        dict_for_header["InitT"] = initial_time_of_observation
        dict_for_header["ChannW"] = channel_width.value
        dict_for_header["Nchan"] = n_channels
        dict_for_header['seed'] = seed

        headers.append(dict_for_header)

    return data_for_injection + cumulative_injection, headers


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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate and plot 2D pulse')
    
    # Add command-line arguments
    parser.add_argument('--bandpass', type=float, nargs=2, default=[1200, 1800],
                        help='Min and max frequency of the bandpass (MHz)')
    parser.add_argument('--fref', type=int, default=1800
                        , help='Reference frequency (MHz)')
    parser.add_argument('--t_i', type=float, default=0,  # change this to start in seconds
                        help='Start time of the observation (s)')
    parser.add_argument('--t_tot', type=float, default=1,
                        help='Total time of the observation (s)')
    parser.add_argument('--t_p', type=float, nargs='+', default=[0.5],
                        help='Time of the pulse(s) (s)')
    parser.add_argument('--f_m', type=float, default=[1400], nargs='+',
                        help='Frequency at which the pulse(s) is centered (MHz)')
    parser.add_argument('--fwhm_f', type=float, default=[400], nargs='+',
                        help='Full width half max of the pulse(s) in frequency (MHz)')
    parser.add_argument('--fwhm_t', type=float, default=[0.015], nargs='+',
                        help='Full width half max of the pulse(s) in time (s)')
    parser.add_argument('--channel_width', type=float, default=0.29296875,
                        help='Width of the frequency channels (MHz)')
    parser.add_argument('--time_resolution', type=float, default=0.001,
                        help='Time resolution of the instrument (s)')
    parser.add_argument('--dm', type=float, default=300,
                        help='Dispersion measure (pc / cm^3)')
    parser.add_argument('--amplitude', type=float, default=50,
                        help='Amplitude of the pulse(s)')
    parser.add_argument('--scatt', action='store_true',
                        help='Add scattering broadening')
    parser.add_argument('--n_of_events', type=int, default=1,
                        help='Number of pulses to generate')
    parser.add_argument('--subburst', action='store_true',
                        help='Add subburst structure')
    parser.add_argument('--ts', action='store_true',
                        help='Generate time series plot')
    parser.add_argument('--sed', action='store_true',
                        help='Generate spectral energy density plot')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--noise', action='store_true',
                        help='Add noise to the pulse')
    parser.add_argument('--plot', action='store_true',
                        help='Plot pulse')
    parser.add_argument('--injection', action='store_true',
                        help='Inject pulse into data')
    parser.add_argument('--filename', type=str, help='Name of the file to inject to')
    parser.add_argument('--dedispersed', action='store_true', help='Generate dedispersed pulse')
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.injection:
        file_extension = Path(args.filename).suffix
        if args.filename is None:
            raise ValueError('Filename must be provided')
        elif file_extension != '.fits' and file_extension != '.fil':
            raise ValueError('Filename must have .fits or .fil extension')
        else:
            filename_without_extension = Path(args.filename).stem
            new_filename = filename_without_extension + '_injected' + file_extension
            file = Your(args.filename)
            t_start = file.your_header.tstart_utc
            total_sample = args.t_tot / args.time_resolution
            file_read = file.get_data(nstart=0, nsamp=total_sample)
            # calculate noise level by obtainig a normalized time series of a region without signal
            time_series = file_read.T[1360:1490, :]
            #normalized_time_series = (time_series - np.mean(time_series)) / np.std(time_series)
            noise_level = np.std(time_series)
            # obtain amplitude values for the pulse between 5 and 30 times the noise level
            five_times_noise =  noise_level / 3
            thirty_times_noise =  noise_level / 2

            data_injection, header = make_injection(file_read.T, number_of_events=args.n_of_events, 
                                                    bandpass=args.bandpass*u.MHz, initial_time_of_observation=t_start,
                                                    time_resolution=args.time_resolution*u.s, 
                                                    amplitude=np.array([five_times_noise, thirty_times_noise]),
                                                    f_m=args.f_m*u.MHz,
                                                    channel_width=args.channel_width*u.MHz, mask=False,
                                                    seed=args.seed)
            sigproc_object = make_sigproc_object(
                rawdatafile=new_filename,
                source_name=file.your_header.source_name,
                nchans=file.your_header.nchans, 
                foff=file.your_header.foff,
                fch1=file.your_header.fch1,
                tsamp=file.your_header.tsamp,
                tstart=file.your_header.tstart,
                src_raj=230909.6,  # HHMMSS.SS
                src_dej=484201.0,  # DDMMSS.SS
                machine_id=0,
                nbeams=0,
                ibeam=0,
                nbits=32,
                nifs=1,
                barycentric=0,
                pulsarcentric=0,
                telescope_id=1,
                data_type=0,
                az_start=-1,
                za_start=-1,
            )
            
            f = open('file.txt', 'w')
            f.write(str(header))
            f.close()
            
            sigproc_object.write_header(new_filename)
            sigproc_object.append_spectra(spectra=data_injection.T, filename=new_filename) 
            print('Data injected in', new_filename)
            
            sys.exit(0)

    else:
        
        pulse = pulse2d(args.bandpass*u.MHz, args.t_tot*u.s, args.t_p*u.s, args.f_m*u.MHz, args.fwhm_f*u.MHz,
                     args.fwhm_t*u.s, args.channel_width*u.MHz,args.time_resolution*u.s, args.amplitude, 
                     args.scatt, args.n_of_events, args.subburst)
        
        disp_pulse = incoherent_dispersion_pulse(pulse, args.bandpass*u.MHz, args.fref*u.MHz, 
                                             args.dm*(u.pc / u.cm**3) , args.time_resolution*u.s, args.channel_width*u.MHz)
        if args.noise:
            noise = np.random.normal(50, 20, disp_pulse.shape)
            pulse += noise
            disp_pulse = disp_pulse + noise
        if args.plot and not args.dedispersed:
            fig = plot_pulse(disp_pulse, args.bandpass*u.MHz, args.t_i, args.t_p*u.s, args.t_tot*u.s, args.ts, args.sed)   
            plt.show()
        elif args.plot and args.dedispersed:
            fig = plot_pulse(pulse, args.bandpass*u.MHz, args.t_i, args.t_p*u.s, args.t_tot*u.s, args.ts, args.sed)
            plt.show()
    
        
    

if __name__ == '__main__':
    main()