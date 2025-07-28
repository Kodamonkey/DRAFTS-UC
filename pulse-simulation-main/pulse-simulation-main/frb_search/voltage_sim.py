import numpy as np
import matplotlib.pyplot as plt

import frb_search.simulate_pulse as sp
import astropy.units as u
import scipy
import argparse


def de_channelize(vp):
    """
    This functions is NOT an inverse for the PFB, only an approximation.
    Returns real voltage by de-channelize baseband data. This is a rough
    approximation for the Polyphase Filter Bank (PFB). Due to the Nyquist
    sampling it is necessary to add an extra zero term on the baseband data,
    getting a 1025 channels. This real voltage is particularly useful for
    using the mathematical cross-correlation and finding the time lag between
    two sets of baseband data up to 1.25 $\\mathrm{ns}$ time bin.
    Parameters
    """

    vp = np.nan_to_num(vp)

    zeros_add = np.zeros(vp.shape[-1], dtype=vp.dtype).reshape(1, -1)
    vp_zeros_add = np.vstack((vp, zeros_add))
    
    # inverting the array so delay matches the regular visibility
    #ts = scipy.fft.irfft(vp_zeros_add, axis=0).ravel(order="F")[::-1]
    ts = scipy.fft.irfft(vp_zeros_add, axis=0).ravel(order="F")
    
    return ts


def parse_args():
    parser = argparse.ArgumentParser(description='Generate and plot 2D pulse')
    
    # Add command-line arguments
    parser.add_argument('--bandpass', type=float, nargs=2, default=[1200, 1800],
                        help='Min and max frequency of the bandpass (MHz)')
    parser.add_argument('--fref', type=int, default=1800
                        , help='Reference frequency (MHz)')
    parser.add_argument('--t_i', type=str, default='2021-01-01T00:00:00.123456789',
                        help='Start time of the observation (s)')
    parser.add_argument('--t_tot', type=float, default=1,
                        help='Total time of the observation (s)')
    parser.add_argument('--t_p', type=float, nargs='+', default=[0.5],
                        help='Time of the pulse(s) (s)')
    parser.add_argument('--f_m', type=float, default=1400,
                        help='Frequency at which the pulse(s) is centered (MHz)')
    parser.add_argument('--fwhm_f', type=float, default=400,
                        help='Full width half max of the pulse(s) in frequency (MHz)')
    parser.add_argument('--fwhm_t', type=float, default=0.015,
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
    parser.add_argument('--axis', type=int, default=0,
                        help='Axis for time series plot (0 or 1)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    pulse = sp.pulse2d(args.bandpass*u.MHz, args.t_tot*u.s, args.t_p*u.s, args.f_m*u.MHz, args.fwhm_f*u.MHz, args.fwhm_t*u.s, args.channel_width*u.MHz,
                    args.time_resolution*u.s, args.amplitude, args.scatt, args.n_of_events, args.subburst)
    
    noise = np.random.normal(50, 20, pulse.shape)
    noise = 0
    pulse = pulse + noise

    pulse_c = pulse * np.exp(1j*0)
    
    voltage = de_channelize(pulse_c)
    #t = np.linspace(0, args.t_tot, len(voltage))
    plt.figure()
    plt.plot(voltage)
    plt.show()
    

if __name__ == '__main__':
    main()
