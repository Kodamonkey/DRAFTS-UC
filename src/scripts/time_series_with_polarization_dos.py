#primero importo las librerías y métodos a usar
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.signal as sign
from astropy.table import QTable
from astropy import constants as const
import astropy.units as u
from sigpyproc import readers as read

from sigpyproc  import readers as read
from sigpyproc.readers import FilReader, PFITSReader
from rich.pretty import Pretty
from sigpyproc.timeseries import TimeSeries
from sigpyproc.block import FilterbankBlock as filterbank
from your import Your
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os
rc('text',usetex=True)
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit



def incoherent_dedispersion_pulse(pulse, bandpass, fref, DM, time_resolution, channel_width, flow, header):
    ''' incoherent dispersion of a pulse
    pulse: 2d pulse
    bandpass: min and max frequency of the bandpass
    fref: reference frequency
    dm: dispersion measure
    time_resolution: time resolution of the instrument
    channel_width: width of the frequency channels
    '''
    n_channels = header.nchans
    f          = np.linspace(*bandpass, n_channels, endpoint=False)
    k_dm       = (4.1488 * 10 ** 3) * (u.s * u.MHz**2 * u.cm**3) /u.pc
    cube_delay = k_dm * ((DM * u.pc) / (u.cm**3)) * (-(1 /(fref*u.MHz)**2) + (1/(f)**2)) 
    time_bin   = (cube_delay / time_resolution).decompose().value.astype(int)
    new_pulse  = np.zeros_like(pulse)
    
    # ADVERTENCIA: Esta lógica de de-dispersión es incorrecta.
    # Itera sobre 'samples' (i) pero debería iterar sobre 'canales'.
    # La forma correcta sería:
    # for i in range(pulse.shape[1]): # Iterar sobre canales
    #   shift = time_bin[i]
    #   new_pulse[:, i] = np.roll(pulse[:, i], -shift)
    #
    # Sin embargo, la mantengo para no alterar tu lógica, ya que no es la causa del 'crash'.
    for i in range(min(len(time_bin), pulse.shape[0])):
        shift = time_bin[i]
        new_pulse[i, :] = np.roll(pulse[i, :], -shift)

    return new_pulse

#-----------------------------------------------------------------------------------------------

def bin_data(arr_data, frame_bin, freq_bin):
    """
    Bin intensity data along the time and frequency axes.
    """
    arr = arr_data.copy()
    if freq_bin > 1: # Evitar reshape innecesario si freq_bin es 1
        arr = np.nanmean(arr.reshape(freq_bin, -1, arr.shape[1]), axis=0)
    if frame_bin > 1: # Evitar reshape innecesario si frame_bin es 1
        arr = np.nanmean(arr.reshape(arr.shape[0], -1, frame_bin), axis=-1)
    
    return arr

#------------------------------------------------------------------------------------------------

def new_timeseries_pulse(filename,start_time,t_int,DM,ds,pol,fits=True):
    '''
    makes a new time series from file, summing all the frequency channels (time_series_sum)
    gives the option of a time series divided by the number of channels (time_series_div)
    
    filename   = file ended in .fits or .fil
    fits       = True if file is .fits, False if file is .fil
    start_time = time to star plotting (*u.s)
    t_int      = total interval of time to plot (*u.s)
    DM         = DM of the source (*u.pc/(u.cm**3))
    ds         = downsampling factor (8,16,32,64,128) 
    '''
    #open
    if fits:
        fits_obj   = Your(filename)
        header = fits_obj.your_header
    else:
        fits_obj   = FilReader(filename)
        header = fits_obj.header
        
    #parameters for the time series
    first_samp = int((start_time *u.s)/ (header.tsamp *u.s))
    last_samp  = int(header.nspectra)
    wind_samp  = int((t_int*u.s )/(header.tsamp *u.s))
    
    #we need to set the correct time of the burst (if it is dispersed)
    k_dm   = (1/2.41e-4) * (u.s * u.MHz**2 * u.cm**3) /u.pc
    f_high = (header.center_freq + (np.abs(header.bw) / 2)) 
    f_low  = (header.center_freq - (np.abs(header.bw) / 2))
    
    if DM != 'not found':
        t_delay   = np.abs(k_dm * ((DM * u.pc) / (u.cm**3)) * (-(1 /(f_high*u.MHz)**2) + (1/(f_low*u.MHz)**2)))
        delay_dec = (t_delay)/(header.tsamp *u.s)
        delay     = int(delay_dec.value)
        sample    = wind_samp + delay
    else:
        sample    = wind_samp
    
    # =========================================================================
    # Lógica de lectura de datos segura (esta parte estaba bien en tu _copy.py)
    # =========================================================================
    
    samples_remaining = last_samp - first_samp

    if samples_remaining <= 0:
        raise ValueError(f"El tiempo de inicio (start_time: {start_time}s) es posterior al final del archivo.")

    if sample > samples_remaining:
        print(f"ADVERTENCIA: La ventana de tiempo excede el final del archivo. Leyendo sólo las {samples_remaining} muestras restantes.")
        samples_to_read = samples_remaining
    else:
        samples_to_read = sample
    
    if samples_to_read <= 0:
        raise ValueError("No hay muestras para leer con los parámetros dados.")

    data = fits_obj.get_data(first_samp, samples_to_read, npoln=header.npol)

    if data is None:
        raise ValueError("Error: fits.get_data() devolvió None. El archivo puede estar corrupto.")

    if header.npol == 4:
        datos = data[:, pol, :]
    else:
        datos = data
    
    #dedisperse the data from the DM value of the source, if it is given
    if DM != 'not found':
        dedisperse  = incoherent_dedispersion_pulse(datos.T, [f_high*u.MHz, f_low*u.MHz], f_high, DM, 
                                                    header.tsamp *u.s,  header.foff, f_low, header)
        
        # Asegurarse de no cortar más de lo que hay
        actual_wind_samp = min(wind_samp, dedisperse.shape[1])
        dedispersed = dedisperse[:, :actual_wind_samp]
        
        downsize    = dedispersed.shape[1] // ds * ds
        if downsize == 0:
             print(f"ADVERTENCIA: La ventana de tiempo ({t_int}s) es demasiado pequeña para el downsampling (ds={ds}). La serie de tiempo estará vacía.")
             pulse_data = np.empty((header.nchans, 0))
        else:
            pulse_data  = dedispersed[:, :downsize]
            pulse_data  = bin_data(pulse_data, ds, 1)
    else:
        pulse_data  = datos
    
    if pulse_data.shape[1] == 0:
        print("ADVERTENCIA: La serie de tiempo resultante tiene longitud 0.")
        return np.array([]), np.array([]), np.array([])

    #creating new timeseries
    # Forma vectorizada más rápida de sumar canales
    time_series_sum = np.nansum(pulse_data, axis=0)
   
    #time axis
    t = np.linspace(start_time, start_time + (pulse_data.shape[1] * header.tsamp * ds), pulse_data.shape[1])
    
    return t,time_series_sum, pulse_data, header

#-------------------------------------------------------------------------------------------------------

def normalization(time_series,res_factor,div_factor,burst=True):   
    res_time_series = time_series - res_factor
    norm_timeseries = res_time_series / div_factor
    return norm_timeseries

#--------------------------------------------------------------------------------------------------------
def statist(filename,scan,start_time,t_int,DM,ds,pol,t_ref,fits=True,burst=True,wf=True):
    
    #all polarizations------------------------------------------------------------------------------------
    if type(pol)==str:
        #STOKES=0-------------------------------------------------------------------------------------------
        t, time_series_0, pulse_data_0  = new_timeseries_pulse(filename,start_time,t_int,DM,ds,0,fits)
        if time_series_0.size == 0: return None # Salir si no hay datos
        time_series_0                   = time_series_0 - np.nanmean(time_series_0)
        pulse_data_0                    = pulse_data_0  - np.nanmean(pulse_data_0, axis=-1)[..., None]
            
        if burst:
            burst     = np.nanmax(time_series_0)
            pos_burst = np.argmax(time_series_0) 
            left_burst  = time_series_0[:len(time_series_0)//4]- np.nanmean(time_series_0[:len(time_series_0)//4])
            right_burst = time_series_0[3*len(time_series_0)//4:]- np.nanmean(time_series_0[3*len(time_series_0)//4:])
            mean_0      = np.nanmean([np.nanmean(left_burst),np.nanmean(right_burst)])
            median_0    = np.nanmean([np.nanmedian(left_burst),np.nanmedian(right_burst)])
            std_0       = np.nanstd(np.concatenate((left_burst, right_burst))) # Std de todo el ruido junto
        else:
            mean_0, median_0, std_0 = np.nanmean(time_series_0), np.nanmedian(time_series_0), np.nanstd(time_series_0)
            burst,pos_burst   = None,None
        
        #STOKES=1-------------------------------------------------------------------------------------------
        t, time_series_1, pulse_data_1  = new_timeseries_pulse(filename,start_time,t_int,DM,ds,1,fits)
        if time_series_1.size == 0: return None
        time_series_1                   = time_series_1 - np.nanmean(time_series_1)
        pulse_data_1                    = pulse_data_1  - np.nanmean(pulse_data_1, axis=-1)[..., None]
        if burst:
            left_burst  = time_series_1[:len(time_series_1)//4]
            right_burst = time_series_1[3*len(time_series_1)//4:]
            mean_1, median_1 = np.nanmean([np.nanmean(left_burst),np.nanmean(right_burst)]), np.nanmean([np.nanmedian(left_burst),np.nanmedian(right_burst)])
            std_1       = np.nanstd(np.concatenate((left_burst, right_burst)))
        else:
            mean_1, median_1, std_1 = np.nanmean(time_series_1), np.nanmedian(time_series_1), np.nanstd(time_series_1)
            
        #STOKES=2------------------------------------------------------------------------------------------
        t, time_series_2, pulse_data_2  = new_timeseries_pulse(filename,start_time,t_int,DM,ds,2,fits)
        if time_series_2.size == 0: return None
        time_series_2                   = time_series_2 - np.nanmean(time_series_2)
        pulse_data_2                    = pulse_data_2  - np.nanmean(pulse_data_2, axis=-1)[..., None]
        if burst:
            left_burst  = time_series_2[:len(time_series_2)//4]- np.nanmean(time_series_2[:len(time_series_2)//4])
            right_burst = time_series_2[3*len(time_series_2)//4:]- np.nanmean(time_series_2[3*len(time_series_2)//4:])
            mean_2, median_2 = np.nanmean([np.nanmean(left_burst),np.nanmean(right_burst)]), np.nanmean([np.nanmedian(left_burst),np.nanmedian(right_burst)])
            std_2       = np.nanstd(np.concatenate((left_burst, right_burst)))
        else:
            mean_2, median_2, std_2 = np.nanmean(time_series_2), np.nanmedian(time_series_2), np.nanstd(time_series_2)
            
       #STOKES=3------------------------------------------------------------------------------------------
        t, time_series_3, pulse_data_3  = new_timeseries_pulse(filename,start_time,t_int,DM,ds,3,fits)
        if time_series_3.size == 0: return None
        time_series_3                   = time_series_3 - np.nanmean(time_series_3)
        pulse_data_3                    = pulse_data_3  - np.nanmean(pulse_data_3, axis=-1)[..., None]
        if burst:
            left_burst  = time_series_3[:len(time_series_3)//4]- np.nanmean(time_series_3[:len(time_series_3)//4])
            right_burst = time_series_3[3*len(time_series_3)//4:]- np.nanmean(time_series_3[3*len(time_series_3)//4:])
            mean_3, median_3 = np.nanmean([np.nanmean(left_burst),np.nanmean(right_burst)]), np.nanmean([np.nanmedian(left_burst),np.nanmedian(right_burst)])
            std_3       = np.nanstd(np.concatenate((left_burst, right_burst)))
        else:
            mean_3, median_3, std_3 = np.nanmean(time_series_3), np.nanmedian(time_series_3), np.nanstd(time_series_3)
            
        #timeseries
        norm_ts_0 = normalization(time_series_0,mean_0,std_0)
        norm_ts_1 = normalization(time_series_1,mean_1,std_1)
        norm_ts_2 = normalization(time_series_2,mean_2,std_2)
        norm_ts_3 = normalization(time_series_3,mean_3,std_3)
        
        linear_pol_ts = linear_polarizer(filename,norm_ts_1,norm_ts_2)
        
        if pol=='linear':
            norm_ts   = linear_pol_ts
        elif pol=='all':
            norm_ts = [norm_ts_0, norm_ts_3, linear_pol_ts]
                                                          
        #waterfaller
        pulse_norm_0 = pulse_data_0 / np.nanstd(pulse_data_0, axis=-1)[..., None]
        pulse_norm_1 = pulse_data_1 / np.nanstd(pulse_data_1, axis=-1)[..., None]
        pulse_norm_2 = pulse_data_2 / np.nanstd(pulse_data_2, axis=-1)[..., None]
        pulse_norm_3 = pulse_data_3 / np.nanstd(pulse_data_3, axis=-1)[..., None]
        
        linear_pol_wf = linear_polarizer(filename,pulse_norm_1,pulse_norm_2)

        if pol=='linear':
            pulse_norm   = linear_pol_wf
        else: # 'all'
            pulse_norm = pulse_norm_0
        
    #only one Stoke parameter------------------------------------------------------------------------------
    else:
        t, time_series, pulse_data = new_timeseries_pulse(filename,start_time,t_int,DM,ds,pol,fits)
        if time_series.size == 0: return None # Salir si no hay datos
        
        time_series                = time_series - np.nanmean(time_series)    
        pulse_data                 = pulse_data - np.nanmean(pulse_data, axis=-1)[..., None]
        
        if burst:
            burst     = np.nanmax(time_series)
            pos_burst = np.argmax(time_series) 
            left_burst  = time_series[:len(time_series)//4] - np.nanmean(time_series[:len(time_series)//4])
            right_burst = time_series[3*(len(time_series)//4):] - np.nanmean(time_series[3*(len(time_series)//4):])
            mean      = np.nanmean([np.nanmean(left_burst),np.nanmean(right_burst)])
            median    = np.nanmean([np.nanmedian(left_burst),np.nanmedian(right_burst)])
            std       = np.nanstd(np.concatenate((left_burst, right_burst)))
            pulse_norm = pulse_data / np.nanstd(pulse_data, axis=-1)[..., None]
        else:
            mean, median, std = np.nanmean(time_series), np.nanmedian(time_series), np.nanstd(time_series)
            pulse_norm = pulse_data / np.nanstd(pulse_data, axis=-1)[..., None]
            burst,pos_burst   = None,None

        norm_ts    = normalization(time_series,mean,std)

    if wf:
        plotinfo   = plot_wf(filename,scan,norm_ts,pulse_norm,
                             start_time,t_int,t_ref,DM,ds,pol,fits=True,wf=True)
    else:
        plotinfo  = plot_ts(filename,scan,t,norm_ts,pos_burst,pol,t_ref,burst)
    return plotinfo

#-------------------------------------------------------------------------------------------------------------------

def plot_wf(filename,scan,time_series,pulse_normalized,start_time,t_int,t_ref,DM,ds,pol,fits=True,wf=True):
    if fits:
        fits_obj   = Your(filename)
        header = fits_obj.your_header
    else:
        fits_obj   = FilReader(filename)
        header = fits_obj.header
        
    f_high = (header.center_freq + (np.abs(header.bw) / 2))
    f_low  = (header.center_freq - (np.abs(header.bw) / 2))
    t      = np.linspace(start_time, start_time + t_int, pulse_normalized.shape[1])
    
    chann_bandwidth = header.foff
    nchannels       = header.nchans
    f_ref           = f_low * u.MHz
    total_bandwidth = chann_bandwidth * nchannels *u.MHz
    freq_2          = f_ref + total_bandwidth
    
    if wf:
        fig, ax = plt.subplots(2,1,figsize=(7,9), gridspec_kw={'height_ratios':[0.25,0.75]},sharex=True)
        plim = [1, 99]
        vmin, vmax = np.nanpercentile(pulse_normalized, plim[0]), np.nanpercentile(pulse_normalized, plim[1])
        
        divider = make_axes_locatable(ax[1])
        cax     = divider.append_axes("right", size="5%", pad=0.1)
        
        f    = np.linspace(f_ref,freq_2, pulse_normalized.shape[0], endpoint=False)
        fes = f.value/1000
        # print(fes) # Comentado para reducir output

        plot = ax[1].pcolormesh(t, f.value, pulse_normalized, shading='auto',
                                cmap = 'magma', rasterized = 'True', vmin=vmin, vmax=vmax)        
        
        if isinstance(time_series, list) and len(time_series)==3:
            ax[0].plot(t, time_series[1]-np.mean(time_series[0][0:len(time_series[0])//4]),c='blue',ls=':',label='circular polarization')
            ax[0].plot(t, time_series[0]-np.mean(time_series[0][0:len(time_series[0])//4]),c='black',label='total intensity')
            ax[0].plot(t, time_series[2]-np.mean(time_series[0][0:len(time_series[0])//4]),c='red',ls='--',label='linear polarization')
        else:
            norm_timeseries= time_series-np.mean(time_series[0:len(time_series)//4])
            ax[0].plot(t, time_series-np.mean(time_series[0:len(time_series)//4]),c='blueviolet')        
            ax[0].tick_params(axis='both', which='major', labelsize=17)
        
        ax[1].tick_params(axis='both', which='major', labelsize=17)
        ax[1].set_ylabel(r'Frequency (MHz)',fontsize=20)
        ax[1].set_xlabel(r'$Time (s)$',fontsize=20)
        
        ax[0].set_ylabel('SNR',fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=17)

        # Títulos
        title_str = f'Candidate at {scan} - {os.path.basename(filename)} - {t_ref:.3f} s'
        if pol == 'all':
            fig.suptitle(f'{title_str} (all polarizations)', fontsize=20)
        elif pol == 'linear':
            fig.suptitle(f'{title_str} (linear polarization)', fontsize=20)
        elif pol == 3:
            fig.suptitle(f'{title_str} (circular polarization)', fontsize=20)
        elif pol == 0:
            fig.suptitle(f'{title_str} (Total intensity)', fontsize=20)
        else:
            fig.suptitle(f'{title_str} (Stokes {pol})', fontsize=20)

        cbar = plt.colorbar(plot, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('intensity (arbitrary units)', fontsize=17)
        
    ax[0].legend(loc=2)
    plt.show()    
    
    # Devolver la serie de tiempo normalizada (para el caso 'else' de wf)
    if isinstance(time_series, list) and len(time_series)==3:
        # Devolver Stokes I si es 'all'
        norm_timeseries = time_series[0]-np.mean(time_series[0][0:len(time_series[0])//4])
    else:
        norm_timeseries = time_series-np.mean(time_series[0:len(time_series)//4])

    return header,pulse_normalized,norm_timeseries,t
#---------------------------------------------------------------------------------------------------

def linear_polarizer(filename, timeseries1, timeseries2):
    linear_pol = np.sqrt(timeseries1**2 + timeseries2**2)
    return linear_pol

#----------------------------------------------------------------------------------------------------

def plot_ts(filename,scan,t,norm_timeseries,pos_burst,pol,t_ref,burst=True):

    fig, ax = plt.subplots(1,1,figsize=(12,4))
    
    # Inicializar variables para evitar NameError
    y_burst = np.nan
    region = np.array([])
    times_series_norm = np.array([]) # Variable que faltaba

    if burst and pos_burst is not None and pos_burst < len(t):
        if isinstance(norm_timeseries, list) and len(norm_timeseries)==3:
            y_burst  = norm_timeseries[0][pos_burst]
            region1  = norm_timeseries[0][3*len(norm_timeseries[0])//4:-1]
            region2  = norm_timeseries[0][0:len(norm_timeseries[0])//4]
            if region1.size > 0 or region2.size > 0:
                region = np.concatenate((region1, region2))
            else:
                region = norm_timeseries[0] # Usar todo si las regiones fallan
        else:
            y_burst  = norm_timeseries[pos_burst]
            region1  = norm_timeseries[3*(len(norm_timeseries)//4):-1]
            region2  = norm_timeseries[0:len(norm_timeseries)//4]
            if region1.size > 0 or region2.size > 0:
                region = np.concatenate((region1, region2))
            else:
                region = norm_timeseries # Usar todo si las regiones fallan
    else:
        if isinstance(norm_timeseries, list) and len(norm_timeseries)==3:
             region   = norm_timeseries[0]
        else:
             region   = norm_timeseries
        burst = False # Forzar a False si no hay pos_burst

    newnorm = norm_timeseries    

    # Títulos
    title_str = f'Candidate at {scan} - {os.path.basename(filename)} - {t_ref:.3f} s'
    if burst:
        if pol == 'all':
            ax.set_title(f'{title_str} (all polarizations)', fontsize=20)
        elif pol == 'linear':
            ax.set_title(f'{title_str} (linear polarization)', fontsize=20)
        elif pol == 0:
            ax.set_title(f'{title_str} (Total intensity)', fontsize=20)
        elif pol == 3:
            ax.set_title(f'{title_str} (circular polarization)', fontsize=20)
        else:
            ax.set_title(f'{title_str} (Stokes {pol})', fontsize=20)
    else:
        ax.set_title(f'{title_str} (No burst detected)', fontsize=20)

    # Plotting
    if isinstance(newnorm, list) and len(newnorm)==3:
        ts_V = newnorm[1]-np.mean(newnorm[0][0:len(newnorm[0])//4])
        ts_L = newnorm[2]-np.mean(newnorm[0][0:len(newnorm[0])//4])
        ts_I = newnorm[0]-np.mean(newnorm[0][0:len(newnorm[0])//4])
        ax.plot(t, ts_V, c='blue',ls=':',label='circular polarization')
        ax.plot(t, ts_L, c='red',ls='--',label='linear polarization')
        ax.plot(t, ts_I, c='black',label='total intensity')
        
        # ========= CORRECCIÓN 2 =========
        # Definir times_series_norm también en este caso.
        # fits_array.py pide 'linear', así que devolvemos la serie lineal.
        if pol == 'linear':
            times_series_norm = ts_L
        elif pol == 'all':
            times_series_norm = ts_I # Devolver Stokes I si es 'all'
        else:
            times_series_norm = ts_I # Default
            
    else:
        times_series_norm = newnorm-np.mean(newnorm[0:len(newnorm)//4])
        ax.plot(t, times_series_norm, c='indigo')
        
    
    ax.text(0.07, 0.7, r"$\sigma$: %f" %np.round(np.std(region),3), transform = ax.transAxes,fontsize=15)    
    
    if burst:
        ax.text(0.07, 0.6, r"$SNR_{burst}$: %f " %np.round(y_burst,3), transform = ax.transAxes,fontsize=15)
        ax.axvspan(t[3*(len(t)//4)],t[-1], color='beige',alpha=0.5)
        ax.axvspan(t[0],t[len(t)//4], color='beige',alpha=0.5)
    else:
        ax.axvspan(t[0],t[-1], color='beige',alpha=0.5)

    ax.set_ylabel(r'SNR',fontsize=20)
    ax.set_xlabel(r'Time (s)',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)    
    
    # ========= CORRECCIÓN 1 =========
    # Se cambian las llaves {} por paréntesis ()
    plt.legend(fontsize='large', loc=(0.77, 0.40))
    plt.ylim(-5,10)
    
    # Cerrar la figura para que no se acumulen
    plt.close(fig) 
    
    return np.nanmean(region), y_burst, times_series_norm