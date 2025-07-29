from __future__ import division
import sys,os,numpy as np, matplotlib.pyplot as plt, pylab, matplotlib.mlab as mlab, matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.dates as md
import dateutil
import matplotlib
from matplotlib import pyplot as plt
import math
from datetime import datetime
import time
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from astropy.coordinates import *
from astropy.timeseries import LombScargle
from astropy import units as au
import astropy.time as at
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.optimize import curve_fit
import argparse
from scipy import constants
plt.style.use('C:\\Users\\Crist\\OneDrive\\Escritorio\\CBV\\Universidad\\Otoño 2023\\Proyecto Astronómico de Investigación\\pulse-simulation\\frb_search\\paper-sty.mplstyle')

width = 120 / 25.4                 # in. (180 mm max A&A)
height = width / constants.golden  # Golden ratio
figsize = (width, height)

SNR_MIN = 7
BURST_FWHM = 0.001
BURST_FWHM_MILLI_SEC = 1
N_POL = 2

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def sigma(fwhm, n, sn):
    return (fwhm / 2) * np.sqrt(2 / (n * (sn)**2))


def norm_rates(ref_rate, fluence_ref, fluence_100, gamma):
    return ref_rate * (fluence_100 / fluence_ref) ** gamma


def fluence(flux, time):
    return flux * time


def flux(snr, sefd, n_p, t, bw):
    return snr * sefd / np.sqrt(n_p * t * bw)


def calc_rates(n_of_events, duration):
    return n_of_events / duration


burst_fwhm = 0.001  # Full width at half maximum (FWHM) of the burst in Jy (or another appropriate unit)
burst_fwhm_sec = 1  # FWHM of the burst in milli seconds

# Use the variables in your function calls
flux_eff = flux(7, 17, 2, burst_fwhm, 300*10**6)  # checkeado
fluence_eff = fluence(flux_eff, burst_fwhm_sec)

flux_gbt = flux(7, 10, 2, burst_fwhm, 740*10**6)  # checkeado
fluence_gbt = fluence(flux_gbt, burst_fwhm_sec)

flux_ao_wide = flux(7, 3.5, 2, burst_fwhm, 580*10**6)  # checkeado
flux__ao_alfa = flux(7, 3.5, 2, burst_fwhm, 300*10**6)  # checkeado

flux_ao = flux(7, 3.5, 2, burst_fwhm, 580*10**6)  # original 10 s/n
fluence_ao = fluence(flux_ao, burst_fwhm_sec)

fluence_ao_wide = fluence(flux_ao_wide, burst_fwhm_sec)
fluence_ao_alfa = fluence(flux__ao_alfa, burst_fwhm_sec)

flux_apertiff = flux(7, 23.9, 2, burst_fwhm, 200*10**6)  # conflictive
flux_vla = flux(7, 14.2, 2, burst_fwhm, 128*10**6)  # conflictive scholz 2016 ~ 16.2, 14.2 webpage

fluence_apertiff = fluence(flux_apertiff, burst_fwhm_sec)
fluence_vla = fluence(flux_vla, burst_fwhm_sec)

flux_lt = flux(7, 30, 2, burst_fwhm, 400*10**6)  # 30 scholz but rajwade 50
fluence_lt = fluence(flux_lt, burst_fwhm_sec)

flux_fast = flux(7, 1.56, 2, burst_fwhm, 400*10**6)  # 1.56 Jy  # in paper 2, 1.25 original
fluence_fast = fluence(flux_fast, burst_fwhm_sec)

flux_meerkat = flux(7, 11.9, 2, burst_fwhm, 770*10**6)  # close enough, need to check
fluence_meerkat = fluence(flux_meerkat, burst_fwhm_sec)

flux_100 = flux(7, 17, 2, 0.001, 300*10**6)  # Assume a 100-meter telescope identical to Effelsberg
fluence_100 = fluence(flux_100, 1)

flux_dss43 = flux(7, 16, 2, burst_fwhm, 115*10**6)  # original in paper 16, checkeado 17
fluence_dss43 = fluence(flux_dss43, burst_fwhm_sec)

flux_dss63 = flux(7, 25, 1, burst_fwhm, 120*10**6)  #original in paper 25, checkeado 20
fluence_dss63 = fluence(flux_dss63, burst_fwhm_sec)

flux_vla_sband = flux(7, 13.2, 2, burst_fwhm, 1024*10**6)  # 
fluence_vla_sband = fluence(flux_vla_sband, burst_fwhm_sec)

print('Fluence for AO:', fluence_ao)
print('Fluence for AO/L-wide:', fluence_ao_wide)
print('Fluence for AO/ALFA:', fluence_ao_alfa)
print('Fluence for WSRT:', fluence_apertiff)
print('Fluence for FAST:', fluence_fast)
print('Fluence for GBT:', fluence_gbt)
print('Fluence for LT:', fluence_lt)
print('Fluence for MeerKAT:', fluence_meerkat)
print('Fluence for VLA:', fluence_vla)
print('Fluence for DSS-43:', fluence_dss43)
print('Fluence for DSS-63:', fluence_dss63)
print('Fluence for Effelsberg:', fluence_eff)
print('Fluence for VLA S-band:', fluence_vla_sband)

def calculate_fluence(row):
    bandwidth = row['bandwidth (MHz)']
    sefd = row['SEFD (Jy)']
    flux_row = flux(SNR_MIN, sefd, N_POL, BURST_FWHM, bandwidth*10**6)
    return fluence(flux_row, BURST_FWHM_MILLI_SEC)
        

def add_rates(og_dataframe, spectral_index):
    new_dataframe = og_dataframe.copy()
    duration_hours = og_dataframe['duration (s)'] / 3600
    rates = calc_rates(og_dataframe['bursts'], duration_hours)
    new_dataframe['rates'] = rates
    new_dataframe['fluence'] = new_dataframe.apply(calculate_fluence, axis=1)
    new_dataframe['norm_rates'] = norm_rates(new_dataframe['rates'], new_dataframe['fluence'], fluence_100, spectral_index)
    return new_dataframe


def transform_dataset_Lband(og_dataframe, columns):
    assert len(columns) == 12, "The number of columns must be 12"
    new_dataframe = pd.DataFrame(columns=columns)
    new_dataframe[columns[0]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).year
    new_dataframe[columns[1]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).month
    new_dataframe[columns[2]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).day
    new_dataframe[columns[3]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).hour
    new_dataframe[columns[4]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).minute
    new_dataframe[columns[5]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).second
    new_dataframe[columns[6]] = og_dataframe['telescope'].replace(['AO', 'AO/ALFA', 'AO/L-wide','EFF','GBT','VLA','LT','FAST','WSRT','MeerKAT', 'DSS-43','ALMA', 'DSS-63', 'NRT'], [0,0,0,1,2,3,4,5,6,7,8,9,10,11])
    new_dataframe[columns[7]] = og_dataframe['frequency (MHz)']
    new_dataframe[columns[8]] = og_dataframe['duration (s)']
    new_dataframe[columns[9]] = np.where(og_dataframe['bursts'] >= 1, 1, 0)
    new_dataframe[columns[10]] = 560
    new_dataframe[columns[11]] = og_dataframe['cite'].replace(['L.G.Spitler-2014/P.Scholz-2016', 'P.Scholz-2016', 'L.G.Spitler-2016',
       'D.M.Hewitt-2022', 'K.M.Rajwade-2020', 'L.Houben-2019',
       'K.Gourdji-2019', 'P.Scholz-2017', 'M.Cruces-2021',
       'J.N.Jahns-2023', 'L.Oostrum-2020', 'D.Li-2021', 'M.Caleb-2020', 'Braga-2014', 'Atel 15981', 'L.G.Spitler-2016/P.Scholz-2016','Liu-2021','Pearlman-2020','Hardy-2017','Y.K.Zhang-2024', 'Gouiffes-2024'], 
       [0,1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15,16,17,18,19,20]) # this parameter doesn't matter for periodicity calculation but for the plot phase it does
    return new_dataframe


def transform_dataset_Lband_norm_rates(og_dataframe, columns):
    assert len(columns) == 12, "The number of columns must be 12"
    new_dataframe = pd.DataFrame(columns=columns)
    new_dataframe[columns[0]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).year
    new_dataframe[columns[1]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).month
    new_dataframe[columns[2]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).day
    new_dataframe[columns[3]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).hour
    new_dataframe[columns[4]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).minute
    new_dataframe[columns[5]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).second
    new_dataframe[columns[6]] = og_dataframe['telescope'].replace(['AO', 'AO/ALFA', 'AO/L-wide', 'EFF','GBT','VLA','LT','FAST','WSRT',
                                                                   'MeerKAT', 'DSS-43', 'ALMA', 'DSS-63','NRT'], [0,0,0,1,2,3,4,5,6,7,8, 9,10,11])
    new_dataframe[columns[7]] = og_dataframe['frequency (MHz)']
    new_dataframe[columns[8]] = og_dataframe['duration (s)']
    new_dataframe[columns[9]] = og_dataframe['norm_rates']
    new_dataframe[columns[10]] = 560
    new_dataframe[columns[11]] = og_dataframe['cite'].replace(['L.G.Spitler-2014/P.Scholz-2016', 'P.Scholz-2016', 'L.G.Spitler-2016',
       'D.M.Hewitt-2022', 'K.M.Rajwade-2020', 'L.Houben-2019',
       'K.Gourdji-2019', 'P.Scholz-2017', 'M.Cruces-2021',
       'J.N.Jahns-2023', 'L.Oostrum-2020', 'D.Li-2021', 'M.Caleb-2020', 'Braga-2014', 'Atel 15981', 'L.G.Spitler-2016/P.Scholz-2016','Liu-2021','Pearlman-2020','Hardy-2017','Y.K.Zhang-2024','Gouiffes-2024'],
         [0,1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15,16,17,18,19,20])
    return new_dataframe 


def dataset_window_transform(og_dataframe, columns):
    assert len(columns) == 12, "The number of columns must be 12"
    new_dataframe = pd.DataFrame(columns=columns)
    new_dataframe[columns[0]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).year
    new_dataframe[columns[1]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).month
    new_dataframe[columns[2]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).day
    new_dataframe[columns[3]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).hour
    new_dataframe[columns[4]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).minute
    new_dataframe[columns[5]] = pd.DatetimeIndex(og_dataframe['start_time (UTC)']).second
    new_dataframe[columns[6]] = og_dataframe['telescope'].replace(['AO','EFF','GBT','VLA','LT','FAST','WSRT','MeerKAT', 'DSS-43', 'ALMA'], [0,1,2,3,4,5,6,7,8,9])
    new_dataframe[columns[7]] = og_dataframe['frequency (MHz)']
    new_dataframe[columns[8]] = og_dataframe['duration (s)']
    new_dataframe[columns[9]] = np.ones(len(og_dataframe))
    new_dataframe[columns[10]] = 560
    new_dataframe[columns[11]] = og_dataframe['cite'].replace(['L.G.Spitler-2014/P.Scholz-2016', 'P.Scholz-2016', 'L.G.Spitler-2016',
       'D.M.Hewitt-2022', 'K.M.Rajwade-2020', 'L.Houben-2019',
       'K.Gourdji-2019', 'P.Scholz-2017', 'M.Cruces-2021',
       'J.N.Jahns-2023', 'L.Oostrum-2020', 'D.Li-2021', 'M.Caleb-2020', 'Braga-2014', 'Atel 15981', 'L.G.Spitler-2016/P.Scholz-2016'],
         [0,1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15])
    return new_dataframe


def read_obsinfo(file, det_model):
    data = np.genfromtxt(file)
    years = data[:,0]
    months = data[:,1]
    days = data[:,2]
    hours = data[:,3]
    minutes = data[:,4]
    seconds = data[:,5]
    duration = data[:,8]
    nFRB = data[:,9]
    tel = data[:,6]
    freq = data[:,7]
    dmval = data[:,10]
    obsset= data[:,11]
    starts = []
    det = []
    FRB_loc = SkyCoord('05:31:59','+33:08:50',unit=(au.hourangle,au.deg),equinox='J2000')
    AO_loc = EarthLocation.from_geodetic(lon='-66.7528',lat='18.3464') 
    GBT_loc = EarthLocation.from_geodetic(lon='-79.8398',lat='38.4322') 
    Eff_loc = EarthLocation.from_geodetic(lon='6.882778',lat='50.52472')
    VLA_loc = EarthLocation.from_geodetic(lon='-107.6184',lat='34.0784') 
    Lov_loc = EarthLocation.from_geodetic(lon='-2.3085',lat='53.2367')
    WSRT_loc = EarthLocation.from_geodetic(lon='6.6033',lat='52.91472')
    FAST_loc = EarthLocation.from_geodetic(lon='106.9283',lat='25.6528')
    MeerKAT_loc = EarthLocation.from_geodetic(lon='21.4431',lat='-30.7135')
    dss_43_loc = EarthLocation.from_geodetic(lon='148.9816',lat='-35.4013')
    ALMA_loc = EarthLocation.from_geodetic(lon='-67.703',lat='-23.029')
    dss_63_loc = EarthLocation.from_geodetic(lon='-4.2480', lat='40.4313')
    Nancay_loc = EarthLocation.from_geodetic(lon='2.1972', lat='47.3860')
    tel_locs = [AO_loc,Eff_loc,GBT_loc,VLA_loc,Lov_loc, FAST_loc, WSRT_loc, MeerKAT_loc, dss_43_loc, ALMA_loc, dss_63_loc, Nancay_loc]
    for i in range(len(years)):
        startstr = '%04i-%02i-%02iT%02i:%02i:%02i'%(years[i],months[i],days[i],
                                                    hours[i],minutes[i],
                                                    seconds[i])
        if det_model == 'binary':
            if nFRB[i]!=0:
                det.append(1)
            else:
                det.append(0)
        if det_model == 'norm_rate':
            if nFRB[i]!=0:
                det.append(nFRB[i])
            else:
                det.append(0)
        start = at.Time(startstr,format='isot',scale='utc',location=tel_locs[int(tel[i])])
        starts.append(start.mjd)
    return starts,det,duration,obsset,tel,nFRB


def read_toas(file):
    data = np.genfromtxt(file)
    toas = data[:,0]
    dataset = data[:,1]
    return toas,dataset


def periodogram(time,y,data_type='obs',plot=True,top_vals=10, discard_vals=False):
    #Lomb-Scargle for unevenly sampled data from astropy
    if data_type=='obs':
        print("The analysis will be carried for observations")
        frequency, power = LombScargle(time, y,center_data=True,
                                       fit_mean=True).autopower(normalization='psd',nyquist_factor=16)
    elif data_type=='win':
        print("The analysis will be carried for the window")
        frequency, power = LombScargle(time, y,center_data=False,
                                       fit_mean=False).autopower(normalization='psd',nyquist_factor=16)
    if not discard_vals:
        i=np.argmax(power)
        period_l = float("{:.2f}".format(1./frequency[np.argmax(power)]))
        print("Lomb-Scargle prediction: ",period_l)
        if top_vals is not None:
            print("Top "+str(top_vals)+" periods:")
            for i in range(top_vals):
                idx = (-power).argsort()[:top_vals]
                print(i+1,") ",float("{:.2f}".format(1./frequency[idx[i]]))," Power: ",np.round(power[idx[i]],1))
    else:
        total_days = int(max(time) - min(time))
        max_period = total_days / 3.0

        # Filter out periods that exceed the maximum allowed period
        valid_indices = 1.0 / frequency <= max_period
        valid_frequency = frequency[valid_indices]
        valid_power = power[valid_indices]

        if len(valid_frequency) == 0:
            print("No valid periods found.")
            return

        # Find the best period
        i = np.argmax(valid_power)
        period_l = float("{:.2f}".format(1. / valid_frequency[i]))
        print("Lomb-Scargle prediction: ", period_l)

        if top_vals is not None:
            print("Top " + str(top_vals) + " periods:")
            idx = (-valid_power).argsort()[:top_vals]
            for i in range(min(top_vals, len(idx))):
                print(i + 1, ") ", float("{:.2f}".format(1. / valid_frequency[idx[i]])), " Power: ", np.round(valid_power[idx[i]], 1))


  
    #plt.savefig('FRB121102_periodogram_Eff.png',dpi=300)
    
    if plot:  
        fig, axs = plt.subplots(1, 2,facecolor='w',figsize=(15,5))
        #Data vizualization
        axs[0].plot(time, y,'ko')
        axs[0].set_title("Data",size=14)
        axs[0].set_xlabel("Time",size=14)
        axs[0].set_ylabel("",size=14)
        axs[0].set_yticks([0])
        
        axs[1].plot(frequency, power,'-',label="Prediction: "+str(1./frequency[i])+" days")
        axs[1].set_title("Lomb-Scargle periodogram",size=14)
        axs[1].set_xlabel("Freq, Hz",size=14)
        axs[1].set_ylabel("Power",size=14)
        
        plt.tight_layout()
        return period_l
    else:
        return period_l,frequency, power
    

def folding(epochs,detection,period):
    A = []
    A_i= []
    B = []
    B_i =[]
    e_pos = []
    e_neg = []
    i=0
    while i<len(epochs):
        phase=epochs[i]/period-int(epochs[i]/period)
        if detection[i]==1:
            A.append(phase)
            A_i.append(i)
            e_pos.append(epochs[i])
        elif detection[i]==0:
            B.append(phase)
            B_i.append(i)
            e_neg.append(epochs[i])
        i+=1
    return A,A_i,e_pos,B,B_i,e_neg


def folding_norm(epochs,detection,period):
    A = []
    A_i= []
    B = []
    B_i =[]
    e_pos = []
    e_neg = []
    i=0
    while i<len(epochs):
        phase=epochs[i]/period-int(epochs[i]/period)
        if detection[i]>0:
            A.append(phase)
            A_i.append(i)
            e_pos.append(epochs[i])
        elif detection[i]==0:
            B.append(phase)
            B_i.append(i)
            e_neg.append(epochs[i])
        i+=1
    return A,A_i,e_pos,B,B_i,e_neg


def phaseshift(array,shift):
    i = 0
    arraymod = []
    while i < len(array):
        a_shifted=array[i]+shift
        if a_shifted < 0:
            a_shifted=1+a_shifted
        elif a_shifted > 1:
            a_shifted=a_shifted-1
        arraymod.append(a_shifted)
        #print array[i],arraymod[i]
        i+=1
    return arraymod


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def periodicity(obsinfo,toas, amplitude=0, mean=0, sig=0, data_type='obs',top_cand=10,shift=0,period='auto',plot='all',
                significance=True, det_model='binary', fit=False, discard_vals=False, plot_fit=False, gamma=-1.5, save=False, plot_two=False):
    
    epochs,det,duration,obsset,tel,bursts=read_obsinfo(obsinfo, det_model)
    rate=bursts/duration*24*3600
    if data_type=='obs':
        y=det       #-np.mean(det)
    elif data_type=='win':
        y=np.ones(len(det))
        plot=='all'
    
    if plot=='all':
        period_l=periodogram(epochs,y,data_type=data_type,top_vals=top_cand,plot=True)
    else:
        period_l,F_l,A_l=periodogram(epochs,y,data_type=data_type,top_vals=top_cand,plot=None, discard_vals=discard_vals)
    
    if data_type=='win':
        print("***Window analysis done***")
        return
    
    peaks = np.zeros(10000)  # try 20000
    for i in range(10000):  # original 10000
        y_boostraped = np.random.choice(y, size=len(y), replace=True)
        p,a = LombScargle(epochs,y_boostraped,center_data=True, fit_mean=False).autopower(normalization='psd')
        peaks[i] = a.max()
    sig5, sig3, sig2, sig1 = np.percentile(peaks, [99.9999426, 99.7, 95.4, 68.2])
    print(sig5, sig3,sig2,sig1)
    
    if period=='auto':
        period=period_l
        print("Period set to: ",period)
    else:
        period=float(period)
    if det_model == 'binary':
        pos,pos_i,e_pos,neg,neg_i,e_neg = folding(epochs,det,period)
    elif det_model == 'norm_rate':
        pos,pos_i,e_pos,neg,neg_i,e_neg = folding_norm(epochs,det,period)
    bins=15
    ref= np.array([57075])  #np.array([57075]) # original
    ref_arr = ref/period-int(ref/period)
    ref_shifted=phaseshift(ref_arr,shift)
    pos_shifted=phaseshift(pos,shift)
    neg_shifted=phaseshift(neg,shift)
    times, dataset = read_toas(toas)
    if plot=='all':
        fig, axs = plt.subplots(1, 2,facecolor='w',figsize=(18,5))
        s3=axs[1].hist(neg_shifted,bins=bins,label='no detections')
        s3=axs[1].hist(pos_shifted,bins=bins,label='detections')
        for l in range(len(neg_shifted)):
            axs[0].bar(neg[l],duration[neg_i[l]]/3600,color='grey',width=0.01)
           
        
        for j in range(len(pos_shifted)):
            if obsset[pos_i[j]]!=0:
                axs[0].bar(pos[j],duration[pos_i[j]]/3600,color='mediumvioletred',width=0.01)
                for idx,i in enumerate(dataset):
                    if i == obsset[pos_i[j]]:
                        position=abs(times[idx]-e_pos[j])*24
                        axs[0].plot(pos[j],position,marker='*',color='goldenrod')
            else:
                axs[0].bar(pos[j],duration[pos_i[j]]/3600,color='goldenrod',width=0.01)
    
        axs[0].axvline(x=ref_arr[0],color='red',linewidth=1,ls='--')
        
        axs[0].set_ylabel(r'Obs. length [hr]',size=14)
        axs[0].set_ylim(0.,12.)
        axs[1].set_title(r'Phase shifted',size=14)
        axs[1].set_ylabel(r'Obs. count',size=14)
        axs[1].set_xlim(0.,1.)
        axs[1].set_xlabel(r'$\phi$',size=14)
        axs[1].legend(numpoints=1,fontsize=10,loc=2)
       
        

   
    
    elif plot=='paper':
        fig, axs = plt.subplots(2, 1,facecolor='w',figsize=(width, height))
        i=np.argmax(A_l)
        axs[0].axvline(x=period,c='mediumvioletred',linewidth=1.,ls='--')
        #axs[0].axvline(x=1/period,c='mediumvioletred',linewidth=1.5,ls='--')
        axs[0].axhline(y=sig1,c='k',linewidth=1.,ls=':', alpha=0.5)
        axs[0].text(period-0.2*period,sig1,r'$1\sigma$')
        # axs[0].axhline(y=sig2,c='k',linewidth=1.,ls=':', alpha=0.5)
        # axs[0].text(period-0.2*period,sig2,r'$2\sigma$',)  # included in S band norm rate periodogram
        axs[0].axhline(y=sig3,c='k',linewidth=1.,ls=':', alpha=0.5)
        axs[0].text(period-0.2*period,sig3,r'$3\sigma$')
        axs[0].axhline(y=sig5,c='k',linewidth=1.,ls=':', alpha=0.5)
        axs[0].text(period-0.2*period, sig5-0.1, r'$5\sigma$')  # -0.1 for s band plot with width 12cm
        axs[0].plot(1/F_l, A_l,'-')
        if plot_two:
            data_loaded = np.load('periodogram_tests/periodogram_data.npz')
            power_loaded = data_loaded['power']
            frequency_loaded = data_loaded['frequency']
            axs[0].plot(1/frequency_loaded[30:300], 4*power_loaded[30:300], label='Data with removed observations', alpha=0.4)
            axs[0].legend(loc='upper left')
        #guess1 = [14, 160, 1]
        if fit:
            guess = [amplitude, mean, sig]
            popt, _ = curve_fit(gaussian, 1/F_l, A_l, p0=guess)
            if abs(popt[1] - guess[1]) > 5 or abs(popt[2] - guess[2]) > 5:
                print('The data was not fitted properly, trying again with a smaller range')
                if det_model == 'binary':
                    popt, _ = curve_fit(gaussian, 1/F_l[30:50], A_l[30:50], p0=guess)  # for L band [30:50] for S old band [36:60], new [44:52]
                    if plot_fit:
                        axs[0].plot(1/F_l[30:50], gaussian(1/F_l[30:50], *popt), label='fit')
                else:
                    popt, _ = curve_fit(gaussian, 1/F_l[110:140], A_l[110:140], p0=guess)  # for L band 160 peak [110:140], for 290 peak [60, 80] for old S band [36:60], new [45:55] 
                    if plot_fit:
                     axs[0].plot(1/F_l[110:140], gaussian(1/F_l[110:140], *popt), label='fit')
            elif plot_fit:
                axs[0].plot(1/F_l, gaussian(1/F_l, *popt), label='fit')
            print(f'Parameters for peak: A={popt[0]}, mu={popt[1]}, sigma={popt[2]}')
            fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
            mean_amp = np.mean(A_l)
            timeseries_std = np.std(y)
            timeseries_mean = np.mean(y)
            print(timeseries_mean)
            timeseries_minus_mean = y - timeseries_mean
            average_signal_to_noise = np.mean(timeseries_minus_mean[timeseries_minus_mean > 0])
            timseries_snr = y / timeseries_std
            timeseries_avg_snr = np.mean(timseries_snr)
            one_sigma = sigma(fwhm, len(epochs), average_signal_to_noise)
            print('The FWHM is:', fwhm)
            print('The average SNR is:', average_signal_to_noise)
            print('The uncertainty in the period at one sigma is:', one_sigma)
            print('The uncertainty in the period at three sigma is:', 3 * one_sigma)
            axs[0].text(period+0.2*period,np.max(A_l)+0.005*np.max(A_l), f'P={period:.1f}' + ' ± ' + f'{one_sigma:.1f}' + ' days',backgroundcolor='1')
        #axs[0].plot(F_l, A_l,'-')
        #axs[0].text(period+0.05*period,np.max(A_l)+0.05*np.max(A_l),'P='+str(int(period_l))+' days',size=10,backgroundcolor='1')
        harmonics = [period / 2, period/3, period / 4] + [n * period for n in range(2, 3)]  # Harmonics 161/2, 161/4, and up to 6*161
        indices = [find_nearest(1/F_l, val) for val in harmonics]
        height_bar_non_det = 0.1 * np.max(y)
        #y_values_for_harmonics = A_l[indices]
        if det_model == 'binary':
            peak_y = np.max(A_l) / 6 # 6 lband 2 sband
            arrow_length = peak_y / 2 # 2 lband 0.5 sband
        elif det_model == 'norm_rate':
            peak_y = np.max(A_l) / 2 # 270 lband 6 sband
            arrow_length = peak_y / 2  # 50 lband 1 sband
        for harmonic in harmonics:
            axs[0].annotate(
            '', 
            xy=(harmonic, peak_y), 
            xytext=(harmonic, peak_y + arrow_length), 
            arrowprops=dict(facecolor='purple', edgecolor='purple', arrowstyle='->', lw=0.7))
        axs[0]
        if det_model == 'binary':
            for l in range(len(neg_shifted)):
                #axs[1].bar(neg_shifted[l],duration[neg_i[l]]/3600,color='grey',width=0.01)  # original plot for duration
                axs[1].bar(neg_shifted[l],bursts[neg_i[l]]-height_bar_non_det,color='grey',width=0.01)  # plot for number of bursts
            for j in range(len(pos_shifted)):
                # if obsset[pos_i[j]]!=0:
                #     axs[1].bar(pos_shifted[j],duration[pos_i[j]]/3600,color='mediumvioletred',width=0.015)
                #     for idx,i in enumerate(dataset):
                #         if i == obsset[pos_i[j]]:
                #             position=abs(times[idx]-e_pos[j])*24
                #             axs[1].plot(pos_shifted[j],position,marker='*',color='gold',ms=10)  # original plot for duration and markers for detections
                if obsset[pos_i[j]]!=0:
                    axs[1].bar(pos_shifted[j],bursts[pos_i[j]],color='mediumvioletred',width=0.015) 
        elif det_model == 'norm_rate':
            for l in range(len(neg_shifted)):
                axs[1].bar(neg_shifted[l],bursts[neg_i[l]]-height_bar_non_det,color='grey',width=0.01)
            
            for j in range(len(pos_shifted)):
                if obsset[pos_i[j]]!=0:
                    axs[1].bar(pos_shifted[j],bursts[pos_i[j]],color='mediumvioletred',width=0.015)      

        axs[0].set_xlabel(r"Period (days)", size=12)
        axs[0].set_ylabel("Power", size=12)
        axs[0].set_ylim(0.,np.max(A_l)+0.2*np.max(A_l))
        axs[0].set_xscale('log')
        axs[0].set_xlim(5.,1000)
        if det_model == 'binary':
            #axs[1].set_ylabel(r'Observation (hours)',size=12)  # original plot for duration
            #axs[1].set_ylim(0.,12.)  # original plot for duration
            axs[1].set_ylabel(r'Detections', size=12)
            axs[1].set_ylim(-height_bar_non_det, 1.0)  # plot for number of bursts
            axs[1].set_yticks([0, 1])
            axs[1].set_yticklabels(['0', '1'])
            axs[0].set_title(r'Periodogram for binary detection model', size=12)
            axs[1].hlines(0, 0, 1, color='black', linestyle='--', lw=1)
        else:
            axs[1].set_ylabel(r'Normalised rates', size=12)
            axs[0].set_title(f'Periodogram for normalised rates with $\gamma = {gamma}$', size=12)
            axs[1].set_yticks([0, np.max(bursts)])
            axs[1].set_ylim(-height_bar_non_det, np.max(bursts))
            axs[1].hlines(0, 0, 1, color='black', linestyle='--', lw=1)
            #axs[1].set_yticklabels(['0', '20'])
            #axs[1].set_ylim(-0.5, 10)
        axs[1].set_xlabel(r'$\phi$', size=12)
        axs[1].set_xlim(0.,1.)
        # axs[0].tick_params(axis="x", labelsize=14)
        # axs[0].tick_params(axis="y", labelsize=14)
        # axs[1].tick_params(axis="x", labelsize=14)
        # axs[1].tick_params(axis="y", labelsize=14)
        plt.tight_layout()
        if save:  # save data from periodogram as npz file
            np.savez('periodogram_tests/periodogram_data.npz', period=period, power=A_l, frequency=F_l, peaks=peaks, sig1=sig1, sig3=sig3, sig5=sig5)
        
    active_per = float("{:.2f}".format((np.max(pos_shifted)-np.min(pos_shifted))*100))
    active_days = float("{:.2f}".format((np.max(pos_shifted)-np.min(pos_shifted))*159.3))
    print('The minimun phase is:', np.min(pos_shifted))
    print('The maximum phase is:', np.max(pos_shifted))
    print("Middle phase:",float("{:.2f}".format((np.max(pos_shifted)+np.min(pos_shifted))/2)))
    print("Active window (%): ", active_per)
    print("Active window (days):", active_days)
    print("Epochs in total: ",len(epochs))
    print("Non-detections:",len(neg))
    print("Detections:",len(pos))
    print("Time span (days):",int(np.max(epochs)-np.min(epochs)))
    return


def arg_parser():
    parser = argparse.ArgumentParser(description='Periodicity analysis for FRB121102')
    parser.add_argument('--obsinfo', type=str, help='File with observation info')
    parser.add_argument('--obstimes', type=str, help='File with TOAs')
    parser.add_argument('--data_type', type=str, default='obs', help='Type of data to analyze (obs or win)')
    parser.add_argument('--top_cand', type=int, default=15, help='Number of top candidates to show')
    parser.add_argument('--shift', type=float, default=0, help='Phase shift')
    parser.add_argument('--period', default='auto', help='Period to analyze')
    parser.add_argument('--plot', type=str, default='all', help='Plot type (all or paper)')
    parser.add_argument('--guess', type=float, nargs='+', default=[0, 0, 0], help='Guess for Gaussian fit, in the form A mu sigma')
    parser.add_argument('--det_model', type=str, default='binary', help='Detection model (binary or norm_rate)')
    parser.add_argument('--fit', action='store_true', help='Fit a Gaussian to the periodogram')
    parser.add_argument('--from_dataset', type=str, help='Transform dataset to the desired format')
    parser.add_argument('--spectral_index', type=float, default=-1.5, help='Spectral index for the normalization of the rates')
    parser.add_argument('--discard_vals', action='store_true', help='Show all periods in the periodogram')
    parser.add_argument('--plot_fit', action='store_true', help='Plot the Gaussian fit')
    parser.add_argument('--window_transform', action='store_true', help='Transform the dataset to fill it with ones')
    parser.add_argument('--save', action='store_true', help='Save the data from the periodogram')
    parser.add_argument('--plot_two', action='store_true', help='Plot two periodograms')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    # if args.period == 'auto':
    #     period=args.period
    # else:
    #     period=int(args.period)
    period = args.period
    save = args.save
    plot_two = args.plot_two
    if args.from_dataset:
        if args.window_transform:
            columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'observatory', 'frequency', 'duration (s)', 'number_of_bursts', 'dm', 'counter']
            df = pd.read_csv(args.from_dataset)
            new_dataset = dataset_window_transform(df, columns)
            new_dataset.to_csv('periodogram_tests/dataset_for_periodicity.txt', sep=' ', header=None, index=False)
            periodicity(f'periodogram_tests/dataset_for_periodicity.txt', args.obstimes, amplitude=args.guess[0], mean=args.guess[1], 
                        sig=args.guess[2],data_type=args.data_type, top_cand=args.top_cand, shift=args.shift, 
                        period=period, plot=args.plot, det_model=args.det_model, fit=args.fit, discard_vals=args.discard_vals, plot_fit=args.plot_fit, save=save, plot_two=plot_two)
            plt.show()
            return
        if args.det_model == 'binary':
            columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'observatory', 'frequency', 'duration (s)', 'number_of_bursts', 'dm', 'counter']
            df = pd.read_csv(args.from_dataset)
            new_dataset = transform_dataset_Lband(df, columns)
            new_dataset.to_csv('periodogram_tests/dataset_for_periodicity.txt', sep=' ', header=None, index=False)
            periodicity(f'periodogram_tests/dataset_for_periodicity.txt', args.obstimes, amplitude=args.guess[0], mean=args.guess[1], 
                        sig=args.guess[2],data_type=args.data_type, top_cand=args.top_cand, shift=args.shift, 
                        period=period, plot=args.plot, det_model=args.det_model, fit=args.fit, discard_vals=args.discard_vals, plot_fit=args.plot_fit, save=save, plot_two=plot_two)
            plt.show()
            return
        elif args.det_model == 'norm_rate':
            columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'observatory', 'frequency', 'duration (s)', 'number_of_bursts', 'dm', 'counter']
            df = pd.read_csv(args.from_dataset)
            dataset_with_rates = add_rates(df, args.spectral_index)
            new_dataset = transform_dataset_Lband_norm_rates(dataset_with_rates, columns)
            new_dataset.to_csv('periodogram_tests/dataset_for_periodicity.txt', sep=' ', header=None, index=False)
            periodicity(f'periodogram_tests/dataset_for_periodicity.txt', args.obstimes, amplitude=args.guess[0], mean=args.guess[1],
                         sig=args.guess[2],data_type=args.data_type, top_cand=args.top_cand, shift=args.shift, 
                        period=period, plot=args.plot, det_model=args.det_model, fit=args.fit, discard_vals=args.discard_vals, plot_fit=args.plot_fit,
                          gamma=args.spectral_index, save=save, plot_two=plot_two)
            plt.show()
            return
    periodicity(args.obsinfo,args.obstimes,amplitude=args.guess[0], mean=args.guess[1], sig=args.guess[2], 
                data_type=args.data_type,top_cand=args.top_cand,shift=args.shift,period=period,plot=args.plot,
                  det_model=args.det_model, fit=args.fit, discard_vals=args.discard_vals, plot_fit=args.plot_fit, gamma=args.spectral_index, save=save, plot_two=plot_two)
    plt.show()


if __name__ == '__main__':
    main()