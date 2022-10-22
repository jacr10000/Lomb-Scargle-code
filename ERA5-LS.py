#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:33:34 2022

@author: castro
"""

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import fft, fftfreq, ifft
from astropy.timeseries import LombScargle
import scipy

def LS1(file_path,category1,title='Lomb-Scargle Power Spectral Density'):
    '''Plots Lomb-Scargle powers for ERA5 file in logarithmic scales'''
    
    openfile = h5py.File(file_path, 'r')
    cat1 = openfile.get('AltitudeInterpolated')
    cat2 = cat1.get(category1)
    df = pd.DataFrame(np.array(cat2.get('data')))
    df = df.transpose()
    df.sort_index(axis=0, ascending=False)
    
    alt20 = df.iloc[:131]
    alt20 = alt20.mean()
    alt40 = df.iloc[131:264]
    alt40 = alt40.mean()
    alt60 = df.iloc[264:398]
    alt60 = alt60.mean()
    alt75 = df.iloc[398:]
    alt75 = alt75.mean()
    
    freq20, power20 = LombScargle(alt20.keys(), alt20, normalization='psd').autopower()
    freq40, power40 = LombScargle(alt40.keys(), alt40, normalization='psd').autopower()
    freq60, power60 = LombScargle(alt60.keys(), alt60, normalization='psd').autopower()
    freq75, power75 = LombScargle(alt75.keys(), alt75, normalization='psd').autopower()
    
    m20, c20 = np.polyfit(np.log(freq20[1000:3800]), np.log(power20[1000:3800]), 1)
    fit20 = np.exp(m20*np.log(freq20)+ c20)
    m40, c40 = np.polyfit(np.log(freq40[50:800]), np.log(power40[50:800]), 1)
    fit40 = np.exp(m40*np.log(freq40)+ c40)
    m60, c60 = np.polyfit(np.log(freq60[50:800]), np.log(power60[50:800]), 1)
    fit60 = np.exp(m60*np.log(freq60)+ c60)
    m75, c75 = np.polyfit(np.log(freq75[50:800]), np.log(power75[50:800]), 1)
    fit75 = np.exp(m75*np.log(freq75)+ c75)
    
    
    fig, axs = plt.subplots(nrows=4, figsize=(15,15))
    sns.lineplot(x=freq20,y=power20,ax=axs[0], color='r')
    sns.lineplot(x=freq40,y=power40,ax=axs[1], color='r')
    sns.lineplot(x=freq60,y=power60,ax=axs[2], color='r')
    sns.lineplot(x=freq75,y=np.abs(power75),ax=axs[3], color='r')
    sns.lineplot(x=freq20,y=fit20, ax=axs[0])
    sns.lineplot(x=freq40, y=fit40,ax=axs[1])
    sns.lineplot(x=freq60, y=fit60,ax=axs[2])
    sns.lineplot(x=freq75, y=fit75,ax=axs[3])
    for i in axs:
        i.set(xscale='log', yscale='log', ylabel='Power Spectral Density')
    plt.legend(['Data','Best fit line'])
    axs[0].set_title(title)
    axs[3].set_xlabel('Frequency [$Hours^{-1}$]')
    axs[0].text(0.00001, 1, '[0-20]km', fontsize=15)
    axs[0].text(0.001, 1000000, 'Equation of line: P = exp('+'%.2f' % m20+'*ln(f)'+'%.2f' % c20+')', fontsize=15)
    axs[1].text(0.00001, 100, '[20-40]km', fontsize=15)
    axs[1].text(0.001, 1000000, 'Equation of line: P = exp('+'%.2f' % m40+'*ln(f)'+'%.2f' % c40+')', fontsize=15)
    axs[2].text(0.00001, 100, '[40-60]km', fontsize=15)
    axs[2].text(0.001, 1000000, 'Equation of line: P = exp('+'%.2f' % m60+'*ln(f)'+'%.2f' % c60+')', fontsize=15)
    axs[3].text(0.00001, 100, '[60-75]km', fontsize=15)
    axs[3].text(0.001, 1000000, 'Equation of line: P = exp('+'%.2f' % m75+'*ln(f)-'+'%.2f' % c75+')', fontsize=15) 
    plt.show()


    
def LS2(file_path, category1, minfreq, maxfreq, title='Lomb-Scargle Power Spectral Density'):
    '''Plots Lomb-Scargle powers for ERA5 file'''
    
    #Open file and create dataframe
    openfile = h5py.File(file_path, 'r')
    cat1 = openfile.get('AltitudeInterpolated')
    cat2 = cat1.get(category1)
    df = pd.DataFrame(np.array(cat2.get('data')))
    df = df.transpose()
    df.sort_index(axis=0, ascending=False)
    
    #Find means for each altitude range
    alt20 = df.iloc[:131]
    alt20 = alt20.mean()
    alt40 = df.iloc[131:264]
    alt40 = alt40.mean()
    alt60 = df.iloc[264:398]
    alt60 = alt60.mean()
    alt75 = df.iloc[398:]
    alt75 = alt75.mean()
    
    #LS, adjust frequency range in function
    freq20, power20 = LombScargle(alt20.keys(), alt20, normalization='psd').autopower(minimum_frequency=minfreq,
                                                                                      maximum_frequency=maxfreq)
    freq40, power40 = LombScargle(alt40.keys(), alt40, normalization='psd').autopower(minimum_frequency=minfreq,
                                                                                      maximum_frequency=maxfreq)
    freq60, power60 = LombScargle(alt60.keys(), alt60, normalization='psd').autopower(minimum_frequency=minfreq,
                                                                                      maximum_frequency=maxfreq)
    freq75, power75 = LombScargle(alt75.keys(), alt75, normalization='psd').autopower(minimum_frequency=minfreq,
                                                                                      maximum_frequency=maxfreq)    
    #Create subplots
    fig, axs = plt.subplots(nrows=4, figsize=(15,15))
    sns.lineplot(x=freq20,y=power20,ax=axs[0], color='r')
    sns.lineplot(x=freq40,y=power40,ax=axs[1], color='r')
    sns.lineplot(x=freq60,y=power60,ax=axs[2], color='r')
    sns.lineplot(x=freq75,y=power75,ax=axs[3], color='r')
    for i in axs:
        i.set(ylabel='Power Spectral Density')
    axs[0].set_title(title)
    axs[3].set_xlabel('Frequency [$Hours^{-1}$]')
    axs[0].text(0.1, 100000, '[0-20]km', fontsize=15)
    axs[1].text(0.1, 10000000, '[20-40]km', fontsize=15)
    axs[2].text(0.1, 10000000, '[40-60]km', fontsize=15)
    axs[3].text(0.1, 10000000, '[60-75]km', fontsize=15)
    plt.show()
    
    
    
    