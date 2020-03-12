# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:12:06 2020

@author: Ovidiu
"""

import csv
import math
import numpy as np
#from scipy import signal
from scipy.io import wavfile
#from Signal_Analysis.features import signal as sg       # https://brookemosby.github.io/Signal_Analysis/index.html
## https://homepage.univie.ac.at/christian.herbst/python/
#import matplotlib.pyplot as plt
#
#number_of_mfcc = 16
#delta_mfcc_order = 2
#
#cpp_window_size = 0.025 # ms
#
recording_rate, recording_signal = wavfile.read('t2_learning_computer_x.wav')
#recording_signal_windowed = recording_signal[499:500+math.floor(recording_rate * cpp_window_size)]
#
#recording_signal_windowed_normalized = recording_signal_windowed - np.min(recording_signal_windowed) # switch signal as desired
#recording_signal_windowed_normalized = recording_signal_windowed_normalized / np.max(recording_signal_windowed_normalized) * 2 - 1
#
## recording_signal_hamming = np.multiply( np.hamming(recording_signal_windowed_normalized.shape[0]), recording_signal_windowed_normalized)
#
recording_length = recording_signal.shape[0] / recording_rate
#recording_F0 = sg.get_F_0( recording_signal, recording_rate )
#recording_HNR = sg.get_HNR( signal = recording_signal, rate = recording_rate, time_step = 0,
#                           min_pitch = 75, silence_threshold  = 0.1, periods_per_window = 4.5 )
#
#recording_segment_frequencies, recording_segment_power_spectrum = signal.periodogram( recording_signal_windowed_normalized, recording_rate )
## recording_segment_frequencies, recording_segment_power_spectrum = signal.periodogram( recording_signal_hamming, recording_rate )
#
number_of_mfcc = 16
delta_mfcc_order = 2
recording_mfccs = np.ndarray( ( math.ceil(recording_length*100), number_of_mfcc + 2), dtype = object )

with open('mfcc.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        
        j = 0
        for collumn in row[0].split(';'):
            
            recording_mfccs[i, j] = collumn 
            j += 1
    
        i += 1
#
recording_mfccs = recording_mfccs[1:math.ceil(recording_length*100), delta_mfcc_order:number_of_mfcc + delta_mfcc_order]
recording_mfccs = recording_mfccs.astype(np.float)
recording_mfccs = recording_mfccs[~np.isnan(recording_mfccs).any(axis=1)]
number_of_mfcc_segments = recording_mfccs.shape[0]
#
if delta_mfcc_order>0:
    
    recording_delta_mfccs_raw = np.hstack((
                                    np.repeat(np.reshape(recording_mfccs[:,0],(number_of_mfcc_segments,1)), delta_mfcc_order, 1),
                                    recording_mfccs, 
                                    np.repeat(np.reshape(recording_mfccs[:,-1],(number_of_mfcc_segments,1)), delta_mfcc_order, 1)
                                    ))
    
    recording_delta_mfccs = np.zeros((number_of_mfcc_segments, number_of_mfcc), float)
    for i in range(number_of_mfcc_segments):
        
        for j in range(number_of_mfcc):
            
            numerator = 0
            denominator = 0
            for k in range(delta_mfcc_order):
                
                numerator = numerator + (k + 1) * (recording_delta_mfccs_raw[i, j+k+1+delta_mfcc_order] - recording_delta_mfccs_raw[i, j-k-1+delta_mfcc_order])
                denominator = denominator + (k + 1) * (k + 1)
                
            recording_delta_mfccs[i,j] = numerator / (denominator * 2)
        
##print( recording_F0 )
##print( recording_HNR )
##print( recording_mfccs )
##print( recording_delta_mfccs_raw )
##print( recording_delta_mfccs )
#
#recording_magnitude_spectrum = np.log(np.abs(recording_segment_power_spectrum))
#recording_cepstrum = np.fft.ifft(recording_magnitude_spectrum)
#peaks, _ = signal.find_peaks(recording_cepstrum)
#prominences = signal.peak_prominences(recording_cepstrum, peaks)
#
#fig, p = plt.subplots( 2, 2 )
#p[0, 0].plot( recording_signal_windowed_normalized )
#p[0, 0].set_title( 'Time Signal' )
#p.flat[0].set(xlabel='Time (frames)', ylabel='Amplitude')
#
#p[0, 1].plot( recording_segment_frequencies, recording_segment_power_spectrum )
#p[0, 1].set_title( 'Power Spectrum' )
#p.flat[1].set(xlabel='Frequency (Hz)', ylabel='Power')
#
#p[1, 0].plot( recording_segment_frequencies, recording_magnitude_spectrum )
#p[1, 0].set_title( 'Magnitude Spectrum' )
#p.flat[2].set(xlabel='Frequency (Hz)', ylabel='Magnitude')
#
#p[1, 1].plot( recording_cepstrum )
#p[1, 1].set_title( 'Cepstrum' )
#p.flat[3].set(xlabel='Quefrency', ylabel='Amplitude')
#
#plt.show()

import os

os.system ('"C:\\Darwin-Project\\Praat.exe" --run C:\\Darwin-Project\\audio\\Praat\\Scripts\\FF "C:\\Darwin-Project\\audio\\t2_learning_computer_x.wav" "C:\\Darwin-Project\\audio\\Praat\\Data\\FF.txt"')
os.system ('"C:\\Darwin-Project\\Praat.exe" --run C:\\Darwin-Project\\audio\\Praat\\Scripts\\CPP "C:\\Darwin-Project\\audio\\t2_learning_computer_x.wav" "C:\\Darwin-Project\\audio\\Praat\\Data\\CPP.txt"')
os.system ('"C:\\Darwin-Project\\Praat.exe" --run C:\\Darwin-Project\\audio\\Praat\\Scripts\\HNR "C:\\Darwin-Project\\audio\\t2_learning_computer_x.wav" "C:\\Darwin-Project\\audio\\Praat\\Data\\HNR.txt"')
#os.system ('"C:\\Darwin-Project\\Praat.exe" --run extractNorthwindDeltaMFCC')
#os.system ('"C:\\Darwin-Project\\Praat.exe" --run extractNorthwindPhonemeDuration')