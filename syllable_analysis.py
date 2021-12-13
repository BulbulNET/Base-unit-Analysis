"""
Created on Sun Nov 29 18:15:11 2020

@author: YL
"""
import numpy as np
import pandas as pd
import sys, os
import sounddevice as sd
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import pandas
import glob
from parse import *
import re 
from scipy import signal


def syllable_analysis(labelsPath = "bulbul", deg = 4, nmfcc = 14, start_zcr = 3 , end_zcr = -2):
    """
    Parameters
    ----------
    labelsPath : directory with all data in folders by syllable name.
    deg : degree of legendre polynomial
    bnfcc : ...
    start_zcr : ...
    end_zcr : ...

    Returns
    -------
    large_df (data frame) - with all parameters extracted - 
    'filename' , 'syllable_number' , 'zcrmean', 'zcrstart' , 'zcrend'
     , 'zcrmid' , 'centmean' , 'bwmean' , 'syl_length' , 'mean_flat'

    """
    # get  list of all files  within labelsPath
    wav_files=glob.glob(labelsPath + "/*/*.wav", recursive=True)

    N = len(wav_files)

    deg = deg - 1 # for range counting
    
    # extract location folder and text file name from each file path

    regexp=re.compile(r"(\d{1,2})\.wav$")
    syl = [regexp.search(x).group(1) for x in wav_files]

    # extract numbers- number of syllable, of files and list of syllable
    y = ['0'+i if len(i) == 1 else i for i in syl]
    syl_list = set(y)
    syl_list = sorted(syl_list)
    num_of_syll = len(syl_list)


    bdir = os.listdir(labelsPath)
    bdir.sort(key = int)
    syllable_df = pd.DataFrame([])


    for i in range(1,num_of_syll+1):
        syl_files = os.listdir(str(labelsPath + '/' + str(i)))
        L = len(syl_files)
        
        data_files = []
        
        for k in range(L):
            wavname = str(labelsPath + '/' + str(i)+ '/' + syl_files[k])
    
            x ,fs = librosa.load(wavname, sr = None)
            sos = signal.butter(35, np.array([1000,3500]), 'bp', fs = fs, output='sos')
            x = signal.sosfilt(sos, x)
            
            filename = syl_files[k]
            syllable_number = i
            
            frameL=512
            hopL=64
            zcr2 = librosa.feature.zero_crossing_rate(x,frame_length=frameL, 
                  hop_length=hopL, center = True, pad = True) 
            f0zcr = zcr2*fs/2
            
            times_zcr = librosa.times_like(f0zcr, hop_length = hopL)
            ntimes_zcr = 2*times_zcr[start_zcr:end_zcr]/times_zcr[-1] - 1
            coeffs = np.polynomial.legendre.legfit(ntimes_zcr, f0zcr[0,start_zcr:end_zcr], deg)
            
            
            zcr = librosa.feature.zero_crossing_rate(x,frame_length=512, hop_length=128)
            zcrmean = np.mean(zcr)
            zcrstart = np.mean(zcr[0,:3])
            zcrend = np.mean(zcr[0, -3:])
            indmid = int(np.floor(zcr.size/2))
            zcrmid = np.mean(zcr[0,indmid-1:indmid+2])
                            
            cent = librosa.feature.spectral_centroid(x, sr = fs, n_fft=512, hop_length=128)
            centmean = np.mean(cent)
                    
            bw = librosa.feature.spectral_bandwidth(x, sr = fs, n_fft=512, hop_length=128)
            bwmean = np.mean(bw)
            
            syl_length = x.size/fs
            
            flatness = librosa.feature.spectral_flatness(y=x, S=None, n_fft=512,
            hop_length=128, window='hann', center=True, pad_mode='reflect',
            amin=1e-10, power=2.0)
            mean_flat = 10*np.log10(np.mean(flatness))
            
            
            data_files.append([filename , syllable_number , zcrmean
                    , zcrstart , zcrend , zcrmid , centmean , bwmean , syl_length , mean_flat]) 
    
            mfcc = librosa.feature.mfcc(x,sr = fs, n_mfcc = nmfcc)
            mfccmean = np.mean(mfcc,1)
            mfccL = mfccmean.tolist()
            data_files[k].extend(mfccL)
            coeffsL = coeffs.tolist()
            data_files[k].extend(coeffsL)
    
            column = ['filename' , 'syllable_number' , 'zcrmean'
                        , 'zcrstart' , 'zcrend' , 'zcrmid' , 'centmean' , 'bwmean' , 'syl_length' , 'mean_flat']
            for j in range(1,nmfcc+1):
                column.append('mfcc'+str(j))
    
            for j in range(deg+1):
                column.append('coeff'+str(j))
                
            
        df = pd.DataFrame(data_files , columns = column)
        
        syllable_df = pd.concat([syllable_df,df], ignore_index=True)
        
    return syllable_df
        


        

    