# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:15:11 2020

CASE 1 - Mel spectrogram with variable length (fixed frame length and hop size)
 - MFBlist
CASE 2 - Mel spectrogram with fixed length (variable hop size and frame length)
 - MFBfixlist
CASE 3 - Mel spectrogram with fixed length with zero padding to largest expected
    word length (fixed frame length and hop size) - M_mbflist

@author: AM and YL
"""


import numpy as np
import sys, os
import sounddevice as sd
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
from parse import *
import re 
from scipy import signal
import Utils as Ut

 
def mfb_analysis(labelsPath, case):
    
 # get  list of all files  within labelsPath
    wav_files=glob.glob(labelsPath + "/*/*.wav", recursive=True)
    
 # parameters
    nmfcc = 14
    N = len(wav_files) #number of all words
    fL = 700
    fH = 3900
    frame_length = 512
    hop_length = 128
    n_fft = 2048
    n_filters = 35
    f_max = 8000
    fs = 44100 # fs is consequently set by librosa load
    max_word = 2
    maxL = int(np.ceil(max_word*fs/hop_length))
    M_mfb = np.zeros([n_filters, maxL])
    average_word = 0.1
    n_frames = int(np.round(average_word*fs/hop_length)) # fixed number of frame for MFBfixlist
    
    
    WordList = os.listdir(labelsPath)
    if 'desktop.ini' in WordList:
            false_dir_index = WordList.index('desktop.ini')
            WordList.pop(false_dir_index)
    WordList.sort(key = int)
    WordLen = len(WordList)
    
    word_df = pd.DataFrame([])
    MFBlist = []
    mfcclist = []
    
    num_syl = {'1':3 ,'12':4 , '14':3 , '17':1 , '40':3 , '41':4 , '50':1
               , '74':3 , '103':2 , '118':2 , '129':3 , '179':3 , '341':3 , '999':0}
    
    ind = 0
    for j in WordList:
        print(j)
        arr = os.listdir(str(labelsPath +'/'+ j ))
        if 'desktop.ini' in arr:
                false_arr_index = arr.index('desktop.ini')
                arr.pop(false_arr_index)
        #print(arr)
        L = len(arr) 
        
        data_files = []
        
        for k in range(L):
            wavname = str(labelsPath +'/'+ str(j)+ '/' + arr[k])
              
            # print(wavname)
              
            x ,fs = librosa.load(wavname, sr = None) 
            sos = signal.butter(35, np.array([fL,fH]), 'bp', fs = fs, output='sos')
            x = signal.sosfilt(sos, x)
            # sd.play(x,fs)
            # time.sleep(1)
            
            filename = arr[k]
            word_number = j
            numSyl = num_syl[word_number]  
            
            word_length = x.size/fs
            
            mfcc = librosa.feature.mfcc(x,sr = fs, n_mfcc = nmfcc)
            
    ######### CASE 1
            if case == '1':
                MFB = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = n_fft,
                                                   hop_length = hop_length,n_mels=n_filters,       
                fmax=f_max)

                
                MFB = Ut.medclip(MFB , 3.5 , 1e-14)
                MFB_dB = librosa.power_to_db(MFB, ref=np.max)
                
                MFBlist.append(MFB_dB) # - case 1
            
            elif case == '2' :    
    ##########  fixed number of frames with variable hop size
                hop_size = int(np.floor(x.size/n_frames))
               
                
                win_length = 2*hop_size
                if win_length > n_fft:
                    win_length = n_fft
                                    
                MFBfix = np.zeros([n_filters, n_frames])
                MFBcal = librosa.feature.melspectrogram(y = x, sr = fs, 
                        win_length = win_length,n_fft = n_fft,
                        hop_length = hop_size,n_mels=n_filters, fmax=f_max)
                MFBcal_dB = librosa.power_to_db(MFBcal, ref=np.max)
                dimMFB = MFBcal.shape
                MFBfix[:, :dimMFB[1]] = MFBcal_dB[:,:n_frames]
                
                MFBfix2 = np.zeros([n_filters, n_frames])
                MFBcal2 = Ut.medclip(MFBcal , 3.5 , 1e-14)
                MFBcal2_dB = librosa.power_to_db(MFBcal2, ref=np.max)
                MFBfix2[:, :dimMFB[1]] = MFBcal2_dB[:,:n_frames]
                
                MFBlist.append(MFBfix2) # - case 2
            
            elif case == '3':

        # Mel spectrogram with fixed length with zero padding to largest expected
        #    word length (fixed frame length and hop size) - M_mbflist 
                M_mfb = np.zeros([n_filters, maxL])
                MFB_last = MFB.shape[1]
                M_mfb[:,:MFB_last] = np.copy(MFB_dB) # fixed length MFB with zero padding for the columns (case 3)
                
                MFBlist.append(M_mfb) # - case 3

            
            mfcclist.append(mfcc)
            
            data_files.append([filename , word_number , word_length, numSyl, ind]) 
            column = ['filename' , 'word_number' , 'word_length', 'number_of_syl' , 'MFB']
            ind = ind + 1
            
               
            
        df = pd.DataFrame(data_files , columns = column)
        
        word_df = pd.concat([word_df,df], ignore_index=True)
    
    return word_df , MFBlist
