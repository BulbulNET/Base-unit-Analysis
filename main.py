# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:16:36 2021

main script for running functions of Words and Syllables analysis,
data reduction , and clustering techniques - 

mfb_analysis - mfb_analysis to obtain all words from directory "words".
The output large_df contains the word's filename, label (word name),
length, number of syllables, and MFB index.
MFBlist - a list with mel-spectrogram of each word.

data_reduction_words - applying PCA on Bulbul's words for data reduction
and PCA transform (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
to map new words on the low dimensional.
users can obtain pca , tsne or other methods.


Authors: AM and YL
"""

import reduction_classify as RC
import word_analysis as WA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import syllable_analysis as SA

words = True
mew_words = False
syllables = False

if words:
####   word analysis for taged words from directory - folders by word name   ####
    word_df, MFBlist = WA.mfb_analysis(labelsPath = "words" , case = '2')

####   words data reduction for initial words - 
    X_r, y2 = RC.data_Reduction_words(word_df ,MFBlist, drMethod = 'tsne', n_components =2 , centroid = False , newWords = False)

if mew_words:
####   and an option to project new words on the initial analysis  ####
    X_r , y2 , N_r , Ny2, new_words_df = RC.data_Reduction_words(word_df ,MFBlist, drMethod = 'pca', n_components =10 , centroid = True , newWords = True)

if syllables:
##   sylable analysis by Lagendre polinomials-

    syllable_df = SA.syllable_analysis("syllables" , 3 , 14 , 3 , -2)

####   syllable data reduction - 

    X_r, y2 = RC.data_Reduction_syllable(syllable_df ,  drMethod = 'tsne' ,  n_components = 2 , centroid = True , newWords = False)


