# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:15:11 2020

@author: AM and YL
"""
import numpy as np
import sys, os
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import pandas as pd
from sklearn.cluster import KMeans
import word_analysis as WA



def comp_centroid (X_r ,y2):
    '''

    Parameters
    ----------
    X_r : metrix of number of words X 2 coordinates
        contains the coordinates of each word after PCA tranformation.
    y2 : array of word number, of all words
        contains all words and describe them as numbers.

    Returns coor_point , point_tag , perc_dist, closes_median
    -------

    '''
    word_names = np.unique(y2)
    word_names = word_names.astype('int')
    Nword = word_names.size
    coldim = X_r.shape[1]
    centroids = np.zeros([Nword, coldim])
    medians = np.zeros([Nword, coldim])
    
    for i in range(Nword):
                                     
        centroids[i,0:coldim] = np.mean(X_r[np.where(y2 == word_names[i]), 0:coldim],axis = 1)  
        medians[i,0:coldim] = np.median(X_r[np.where(y2 == word_names[i]), 0:coldim], axis = 1)
        
    return centroids , medians

def norm(v,p):
    """
        function to  compute the l-p norm of an input vector.
        Inputs: v - a numpy array (n dim vector)
        p - the order of the norm (1,2,3...)
        if p = np.inf the max norm should be computed
        Outputs:  p-th norm of v
     """
    if p == np.inf:
        n = max(abs(v))
    else:
        n = (sum(abs(v)**p))**(1/p)
    return n

def euqli_dist(p, q):
    # Calculates the euclidean distance, the "ordinary" distance between two
    # points
     return sqrt(sum((p-q)**2))
    
def  nearest_point(point, medians , wordlist):
    low_dist = float('inf')
    closest_pos = None
    points_dist = 0
    M = medians.shape[0]
    for i in range(M):
        dist = norm(point-medians[i] , 2) 
        points_dist += dist
        if dist < low_dist:
            low_dist = dist
            closes_median = medians[i]
            coor_point = point
            point_tag = wordlist[i]
            perc_dist = (1-(low_dist/points_dist))*100
    return coor_point , point_tag , perc_dist, closes_median , low_dist

        
def  data_Reduction_words(DF ,MFBlist ,  drMethod , n_components = 2 , centroid = True , newWords = True):
    """
    parameters
    ----------
    DF: a datafram containing files information
    MFBlist : list of MFB from all signals
    drMethod : pca or tsne
    n_components : 2 as default but can be more for clustering
    centroid : if True, centroid are calculate and presented.
    newWords : if True, new words can be applied on the old words data 
    
          
    output
    ----------
    returns: a plot of pca/ tsne
    X_r : coordinates of data
    y2 : labels of data
    
    """
    y = DF.iloc[:,1]
    y1 = y.to_numpy()
    y2 = y1.astype('int')
    y3 = np.unique(y2)
    yunique = np.unique(y)
    
    wlen = DF.iloc[:,2]
    wlen = wlen.to_numpy().reshape(wlen.size,1)
    wlen = wlen - wlen.mean()
    nsyl = DF.iloc[:,3]
    nsyl = nsyl.to_numpy().reshape(nsyl.size,1)
    #test = np.random.uniform(size=(515,1))
    new_words_df = pd.DataFrame([])


    
    
    numSample = int(len(y))
    numParameter = int(MFBlist[0].size)
    Mb = np.zeros([numSample , numParameter ])
    #small_df = DF.iloc[:,2:]
    for i in range(numSample):
        result = MFBlist[i].flatten()
        Mb[i,:] = (result-np.min(result))/(np.max(result)-np.min(result))

    lendir = yunique.size
    z = list(range(1 , lendir+1))
    #z = list(np.unique(y))
    zlist = list(map(int, yunique))
    z1= sorted(zlist)
    z2 = list(range(len(z1)))
    
    # >> insert words names
    
    words_names =['1-tu_ti_tuyu', '12-t_tu_ti_tuyu' , '14-tuyu_tu_ti' , '17-tutitu',
                  '40-ti_tu_ti' ,'41-ti_tu_tu_ti' , '50-watch' , '74-tiyutitiyutitiyu' ,
                  '103-ti_ti' , '118-ti_tuyu' , '129-tuyu_tu_tuyu' ,
                  '179-ti_ti_towy' , '341- wa_towy_ti']

    
    colors = ['blue', 'black', 'cyan' , 'palegreen','purple', 'red'
                   ,'teal' ,'orange', 'yellow','pink', 'silver' ,'sienna',
                     'cornflowerblue'] 
           
    lw = 2

    
    if drMethod == 'pca':
        
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        
        pca = PCA(n_components)
        X_r = pca.fit(Mb).transform(Mb)
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))
        fig = plt.figure(1, figsize=(16, 16))
        for color, i, words_name in zip(colors, z1, words_names):
            plt.scatter(X_r[y2 == i, 0], X_r[y2 == i, 1], color=color ,  s = 250 , alpha=.5,
                        lw=lw, label=words_name)

        plt.legend(bbox_to_anchor=(1.0 , 0.9) , loc='best', shadow= False, scatterpoints=1 )
        # plt.title('PCA of bulbul dataset')
        plt.grid('both')
        # plt.show()

        if centroid == True:
            
            centroids, medians = comp_centroid (X_r ,y2)
            for color, i, words_name in zip(colors, z2, words_names):
                plt.scatter(medians[i, 0], medians[i, 1], color=color , marker = "X" ,  s = 250 , alpha=.9,
                        lw=lw, label=words_name)
            # plt.show()
            
        if newWords == True:
            
            DFNew_Word = []
            
            Nlarge_df, NMFBlist = WA.word_analysis(labelsPath = "RandomWords" , case = '2')
            Ny = Nlarge_df.iloc[:,1]
            Ny1 = Ny.to_numpy()
            Ny2 = Ny1.astype('int')
            Nyunique = np.unique(Ny)
   
            numSample = int(len(Ny))
            numParameter = int(NMFBlist[0].size)
            NMb = np.zeros([numSample , numParameter ])
            #small_df = DF.iloc[:,2:]
            for i in range(numSample):
                result = NMFBlist[i].flatten()
                NMb[i,:] = (result-np.min(result))/(np.max(result)-np.min(result))

            
            N_r = pca.transform(NMb)
            
            new_names =['1-tu_ti_tuyu', '12-t_tu_ti_tuyu' , '14-tuyu_tu_ti' 
                   ,'41-ti_tu_tu_ti' , '118-ti_tuyu' , '129-tuyu_tu_tuyu' ,
                  '179-ti_ti_towy' , '341- wa_towy_ti' , '999-random']
            
            Ncolors = ['blue', 'black', 'cyan' , 'red'
                   ,'pink', 'silver' ,'sienna',
                     'cornflowerblue', 'yellow'] 
            
            N2 = [1,12,14,41,118,129,179,341,999]
            
            for color, i, words_name in zip(Ncolors, N2, new_names):
                plt.scatter(N_r[Ny2 == i, 0], N_r[Ny2 == i, 1], color=color , marker = "d" ,  s = 250 , alpha=.5,
                        lw=lw, label=words_name)
            
            plt.show()
            
            for i in range(Ny2.size):
                wordtag = int(Ny2[i])
                
                coor_point , point_tag , perc_dist, closes_median , low_dist = nearest_point(N_r[i] , centroids , y3 )
                
            
                DFNew_Word.append([wordtag , coor_point , point_tag , perc_dist, low_dist , closes_median ]) 
                column = ['word_number' , 'word_coor' , 'closest_centroid', 'score' , 'lowest_dist' , 'centroid_coor']

            df = pd.DataFrame(DFNew_Word , columns = column)
        
            new_words_df = pd.concat([new_words_df,df], ignore_index=True)
            
            return X_r , y2 , N_r , Ny2, new_words_df
            
        else:
            return X_r , y2
    
    if drMethod == 'pca_len':
        
            font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 22}
    
            plt.rc('font', **font)
            
            n_comp = 2
            pca = PCA(n_components= n_comp)
            X_r = pca.fit(Mb).transform(Mb)
            # Percentage of variance explained for each components
            X_r_len = np.hstack((X_r,wlen))
            X_r_len = np.hstack((X_r_len,nsyl))
            
            
            pca = PCA(n_components= 2)
            X_rl = pca.fit(X_r_len).transform(X_r_len)
            print('explained variance ratio (first two components): %s'
                  % str(pca.explained_variance_ratio_))
            fig = plt.figure(1, figsize=(16, 16))
    
            for color, i, words_name in zip(colors, z1, words_names):
                plt.scatter(X_rl[y2 == i, 0], X_rl[y2 == i, 1] , color=color ,  s = 250 , alpha=.9,
                            lw=lw, label=words_name)
    
            plt.legend(bbox_to_anchor=(1.0 , 0.9) , loc='best', shadow= False, scatterpoints=1 )
            # plt.title('PCA of bulbul dataset')
            plt.grid('both')
            plt.show()

            return X_rl
    
    if drMethod == 'tsne':
        
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components, init='pca', perplexity=50.0, 
        n_iter = 7000, 
        random_state = 0).fit_transform(Mb)
        fig = plt.figure(2, figsize=(16, 16))
        for color, i, words_name in zip(colors, z1, words_names):
            plt.scatter(X_embedded[y2 == i, 0], X_embedded[y2 == i, 1], color=color, s = 120,alpha=.9, lw=lw,
                        label= words_name)

        plt.legend(bbox_to_anchor=(1.0 , 0.9) , loc='best', shadow=True, scatterpoints=1)
        # plt.title('TSNE of bulbul dataset')
        plt.grid('both')
        
        return X_embedded , y   
        
def  data_Reduction_syllable(DF , drMethod , colorMethod = 'syl' , n_components = 2 , centroid = True , newWords = False):
    """
    parameters
    ----------
    DF: a datafram containing files information
    drMethod : pca or tsne
    n_components : 2 as default but can be more for clustering (not more then 4)
    centroid : if True, centroid are calculate and presented.
    newWords : not relevant. sould be set on False
    
          
    output
    ----------
    returns: a plot of pca/ tsne
    X_r : coordinates of data
    y : labels of data
    
    """
    y = DF.iloc[:,1]
    y.to_numpy()
    small_df = DF.iloc[:,2:]
    Mb = small_df.to_numpy()
    Mb = Mb[:,-4:] # 4 coeff
    lendir = np.unique(y).size
    z = list(range(1 , lendir+1))
    
    syllable_names =['A1-t', 'A2-tu' , 'A3-ti' , 'A4-tuyu', 'B5-ti' , 'B6-tuyu' ,
                    'C7-tuyu' , 'C8-ti' , 'C9-tuyu' , 'D10-ti' , 'D11-ti' , 'D12-towy' ,
                    'E13-tu' , 'E14-ti', 'E15-ti', 'F16-watch' , 'G17-tu' , 'G18-ti' , 'G19-tuyu',
                    'H20-wa' , 'H21-towy' , 'H22-ti']
    
    if colorMethod == 'syl':
          colors = ['blue', 'blue', 'green', 'red', 'green', 'tomato',
                'red', 'green', 'tomato', 'lime', 'lime', 'blueviolet',
                'blue', 'green', 'green', 'yellow', 'blue',
              'green', 'red','yellow', 'blueviolet', 'green']

    
    elif colorMethod == 'word':
        colors = ['blue', 'blue', 'blue', 'blue', 'gray', 'gray',
                  'cyan', 'cyan', 'cyan', 'blueviolet', 'blueviolet', 'blueviolet',
                  'limegreen', 'limegreen', 'limegreen', 'yellow', 'darkorange',
                  'darkorange', 'darkorange','red', 'red', 'red']

    lw = 2

    
    if drMethod == 'pca':
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        
        pca = PCA(n_components)
        X_r = pca.fit(Mb).transform(Mb)
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))
        fig = plt.figure(1, figsize=(16, 16))
        for color, i, syllable_name in zip(colors, z, syllable_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, s = 150 ,alpha=.9, lw=lw,
                        label=syllable_name)
            

        plt.legend(bbox_to_anchor=(1.0 , 0.9) , loc='best', shadow= False, scatterpoints=1 )
        # plt.title('PCA of bulbul dataset' , fontsize=15)
        plt.grid('both')



        return X_r , y


    if drMethod == 'tsne':
        
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22}

        plt.rc('font', **font)
        
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components, init='pca', perplexity=50.0, 
        n_iter = 7000, 
        random_state = 0).fit_transform(Mb)
        fig = plt.figure(2, figsize=(16, 16))
        for color, i, syllable_name in zip(colors, z, syllable_names):
            plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], color=color, s = 120,alpha=.9, lw=lw,
                        label=syllable_name)

        plt.legend(bbox_to_anchor=(1.0 , 0.9) , loc='best', shadow=True, scatterpoints=1)
        # plt.title('TSNE of bulbul dataset')
        plt.grid('both')
        
        return X_embedded , y   
    
  

def words_classify(X_train, y_train, X_test, y_labels, K, method = 'NearestCentroid'):
    """
    words_classify use a classification method to classify a set of new words
    using one of several classification methods with a labeld training set 
    (X_train, y_train).
    current methods: 'knn', 'NearestCentroid', 'svm', 'logreg', 'nn'
    see: https://scikit-learn.org/stable/supervised_learning.html
    Input parameters:
        X_train - labeled input dataset of points (each point in one row)
        y_train - labels of X_train points
        X_test - original (high dimensional) data) to classify
        y_labels - labels of X_test
        K - number of k neighbors
        method - classification method [str]
    Return:
        y_test - the predicted classes
    """
    
    #X_test_r = pca.transform(X_test) # applying the same transform as used to generate X_train
    if method == 'knn': # k nearest neighbors
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(X_train, y_train)
        KNeighborsClassifier(...)
        y_test = neigh.predict(X_test)
    elif method == 'nearestcent': # nearest centroid
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        NearestCentroid()
        y_test = clf.predict(X_test)
    elif method == 'svm': # support vector machine
        from sklearn import svm
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        y_test = clf.predict(X_test)
    elif method == 'logreg': # logistic regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_test = clf.predict(X_test)
    elif method == 'nn': # multi-layer neural network
        from sklearn.neural_network import MLPClassifier
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-3, max_iter = 25000,
                     hidden_layer_sizes=(7, np.unique(y_labels).size ), random_state=1)
        clf.fit(X_train_s, y_train)
        y_test = clf.predict(X_test_s)
    
    acc = (np.sum((y_test-y_labels)==0))/len(y_labels)*100
    print('acc = ', acc)
    
    return y_test, acc
    
def unit_clustering(x , y):
    """
    Parameters
    ----------
    x : each word coordinate on the pca space.
        dimentions vary acording to dimentionality reduction.
    y : lables
        the correct lable for each word.

    Returns
    -------
    scatter plot with clusters.

    """
    yunique = np.unique(y)
    kmeans = KMeans(n_clusters = yunique.size)
    KMeans.fit(x)
    y_kmeans = kmeans.predict(x)
    
    fig = plt.figure(2, figsize = (16,16))
    # plt.figure(2)
    plt.scatter(x[:, 0], x[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 200, alpha = 0.5)
    
    return fig


