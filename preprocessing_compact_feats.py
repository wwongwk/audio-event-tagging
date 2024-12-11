# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:10:01 2023

@author: anusha
"""
import pandas as pd
import numpy as np
import librosa


def clean_data(data):
    X_filenames = data.loc[:, ["filename"]]
    y = data.loc[:, ["target"]]
    
    return X_filenames, y

def return_mfcc(audio, sr, n_mfcc=12):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, fmax=sr/2.0)
    a_mfcc = librosa.feature.mfcc(S = librosa.power_to_db(mel_spect), sr = sr, n_mfcc=n_mfcc)
    return a_mfcc

def return_compact_features(X):
    '''
        calculate 12MFCCs + 1 energy + 1 ZCR per frame
        calculate mean of all 14 features over all the frames for a given audio sample
        calculate stddev of all 14 features over all frames for a given audio sample
        calculate mean of delta of MFCC_x over all frames (x in {1,2,...,12}), and mean of delta of energy over all frames, 
            and mean of delta of ZCR over all frames.
        calculate mean of delta-delta of MFCC_x over all frames (x in {1,2,...,12}), and mean of delta-delta of energy over all frames, 
            and mean of delta-delta of ZCR over all frames.
    
        Totally that's 14x4 features per audio file = 56 features.
    '''
    #input_path = Path('data/ESC-50-master/audio/')
    X_new = []
    
    for i in range(len(X)):
        f = X.loc[i,"filename"]
        path = ".\\data\\ESC-50-master\\audio\\" + f
        y, sr = librosa.load(str(path), duration=5, sr=None)
        x_i = []
        
        #-------------MEAN---------------
        #returns 12 MFCCs for 216 frames:
        mfcc = return_mfcc(y, sr, 12)
        #takes mean of 12 MFCCs over all frames:
        x_i.extend(np.mean(mfcc, axis=1))
        #print(x_i)
        
        #returns 1 energy value per frame:
        energy = librosa.feature.rms(y=y)
        #takes mean of energies over all frames:
        x_i.extend([np.mean(energy)])
        
        #returns 1 ZCR per frame:
        ZCR = librosa.feature.zero_crossing_rate(y=y)
        #takes the mean of ZCRs over all frames:
        x_i.extend([np.mean(ZCR)])
        
        #--------------STD DEV----------------
        #takes std dev of 12 MFCCs over all frames:
        x_i.extend(np.std(mfcc, axis=1))
        #takes the std dev of energy over all frames:
        x_i.extend([np.std(energy)])
        #takes the std dev of ZCR over all frames:
        x_i.extend([np.std(ZCR)])
        
        #-----------------DELTA---------------
        #takes the delta of MFCCs over all frames:
        dx_mfcc = librosa.feature.delta(mfcc)
        #takes the mean of delta_MFCC over all frames:
        x_i.extend(np.mean(dx_mfcc, axis=1))
        
        #takes the delta of energy values for each frame:
        dx_energy = librosa.feature.delta(energy)
        #mean of delta energy:
        x_i.extend([np.mean(dx_energy)])
        
        #takes the delta of ZCR over all frames:
        dx_ZCR = librosa.feature.delta(ZCR)
        #mean of delta ZCR:
        x_i.extend([np.mean(dx_ZCR)])
        
        #------------------DELTA DELTA----------------
        #takes the delts-delta of MFCCs over all frames:
        ddx_mfcc = librosa.feature.delta(mfcc, order = 2)
        #takes the mean of delta-delta_MFCC over all frames:
        x_i.extend(np.mean(ddx_mfcc, axis=1))     
        
        #takes the delta-delta of energy values for each frame:
        ddx_energy = librosa.feature.delta(energy, order=2)
        #mean of delta-delta energy:
        x_i.extend([np.mean(ddx_energy)])
        
        #takes the delta-delta of ZCR over all frames:
        ddx_ZCR = librosa.feature.delta(ZCR, order=2)
        #mean of delta-delta ZCR:
        x_i.extend([np.mean(ddx_ZCR)])
        
        X_new.append(x_i)
    
    X_new = np.array(X_new)
    
    return X_new


data = pd.read_csv(".\\data\\ESC-50-master\\meta\\esc50.csv")
Xfiles, y = clean_data(data)
X_feats = return_compact_features(Xfiles)