# -*- coding: utf-8 -*-
"""
@author: anusha
@title: Audio Event Tagging - Feature extraction & model training
"""

from functionsCode import clean_data, return_Xdata, return_compact_features, random_frames_dataset, first_nframes_dataset, every_nth_frame_dataset
from functionsCode import SVM_PCA_loop, SVM_model_ovo_5nested_5cv, SVM_model_ovr_5nested_5cv
import pandas as pd
import numpy as np

#Import data and perform basic preprocessing
data = pd.read_csv(".\\data\\ESC-50-master\\meta\\esc50.csv")
Xfiles, y = clean_data(data)
y = np.array(y)

data_10 = data.loc[data['esc10']==True]
data_10 = data_10.reset_index()
data_10 = data_10.drop(["index"], axis=1)
Xfiles_10, y_10 = clean_data(data_10)
y_10 = np.array(y_10)

metadata = pd.read_csv(".\\data\\ESC-50-master\\meta\\esc50.csv")
#mapping to string labels
label_mapping = dict(metadata[['target', 'category']].drop_duplicates().sort_values(by = ['target']).to_numpy())

#mapping to string labels for ESC10
label_mapping_esc10 = dict(data_10[['target', 'category']].drop_duplicates().sort_values(by = ['target']).to_numpy())


labels = [v for v in label_mapping.values()]
labels_esc10 = [v for v in label_mapping_esc10.values()]



#--------------feature extraction methods----------------
'''
(1) Returns full X data. No Train test split.
    None of the feature extractions depend on other datapoints.
    Features are extracted independently for each datapoint, hence the 
    feature extraction is done in one go for the whole data.

(2) Output label remains the same.
'''

#-----method 1-----
'''
Extracting 12 MFCCs for each frame, and concatenating them into a 1D array
'''
#ESC50
X_m1 = return_Xdata(Xfiles)
#ESC10
X_m1_10 = return_Xdata(Xfiles_10)

#-----method 2-----
'''
calculate 12MFCCs + 1 energy + 1 ZCR per frame
calculate mean of all 14 features over all the frames for a given audio sample
calculate stddev of all 14 features over all frames for a given audio sample
calculate mean of delta of MFCC_x over all frames (x in {1,2,...,12}), and mean of delta of energy over all frames, 
    and mean of delta of ZCR over all frames.
calculate mean of delta-delta of MFCC_x over all frames (x in {1,2,...,12}), and mean of delta-delta of energy over all frames, 
    and mean of delta-delta of ZCR over all frames.

Totally 14x4 features per audio file = 56 features.
'''
#ESC50
X_m2 = return_compact_features(Xfiles)
#ESC10
X_m2_10 = return_compact_features(Xfiles_10)

#-----method 3-----
'''
Given n_frames, select n_frames number of random frame indices.
Use the 12 MFCCs from those frames, and concatenate into a 1D array
n_frames * 12 features for each audio file
'''
#ESC50
X_m3, X_m3_idxlst = random_frames_dataset(Xfiles, 100)
#ESC10
X_m3_10, X_m3_10_idxlst = random_frames_dataset(Xfiles_10, 100)

#-----method 4-----
'''
Given n_frames, select the first n_frames number of frames.
Use the 12 MFCCs from those frames, and concatenate into a 1D array
n_frames * 12 features for each audio file
'''
#ESC50
X_m4 = first_nframes_dataset(Xfiles, 100, 0)
#ESC10
X_m4_10 = first_nframes_dataset(Xfiles_10, 100, 0)

#-----method 5-----
'''
Pick every 15th Frame out of all the 431 frames, for each audio file
Use the 12 MFCCs from those frames, and concatenate into a 1D array
n_frames * 12 features for each audio file
'''
#ESC50
X_m5 = every_nth_frame_dataset(Xfiles, 15)
#ESC10
X_m5_10 = every_nth_frame_dataset(Xfiles_10, 15)

#-----method 6-----
'''
Given n_frames and k, select the first n_frames number of frames after leaving k frames.
Use the 12 MFCCs from those frames, and concatenate into a 1D array
n_frames * 12 features for each audio file
'''
#ESC50
X_m6 = first_nframes_dataset(Xfiles, 100, k = 100)
#ESC10
X_m6_10 = first_nframes_dataset(Xfiles_10, 100, k = 100)



#--------------SVM models to be trained----------------


#-----model 1-----
'''
PCA on method 1 data
OVO SVM model
'''
#ESC50
print("PCA + OVO SVM: ESC50:")
model1_esc50_acc = SVM_PCA_loop(X_m1, y, labels, "PCA + OVO SVM: ESC50")
#ESC10
print("PCA + OVO SVM: ESC10:")
model1_esc10_acc = SVM_PCA_loop(X_m1_10, y_10, labels_esc10, "PCA + OVO SVM: ESC10")


#-----model 2-----
'''
OVO SVM on method 2 data
'''
#ESC50
print("Method 2 + OVO SVM 5fold nested CV: ESC50:")
model2_esc50_acc, model2_esc50_C = SVM_model_ovo_5nested_5cv(X_m2, y, labels, "Method 2 + OVO SVM 5fold nested CV: ESC50")

#ESC10
print("Method 2 + OVO SVM 5fold nested CV: ESC10:")
model2_esc10_acc, model2_esc10_C = SVM_model_ovo_5nested_5cv(X_m2_10, y_10, labels_esc10, "Method 2 + OVO SVM 5fold nested CV: ESC10:")

#-----model 3-----
'''
OVR SVM on method 2 data
'''
#ESC50
print("Method 2 + OVR SVM 5fold nested CV: ESC50:")
model3_esc50_acc, model3_esc50_C = SVM_model_ovr_5nested_5cv(X_m2, y, labels, "Method 2 + OVR SVM 5fold nested CV: ESC50")
#ESC10
print("Method 2 + OVR SVM 5fold nested CV: ESC10:")
model3_esc10_acc, model3_esc10_C = SVM_model_ovr_5nested_5cv(X_m2_10, y_10, labels_esc10, "Method 2 + OVR SVM 5fold nested CV: ESC10:")

#-----model 4-----
'''
OVO SVM on method 3 data
'''
#ESC50
print("Method 3 + OVO SVM 5fold nested CV: ESC50:")
model4_esc50_acc, model4_esc50_C = SVM_model_ovo_5nested_5cv(X_m3, y, labels, "Method 3 + OVO SVM 5fold nested CV: ESC50")
#ESC10
print("Method 3 + OVO SVM 5fold nested CV: ESC10:")
model4_esc10_acc, model4_esc10_C = SVM_model_ovo_5nested_5cv(X_m3_10, y_10, labels_esc10, "Method 3 + OVO SVM 5fold nested CV: ESC10")

#-----model 5-----
'''
OVO SVM on method 4 data
'''
#ESC50
print("Method 4 + OVO SVM 5fold nested CV: ESC50:")
model5_esc50_acc, model5_esc50_C = SVM_model_ovo_5nested_5cv(X_m4, y, labels, "Method 4 + OVO SVM 5fold nested CV: ESC50")
#ESC10
print("Method 4 + OVO SVM 5fold nested CV: ESC10:")
model5_esc10_acc, model5_esc10_C = SVM_model_ovo_5nested_5cv(X_m4_10, y_10, labels_esc10, "Method 4 + OVO SVM 5fold nested CV: ESC10")

#-----model 6-----
'''
OVO SVM on method 5 data
'''
#ESC50
print("Method 5 + OVO SVM 5fold nested CV: ESC50:")
model6_esc50_acc, model6_esc50_C = SVM_model_ovo_5nested_5cv(X_m5, y, labels, "Method 5 + OVO SVM 5fold nested CV: ESC50")
#ESC10
print("Method 5 + OVO SVM 5fold nested CV: ESC10:")
model6_esc10_acc, model6_esc10_C = SVM_model_ovo_5nested_5cv(X_m5_10, y_10, labels_esc10, "Method 5 + OVO SVM 5fold nested CV: ESC10")


#-----model 7-----
'''
OVO SVM on method 6 data
'''
#ESC50
print("Method 6 + OVO SVM 5fold nested CV: ESC50:")
model7_esc50_acc, model7_esc50_C = SVM_model_ovo_5nested_5cv(X_m6, y, labels, "Method 6 + OVO SVM 5fold nested CV: ESC50")
#ESC10
print("Method 6 + OVO SVM 5fold nested CV: ESC10:")
model7_esc10_acc, model7_esc10_C = SVM_model_ovo_5nested_5cv(X_m6_10, y_10, labels_esc10, "Method 6 + OVO SVM 5fold nested CV: ESC10")
