# -*- coding: utf-8 -*-
"""
@author: anusha
@title: Audio Event Tagging - function defs
"""

import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
from itertools import chain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
import random
import itertools

def return_Xdata(X_files):
    '''
    #Combine all flattened MFCCs into a single array, to serve as input.
    #final dimensions are (#examples, 5603)
    #5172 = 12x431
    #431 : number of frames per audio file; 12-number of mfccs extracted per frame
    '''
    
    #input_path = Path('data/ESC-50-master/MFCCs/')
    #file_names = list(input_path.glob('**/*.pickle'))

    Xdata = []
    for i in range(len(X_files)):
        f = X_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        Xdata.append(list(chain.from_iterable(data1.T)))

    Xdata = np.array(Xdata)
    
    return Xdata

def return_Xtr_Xtst(Xtr_files, Xtst_files):
    '''
    #Combine all flattened MFCCs into a single array, to serve as input.
    #final dimensions are (#examples, 5603)
    #5172 = 12x431
    #431 : number of frames per audio file; 12-number of mfccs extracted per frame
    '''
    
    #input_path = Path('data/ESC-50-master/MFCCs/')
    #file_names = list(input_path.glob('**/*.pickle'))

    Xtr = []
    for i in range(len(Xtr_files)):
        f = Xtr_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        Xtr.append(list(chain.from_iterable(data1.T)))

    Xtr = np.array(Xtr)
    
    Xtst = []
    for i in range(len(Xtst_files)):
        f = Xtst_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        Xtst.append(list(chain.from_iterable(data1.T)))

    Xtst = np.array(Xtst)
    
    return Xtr, Xtst

def get_train_test_split(data, split=0.2):
    '''
    Split once to get the test and training set
    '''
    X_filenames = data.loc[:, ["filename"]]
    y = data.loc[:, ["target"]]
    X_train_files, X_test_files, y_train, y_test = train_test_split(X_filenames, y, test_size=split, random_state=123, stratify=y)
    #print(X_train_files.shape,X_test_files.shape)
    
    X_train_files = X_train_files.reset_index()
    X_train_files = X_train_files.drop(["index"], axis=1)
    
    X_test_files = X_test_files.reset_index()
    X_test_files = X_test_files.drop(["index"], axis=1)
    
    y_train = y_train.reset_index()
    y_train = y_train.drop(["index"], axis=1)
    
    y_test = y_test.reset_index()
    y_test = y_test.drop(["index"], axis=1)
    
    return X_train_files, X_test_files, y_train, y_test


def return_PCA_tr_tst(Xtr, Xtst, PCAcomponents):
    pca = decomposition.PCA(n_components = PCAcomponents)
    Xtr_pca = pca.fit_transform(Xtr)
    Xtst_pca = pca.transform(Xtst)
    return Xtr_pca, Xtst_pca

def return_PCA_tr_tst_val(Xtr, Xtst, Xval, PCAcomponents):
    pca = decomposition.PCA(n_components = PCAcomponents)
    Xtr_pca = pca.fit_transform(Xtr)
    Xtst_pca = pca.transform(Xtst)
    Xval_pca = pca.transform(Xval)
    return Xtr_pca, Xtst_pca, Xval_pca

def clean_data(data):
    X_filenames = data.loc[:, ["filename"]]
    y = data.loc[:, ["target"]]
    
    return X_filenames, y



def SVM_PCA_loop(X, y, labels_classes, title):
    #Input: X: 12 MFCCs are extracted for each frame of an audio file, they are flattened and a 1D feature vector is created
    # y: 1 label for each input

    #PCA is applied. PCA components are looped through - from 10 to 1000
    #best number of features are piced based on highest generalization error.
    
    tst_acc_PCA = []
    # Xfiles, y = clean_data(data)
    # X = return_Xdata(Xfiles)
    
    for no_PCA in range(10,110):
        #looping over how many PCA components we want
        
        #Split data into train-test
        X_tr_val, Xtst, y_tr_val, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        #perform PCA on the train-test datasets
        X_tr_val_pca, Xtst_pca = return_PCA_tr_tst(X_tr_val, Xtst, no_PCA)
        
        #Split the transformed data - Xtr_pca - into train and val:
        Xtr, Xval, ytr, yval = train_test_split(X_tr_val_pca, y_tr_val, test_size=0.2, random_state=42, stratify=y_tr_val)
        
        acc_for_C = []
        for c in C:
            clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=c)
            clf.fit(Xtr, np.array(ytr).flatten())
            yval_pred = clf.predict(Xval)
            
            acc_for_C.append(accuracy_score(np.array(yval).flatten(), yval_pred))
            
        bestC_idx = np.argmax(acc_for_C)
        best_clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=C[bestC_idx])
        best_clf.fit(X_tr_val_pca, np.array(y_tr_val).flatten())
        y_pred = best_clf.predict(Xtst_pca)
        
        tst_acc_PCA.append(round(accuracy_score(np.array(ytst).flatten(), y_pred)*100))
        
        print("#PCA_feats:: " + str(no_PCA) + " best tr acc:: " + str(round(np.max(acc_for_C)*100, 2)) + " at C = " + str(C[np.argmax(acc_for_C)]) + " tst acc:: " + str(round(accuracy_score(ytst, y_pred)*100, 2)))
    
    tst_acc_PCA = np.array(tst_acc_PCA)
    with open('testAcc_diffPCAfeats_esc50_1.pkl', 'wb') as file: 
        pickle.dump(tst_acc_PCA, file)
        
    ind = np.argpartition(tst_acc_PCA, -5)[-5:]
    print("RESULTS")
    print("top 5 number of PCA features - best accuracies::")
    print("best accuracies:")
    print(tst_acc_PCA[ind])
    print("# of PCA features = Indices + 10. Indices:")
    print(ind)
    
    best_pca = np.argmax(tst_acc_PCA) + 10
    xtr, xtst, ytr, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    #perform PCA on the train-test datasets
    xtr_pca, xtst_pca = return_PCA_tr_tst(xtr, xtst, best_pca)
    
    clf_best = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=10)
    clf_best.fit(xtr_pca, np.array(ytr).flatten())
    y_pred = clf_best.predict(xtst_pca)
    
    final_accuracy = round(accuracy_score(np.array(ytst).flatten(), y_pred)*100, 2)
    
    cnf_matrix = confusion_matrix(np.array(ytst).flatten(), y_pred)
    
    plot_confusion_matrix(cnf_matrix,classes=labels_classes, title_1=title)
    
    
    print("Final accuracy:: " + str(final_accuracy))
    return final_accuracy


def SVM_ovo_adding1class(data):
    #----------------not adding to evaluation : just for testing--------------------
    #increasing by 1 class at a time and training the model
    #PCA is applied to data
    #finding the best C value for each model and accuracy score on it.
    
    C = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    results = []
    for k in range(2,51):
        #looping over how many classes we are using
        data_sub = data.loc[data['target']==0]
        for i in range(1,k):
            d = data.loc[data['target']==i]
            data_sub = pd.concat([data_sub, d], axis=0)
            
        X_train_files, X_test_files, y_train, y_test = get_train_test_split(data_sub, 0.2)
        Xtr_1, Xtst_1 = return_Xtr_Xtst(X_train_files, X_test_files)
        
        Xtr, Xtst = return_PCA_tr_tst(Xtr_1, Xtst_1, int((k)*0.8*40))
        
        acc_for_C = []
        
        for c in C:
            clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=c)
            clf.fit(Xtr, np.array(y_train).flatten())
            y_pred = clf.predict(Xtst)
            
            acc_for_C.append(accuracy_score(np.array(y_test).flatten(), y_pred))
            
        results_str = "#features:: " + str(k) + " best acc:: " + str(round(np.max(acc_for_C)*100,2)) + " at C = " + str(C[np.argmax(acc_for_C)])
        results.append(results_str)
        print(results_str)
    
    return results
    
def SVM_ovr_adding1class(data):
    #----------------not adding to evaluation : just for testing--------------------
    #same as SVM_ovo_adding1class excpet ovr instead of ovo
    
    results = []
    for k in range(2,20):
        #looping over how many classes we are using
        data_sub = data.loc[data['target']==0]
        for i in range(1,k):
            d = data.loc[data['target']==i]
            data_sub = pd.concat([data_sub, d], axis=0)
            
        X_train_files, X_test_files, y_train, y_test = get_train_test_split(data_sub, 0.2)
        Xtr_1, Xtst_1 = return_Xtr_Xtst(X_train_files, X_test_files)
        
        Xtr, Xtst = return_PCA_tr_tst(Xtr_1, Xtst_1, min(int((k)*0.6*40), 1000))
        
        acc_for_C = []
        
        for c in C:
            lin_clf = svm.LinearSVC(max_iter=10000, multi_class="ovr", dual="auto", C=c)
            lin_clf.fit(Xtr, np.array(y_train).flatten())
            y_pred = lin_clf.predict(Xtst)
            
            acc_for_C.append(accuracy_score(np.array(y_test).flatten(), y_pred)) #looking at the best generalization error
            
        results_str = "#features:: " + str(k) + " best acc:: " + str(np.max(acc_for_C)) + " at C = " + str(C[np.argmax(acc_for_C)])
        results.append(results_str)
        print(results_str)
        
    return results


def return_mfcc(audio, sr, n_mfcc=12):
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, fmax=sr/2.0)
    a_mfcc = librosa.feature.mfcc(S = librosa.power_to_db(mel_spect), sr = sr, n_mfcc=n_mfcc)
    return a_mfcc

def return_compact_features(X_files):
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
    
    for i in range(0, len(X_files)):
        f = X_files.loc[i,"filename"]
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


def SVM_model1_ovo(X_feats, y, mappings):    
    Xtr_val, Xtst, ytr_val, ytst = train_test_split(X_feats, y, test_size=0.2, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtr_val, ytr_val, test_size=0.2, stratify=ytr_val)
    
    acc_for_C = []
    for c in C:
        clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=c)
        clf.fit(Xtr, np.array(ytr).flatten())
        ypred = clf.predict(Xval)
        acc_for_C.append(accuracy_score(np.array(yval).flatten(), ypred))
    
    bestC_idx = np.argmax(acc_for_C)
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=C[bestC_idx])
    clf.fit(Xtr_val, np.array(ytr_val).flatten())
    ypred = clf.predict(Xtst)
    final_acc = accuracy_score(np.array(ytst).flatten(), ypred)
    '''
    cm = confusion_matrix(np.array(ytst).flatten(), ypred, labels=mappings)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mappings)
    disp.plot()'''
    
    print("Final accuracy:: " + str(round(final_acc*100, 2)))
    return final_acc, C[bestC_idx]
    

def SVM_model2_ovr(X_feats, y):   
    Xtr_val, Xtst, ytr_val, ytst = train_test_split(X_feats, y, test_size=0.2, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtr_val, ytr_val, test_size=0.2, stratify=ytr_val)
    
    acc_for_C = []
    for c in C:
        lin_clf = svm.LinearSVC(max_iter=10000, multi_class="ovr", dual="auto", C=c)
        lin_clf.fit(Xtr, np.array(ytr).flatten())
        ypred = lin_clf.predict(Xval)
        acc_for_C.append(accuracy_score(np.array(yval).flatten(), ypred))
    
    bestC_idx = np.argmax(acc_for_C)
    lin_clf = svm.LinearSVC(max_iter=10000, multi_class="ovr", dual="auto", C=C[bestC_idx])
    lin_clf.fit(Xtr_val, np.array(ytr_val).flatten())
    ypred = lin_clf.predict(Xtst)
    final_acc = accuracy_score(np.array(ytst).flatten(), ypred)
    
    print("Final accuracy:: " + str(round(final_acc*100, 2)))
    return final_acc, C[bestC_idx]


def return_new_label_encodings(n_feats, y_col):
    #ENCODE OUTPUT TO GET BINARY TREE FORMAT
    #for 50 columns, add 6 new columns
    # add 0/1 labels -> all 6 together correspond to the original label's binary representation
    n_cols = 0
    n_features = n_feats

    while n_features > 2**n_cols:
        n_cols +=1
    #print(n_cols)

    m = y_col.shape[0]
    new_feature_encodings = np.zeros((m, n_cols), dtype=int)
    for idx in range(0, m):
        y_i = y_col[idx]
        for i in range(n_cols-1, -1, -1):
            #print(i)
            if y_i - 2**i >=0:
                new_feature_encodings[idx][n_cols-i-1] = 1
                y_i = y_i - 2**i
    
        #print(str(y_col[idx]) + "::" + str(new_feature_encodings[idx]))
    return new_feature_encodings


def get_labels_from_binaryEncodings(pred_matrix):
    y_pred = []
    for i in range(0, pred_matrix.shape[0]):
        y_i = 0
        for j in range(0, pred_matrix.shape[1]):
            y_i += pred_matrix[i][j] * 2**(pred_matrix.shape[1]-1-j)
        y_pred.append(y_i)
    
    return y_pred


def binaryTreeforSVM(X_feats, y):
    #model 4 - not giving good results
    y_enc = return_new_label_encodings(50, y_col=np.array(y))
    Xtr, Xtst, ytr, ytst = train_test_split(X_feats, y_enc, test_size=0.2, stratify=y)
    
    svm_list = []
    for i in range(0, y_enc.shape[1]):
        svm_i = svm.SVC(kernel='rbf', C=1000)
        svm_i.fit(Xtr, ytr[:,i])
        ypred_i = svm_i.predict(Xtst)
        acc = round(accuracy_score(ytst[:,i], ypred_i)*100, 2)
        print(str(i) + " :: " + str(acc))
        svm_list.append(svm_i)
        
        
    #for idx in range(0, Xtst.shape[0]):
    pred_matrix = []
    for svm_i in svm_list:
        y_pred = svm_i.predict(Xtst)
        pred_matrix.append(y_pred)
        
    pred_matrix = np.array(pred_matrix)
    pred_matrix = pred_matrix.T
    
    y_pred = get_labels_from_binaryEncodings(pred_matrix)
    y_true = get_labels_from_binaryEncodings(ytst)
    
    print(accuracy_score(y_true, y_pred))
    
    return pred_matrix, ytst


def featureTransformation_SVM_compactFeats(X_feats, y):
    transformers = [StandardScaler(with_std = False), # Centering
                StandardScaler(), #Standardisation
                Normalizer(norm = 'l2'), #L2 normalization
                MinMaxScaler()] # Unit Range
    
    Xtr_val, Xtst, ytr_val,  ytst = train_test_split(X_feats, y, test_size=0.2, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtr_val, ytr_val, test_size=0.2, stratify=ytr_val)
    final_accuracies = []
    
    for i in range(0, len(transformers)):
        acc_for_C = []
        for c in C:
            pipeline = Pipeline([('scaler', transformers[i]),
                                     ('classifier', svm.SVC(C=c))])
            pipeline.fit(Xtr, np.array(ytr).flatten())
            pipeline_pred = pipeline.predict(Xval)
            acc_for_C.append(accuracy_score(np.array(yval).flatten(), pipeline_pred))
            
        bestC_idx = np.argmax(acc_for_C)
        best_model = Pipeline([('scaler', transformers[i]),
                                 ('classifier', svm.SVC(C=C[bestC_idx]))])
        best_model.fit(Xtr_val, np.array(ytr_val).flatten())
        ypred = best_model.predict(Xtst)
        
        final_accuracies.append([accuracy_score(np.array(ytst).flatten(), ypred), C[bestC_idx]])
    
    return final_accuracies


def SVM_model_ovo_5nested_5cv(X, y, labels_classes, title):
    n = 5
    outer_folds = StratifiedKFold(n_splits=n, shuffle=True)
    
    final_acc = []
    best_Cs = []
    for index_tr_val, index_tst in outer_folds.split(X, y):
        Xtr_val = X[index_tr_val]
        ytr_val = y[index_tr_val]
        Xtst = X[index_tst]
        ytst = y[index_tst]
    
        acc_for_C = []
        for c in C:
            clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=c)
            
            acc_each_fold_givenC = []
            inner_folds = StratifiedKFold(n_splits=n, shuffle=True)
            for index_train, index_val in inner_folds.split(Xtr_val, ytr_val):
                Xtr = Xtr_val[index_train]
                ytr = ytr_val[index_train]
                Xval = Xtr_val[index_val]
                yval = ytr_val[index_val]
                
                clf.fit(Xtr, np.array(ytr).flatten())
                ypred = clf.predict(Xval)
                acc_each_fold_givenC.append(accuracy_score(np.array(yval).flatten(), ypred))
            acc_for_C.append(np.mean(acc_each_fold_givenC))
            
        
        bestC_idx = np.argmax(acc_for_C)
        clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=C[bestC_idx])
        clf.fit(Xtr_val, np.array(ytr_val).flatten())
        ypred = clf.predict(Xtst)
        
        final_acc.append(accuracy_score(np.array(ytst).flatten(), ypred))
        best_Cs.append(C[bestC_idx])
        
    
    final_best_C_idx = np.argmax(final_acc)
    final_best_C = best_Cs[final_best_C_idx]
    
    xtr, xtst, ytr, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    final_svm = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=final_best_C)
    final_svm.fit(xtr, np.array(ytr).flatten())
    y_pred = final_svm.predict(xtst)
    
    final_accuracy = round(accuracy_score(np.array(ytst).flatten(), y_pred)*100, 2)
    
    cnf_matrix = confusion_matrix(np.array(ytst).flatten(), y_pred)
    # y_true_labels = list(map(label_mapping.get, np.array(ytst).flatten()))
    # y_true_labels_cat = list(map(label_mapping_big.get, np.array(ytst).flatten()))

    # y_pred_labels = list(map(label_mapping.get, y_pred))
    # y_pred_labels_cat = list(map(label_mapping_big.get, y_pred))
    
    
    plot_confusion_matrix(cnf_matrix,classes=labels_classes, title_1=title)
    
    
    print("Final accuracy:: " + str(final_accuracy))
    return final_acc, best_Cs


def SVM_model_ovr_5nested_5cv(X, y, labels_classes, title):    
    #Xtr_val, Xtst, ytr_val, ytst = train_test_split(X_feats, y, test_size=0.2, stratify=y)
    
    n = 5
    outer_folds = KFold(n_splits=n, shuffle=True)
    
    final_acc = []
    best_Cs = []
    for index_tr_val, index_tst in outer_folds.split(X):
        Xtr_val = X[index_tr_val]
        ytr_val = y[index_tr_val]
        Xtst = X[index_tst]
        ytst = y[index_tst]
    
        acc_for_C = []
        for c in C:
            clf = svm.LinearSVC(max_iter=10000, multi_class="ovr", dual="auto", C=c)
            
            acc_each_fold_givenC = []
            inner_folds = KFold(n_splits=n, shuffle=True)
            for index_train, index_val in inner_folds.split(Xtr_val):
                Xtr = Xtr_val[index_train]
                ytr = ytr_val[index_train]
                Xval = Xtr_val[index_val]
                yval = ytr_val[index_val]
                
                clf.fit(Xtr, np.array(ytr).flatten())
                ypred = clf.predict(Xval)
                acc_each_fold_givenC.append(accuracy_score(np.array(yval).flatten(), ypred))
            acc_for_C.append(np.mean(acc_each_fold_givenC))
            
        
        bestC_idx = np.argmax(acc_for_C)
        clf = svm.LinearSVC(max_iter=10000, multi_class="ovr", dual="auto", C=C[bestC_idx])
        clf.fit(Xtr_val, np.array(ytr_val).flatten())
        ypred = clf.predict(Xtst)
        
        final_acc.append(accuracy_score(np.array(ytst).flatten(), ypred))
        best_Cs.append(C[bestC_idx])
        
        
    final_best_C_idx = np.argmax(final_acc)
    final_best_C = best_Cs[final_best_C_idx]
    
    xtr, xtst, ytr, ytst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    final_svm = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=final_best_C)
    final_svm.fit(xtr, np.array(ytr).flatten())
    y_pred = final_svm.predict(xtst)
    
    final_accuracy = round(accuracy_score(np.array(ytst).flatten(), y_pred)*100, 2)
    
    cnf_matrix = confusion_matrix(np.array(ytst).flatten(), y_pred)
    # y_true_labels = list(map(label_mapping.get, np.array(ytst).flatten()))
    # y_true_labels_cat = list(map(label_mapping_big.get, np.array(ytst).flatten()))
    
    # y_pred_labels = list(map(label_mapping.get, y_pred))
    # y_pred_labels_cat = list(map(label_mapping_big.get, y_pred))
    
    
    plot_confusion_matrix(cnf_matrix,classes=labels_classes, title_1=title)
    

    print("Final accuracy:: " + str(final_accuracy))
    return final_acc, best_Cs

def random_frames_dataset(X_files, n_frames):
    Xdata = []
    idx_list = []
    while len(idx_list) < n_frames:
        i = random.randrange(0, 431)
        if i not in idx_list:
            idx_list.append(i)
        
    idx_list.sort()
    #idx_list has the indices of the random frames we want to 
    #use as features for our SVM
    
    for i in range(len(X_files)):
        f = X_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        #Xdata.append(data1)
        Xdata.append(list(chain.from_iterable(data1[:,idx_list].T)))
    
    Xdata = np.array(Xdata)
    return Xdata, idx_list


def first_nframes_dataset(X_files, n_frames, k = 0):
    '''
    Given n_frames, select the first n_frames number of frames.
    Use the 12 MFCCs from those frames, and concatenate into a 1D array
    n_frames * 12 features for each audio file
    '''
    Xdata = []

    for i in range(len(X_files)):
        f = X_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        #Xdata.append(data1)
        Xdata.append(list(chain.from_iterable(data1[:,k:k+n_frames].T)))
    
    Xdata = np.array(Xdata)
    return Xdata

def every_nth_frame_dataset(X_files, n):
    '''
    Pick every 15th Frame out of all the 431 frames, for each audio file
    Use the 12 MFCCs from those frames, and concatenate into a 1D array
    n_frames * 12 features for each audio file
    '''
    Xdata = []

    for i in range(len(X_files)):
        f = X_files.loc[i,"filename"][:-3]+"pickle"
        path = ".\\data\\ESC-50-master\\12_MFCCs\\audio\\" + f
        data1 = pd.read_pickle(path)
        #Xdata.append(data1)
        Xdata.append(list(chain.from_iterable(data1[:,n-1::n].T)))
    
    Xdata = np.array(Xdata)
    return Xdata


def plot_confusion_matrix(cm, classes, title_1,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title="Confusion matrix: " + title_1
    plt.figure(figsize=(15,15), dpi = 300) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize = 12)
    plt.yticks(tick_marks, classes, fontsize = 12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        
C = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
