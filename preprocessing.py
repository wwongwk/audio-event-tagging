import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from AudioLibrary.AudioFeatures import AudioFeatures
from AudioLibrary.AudioSignal import AudioSignal

# Audio features extraction function
def global_feature_statistics(
        y,
        win_size=0.025,
        win_step=0.01,
        nb_mfcc=12,
        mel_filter=40,
        stats=[
            'mean',
            'std',
            'med',
            'kurt',
            'skew',
            'q1',
            'q99',
            'min',
            'max',
            'range'
        ],
        features_list=[
            'zcr',
            'energy',
            'energy_entropy',
            'spectral_centroid',
            'spectral_spread',
            'spectral_entropy',
            'spectral_flux',
            'sprectral_rolloff',
            'mfcc'
        ]):
    # Extract features
    audio_features = AudioFeatures(y, win_size, win_step)
    features, features_names = audio_features.global_feature_extraction(
        stats=stats,
        features_list=features_list,
        nb_mfcc=nb_mfcc,
        nb_filter=mel_filter)
    return features