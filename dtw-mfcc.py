import time
import librosa
import wp as wp

from dtw import dtw
import librosa.display
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import math


# Get feature
def get_mfcc(file_path):
    y, sr = librosa.load(file_path)  # read .wav file
    hop_length = math.floor(sr * 0.010)  # 10ms hop
    win_length = math.floor(sr * 0.025)  # 25ms frame
    # mfcc is 13 x T matrix
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 39 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    # return T x 39 (transpose of X)
    return X.T  # dtw use T x N matrix


# mfcc = get_mfcc('data_output/nhay/2_nhay_5.wav')
# mfcc1 = get_mfcc('data_output/nhay/3_nhay_7.wav')
# D, _, _, _ = dtw(mfcc, mfcc1, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
# print('distance is: ', D)

train_path = 'data_output'
files = [file for file in os.listdir(train_path)]



def predict_result(file_test):
    mfcc = get_mfcc(file_test)
    distance = np.inf
    predicted_label = None
    # cost1 = np.inf
    for file_name in files:
        for audio_name in os.listdir(train_path + '/' + file_name)[0:3]:  # Voi moi tu, lay 3 mau
            mfcc1 = get_mfcc(train_path + '/' + file_name + '/' + audio_name)
            D, _, _, _ = dtw(mfcc, mfcc1, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

            if D < distance:
                distance = D
                predicted_label = file_name
    return distance, predicted_label


dis, result = predict_result('data_output/B/86_B_4.wav')


print('distance: ', dis)
print('The word predict is: ', result)