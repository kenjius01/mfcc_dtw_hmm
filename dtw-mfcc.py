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
import matplotlib.pyplot as plt
# Get feature
from sklearn import metrics


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


predict = []
real_label = []


def cross_validation():
    for file_name in files:
        for audio_name in os.listdir(train_path + '/' + file_name)[4:20]:
            _, predict_label = predict_result(train_path + '/' + file_name + '/' + audio_name)
            real_label.append(file_name)
            predict.append(predict_label)


cross_validation()
print("Classification report: \n\n%s\n"
      % (metrics.classification_report(real_label, predict)))

mat = metrics.confusion_matrix(real_label, predict)
label_names = list(set(real_label))
plt.figure()
plt.imshow(mat, interpolation='nearest', cmap='Blues')
plt.title('normalized confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(label_names))
plt.xticks(tick_marks, label_names, rotation=90)
plt.yticks(tick_marks, label_names)
plt.tight_layout()
plt.show()

accuracy = metrics.accuracy_score(real_label, predict)
print('Accuracy classification score: {0:.2f}%'.format(100 * accuracy))
precision = metrics.precision_score(real_label, predict, average='weighted')
print('Precision classification score: {0:.2f}%'.format(100 * precision))
dis, result = predict_result('test_data/1.wav')

print('distance: ', dis)
print('The word predict is: ', result)
