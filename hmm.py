import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
import os
import warnings
import scipy.stats as sp
from python_speech_features import mfcc
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import operator
import glob
import itertools

num_of_states = 9  # number of hidden states
num_of_mixtures = 2  # number of mixtures for each hidden state
covariance_type = 'diag'  # covariance type
n_iter = 1000  # number of iterations
dimension = 2


# This function returns the initial prior probabilities vector for model
def getPriorProabiblity():
    priorProabiblity = np.zeros(num_of_states)
    priorProabiblity[0: dimension] = 1 / float(dimension)
    return priorProabiblity


# This function returns the initial transition matrix according to left-right model
def getTransitionMatrix():
    transitionMatrix = (1 / float(dimension + 1)) * np.eye(num_of_states)

    for i in range(num_of_states - dimension):
        for j in range(dimension):
            transitionMatrix[i, i + j + 1] = 1. / (dimension + 1)
    j = 0;
    for i in range(num_of_states - dimension, num_of_states):
        for j in range(num_of_states - i - j):
            transitionMatrix[i, i + j] = 1. / (num_of_states - i)

    return transitionMatrix


# Construct GMM + HMM based on passed parameters
def constructGMMHMM():
    return hmm.GMMHMM(n_components=num_of_states, n_mix=num_of_mixtures,
                      transmat_prior=getTransitionMatrix(), startprob_prior=getPriorProabiblity(),
                      covariance_type=covariance_type, n_iter=n_iter)


# Construct Gaussian HMM, i.e. GMM + HMM with 1 mixture model
def constructGaussianHMM():
    return hmm.GaussianHMM(n_components=num_of_states,
                           covariance_type=covariance_type, n_iter=n_iter)


labels = []
words = []
features = []
hmmModels = []
folder_name = 'data_output'


# Get MFCC features based on path of audio
def get_mfcc(audio_path):
    sample_rate, wave = wavfile.read(audio_path)
    return mfcc(wave, nfft=2048, samplerate=sample_rate, numcep=13)


# Loop over all folders inside directory of dataset (those will be our labels)
for file_name in os.listdir(folder_name):
    features = np.array([])
    data_length = len(os.listdir(folder_name + '/' + file_name))
    # Taking the index to split data into two parts, 80% and 20% (training and testing set)
    training_index = int(data_length * 0.8)
    # Loop over audio files in training set, construct their MFCC features and append them
    for audio_name in os.listdir(folder_name + '/' + file_name)[0:training_index]:
        if len(features) == 0:
            features = get_mfcc(folder_name + '/' + file_name + '/' + audio_name)
        else:
            features = np.append(features, get_mfcc(folder_name + '/' + file_name + '/' + audio_name), axis=0)
        labels.append(file_name)
        if file_name not in words:
            words.append(file_name)
    # Construct hmm model for each label
    hmmModel = constructGaussianHMM()
    np.seterr(all='ignore')
    # Train hmm model on MFCC features corresponding to the label
    hmmModel.fit(features)
    print('Finished training for: ', file_name)
    hmmModels.append((hmmModel, file_name))

# Visualization of MFCC features for the first audio in each folder
# figure = plt.figure()
# for idx, word in enumerate(words):
#     mfcc_features = get_mfcc(folder_name + '/' + word + '/' + os.listdir(folder_name + '/' + word)[0])
#     plt.matshow((mfcc_features.T)[:, :50])
#     plt.text(50, -5, word, horizontalalignment='left', fontsize=20)

# Calculate score (log likelihood) for each observation sequence from testing set for each model and take label
# corresponding to max score
count = 0
predicted_labels = []
real_labels = []
for file_name in os.listdir(folder_name):
    data_length = len(os.listdir(folder_name + '/' + file_name))
    # Take testing set 20% of the whole samples under each label
    testing_index = int(data_length * 0.8)
    for audio_name in os.listdir(folder_name + '/' + file_name)[testing_index:data_length]:
        features = get_mfcc(folder_name + '/' + file_name + '/' + audio_name)
        probs = {}
        for item in hmmModels:
            hmm_model, label = item
            # Calculate score of each observation sequence (log likelihood)
            probs[label] = hmm_model.score(features)
        # Get key having the highest score as predicted label
        result = max(probs.items(), key=operator.itemgetter(1))[0]
        predicted_labels.append(result)
        real_labels.append(file_name)
        if (result == file_name):
            count = count + 1
print('Accuracy of the model on testing set is: ', (count * 100 / len(real_labels)))

# Getting confusion matrix of testing set and plotting it
conf_matrix = confusion_matrix(real_labels, predicted_labels)
np.set_printoptions(precision=2)
plt.figure()
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.xticks(range(len(words)), words, rotation=45)
plt.yticks(range(len(words)), words)
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if i == j else "black")
plt.tight_layout()
plt.ylabel('Correct label')
plt.xlabel('Predicted label')
plt.show()


def predict_label(file_name):
    features = get_mfcc(file_name)
    probs = {}
    for item in hmmModels:
        hmm_model, label = item
        # Calculate score of each observation sequence (log likelihood)
        probs[label] = hmm_model.score(features)
    # Get key having the highest score as predicted label
    result = max(probs.items(), key=operator.itemgetter(1))[0]
    return result


result = predict_label('len.wav')
print('predict word is: ', result)
