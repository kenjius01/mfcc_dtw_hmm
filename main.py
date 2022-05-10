import librosa
import matplotlib.pyplot as plt
import re
import os

import numpy as np
import math
from scipy.io import wavfile

from pydub import AudioSegment

# import pywt

WIN_LENGTH = 0.025
WIN_STEP = 0.01
NUM_CEP = 13


# Noi file audio với label

def match_signal_file_with_label_track_file():
    signal_and_label_track_pairs = list()
    for _file in os.listdir("data_speech/"):
        if _file.endswith(".wav"):
            signal_and_label_track_pairs.append({"signal_path": _file, "label": os.path.splitext(_file)[0] + ".txt"})
    return signal_and_label_track_pairs


def get_signal_from_file(signal_file_name):
    signal, fs = librosa.load(signal_file_name)
    return signal, fs


def show_signal(signal, label):
    plt.plot(signal)
    plt.xlabel(label)
    plt.show()


# Lay tung khau lenh don le trong file label

def parse_track_file(label_track_file_name):
    label_track_file = open(label_track_file_name)
    label_track_file_line = label_track_file.readlines()
    label_track_list = list();
    for line in label_track_file_line:
        label_item = re.split(r'\s+', line);
        label_track_list.append({"timestamp_start": float(label_item[0].replace(",", ".")) * 1000,
                                 "timestamp_end": float(label_item[1].replace(",", ".")) * 1000,
                                 "label": label_item[2]})
    return label_track_list  # ["timestamp_start":0, "timestamp_end": 1234, "label": "len", "label_id"=}]


# Ham tach file thanh cac file khau lenh don le

def audio_segment(signal_file_name, label_track_file):
    dataset = list()
    file_number = (os.path.splitext(signal_file_name))[0][12:]
    label_track_list = parse_track_file(label_track_file)
    i = 0
    for label_track in label_track_list:
        try:
            newAudio = AudioSegment.from_wav(signal_file_name);
            newAudio = newAudio[label_track["timestamp_start"]:label_track["timestamp_end"]]
            signal = "data_output/" + label_track["label"] + "/" + file_number + '_' + label_track["label"] + "_" + str(
                i) + ".wav"
            newAudio.export(signal, format="wav")
            dataset.append({"signal_path": signal, "label": label_track["label"]})
            i += 1
        except FileNotFoundError:
            print("Not able to segment %s file. File not found error" % signal)
    return dataset


data = list()
for pair in match_signal_file_with_label_track_file():
    data += audio_segment("data_speech/" + pair['signal_path'], "data_speech/" + pair['label'])

print("It was created %s new .wav files, where each signal represents one word." % len(data))

for d in data:
    signal, fs = get_signal_from_file(d['signal_path'])
    d["fs"] = fs
    d["signal"] = signal


# Ham trich xuat dac trung mfccs

def get_MFCC(signal, sample_rate):
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sample_rate)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta_delta))
    return mfccs_features


# Test xem da du 39 feature chua

mfcc_features = get_MFCC(data[0]['signal'], data[0]['fs'])
print('Shape of feature: ', mfcc_features.shape)

# lay dac trung cua tat ca cac khau lenh don le
files = [file for file in os.listdir('data_output')]
# print(files[1])
dataset = list()
for d in data:
    dataset_features = get_MFCC(d["signal"], d["fs"])
    dataset.append([dataset_features, d['label']])

print('shape: ', dataset[3][0].shape)
print('label: ', dataset[3][1])

print("Example of MFFC result image")
mffc_example = dataset[3][0]
plt.xlabel("frames")
plt.ylabel("frequency")
plt.title(dataset[3][1])
plt.imshow(mffc_example.T)
plt.show()
