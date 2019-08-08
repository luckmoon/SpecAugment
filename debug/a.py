#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-08 14:12
# @Author  : wulala
# @Project : SpecAugment
# @File    : a.py
# @Software: PyCharm

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def plot_spec(_type=None):
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, y_axis=_type)
    plt.colorbar(format='%+2.0f dB')
    if _type == 'linear':
        plt.title('Linear-frequency power spectrogram')
    if _type == 'log':
        plt.title('Log-frequency power spectrogram')
    plt.show()


if __name__ == '__main__':
    wav_file = '../data/61-70968-0002.wav'
    y, sr = librosa.load(wav_file, sr=16000)
    plot_spec('linear')
    plot_spec('log')
    # librosa.feature.melspectrogram(S=D)
