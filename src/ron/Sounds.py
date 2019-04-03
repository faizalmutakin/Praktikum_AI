# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:07:37 2019

@author: IMRON
"""

# In[1]: Import library
import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt

# In[2]: Membuat fungsi mfcc

def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()
    
# In[3]:  Load data dari freesound
display_mfcc('266093__stereo-surgeon__kick-loop-5.wav')

# In[4]: Load data dari GTZAN dengan genre classical
display_mfcc('genres/classical/classical.00067.au')