# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the publication of the following work

"Classification of Progressive Wear on a Multi-Directional Pin-on-Disc 
Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo 
Using Acoustic Emission and Machine Learning"

@any reuse of this code should be authorized by the first owner, code author
"""

# %%
# Libraries to import
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import os
from Utils import *
# %%

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)


# %%
# Inputs to be added

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sample_rate = 2000000
windowsize = 5000
t0 = 0
dt = 1/sample_rate
time = np.arange(0, windowsize) * dt + t0
path = r'C:\Users\srpv\Desktop\RMS Polymer Wear\Feature extraction'

# %%

featurefile = 'Featurespace'+'.npy'
classfile = 'Classlabel'+'.npy'
rawfile = 'Raw'+'.npy'

featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
rawfile = os.path.join(path, rawfile)

Featurespace = np.load(featurefile).astype(np.float64)
classspace = np.load(classfile).astype(np.float64)

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)


Featurespace = pd.DataFrame(Featurespace)
classspace = pd.DataFrame(classspace)

classspace.columns = ['Categorical']
data = pd.concat([rawspace, classspace], axis=1)
print("Respective windows per category", data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())
print("windows of the class: ", minval)
data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique()])
print("Balanced dataset: ", data.Categorical.value_counts())
rawspace = data.iloc[:, :-1]
classspace = data.iloc[:, -1]


# %%

STFT("STFT", sample_rate, rawspace, classspace, time, total_path)
