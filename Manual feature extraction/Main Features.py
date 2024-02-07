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

import os
import numpy as np
from Feature_extraction import *
print(np.__version__)

# %%
# Inputs to be added

windowsize = 5000
sample_rate = 2000000  # sampling rate of the sensor
t0 = 0
dt = 1/sample_rate
time = np.arange(0, windowsize) * dt + t0

band_size = 6  # No of windows on which cumulative energy has to be computed [band_size-1]
peaks_to_count = 7  # Frequency peaks
count = 0

path = r'C:\Users\srpv\Desktop\RMS Polymer Wear\Data'  # datafolder path ---->
classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5',
           'Class 6', 'Class 7', 'Class 8']  # Class names to be included

# %%

featurelist, classlist, rawlist = Timeseries_feature(
    classes, sample_rate, band_size, peaks_to_count)

# %%
# Saving the manual extracted feature as a array
Featurespace = np.asarray(featurelist)
Featurespace = Featurespace.astype(np.float64)

rawspace = np.asarray(rawlist)
rawspace = rawspace.astype(np.float64)

classspace = np.asarray(classlist)

featurefile = 'Featurespace'+'.npy'
classfile = 'Classlabel'+'.npy'
rawfile = 'Raw'+'.npy'

np.save(featurefile, Featurespace, allow_pickle=True)
np.save(classfile, classspace, allow_pickle=True)
np.save(rawfile, rawspace, allow_pickle=True)

# %%
