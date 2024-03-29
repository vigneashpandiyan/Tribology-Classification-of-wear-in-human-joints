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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os

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
Featurespace = pd.DataFrame(Featurespace)

classspace = np.load(classfile).astype(np.float64)
classspace = pd.DataFrame(classspace)
classspace.columns = ['Categorical']

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)


data = pd.concat([rawspace, classspace], axis=1)
print("Respective windows per category", data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())

print("windows of the class: ", minval)
df = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique()])
print("Balanced dataset: ", data.Categorical.value_counts())

minval = 1
df_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique()])


# %%

def plot_time_series(data, class_name, ax, colour, i, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 3 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, color=colour, linewidth=3)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.650
    )
    ax.set_title(class_name)
    # ax.set_ylim([-0.025, 0.025])
    ax.set_ylabel('Amplitude (V)')
    # ax.set_xlabel('Window size (μs)')


# %%

graphname = 'Moving average visualisation'+'.png'
classes = df.Categorical.unique()
classes = np.sort(classes)
color = cm.rainbow(np.linspace(0, 1, len(classes)))

fig, axs = plt.subplots(
    nrows=int(len(classes)/2),
    ncols=int(1*2),
    sharey=False,
    sharex=True,
    figsize=(10, 8),
    dpi=800
)

for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df[df.Categorical == cls].drop(labels='Categorical', axis=1).mean(axis=0).to_numpy()
    plot_time_series(data, classes[i], ax, color[i], i)
plt.xlabel('Window size (μs)')
fig.tight_layout()
plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.clf()

# %%

graphname = 'Window visualisation'+'.png'
fig, axs = plt.subplots(
    nrows=int(len(classes)/2),
    ncols=int(1*2),
    sharey=False,
    sharex=True,
    figsize=(10, 8),
    dpi=800
)

for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df_1[df_1.Categorical == cls].drop(labels='Categorical', axis=1).mean(axis=0).to_numpy()
    ax.plot(data, color=color[i], linewidth=3)
    ax.set_ylabel('Amplitude (V)')

plt.xlabel('Window size (μs)')
fig.tight_layout()
plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.clf()
