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
import os
import seaborn as sns
from matplotlib import cm

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
# total_path=os.path.dirname(os.path.dirname(file))
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


# %%


def boxcomparisonplots(Featurespace, classspace, total_path):

    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)
    print("Respective windows per category", data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())
    print("windows of the class: ", minval)
    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])
    print("Balanced dataset: ", data.Categorical.value_counts())
    Featurespace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]

    Featurespace = Featurespace.to_numpy()
    Featurespace = (Featurespace[:, 34:39]).astype(np.float64)
    classspace = classspace.to_numpy()

    classes = np.unique(classspace)
    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    values, counts = np.unique(classspace, return_counts=True)
    print(values, counts)

    c = len(Featurespace)
    df1 = pd.DataFrame(Featurespace)
    df1 = np.ravel(df1, order='F')
    df1 = pd.DataFrame(df1)

    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']

    filename = '0-200 kHZ'
    numbers = np.random.randn(c)
    df3 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df3 = df3.drop(['numbers'], axis=1)

    filename = '200-400 kHZ'
    numbers = np.random.randn(c)
    df4 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df4 = df4.drop(['numbers'], axis=1)

    filename = '400-600 kHZ'
    numbers = np.random.randn(c)
    df5 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df5 = df5.drop(['numbers'], axis=1)

    filename = '600-800 kHZ'
    numbers = np.random.randn(c)
    df6 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df6 = df6.drop(['numbers'], axis=1)

    filename = '800-1000 kHZ'
    numbers = np.random.randn(c)
    df7 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df7 = df7.drop(['numbers'], axis=1)

    Energyband = np.concatenate((df3, df4, df5, df6, df7), axis=0)
    Modes = np.concatenate((df2, df2, df2, df2, df2), axis=0)

    Energybands = np.concatenate((df1, Energyband, Modes), axis=1)
    Energybands = pd.DataFrame(Energybands)
    Energybands.columns = ['Frequency distribution', 'Frequency levels', 'Categorical']

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    sns.set(font_scale=3.5)
    sns.set_style("whitegrid", {'axes.grid': False})
    ax = sns.catplot(y="Frequency levels", x="Frequency distribution", hue="Categorical", kind="bar", data=Energybands, height=12,
                     aspect=1.8, palette=color)
    ax.set_xticklabels(rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=50)
    ax.tick_params(axis='both', which='minor', labelsize=50)
    ax._legend.remove()
    plt.legend(loc='lower right', frameon=False, fontsize=40)
    # plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.title('Frequency distribution', fontsize=50)
    plotname = "PSD_Frequency distribution"+'.png'
    plt.savefig(os.path.join(total_path, plotname), dpi=800, bbox_inches='tight')
    plt.show()


# %%
boxcomparisonplots(Featurespace, classspace, total_path)
