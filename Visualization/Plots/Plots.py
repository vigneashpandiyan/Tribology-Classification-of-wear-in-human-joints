# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:13:12 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from Utils import *
#%%

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path=os.path.dirname(file) 
print(total_path)

#%%
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})
sample_rate=2000000
windowsize= 5000
t0=0
dt=1/sample_rate
time = np.arange(0, windowsize) * dt + t0
path=r'C:\Users\srpv\Desktop\RMS Polymer Wear\Feature extraction'

#%%

featurefile = 'Featurespace'+'.npy'
classfile = 'Classlabel'+'.npy'
rawfile = 'Raw'+'.npy'

featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
rawfile = os.path.join(path, rawfile)

Featurespace = np.load(featurefile).astype(np.float64)
classspace= np.load(classfile).astype(np.float64)

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)


Featurespace = pd.DataFrame(Featurespace)
classspace = pd.DataFrame(classspace)

classspace.columns = ['Categorical']
data = pd.concat([Featurespace, classspace], axis=1)
print("Respective windows per category",data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())  
print("windows of the class: ",minval)
data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
print("Balanced dataset: ",data.Categorical.value_counts())
Featurespace=data.iloc[:,:-1]
classspace=data.iloc[:,-1]


def plots(i,Featurespace,classspace,total_path,feature):
    # Featurespace = Featurespace.transpose()
    data=(Featurespace[i])
    data=data.astype(np.float64)
    #data= abs(data)
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
    df2 = pd.DataFrame(classspace)
    

    df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
    data = pd.concat([df1, df2], axis=1)
    
    
    kdeplot(data,feature,total_path)
    hist(data,feature,total_path)
    violinplot(data,feature,total_path)
    boxplot(data,feature,total_path)
    barplot(data,feature,total_path)
    ridgeplot(data,feature,total_path)
    
    kdeplotsplit(data,feature,total_path)
    Histplotsplit(data,feature,total_path)#
    return data

#%%
      #4#7  
data=plots(3,Featurespace,classspace,total_path,"RMS") #RMS-3 , #Skewness-6,  #Kurtosis-7



   