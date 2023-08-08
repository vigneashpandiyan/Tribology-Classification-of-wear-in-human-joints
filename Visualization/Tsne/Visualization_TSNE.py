# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from tSNE import *
from matplotlib import animation
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

Featurespace = Featurespace.to_numpy() 
classspace = classspace.to_numpy() 


#%%

#classspace=np.ravel(classspace)
Featurespace = pd.DataFrame(Featurespace)
num_cols = len(list(Featurespace))
rng = range(1, num_cols + 1)
Featurenames = ['Feature_' + str(i) for i in rng] 
Featurespace.columns = Featurenames
feature_cols=list(Featurespace.columns) 
Featurespace.info()
Featurespace.describe()
Featurespace.head()

#%%

df2 = pd.DataFrame(classspace) 
df2.columns = ['Categorical']
df2=df2['Categorical'].replace(0,'Class 1')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(1,'Class 2')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(2,'Class 3')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(3,'Class 4')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(4,'Class 5')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(5,'Class 6')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(6,'Class 7')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(7,'Class 8')
df2 = pd.DataFrame(df2)
classspace = pd.DataFrame(df2)

#%% Training and Testing
standard_scaler = StandardScaler()
Featurespace=standard_scaler.fit_transform(Featurespace)

from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.95, random_state=66)

#%%
graph_name= 'Feature_lower dimensional_representation'+'.png'
#graph_name= str(Material)+'_'+'.png'
ax,fig=TSNEplot(X_train,y_train,graph_name,'Feature dimensions',10)
graph_name= 'Feature_lower dimensional_representation'+'.gif'

#%%
def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))

#%%

