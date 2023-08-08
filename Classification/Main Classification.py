# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:14:40 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os     
os.environ["PATH"] += os.pathsep + 'C:\\Anaconda3\\Library\\bin\\graphviz'
import itertools
import os
from sklearn import metrics
# import pydot
import collections
# import pydotplus
import os
# import pydotplus
# from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler


from IPython.display import Image
from Classifiers.RF import *
from Classifiers.SVM import *
from Classifiers.NeuralNets import *
from Classifiers.kNN import *
from Classifiers.QDA import *
from Classifiers.NavieBayes import *
from Classifiers.Logistic_regression import *
from Classifiers.XGBoost import *

from Utils.plot_roc import *



sample_rate=2000000
windowsize= 5000
t0=0
dt=1/sample_rate
time = np.arange(0, windowsize) * dt + t0

path=r'C:\Users\srpv\Desktop\RMS Polymer Wear\Feature extraction'
path_=r'C:\Users\srpv\Desktop\RMS Polymer Wear\Classification'


folder = os.path.join( path_,'ML_Classifier')
    
try:
    os.makedirs(folder, exist_ok = True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

print(folder)
folder=folder+'/'

#%%

featurefile = 'Featurespace'+'.npy'
classfile = 'Classlabel'+'.npy'
rawfile = 'Raw'+'.npy'
featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
rawfile = os.path.join(path, rawfile)

Featurespace = np.load(featurefile).astype(np.float64)
Featurespace = pd.DataFrame(Featurespace)

classspace= np.load(classfile).astype(np.float64)
classspace = pd.DataFrame(classspace)
classspace.columns = ['Categorical']

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)

Featurespace =Featurespace.to_numpy()
classspace =classspace.to_numpy()


#%%
standard_scaler = StandardScaler()
Featurespace=standard_scaler.fit_transform(Featurespace)
#%%Recursive Feature selection 
Featurespace = RecursiveFeatureElimination(Featurespace, classspace,32)

#%%
Featurespace = pd.DataFrame(Featurespace)
num_cols = len(list(Featurespace))
rng = range(1, num_cols + 1)
Featurenames = ['Feature_' + str(i) for i in rng] 
Featurespace.columns = Featurenames
feature_cols=list(Featurespace.columns) 
Featurespace.info()
Featurespace.describe()
Featurespace.head()

#%% Training and Testing

from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.90, random_state=66)

classes=np.unique(classspace)
classes = list(classes)


#%% Model Training and Testing   
RF(X_train, X_test, y_train, y_test,100,feature_cols,Featurespace, classspace,classes,folder)
NN(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
KNN(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,15, 'distance',folder)
QDA(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
NB(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
LR(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
XGBoost(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
SVM(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder)
#%%

