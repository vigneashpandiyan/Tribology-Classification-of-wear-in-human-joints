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
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

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

#%%
windowsize= 5000
path=r'C:\Users\srpv\Desktop\RMS Polymer Wear\Contrastive learner [Circle loss]'
path_=r'C:\Users\srpv\Desktop\RMS Polymer Wear\Classification'


folder = os.path.join(path_,'Contrastive_Classifier')
    
try:
    os.makedirs(folder, exist_ok = True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

print(folder)
folder=folder+'/'

featurefile_1 = 'train_embeddings_'+'.npy'
classfile_1 = 'train_labels_'+'.npy'

featurefile_1 = os.path.join(path, featurefile_1)
classfile_1 = os.path.join(path, classfile_1)

Featurespace_1 = np.load(featurefile_1).astype(np.float64)
classspace_1= np.load(classfile_1).astype(np.float64)



featurefile_2 = 'test_embeddings_'+'.npy'
classfile_2 = 'test_labels_'+'.npy'

featurefile_2 = os.path.join(path, featurefile_2)
classfile_2 = os.path.join(path, classfile_2)

Featurespace_2 = np.load(featurefile_2).astype(np.float64)
classspace_2= np.load(classfile_2).astype(np.float64)

Featurespace=np.concatenate((Featurespace_1,Featurespace_2))
classspace=np.concatenate((classspace_1,classspace_2))


#%%


Featurespace = pd.DataFrame(Featurespace) 
classspace = pd.DataFrame(classspace) 
classspace.columns = ['Categorical']
data=pd.concat([Featurespace,classspace], axis=1)
minval = min(data.Categorical.value_counts())
minval=np.round(minval,decimals=-3)    
print("windows of the class: ",minval)
# minval=100
data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  

Featurespace=data.iloc[:,:-1]
classspace=data.iloc[:,-1]


Featurespace =Featurespace.to_numpy()
classspace =classspace.to_numpy()



#%%

df2 = pd.DataFrame(classspace) 
df2.columns = ['Categorical']
# df2=df2['Categorical'].replace(1,str(1))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(2,str(2))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(3,str(3))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(4,str(4))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(5,str(5))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(6,str(6))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(7,str(7))
# df2 = pd.DataFrame(df2)
# df2=df2['Categorical'].replace(8,str(8))
classspace = pd.DataFrame(df2) 
classspace=np.ravel(classspace)

#%%
# standard_scaler = StandardScaler()
# Featurespace=standard_scaler.fit_transform(Featurespace)
#%%Recursive Feature selection 
# Featurespace = RecursiveFeatureElimination(Featurespace, classspace,50)

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
X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.25, random_state=66)

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