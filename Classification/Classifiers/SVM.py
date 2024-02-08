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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics

from Utils.Helper import *
from Utils.plot_roc import *

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std


def SVM(X_train, X_test, y_train, y_test, Featurespace, classspace, classes, folder):

    print('Model to be trained is SVM')

    random_state = np.random.RandomState(0)
    #svc_model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',decision_function_shape='ovo', verbose=True,random_state=None)
    svc_model = SVC(kernel='rbf', probability=True, random_state=random_state)
    model = svc_model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    predictions = model.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'SVM'+'_without normalization w/o Opt'
    graph_name2 = 'SVM'

    graph_1 = 'SVM'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'SVM'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=200)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Greens, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')

        # disp.ax_.set_title(title)
        plt.title(title, size=12)
        graphname = folder+graphname
        plt.savefig(graphname, bbox_inches='tight', dpi=200)

    savemodel = 'SVM'+'_model'+'.sav'
    joblib.dump(model, savemodel)

    plt.savefig('SVM_Training.png', bbox_inches='tight', dpi=800)
    plt.show()
