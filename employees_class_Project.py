#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:55:24 2019

@author: mohamed
"""

import os
os.getcwd()
os.chdir('/Volumes/Work/Data Science/Ml/Project/Classification')
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("Dataset_use.csv")

df.info()
df.head(10)
df.describe()
df.isnull().sum()


# difined X, y
X = df.iloc[:,:]
y = df.loc[:,['Attrition']]

# drop Attrition cloumn from x
df.drop(['Attrition'],inplace=True,axis = 1)

#get_dummies ----
X = pd.get_dummies(X,drop_first=True)
y = pd.get_dummies(y,drop_first=True)


# Feature Scaling of training and test set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)


# Feature Scaling of training and test set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# Cross Validation (K-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True,random_state=0)

# creating modling ==================================
#import classifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

clf_logReg = LogisticRegression()
clf_KNN = KNeighborsClassifier(n_neighbors = 10)
model_SVM =SVC(kernel= 'linear')
clf_RForst = RandomForestClassifier(n_estimators= 3, criterion = 'entropy', max_depth = 30 )
clf_adboost = AdaBoostClassifier(random_state=1)


clf_logReg.fit(X_train,y_train)
clf_KNN.fit(X_train, y_train)
model_SVM.fit(X_train, y_train)
clf_RForst.fit (X_train, y_train)
clf_adboost.fit(X_train, y_train)

# Predicting the test set results
y_pred_logReg = clf_logReg.predict(X_test)
y_pred_KNN = clf_KNN.predict(X_test)
y_pred_SVM = model_SVM.predict(X_test)
y_pred_RForst = clf_RForst.predict(X_test)
y_pred_adboost = clf_adboost.predict(X_test)


score_logReg = cross_val_score(clf_logReg, X_train,y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score_logReg)*100, 2) #--> 87.2
score_clf_KNN = cross_val_score(clf_KNN, X_train,y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score_clf_KNN)*100, 2) # --> 84.57
score_model_SVM = cross_val_score(model_SVM, X_train,y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score_model_SVM)*100, 2)  # --> 87.75
score_clf_RForst = cross_val_score(clf_RForst, X_train,y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score_clf_RForst)*100, 2) #--> 81.85
score_clf_adboost = cross_val_score(clf_adboost, X_train,y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
round(np.mean(score_clf_adboost)*100, 2) # --> 87.21


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test,y_pred_logReg) # --> 0.8858
accuracy_score(y_test,y_pred_KNN )   # --> 0.8532
accuracy_score(y_test,y_pred_SVM)    # --> 0.8858
accuracy_score(y_test,y_pred_RForst) # --> 0.84239
accuracy_score(y_test,y_pred_adboost) # --> 0.8614


confusion_matrix(y_test,y_pred_logReg)    # [302,   8], [ 34,  24]
confusion_matrix(y_test,y_pred_KNN )      # [310,   0], [ 54,   4]
confusion_matrix(y_test,y_pred_SVM)       # [301,   9], [ 33,  25]
confusion_matrix(y_test,y_pred_RForst)    # [290,  20],[ 38,  20]
confusion_matrix(y_test,y_pred_adboost)   # [299,  11],[ 40,  18]


classification_report(y_test,y_pred_logReg)
classification_report(y_test,y_pred_KNN )
classification_report(y_test,y_pred_SVM)
classification_report(y_test,y_pred_RForst)
classification_report(y_test,y_pred_adboost)

