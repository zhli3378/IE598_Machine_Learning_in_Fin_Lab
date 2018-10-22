#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:57:05 2018

@author: guomengming
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('https://github.com/zhli3378/IE598_Machine_Learning_in_Fin_Lab/raw/master/IE598_F18_Final_project/MLF_GP1_CreditScore.csv')
X, y = df.iloc[:, 0:26].values, df.iloc[:, 26].values
seed=1
#Split	dataset into 90% train and 10% test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)

feat_labels = df.columns[:26]

params=[1,5,10,100,500,1000,3000,50000]
accurary_score = []


for n in params:
 
    forest = RandomForestClassifier(criterion = 'gini', n_estimators = n, random_state = 42, n_jobs = 2)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    accurary_score.append(metrics.accuracy_score(y_train, y_train_pred))
   
   
plt.plot(params, accurary_score)
plt.ylabel("Accurarcy")
plt.xlabel("N_list")
plt.title("Random Forest")
plt.show()













for i in [1, 10, 50, 100, 500, 1000]:
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Train score", rf.score(X_train, y_train), "n_estimators = ", i)
    print("Test score", rf.score(X_test, y_test))

cross = cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(cross))

feat_labels = df.columns[:26]

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()