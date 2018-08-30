# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets

iris = datasets.load_iris()

X_iris, y_iris = iris.data, iris.target

print (X_iris.shape, y_iris.shape)
print (X_iris[0], y_iris[0])