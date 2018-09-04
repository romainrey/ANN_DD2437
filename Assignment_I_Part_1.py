# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:49:37 2018

@author: Jul
"""

#%% Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Data Generation
meanA = [-10, -10]
covA = [[10, 5], [2, 10]]
x, y = np.random.multivariate_normal(meanA, covA, 100).T
classA = np.array([x, y])
plt.plot(x, y, 'x')
plt.axis('equal')

meanB = [10, 10]
covB = [[8, 2], [2, 9]]
x, y = np.random.multivariate_normal(meanB, covB, 100).T
classB = np.array([x, y])
plt.plot(x, y, 'x')
plt.axis('equal')

plt.show()

pattern = np.append(classA, classB, axis=1)
bias = np.full((1, 200), 1)
pattern = np.append(pattern, bias, axis=0)

targetA=np.full((1, 100), 1)
targetB=np.full((1, 100), 0)
target=np.append(targetA, targetB, axis=1)


#%% Single Layer Perceptron Definition
W = np.array([[0.8], [0.5], [1]])
eta = 0.001
epochs=20
               
#%% Perceptron Learning Rule
#for i in range(epochs):
#    error = target - y    
#    delta_W = eta*error*pattern
#    W = W + delta_W
    
#%% Delta Rule
for i in range(epochs):
    error = target-np.dot(W.T, pattern)   
    delta_W = eta*np.multiply(error, pattern)
    W = W + delta_W


#%% Forward Prop / Test
Input = np.array([[4],[6],[1]]).T
Out_to_Sum = W*Input
Sum = np.sum(Out_to_Sum)
Output = (2/(1+np.exp(-Sum)))-1

               