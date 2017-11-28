# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 08:04:40 2017

@author: Rohil
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

class GradientDescent():
    
    def __init__(self, learning_rate, tolerance, max_epochs):
        
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        
        self.thetas = None
    
    def fit(self, xs, ys):
        
        num_examples, num_features = np.shape(xs)
        
        #initialize all coefficients at 1
        self.thetas = np.ones(num_features) #np.full(num_features, -1)
        
        for i in range(self.max_epochs):
            
            cost = costFunction(self.thetas, xs, ys)
            
            gradient = Gradient(self.thetas, xs, ys)
            
            #update the coefficients
            self.thetas = self.thetas - (self.learning_rate * gradient).transpose()
            
            print ('After {0} epochs, gradient descent ended with an error of {1}'.format(i, cost))
            
            #check if fit is "good enough"
            if cost < self.tolerance:
            
                return self.thetas
        
        return self.thetas


def Sigmoid(z):

    return (1/(1 + np.exp(-z)))
    

def costFunction(thetas, xs, ys):
    #get number of examples,
    m = ys.size
    
    #get hypotheses for each example
    hypothesis = Sigmoid(np.dot(xs, thetas))
    
    #vectorized implementation of cost function for logistic regression
    cost = -1 * (1/m) * (np.log(hypothesis).T.dot(ys) + np.log(1-hypothesis).T.dot(1-ys))
    
    if np.isnan(cost):
        return  (np.inf)
    
    return (cost)


def Gradient(thetas, xs, ys):
    
    m = ys.size
    
    hypothesis = Sigmoid(np.dot(xs, thetas))
    
    gradient = (1/m) * xs.T.dot(hypothesis - ys)
    
    return (gradient)


def getData():

    col_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    
    dataset = pd.read_csv('/Users/Rohil/Documents/breast-cancer-wisconsin.txt', header = None, names = col_names)
    dataset = dataset.drop('Sample code number', axis = 1).replace('?', np.nan).dropna()
    dataset['Class'].replace(2, 0, inplace = True)
    dataset['Class'].replace(4, 1, inplace = True)
    dataset
    
    X_train = dataset[dataset.columns[:-1]]     
    y_train = dataset[dataset.columns[-1]]
    X_train = preprocessing.scale(X_train.values)   
    
    return X_train, y_train


xs, ys = getData()

gd = GradientDescent(learning_rate = .001, tolerance = 0.087, max_epochs = 200000)

print (gd.thetas)


    
