# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:27:29 2017

@author: Rohil
"""
#make imports
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent():
    
    def __init__(self, learning_rate, tolerance, max_epochs):
        
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        
        #thetas is array coefficeints for each term
        #the yint is the last element
        self.thetas = None
        
    def fit(self, xs, ys):
        
        num_examples, num_features = np.shape(xs)
        
        #intializes the features at one
        self.thetas = np.ones(num_features)
               
        for i in range(self.max_epochs):
            
            #difference between predicted and actual values
            error = np.dot(xs, self.thetas) - ys
            
            #caclulating the sum of squared cost
            cost = np.sum(error**2) / (2*num_examples)
            
            #calculate the average gradiet for each sample
            #gets sum of costs for each prediction for all of the training examples divided
            #then divides this sum by number of training examples
            gradient = np.dot(xs.transpose(), error) / num_examples
            
            #update the coefficients
            self.thetas = self.thetas - (self.learning_rate * gradient)
            
            print ('After {0} epochs, gradient descent ended with an error of {1}'.format(i, cost))
            
            #check if fit is "good enough"
            if cost < self.tolerance:
                return self.thetas
                
            
        return self.thetas
    
    def makePrediction(self, x):
        
        return np.dot(x, self.thetas)
    

#load some example data
data = np.loadtxt('/Users/Rohil/Documents/LambertML/testScorePrediction/act_hope.txt', delimiter='\t', skiprows = 1)
col_names = ['hope_eligible', 'act_score']

data_map = dict(zip(col_names, data.transpose()))

#create martix of features
features = np.column_stack((np.ones(len(data_map['hope_eligible'])), data_map['hope_eligible']))

gd = GradientDescent(learning_rate = .0001, tolerance=0.1929005, max_epochs = 50000)
thetas = gd.fit(features, data_map['act_score'])
gradient, intercept = thetas
#predict values according to our model 
ys = gd.makePrediction(features)

plt.scatter(data_map['hope_eligible'], data_map['act_score'])
plt.plot(data_map['hope_eligible'], ys)
plt.show()
print (gd.thetas)
        
        
            
        