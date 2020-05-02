# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:42:46 2020

@author: HP
"""

import numpy as np


class LogisticRegression_inits:
    #Initializations
    def __init__(self,init_method='rand', lr=0.01, num_iter=100000, fit_intercept=False,):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.init_method=init_method
        #self.verbose=verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    #Defining the sigmoid function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Loss: Binary cross entropy
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    #Gradient Descent Implementation
    def fit(self, X, y):
        m = y.size
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # weights initialization
        if self.init_method=='zeroes':
            self.theta = np.float32([0]*X.shape[1])
        elif self.init_method=='ones':
            self.theta = np.float32([1]*X.shape[1])
        elif self.init_method=='rand':
            self.theta = np.random.normal(0,1,size=X.shape[1])
        
        
        #Gradient_Descent
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            if i%50==0:
                print("Iteration: {} \t Logloss: {:.5f}".format(i+1, np.mean(self.__loss(h,y))))
            gradient = np.dot(X.T, (h - y))
            self.theta -= self.lr * gradient/m
            
    #Function to print weights
    def weights(self):
        print(self.theta.values())
    
    #Predicting the probability for a given data between 0-1
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    #Predicts Pass/Fail based on the set threshold value and the calculated probability
    def predict(self, X, threshold=0.5):
        pred=self.predict_prob(X) #>= threshold
        for i in range(pred.shape[0]):
            if pred[i]>=threshold:
                pred[i]=1
            else:
                pred[i]=0
        
        return pred
    
    
#Confusion Matrix Definition
def confusion_matrix(y_true,y_pred):
    tot=y_true.shape[0]
    cfm={}
    tp,tn,fp,fn=0,0,0,0
    for i in range(tot):
        if y_true[i] == y_pred[i]:
            if y_true[i]==1:
                tp+=1
            else:
                tn+=1
        else:
            if y_pred[i]==1:
                fp+=1
            else:
                fn+=1
    
    cfm['True_Positive']=tp
    cfm['True_Negative']=tn
    cfm['False_Positive']=fp
    cfm['False_Negative']=fn
    return  cfm

