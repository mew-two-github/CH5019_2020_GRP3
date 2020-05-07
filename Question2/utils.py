# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:42:46 2020

@author: HP
"""

import numpy as np


class LogisticRegression_inits:
    def __init__(self,init_method='rand', lr=0.01, num_iter=100000, fit_intercept=False,reg_lambda=0.1):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.init_method=init_method
        self.reg_lambda=reg_lambda;
        self.m=10
        #self.verbose=verbose
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    #Loss: Binary cross entropy
    def __loss(self, h, y):
        non_reg = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        t=np.square(self.theta)
        reg=(self.reg_lambda/(2*self.m))*np.sum(t)
        return non_reg+reg
    
    def fit(self, X, y):
        
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # weights initialization
        if self.init_method=='zeroes':
            self.theta = np.float32([0]*X.shape[1])
        elif self.init_method=='ones':
            self.theta = np.float32([1]*X.shape[1])
        elif self.init_method=='rand':
            self.theta = np.random.normal(0,1,size=X.shape[1])
        
        
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            
            if (i+1)%(self.num_iter/10)==0:
                print("Iteration: {} \t Logloss: {:.5f}".format(i+1, np.mean(self.__loss(h,y))))
            gradient = np.dot(X.T, (h - y)) + 2*self.reg_lambda*self.theta
            self.theta -= self.lr * gradient/self.m
            for j in range(self.theta.size):
                if(j==0):
                    break
                self.theta[j]-=(self.theta[j]*self.reg_lambda*self.lr)/self.m
            
            
    def weights(self):
        return(self.theta)
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
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

