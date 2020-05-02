# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:42:46 2020

@author: HP
"""

import pandas as pd
import numpy as np

import utils

data=pd.read_excel('Dataset_Question2.xlsx')
row, cols = data.shape

#normalising the data
for column in data.columns:
    if column == 'Test':
        break
    mean = data[column].mean()
    variance = data[column].var()
    data[column] = (data[column]-mean)/np.sqrt(variance)

#Converting Pass/Fail as 0/1
for i in range(row):
    if data.iloc[i,cols-1]=='Pass':
        data.iloc[i,cols-1] = 1
    else:
        data.iloc[i,cols-1] = 0
data.head()

#since the data is distributed randomly, we just split it at 70%
train_data = data.copy().head(700)
print("Train data size = {} ".format(train_data.shape[0]))
x_train = train_data.drop(columns=['Test'],inplace=False)
y_train = train_data['Test']

test_data = data.copy().tail(300)
x_test = test_data.drop(columns=['Test'],inplace=False)
y_test = test_data['Test']
print("Test data size = {} ".format(test_data.shape[0]))
y_test = y_test.to_numpy(dtype='int64')

#Predicting with a  Gaussian initialisation
model_rand=utils.LogisticRegression_inits('rand',lr = 0.1, num_iter=2000)
model_rand.fit(x_train,y_train)
y_pred=model_rand.predict(x_test,threshold = 0.4)
cm=utils.confusion_matrix(y_test,y_pred)
acc=(cm['True_Positive']+cm['True_Negative'])/sum(cm.values())
pre=(cm['True_Positive'])/(cm['True_Positive']+cm['False_Positive'])
rec=(cm['True_Positive'])/(cm['True_Positive']+cm['False_Negative'])
fsc= 2*pre*rec/(pre+rec)
#Printing confusion matrix
print('accuracy: ' ,acc,'\nprecision: ',pre,'\trecall: ',rec,'\nf_score: ',fsc)
print("Confusion Matrix")
print(cm)

