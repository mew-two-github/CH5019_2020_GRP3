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

#Choose 700 rows randomly as training data and rest as test data
train_data = data.sample(700)
#print(train_data.describe())
print("Train data size = {} ".format(train_data.shape[0]))
x_train = train_data.drop(columns=['Test'],inplace=False)
y_train = train_data['Test']

test_data = data.drop(train_data.index)
x_test = test_data.drop(columns=['Test'],inplace=False)
y_test = test_data['Test']

print("Test data size = {} ".format(test_data.shape[0]))

#converting to numpy arrays
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


#Predicting with a  Gaussian initialisation without regularisation
model_rand=utils.LogisticRegression_inits('rand',lr = 0.0015, num_iter=2000, reg_lambda=0)
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

#Predicting with a  Gaussian initialisation with regularisation

model_rand2=utils.LogisticRegression_inits('rand',lr = 0.1, num_iter=2000,reg_lambda=0.25)
model_rand2.fit(x_train,y_train)
y_pred2=model_rand2.predict(x_test,threshold = 0.4)
cm=utils.confusion_matrix(y_test,y_pred2)
acc=(cm['True_Positive']+cm['True_Negative'])/sum(cm.values())
pre=(cm['True_Positive'])/(cm['True_Positive']+cm['False_Positive'])
rec=(cm['True_Positive'])/(cm['True_Positive']+cm['False_Negative'])
fsc= 2*pre*rec/(pre+rec)
#Printing confusion matrix
print('accuracy: ' ,acc,'\nprecision: ',pre,'\trecall: ',rec,'\nf_score: ',fsc)
print("Confusion Matrix")
print(cm)

print("Difference in their weights",(model_rand.weights()-model_rand2.weights()))

    


