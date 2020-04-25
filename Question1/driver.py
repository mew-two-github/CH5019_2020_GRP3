# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:20:23 2020

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
import utils


rep_images = np.zeros(shape=(15,64,64),dtype='float32')
no_comps = np.zeros(shape=(15,1),dtype='int32')
unrolled_images = np.zeros(shape = (4096,10))
for img_id in range (15):
    for i in range(10):
        rep_image = np.zeros(shape=(64,64), dtype = 'float32')
        matrix = np.ndarray(shape=(64,64), dtype = 'float32')
        matrix = utils.read_pgm('.\\Dataset_Question1\\'+str(img_id + 1)+'\\'+str(i+1)+'.pgm',matrix)
        unrolled_images[:,i] = utils.unroll(matrix)
    no_comps[img_id],reduced_matrix = utils.matrix_reduction(unrolled_images)
    weightages = np.asarray([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    rep_images[img_id] = np.matmul(reduced_matrix,weightages).reshape(64,64)

#Printing all the representative images
    
plt.figure(figsize = (20,20))
for i in range(15):
    plt.subplot(5,3,i+1)
    plt.axis('off')
    plt.title(i+1)
    plt.imshow(rep_images[i],cmap='gray')
plt.show()

#Predictions + Testing
img_wise_correct = 0
accuracies = []
predictions = []
correct_predictions = 0
for i in range(15):
    img_wise_correct = 0
    for j in range(10):
        matrix = np.ndarray(shape=(64,64), dtype = 'float32')
        matrix = utils.read_pgm('.\\Dataset_Question1\\'+str(i + 1)+'\\'+str(j+1)+'.pgm',matrix)
        match = 1
        #image_distance = np.linalg.norm(matrix-rep_images[0],1)
        minimum = np.linalg.norm(matrix-rep_images[0],1)
        #minimum = 10**9
        for img_id in range(15):
            image_distance = np.linalg.norm(matrix-rep_images[img_id],1)
            if minimum > image_distance:
                match = img_id + 1
                minimum = image_distance
        predictions.append(match)
        if match == i + 1:
            img_wise_correct += 1
            correct_predictions += 1
        #print(match)
    accuracies.append(img_wise_correct)
accuracy = (correct_predictions/150)*100

print("Accuracy =",accuracy)