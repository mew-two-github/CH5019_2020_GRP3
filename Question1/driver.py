# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:20:23 2020

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
import utils

rep_images = np.ndarray(shape=(15,64,64))

for img_id in range (15):
    for i in range(10):
        rep_image = np.zeros(shape=(64,64), dtype = 'float32')
        matrix = np.ndarray(shape=(64,64), dtype = 'float32')
        matrix = utils.read_pgm('.\\Dataset_Question1\\'+str(img_id + 1)+'\\'+str(i+1)+'.pgm',matrix)
        reduced_matrix = utils.matrix_reduction(matrix)
        rep_image = rep_image + reduced_matrix/10 #Giving equal weightage to each image
    rep_images[img_id] = rep_image
for i in range(15):
    plt.subplot(5,3,i+1)
    plt.axis('off')
    plt.imshow(rep_images[i],cmap='gray')
correct_predictions = 0
for i in range(15):
    for j in range(10):
        matrix = np.ndarray(shape=(64,64), dtype = 'float32')
        matrix = utils.read_pgm('.\\Dataset_Question1\\'+str(i + 1)+'\\'+str(j+1)+'.pgm',matrix)
        match = 1
        minimum = 1000000007
        for img_id in range(15):
            image_distance = np.linalg.norm(matrix-rep_images[img_id],1)
            if minimum > image_distance:
                match = img_id + 1
                minimum = image_distance
        if match == i + 1:
            correct_predictions += 1
accuracy = (correct_predictions/150)*100
print("Accuracy =")
print(accuracy)