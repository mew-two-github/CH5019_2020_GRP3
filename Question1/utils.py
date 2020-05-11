# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:17:26 2020

@author: HP
"""


#Realized that the first line contained P5 followed by the dimensions. Decided to store the dimensions in a List
import numpy as np


#function to convert the pgm file into an image matrix
def read_pgm(path,img_matrix):
    f = open(path,'rb')
    dim = []
    for i in f.readline().split():
        try:
            dim.append(int(i))
        except:
            continue
    img_matrix = np.ndarray(shape = (dim[1],dim[0]),dtype='int32')
    for x in range(dim[1]):
        for y in range(dim[0]):
            img_matrix[x,y] = (ord(f.read(1)))
    return img_matrix
#function to perform svd and use only the high singular valued components
def matrix_reduction(img_matrix):
    u, s, vh = np.linalg.svd(img_matrix)
    total = sum(s)
    partial_sum = 0
    i = 0
    while partial_sum < 0.7*total:
        partial_sum += s[i]
        i += 1
    elbow = i
    reduced_matrix = np.ndarray(shape = (64,64), dtype = 'float32')
    reduced_matrix = u[:,:elbow+1] @ np.diag(s[:elbow+1]) @vh[:elbow+1,:] 
    return elbow, reduced_matrix

def unroll(img):
    img=img.reshape(img.shape[0]*img.shape[1],1)
    img=np.transpose(img)
    return img