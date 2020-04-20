# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:17:26 2020

@author: HP
"""


#Realized that the first line contained P5 followed by the dimensions. Decided to store the dimensions in a List
import numpy as np
import matplotlib.pyplot as plt

def read_pgm(path):
    f = open(path,'rb')
    dim = []
    for i in f.readline().split():
        try:
            dim.append(int(i))
        except:
            continue
    print(dim)
    img_matrix = np.ndarray(shape = (dim[1],dim[0]),dtype='int32')
    for x in range(dim[1]):
        for y in range(dim[0]):
            img_matrix[x,y] = (ord(f.read(1)))
    plt.imshow(img_matrix,cmap='Greys')