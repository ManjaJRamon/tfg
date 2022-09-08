import torch
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from array import *
import sys
import gzip, pickle
from keras.datasets import mnist

fich = torch.load("i5_train.pt", map_location="cpu")
var = fich.numpy()
# var ahora es un tensor con todos las formas de representar un 9 en el dataset de MNIST

(x_train, y_train), (x_test, y_test)=mnist.load_data()
img = x_train[0]
plt.imshow(img, cmap='gray')

print("Representaciones dígito 5: formato binario \n")
"""
plt.imshow(var[0], cmap='gray')

for k in range(len(var)):
  
  for i in range(28):
    for j in range(28):
      if var[k][i][j]!=0:
        var[k][i][j]=1
    print(var[k][i][:])
  print("        ")  
"""




