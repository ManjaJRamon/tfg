import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from array import *
import gzip, pickle

print(" ")

############################## TRAINING ########################################

############# PARSE THE SIZE OF THE DATASET YOU WANT TO GENERATE ###############
############# !pyhton assembler_train.py --size X ###############################

parser = argparse.ArgumentParser()

# Add the desired arguments, defining their type (and default value of 60.000 equal to the actual size of the MNIST dataset).
parser.add_argument("--size", type=int, default=60000) 

# Parse the arguments and store them in a variable.
args = parser.parse_args()

# Store all arguments in variables.
size = args.size
print("Dataset size is: %d" %size)

x = torch.zeros(size,56,28, dtype=torch.uint8)
y = torch.zeros(size, dtype=torch.uint8)

dg0_train = torch.load("imagenes/digitos_dataset/training/dg0_train.pt", map_location = "cpu")
dg1_train = torch.load("imagenes/digitos_dataset/training/dg1_train.pt", map_location = "cpu")
dg2_train = torch.load("imagenes/digitos_dataset/training/dg2_train.pt", map_location = "cpu")
dg3_train = torch.load("imagenes/digitos_dataset/training/dg3_train.pt", map_location = "cpu")
dg4_train = torch.load("imagenes/digitos_dataset/training/dg4_train.pt", map_location = "cpu")
dg5_train = torch.load("imagenes/digitos_dataset/training/dg5_train.pt", map_location = "cpu")
dg6_train = torch.load("imagenes/digitos_dataset/training/dg6_train.pt", map_location = "cpu")
dg7_train = torch.load("imagenes/digitos_dataset/training/dg7_train.pt", map_location = "cpu")
dg8_train = torch.load("imagenes/digitos_dataset/training/dg8_train.pt", map_location = "cpu")
dg9_train = torch.load("imagenes/digitos_dataset/training/dg9_train.pt", map_location = "cpu")


for i in range(size):
  ## Genera numero entero aleatorio entre 0 y 9
  j = random.randint(0,9)
  k = random.randint(0,5999)
  y[i] = j
  if j==0:
    x[i]=dg0_train[k]
  if j==1:
    x[i]=dg1_train[k]
  if j==2:
    x[i]=dg2_train[k]
  if j==3:
    x[i]=dg3_train[k]
  if j==4:
    x[i]=dg4_train[k]
  if j==5:
    x[i]=dg5_train[k]
  if j==6:
    x[i]=dg6_train[k]
  if j==7:
    x[i]=dg7_train[k]
  if j==8:
    x[i]=dg8_train[k]
  if j==9:
    x[i]=dg9_train[k]

# Comprobación rápida

var = x.numpy()
for i in range(56):
  for j in range(28):
    if var[0][i][j]!=0:
      var[0][i][j] = 1
  print(var[0][i][:])
print("         ")
etiq = y[0]
print("Etiqueta es igual a: %d" %etiq)


# En la variable X se tienen todas las imágenes del dataset
# y en la variable Y las respectivas etiquetas de las imágenes


data_image = array('B')
data_label = array('B')

for i in range(size):
  data_label.append(y[i])

for i in range(size):
  for j in range(56):
    for m in range(28):
        data_image.append(x[i][j][m])

hexval = "{0:#0{1}x}".format(size,6) # number of files in HEX

	# header for label array

header = array('B')
header.extend([0,0,8,1,0,0])
header.append(int('0x'+hexval[2:][:2],16))
header.append(int('0x'+hexval[2:][2:],16))
	
data_label = header + data_label

# additional header for images array
	
header.extend([0,0,0,56,0,0,0,28])


header[3] = 3 # Changing MSB for image data (0x00000803)
	
data_image = header + data_image

output_file = open('train-images-idx3-ubyte', 'wb')
data_image.tofile(output_file)
output_file.close()

output_file = open('zip/train-images-idx3-ubyte', 'wb')
data_label.tofile(output_file)
output_file.close()


output_file = open('train-labels-idx1-ubyte', 'wb')
data_label.tofile(output_file)
output_file.close()

output_file = open('zip/train-labels-idx1-ubyte', 'wb')
data_label.tofile(output_file)
output_file.close()

"""
os.system('gzip test-images-idx3-ubyte')
os.system('gzip test-labels-idx1-ubyte')
"""

print("final")






