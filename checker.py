from numpy.core.getlimits import _KNOWN_TYPES
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
print(" ")



lista1 = ["a0_train.pt","a1_train.pt","a2_train.pt","a3_train.pt","a4_train.pt","a5_train.pt","a6_train.pt","a7_train.pt","a8_train.pt","a9_train.pt"]
lista2 = ["i0_train.pt","i1_train.pt","i2_train.pt","i3_train.pt","i4_train.pt","i5_train.pt","i6_train.pt","i7_train.pt","i8_train.pt","i9_train.pt"]
lista3 = ["dg0_train.pt","dg1_train.pt","dg2_train.pt","dg3_train.pt","dg4_train.pt","dg5_train.pt","dg6_train.pt","dg7_train.pt","dg8_train.pt","dg9_train.pt"]


lista4 = ["a0_test.pt","a1_test.pt","a2_test.pt","a3_test.pt","a4_test.pt","a5_test.pt","a6_test.pt","a7_test.pt","a8_test.pt","a9_test.pt"]
lista5 = ["i0_test.pt","i1_test.pt","i2_test.pt","i3_test.pt","i4_test.pt","i5_test.pt","i6_test.pt","i7_test.pt","i8_test.pt","i9_test.pt"]
lista6 = ["dg0_test.pt","dg1_test.pt","dg2_test.pt","dg3_test.pt","dg4_test.pt","dg5_test.pt","dg6_test.pt","dg7_test.pt","dg8_test.pt","dg9_test.pt"]


############################### TRAINING #######################################

"""

For training : 

1. Image samples:  6000 samples/digit -> 60.000 samples
2. Audio samples:  2400 samples/digit -> 24.000 samples

"""

#Esta variable define cuántas muestras de 56x28 quiero tener de cada dígito 
digito_completo = torch.zeros(6000,56,28,dtype=torch.uint8)


dirName = "imagenes/digitos_dataset/training/"
os.makedirs(dirName)

for i in range(10):
  fich_imag = torch.load("imagenes/digitos_imagen_train/"+lista2[i], map_location= "cpu")
  fich_audio = torch.load("imagenes/digitos_audio_train/"+lista1[i], map_location= "cpu")  

  for j in range(6000):
    idx_m = random.randint(0,5999)
    idx_i = random.randint(0,2399)
    imag_np = np.concatenate((fich_imag[idx_m].numpy(),fich_audio[idx_i].numpy()), axis=0)
    imag_tensor = torch.from_numpy(imag_np)
    digito_completo[j] = imag_tensor
  torch.save(digito_completo, dirName+lista3[i])
  
############################### TEST ###########################################

"""

Tengo para test : 

1. Image samples:  1000 samples/digit -> 10.000 samples
2. Audio samples:  600 samples/digit -> 6.000 samples

"""

digito_completo_x = torch.zeros(1000,56,28,dtype=torch.uint8)

dirName2 = "imagenes/digitos_dataset/test/"
os.makedirs(dirName2)


for i in range(10):
  fich_imag = torch.load("imagenes/digitos_imagen_test/"+lista5[i], map_location= "cpu")
  fich_audio = torch.load("imagenes/digitos_audio_test/"+lista4[i], map_location= "cpu")

  for j in range(1000):
      idx_m = random.randint(0,999)
      idx_i = random.randint(0,599)
      imag_np = np.concatenate((fich_imag[idx_m].numpy(),fich_audio[idx_i].numpy()), axis=0)
      imag_tensor = torch.from_numpy(imag_np)
      digito_completo_x[j] = imag_tensor
  torch.save(digito_completo_x, dirName2+lista6[i])
  
print("final")







