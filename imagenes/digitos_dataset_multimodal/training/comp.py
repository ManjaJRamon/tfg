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

#archivo = torch.load("pesos.pt", map_location= "cpu")
#index = torch.nonzero(archivo).size(0)

lista = ["dg0_train.pt","dg1_train.pt","dg2_train.pt","dg3_train.pt","dg4_train.pt","dg5_train.pt","dg6_train.pt","dg7_train.pt","dg8_train.pt","dg9_train.pt"]

for k in range(10): # voy a comprobar en todos los ficheros si se han llenado los tensores hasta el último elemento y no hay ninguna posición sin imagen
  arch = torch.load(lista[k], map_location="cpu")
  var = arch.numpy() # paso a array porque en tensor solo me deja imprimir 14 elementos por línea y no se aprecia el dígito con claridad. Luego imprimiré el array línea a línea por la misma razón

  print(arch.size())
  #for k in range(10):
    #voy a sacar 10 modelos del dígito a ver si salen bien de forma aleatoria
  idx = 5999
  print("ÍNDICE: %d" %idx)
  print("              ")
  for i in range(56):
    for j in range(28):
      if var[idx][i][j]!=0: # paso los elemtentos distintos de 0 a binario para que se vea mejor el dígito con más contraste
          var[idx][i][j]=1
    print(var[idx][i][:]) # imprimo línea a línea para poder ver el dígito en las dimensiones adecuadas
  print("       ")