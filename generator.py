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


"""

Estos archivos están descargados de las Rutas
data/MNIST/TorchvisionDatasetWrapper/processed/training.pt
data/MNIST/TorchvisionDatasetWrapper/processed/test.pt

de los proyectos de Miguel e Isabel

"""


test_isa = torch.load("test_isabel.pt", map_location = "cpu")
test_mig = torch.load("test_miguel.pt", map_location = "cpu")

train_isa = torch.load("training_isabel.pt", map_location = "cpu")
train_mig = torch.load("training_miguel.pt", map_location = "cpu")

print("Muestras de training de Isabel: %d" % (len(train_isa[0])))
print("Muestras de training de Miguel: %d" % (len(train_mig[0])))

print("Muestras de test de Isabel: %d" % (len(test_isa[0])))
print("Muestras de test de Miguel: %d" % (len(test_mig[0])))

print("intento unir dos imágenes cualesquiera y guardar sus etiquetas")


# Declaración de tensores para guardar los dígitos de training y test por separado

######################## VARIABLES MIGUEL #####################################

# Training Miguel
imag_0_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_1_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_2_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_3_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_4_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_5_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_6_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_7_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_8_mig = torch.zeros(6000,28,28, dtype=torch.uint8)
imag_9_mig = torch.zeros(6000,28,28, dtype=torch.uint8)

# Test Miguel
imag_0_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_1_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_2_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_3_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_4_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_5_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_6_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_7_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_8_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)
imag_9_mig_test = torch.zeros(1000,28,28, dtype=torch.uint8)

###############################################################################
############################## TRAINING MIGUEL ################################
###############################################################################


zero=0
one=0
two=0
three=0
four=0
five=0
six=0
seven=0
eight=0
nine=0

for i in range(len(train_mig[1])):
  if train_mig[1][i]==0 and zero<6000:
    imag_0_mig[zero]=train_mig[0][i]
    zero+=1
  if train_mig[1][i]==1 and one<6000:
    imag_1_mig[one]=train_mig[0][i]

  if train_mig[1][i]==2 and two<6000:
    imag_2_mig[two]=train_mig[0][i]
    two+=1
  if train_mig[1][i]==3 and three<6000:
    imag_3_mig[three]=train_mig[0][i]
    three+=1
  if train_mig[1][i]==4 and four<6000:
    imag_4_mig[four]=train_mig[0][i]
    four+=1
  if train_mig[1][i]==5 and five<6000:
    imag_5_mig[five]=train_mig[0][i]
    five+=1
  if train_mig[1][i]==6 and six<6000:
    imag_6_mig[six]=train_mig[0][i]
    six+=1
  if train_mig[1][i]==7 and seven<6000:
    imag_7_mig[seven]=train_mig[0][i]
    seven+=1
  if train_mig[1][i]==8 and eight<6000:
    imag_8_mig[eight]=train_mig[0][i]
    eight+=1
  if train_mig[1][i]==9 and nine<6000:
    imag_9_mig[nine]=train_mig[0][i]
    nine+=1


###############################################################################
############################## TEST MIGUEL ####################################
###############################################################################

zero=0
one=0
two=0
three=0
four=0
five=0
six=0
seven=0
eight=0
nine=0

for i in range(len(test_mig[1])):
  if test_mig[1][i]==0 and zero<1000:
    imag_0_mig_test[zero]=test_mig[0][i]
    zero+=1
  if test_mig[1][i]==1 and one<1000:
    imag_1_mig_test[one]=test_mig[0][i]
    one+=1
  if test_mig[1][i]==2 and two<1000:
    imag_2_mig_test[two]=test_mig[0][i]
    two+=1
  if test_mig[1][i]==3 and three<1000:
    imag_3_mig_test[three]=test_mig[0][i]
    three+=1
  if test_mig[1][i]==4 and four<1000:
    imag_4_mig_test[four]=test_mig[0][i]
    four+=1
  if test_mig[1][i]==5 and five<1000:
    imag_5_mig_test[five]=test_mig[0][i]
    five+=1
  if test_mig[1][i]==6 and six<1000:
    imag_6_mig_test[six]=test_mig[0][i]
    six+=1
  if test_mig[1][i]==7 and seven<1000:
    imag_7_mig_test[seven]=test_mig[0][i]
    seven+=1
  if test_mig[1][i]==8 and eight<1000:
    imag_8_mig_test[eight]=test_mig[0][i]
    eight+=1
  if test_mig[1][i]==9 and nine<1000:
    imag_9_mig_test[nine]=test_mig[0][i]
    nine+=1
    
######################## VARIABLES ISABEL #####################################

# Training Isabel
imag_0_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_1_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_2_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_3_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_4_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_5_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_6_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_7_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_8_isa = torch.zeros(2400,28,28, dtype=torch.uint8)
imag_9_isa = torch.zeros(2400,28,28, dtype=torch.uint8)

# Test Isabel
imag_0_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_1_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_2_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_3_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_4_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_5_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_6_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_7_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_8_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)
imag_9_isa_test = torch.zeros(600,28,28, dtype=torch.uint8)



###############################################################################
############################## TRAINING ISABEL ################################
###############################################################################


zero=0
one=0
two=0
three=0
four=0
five=0
six=0
seven=0
eight=0
nine=0

for i in range(len(train_isa[1])):
  if train_isa[1][i]==0 and zero<2400:
    imag_0_isa[zero]=train_isa[0][i]
    zero+=1
  if train_isa[1][i]==1 and one<2400:
    imag_1_isa[one]=train_isa[0][i]
    one+=1
  if train_isa[1][i]==2 and two<2400:
    imag_2_isa[two]=train_isa[0][i]
    two+=1
  if train_isa[1][i]==3 and three<2400:
    imag_3_isa[three]=train_isa[0][i]
    three+=1
  if train_isa[1][i]==4 and four<2400:
    imag_4_isa[four]=train_isa[0][i]
    four+=1
  if train_isa[1][i]==5 and five<2400:
    imag_5_isa[five]=train_isa[0][i]
    five+=1
  if train_isa[1][i]==6 and six<2400:
    imag_6_isa[six]=train_isa[0][i]
    six+=1
  if train_isa[1][i]==7 and seven<2400:
    imag_7_isa[seven]=train_isa[0][i]
    seven+=1
  if train_isa[1][i]==8 and eight<2400:
    imag_8_isa[eight]=train_isa[0][i]
    eight+=1
  if train_isa[1][i]==9 and  nine<2400:
    imag_9_isa[nine]=train_isa[0][i]
    nine+=1
    

###############################################################################
############################## TEST ISABEL ####################################
###############################################################################

zero=0
one=0
two=0
three=0
four=0
five=0
six=0
seven=0
eight=0
nine=0

for i in range(len(test_isa[1])):
  if test_isa[1][i]==0 and zero<600:
    imag_0_isa_test[zero]=test_isa[0][i]
    zero+=1
  if test_isa[1][i]==1 and one<600:
    imag_1_isa_test[one]=test_isa[0][i]
    one+=1
  if test_isa[1][i]==2 and two<600:
    imag_2_isa_test[two]=test_isa[0][i]
    two+=1
  if test_isa[1][i]==3 and three<600:
    imag_3_isa_test[three]=test_isa[0][i]
    three+=1
  if test_isa[1][i]==4 and four<600:
    imag_4_isa_test[four]=test_isa[0][i]
    four+=1
  if test_isa[1][i]==5 and five<600:
    imag_5_isa_test[five]=test_isa[0][i]
    five+=1
  if test_isa[1][i]==6 and six<600:
    imag_6_isa_test[six]=test_isa[0][i]
    six+=1
  if test_isa[1][i]==7 and seven<600:
    imag_7_isa_test[seven]=test_isa[0][i]
    seven+=1
  if test_isa[1][i]==8 and eight<600:
    imag_8_isa_test[eight]=test_isa[0][i]
    eight+=1
  if test_isa[1][i]==9 and  nine<600:
    imag_9_isa_test[nine]=test_isa[0][i]
    nine+=1
    

################## CREACIÓN BASE DE DATOS DE DÍGITOS DE TRAINING ##############

# Se supone que ahora tengo todos los dígitos por separado tanto de Isa como Miguel
# guardados en 20 variables que ahora quiero que sean ficheros

dirName = "imagenes/digitos_imagen_train/"
os.makedirs(dirName)
torch.save(imag_0_mig, dirName+"i0_train.pt")
torch.save(imag_1_mig, dirName+"i1_train.pt")
torch.save(imag_2_mig, dirName+"i2_train.pt")
torch.save(imag_3_mig, dirName+"i3_train.pt")
torch.save(imag_4_mig, dirName+"i4_train.pt")
torch.save(imag_5_mig, dirName+"i5_train.pt")
torch.save(imag_6_mig, dirName+"i6_train.pt")
torch.save(imag_7_mig, dirName+"i7_train.pt")
torch.save(imag_8_mig, dirName+"i8_train.pt")
torch.save(imag_9_mig, dirName+"i9_train.pt")



dirName2 = "imagenes/digitos_audio_train/"
os.makedirs(dirName2)
torch.save(imag_0_isa, dirName2+"a0_train.pt")
torch.save(imag_1_isa, dirName2+"a1_train.pt")
torch.save(imag_2_isa, dirName2+"a2_train.pt")
torch.save(imag_3_isa, dirName2+"a3_train.pt")
torch.save(imag_4_isa, dirName2+"a4_train.pt")
torch.save(imag_5_isa, dirName2+"a5_train.pt")
torch.save(imag_6_isa, dirName2+"a6_train.pt")
torch.save(imag_7_isa, dirName2+"a7_train.pt")
torch.save(imag_8_isa, dirName2+"a8_train.pt")
torch.save(imag_9_isa, dirName2+"a9_train.pt")



################## CREACIÓN BASE DE DATOS DE DÍGITOS DE TEST ###################


dirName = "imagenes/digitos_imagen_test/"
os.makedirs(dirName)
torch.save(imag_0_mig, dirName+"i0_test.pt")
torch.save(imag_1_mig, dirName+"i1_test.pt")
torch.save(imag_2_mig, dirName+"i2_test.pt")
torch.save(imag_3_mig, dirName+"i3_test.pt")
torch.save(imag_4_mig, dirName+"i4_test.pt")
torch.save(imag_5_mig, dirName+"i5_test.pt")
torch.save(imag_6_mig, dirName+"i6_test.pt")
torch.save(imag_7_mig, dirName+"i7_test.pt")
torch.save(imag_8_mig, dirName+"i8_test.pt")
torch.save(imag_9_mig, dirName+"i9_test.pt")



dirName2 = "imagenes/digitos_audio_test/"
os.makedirs(dirName2)
torch.save(imag_0_isa, dirName2+"a0_test.pt")
torch.save(imag_1_isa, dirName2+"a1_test.pt")
torch.save(imag_2_isa, dirName2+"a2_test.pt")
torch.save(imag_3_isa, dirName2+"a3_test.pt")
torch.save(imag_4_isa, dirName2+"a4_test.pt")
torch.save(imag_5_isa, dirName2+"a5_test.pt")
torch.save(imag_6_isa, dirName2+"a6_test.pt")
torch.save(imag_7_isa, dirName2+"a7_test.pt")
torch.save(imag_8_isa, dirName2+"a8_test.pt")
torch.save(imag_9_isa, dirName2+"a9_test.pt")




