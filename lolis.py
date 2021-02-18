'''
Autor:      Lázaro Martínez Abraham Josué
Fecha:      18 de febrero de 2021
Titulo:     lolis.py
'''
import os
import sys
import tensorflow as tf
# keras es una api que funciona sobre tensorflow, lo hace más sencillo, pero también funciona sobre otros
# esto lo hace portable
import keras
import matplotlib.pyplot as plt
import numpy as np
import h5py
from modAux import load_data
from funciones import *
import time
from scipy import ndimage
from keras.preprocessing import image
from PIL import Image

from ser import guardar,cargar

# para descargar los datos
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

# las listas que contendrán las imagenes y sus respuestas
x_train = []
y_train = []

# función para descargar y transformar las imagenes
def descargarDatos(nombre, num, y):

  arguments = {"keywords":nombre ,"limit":num, "format":"jpg", "print_urls":True}
  imagenes = response.download(arguments)

  for name in os.listdir("downloads/"+nombre):
    x = image.img_to_array(image.load_img("downloads/"+nombre+"/"+name,target_size=(299,299)))
    x /= 255
    x = x.reshape([x.shape[0],x.shape[1],x.shape[2]])
    #print(x.shape)
    #plt.imshow(x)
    #plt.show()
    x_train.append(x)
    y_train.append(y)

# función para imprimir una imagen
def imprimirImagen(index):
  plt.imshow(x_train[index])
  plt.show()

# definción de la arquitectura de la red
layers_dims = [299*299*3, 20, 7, 5, 1]

# función para entrenar
def modeloProfundo(X, Y, layers_dims, learning_rate = 0.0075, iteraciones = 3000, print_cost=False):
  """
  Implementa un modelo de deep learning
  
  Parámetros
  X: datos a probar, dimensiones (64x64x3,m)
  Y: valores correctos de si se trata o no de un gato
  layers_dims: definición de la arquitectura de la red
  learning_rate: factor de aprendizaje
  iteraciones: numero de iteraciones que se realizarán para el aprendizaje
  print_cost: si es True, imprime el costo cada 100 iteraciones
  
  return parametros aprendidos por el modelo
  """

  np.random.seed(1)
  costos = []
  
  # Inicializamos los parametros
  parametros = inicializar(layers_dims)
  
  # for para el forward and backward propagation
  for i in range(iteraciones):
    # forward propagations
    AL, caches = forward(X, parametros)
    
    # calculamos el costo
    costo = calcular_costo(AL, Y)
  
    # backward propagation
    grads = backpropagation(AL, Y, caches)

    # Actualizamos parámetros
    parametros = actualizar_parametros(parametros, grads, learning_rate)
    
    # imprimimos el costo cada 100 iteraciones
    if print_cost and i % 100 == 0:
      print ("Costo después de iteración %i: %f" %(i, costo))
    if print_cost and i % 100 == 0:
      costos.append(costo)
          
  # grafica del costo
  '''
  plt.plot(np.squeeze(costos))
  plt.ylabel('Costo')
  plt.xlabel('Iteraciones (por cada 100)')
  plt.title("Factor de aprendizaje =" + str(learning_rate))
  plt.show()'''
  
  # retornamos los parámetros aprendidos
  return parametros

# función para entrenar el modelo y probar
def entrenar():
  parametros = modeloProfundo(x_train_p,y_train_p,layers_dims,0.0075,3000,True)

  a = 0
  while (a>-1):
    a = int(input("Un número "))
    if a> -1 : revision(a,parametros)

  return parametros

# función para revisar los resultados con el conjunto de prueba
def revision(index,parametros):
  AL, caches = forward(x_train_p[:,index].reshape(x_train_p.shape[0],1), parametros)
  if (AL > 0.5):
    print("Es una loli")
  else:
    print("No es una loli")

  imprimirImagen(index)

# función para entrenar la red, guarda los parámetros del modelo 
def entrenamiento():
  # se manda a llamar dicha función con diferentes tipos de imagenes
  descargarDatos("Loli", 50, 1)
  descargarDatos("Car", 50, 0)
  descargarDatos("Mountain", 50, 0)
  descargarDatos("Beach", 50, 0)

  # se da formato a los conjuntos de entrenamiento
  # x, inputs
  x_train_p = np.array(x_train)
  x_train_p = x_train_p.reshape(x_train_p.shape[0],-1).T
  print(x_train_p.shape)
  # y, salidas
  y_train_p = np.array(y_train)
  y_train_p = y_train_p.reshape(1,y_train_p.shape[0])
  print(y_train_p.shape)

  param = entrenar()
  guardar("lolimodel",param)

# carga los datos del modelo
def carga():
  return cargar("lolimodel")

if __name__ == '__main__':
  param = carga()
  
  x = image.img_to_array(image.load_img("downloads/jess4.jpg",target_size=(299,299)))
  x /= 255
  x = x.reshape([x.shape[0],x.shape[1],x.shape[2]])
  print(x.shape)

  AL, caches = forward(x.reshape(-1,1), param)
  if (AL > 0.5):
    print("Es una loli")
  else:
    print("No es una loli")

  plt.imshow(x)
  plt.show()

# BIBLIOGRAFIA
# https://pub.towardsai.net/building-a-custom-image-dataset-for-deep-learning-projects-7f759d069877
