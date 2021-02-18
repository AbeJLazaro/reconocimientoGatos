'''
Autor:      Lázaro Martínez Abraham Josué
Fecha:      17 de febrero de 2021
Titulo:     redGatos.py
'''
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from modAux import load_data
from funciones import *
import time
import scipy
from scipy import ndimage
from PIL import Image

# datos para el modelo 
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# funciones utiles
def imprimirImagen(index):
  plt.imshow(train_x_orig[index])
  plt.show()

def informacion():
  m_train = train_x_orig.shape[0]
  num_px = train_x_orig.shape[1]
  m_test = test_x_orig.shape[0]

  print ("Numero de ejemplos de entrenamiento: " + str(m_train))
  print ("Numero de ejemplos para pruebas: " + str(m_test))
  print ("Tamaño de cada imagen: (" + str(num_px) + ", " + str(num_px) + ", 3)")
  print ("train_x_orig shape: " + str(train_x_orig.shape))
  print ("train_y shape: " + str(train_y.shape))
  print ("test_x_orig shape: " + str(test_x_orig.shape))
  print ("test_y shape: " + str(test_y.shape))

# para entrenar el modelo **************************************************************************
# redimensionamiento de los ejemplos de prueba 
# El -1 hace que se redimensionen las dimesiones sobrantes
# (64,64,3,m) -> (64x64x3,m)
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Movemos los pixeles para que se encuentren entre 0 y 1 dividiendo entre 255
# que es la intensidad de cada picel
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# definimos los datos para la arquitectura de la red
layers_dims = [12288, 20, 7, 5, 1]

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

# función para entrenar el modelo y mostrar sus resultados en precisión
def entrenar():
  parametros = modeloProfundo(train_x,train_y,layers_dims,0.0075,2500,True)

  #pred_train = prediccion(train_x, train_y, parametros)

  #pred_test = prediccion(test_x, test_y, parametros)

  a = 0
  while (a>-1):
    a = int(input("Un número "))
    if a> -1 : revision(a,parametros)

def revision(index,parametros):
  AL, caches = forward(train_x[:,index].reshape(train_x.shape[0],1), parametros)
  if (AL > 0.5):
    print("Es un gato")
  else:
    print("No es un gato")

  imprimirImagen(index)

if __name__ == '__main__':
  #for a in range(200):
  #  imprimirImagen(a)

  entrenar()

