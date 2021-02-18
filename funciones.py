import numpy as np
import h5py
import matplotlib.pyplot as plt 

# MATEMÁTICA *****************************************************************************
# funciones de activación
def sigmoid(X):
  return 1/(1+np.exp(-(X)))

def relu(X):
  return np.maximum(0,X)

funcAct = {"sigmoid":sigmoid,
          "relu":relu}

def relu_backward(dA, cache):
  """
  Implement the backward propagation for a single RELU unit.

  Arguments:
  dA -- post-activation gradient, of any shape
  cache -- 'Z' where we store for computing backward propagation efficiently

  Returns:
  dZ -- Gradient of the cost with respect to Z
  """
  
  Z = cache
  dZ = np.array(dA, copy=True) # just converting dz to a correct object.
  
  # When z <= 0, you should set dz to 0 as well. 
  dZ[Z <= 0] = 0
  
  assert (dZ.shape == Z.shape)
  
  return dZ

def sigmoid_backward(dA, cache):
  """
  Implement the backward propagation for a single SIGMOID unit.

  Arguments:
  dA -- post-activation gradient, of any shape
  cache -- 'Z' where we store for computing backward propagation efficiently

  Returns:
  dZ -- Gradient of the cost with respect to Z
  """
  
  Z = cache
  
  s = 1/(1+np.exp(-Z))
  dZ = dA * s * (1-s)
  
  assert (dZ.shape == Z.shape)
  
  return dZ

funcActB = {"sigmoid":sigmoid_backward,
            "relu":relu_backward}
# FUNCIONES ******************************************************************************
def inicializar(lista):
  '''
  Función para regresar la inicialización de una red neuronal profunda

  Parámetros
  lista: lista con la cantidad de nodos/neuronas en cada capa

  return parámetros para cada capa
    regresa una lista que contiene listas dentro, cada lista tiene dos elementos, la matriz
    W y el vector b de cada capa de modo que si lista = [10,2,2,1]
    parametros = [[W1,b1],[W2,b2],[W3,b3]]

    notar que si la lista tiene n elementos, la lista de parámetros será de n-1 elementos
    ya que el primer elemento de la lista hace referencia a la cantidad de inputs que tiene
    la primer capa, esta no es una capa que podremos modificar
  '''

  parametros = {}
  L = len(lista)

  for l in range(1,L):
    parametros['W' + str(l)] = np.random.randn(lista[l], lista[l-1]) / np.sqrt(lista[l-1]) #*0.01
    parametros['b' + str(l)] = np.zeros((lista[l], 1))

  return parametros 

def compute_Z(A, W, b):
  '''
  Genera el resultado de Z con una A, W y b determinadas

  Parámetros
  A: vector de entradas (tamaño de la capa anterior, numero de ejemplos)
  W: matriz de pesos (tamaño capa actual, tamaño capa anterior)
  b: vector de bias (tamaño capa actual, 1)

  return valor de Z y caché para backpropagation
  '''

  # se calcula Z
  Z = W @ A + b 

  # se crea la tupla con los elementos de cache
  cache = (A, W, b)

  return Z,cache

def forward_activation(A_prev, W, b, activacion):
  '''
  Implementa el calculo de la función de activación

  Parámetros
  A_prev: Activaciones de la capa anterior (tamaño de la capa anterior, numero de ejemplos)
  W: matriz de pesos (tamaño capa actual, tamaño capa anterior)
  b: vector de bias (tamaño de capa actual, 1)
  activacion: función de activación que se ocupará

  return resultado para A y caché
  '''
  # se ocupa la función definida anteriormente para determinar el valor de Z y el cache
  Z, cache = compute_Z(A_prev, W, b)
  
  # se calcula el valor de la función de activación para el resultado Z
  A = funcAct[activacion](Z)

  # se genera un cache nuevo, con el anterior y el valor de Z
  # es importante que se respete este orden, el cache de la generación de Z y la Z misma
  cache = (cache, Z)

  return A, cache

def forward(X, parametros):
  '''
  Implementación del movimiento hacia delante

  Parámetros
  X: datos de prueba (tamaño de entradas, numero de ejemplos)
  parametros: salida de inicializar

  return predicción y caché
  '''

  # lista de caches vacía
  caches = []
  # inicialización de A0 como X
  A = X
  # numero de capas
  L = len(parametros) // 2

  # ciclo para generar las primeras L-1 capas, es importante considerar que aquí podemos hacer
  # cambios para utilizar diferentes funciones de activación, aquí se dispone de relu para las
  # primeras L-1 capas
  for l in range(1,L):
    # se toma el valor de A_prev como el que tenía A
    A_prev = A 
    # se calcula el valor de la A actual y se guarda el caché que arroja
    A, cache = forward_activation(A_prev, parametros['W' + str(l)], parametros['b' + str(l)], "relu")
    caches.append(cache)

  # se hace el mismo procedimiento para la última capa, pero aquí ocupamos la función sigmoid para
  # tener un valor entre 0 y 1
  A_prev = A 
  AL, cache = forward_activation(A_prev, parametros['W' + str(L)], parametros['b' + str(L)], "sigmoid")
  caches.append(cache)

  return AL, caches

def calcular_costo(AL, Y):
  '''
  Implementación de la función de coste

  Parámetros
  AL: predicciones (1,numbero de ejemplos)
  Y: soluciones correctas

  return costo de los parámetros
  '''
  # cantidad de elementos de prueba
  m = Y.shape[1]

  # se calcula el costo con la siguiente función que es la función de regresió logística
  cost = (-1/m) * np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),(1-Y)))

  # nos sirve para transformar matrices tipo [[1]] a [1]
  cost = np.squeeze(cost)

  return cost

def compute_Ds(dZ, cache):
  '''
  Implementación de un calculo del recorrido hacia atrás para encontrar
  ciertos gradientes

  Parámetros
  dZ: gradiente de costo respecto a la función de la salida
  cache: tupla de valores (A_prev, W, b)

  return dA_prev, dW, dB
  '''

  # se toman los valores de A, W y b del cache
  A_prev, W, b = cache

  # se toma el tamaño de ejemplos de A
  m = A_prev.shape[1]

  # se calculan los gradientes con las siguientes formulas
  dW = (1/m)* dZ @ A_prev.T
  db = (1/m)* np.sum(dZ, axis=1,keepdims=True)
  dA_prev = W.T @ dZ
  
  return dA_prev, dW, db

def backward_activation(dA, cache, activacion):
  '''
  Implementación de la función de activación hacia atrás, termina de calcular todos
  los gradientes necesarios, puede verse como un proxy

  Parámetros
  dA: gradiente de la capa siguiente
  cache: tupla de valores (cache linear, cache de activacion)
  activacion: funcion de activación a ser utilizada

  return dA_prev, dW, db
  '''
  # se dividen los dos tipos de cache
  suma_cache, Z = cache

  # multiplicación de posición por posición (element wise) para determinar dZ
  dZ = funcActB[activacion](dA, Z)

  # con el valor de dZ, se obtienen los gradientes con la función anterior
  dA_prev, dW, db = compute_Ds(dZ, suma_cache)

  return dA_prev, dW, db

def backpropagation(AL, Y, caches):
  '''
  Implementación del backpropagation

  Parámetros
  AL: predicciones
  Y: resultados correctos
  caches: cache del forward propagation

  return gradientes
  '''

  # gradientes en forma de diccionario para acceder a ellos de forma más sencilla
  grads = {}
  L = len(caches) # numero de capas
  m = AL.shape[1] # numero de ejemplos
  Y = Y.reshape(AL.shape) # Y es igual a AL

  # derivada del costo total respecto a AL, esta formula esta dada en el curso
  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

  # obtenemos el primer cache, el de la capa de salida
  current_cache = caches[L-1]
  # calculamos sus gradientes
  grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache,"sigmoid")
  
  # con un ciclo for, calculamos los gradientes de las demás capas, podemos interpretar estos dos pasos
  # como la parte al reves de la propagación hacia delante
  for l in reversed(range(L-1)):
    current_cache = caches[l]
    dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l+1)], current_cache,"relu")
    grads["dA" + str(l)] = dA_prev_temp
    grads["dW" + str(l + 1)] = dW_temp
    grads["db" + str(l + 1)] = db_temp

  # se regresan los gradientes
  return grads

def actualizar_parametros(parametros, grads, learning_rate):
  """
  Implementación para modificar los gradientes

  Parámetros
  parametros: lista con las matrices W y vectores b
  grads: gradientes para cada uno de esos parámetros
  learning_rate: factor de aprendizaje

  return parametros modificados
  """
  
  L = len(parametros) // 2

  for l in range(L):
      parametros["W" + str(l+1)] -= (learning_rate * grads["dW" + str(l+1)])
      parametros["b" + str(l+1)] -= (learning_rate * grads["db" + str(l+1)])
  return parametros

def prediccion(X, y, parameters):

  """
  This function is used to predict the results of a  L-layer neural network.
  
  Arguments:
  X -- data set of examples you would like to label
  parameters -- parameters of the trained model
  
  Returns:
  p -- predictions for the given dataset X
  """
  
  m = X.shape[1]
  n = len(parameters) // 2 # number of layers in the neural network
  p = np.zeros((1,m))
  
  # Forward propagation
  probas, caches = L_model_forward(X, parameters)

  
  # convert probas to 0/1 predictions
  for i in range(0, probas.shape[1]):
    if probas[0,i] > 0.5:
      p[0,i] = 1
    else:
      p[0,i] = 0
  
  print("Precisión: "  + str(np.sum((p == y)/m)))
      
  return p

