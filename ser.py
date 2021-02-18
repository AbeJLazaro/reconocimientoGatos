'''
Autor:      Lázaro Martínez Abraham Josué
Fecha:      18 de febrero de 2021
Titulo:     ser.py
'''
import pickle 
import numpy as np 

def guardar(nombre,datos):
  with open(nombre+'.pickle', 'wb') as f:
    pickle.dump(datos, f)

def cargar(nombre):
  with open(nombre+'.pickle', 'rb') as f:
    entry = pickle.load(f)
  
  return entry

if __name__ == '__main__':
  a = {"w1": np.array([[x for x in range(100)]])}
  guardar("entry",a)
  ab = cargar("entry")
  print(a)
  print(ab)