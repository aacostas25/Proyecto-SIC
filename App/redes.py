import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.formats.info import Iterator
from pydataset import data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

#Seleccionamos la base datos que queremos trabajar
def seleccionarDB(numero):
    'opciones para database: titanic, iris,snails,datos sinteticos'
    if numero == 1:
        baseDatos = data('titanic')
        baseDatos = pd.get_dummies(baseDatos,drop_first=True)
        X_train,x_test,Y_train,y_test = train_test_split(baseDatos.drop('survived_yes',axis = 1,), baseDatos['survived_yes'].values.reshape(-1,1),random_state=42)
        x_inicial = 4
    if numero == 2:
        baseDatos = data('iris')
        encoder = LabelEncoder()
        baseDatos['Species'] = encoder.fit_transform(baseDatos['Species'])
        X = baseDatos[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
        Y = baseDatos[['Species']]
        X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_inicial = 4

    if numero == 3:
        baseDatos = data('snails')
        prueba = baseDatos['Deaths']/baseDatos['N']
        baseDatos['Porcentaje_muertes'] = prueba
        baseDatos = pd.get_dummies(baseDatos,drop_first=True)
        X = baseDatos[['Species_B','Exposure','Rel.Hum','Temp']]
        Y = baseDatos[['Porcentaje_muertes']]
        X_train, x_test, Y_train, y_test = train_test_split(X, Y , test_size=0.2, random_state=42)
        x_inicial = 4
    return x_inicial,X_train,Y_train,x_test,y_test
# Funciones de activacion

def nonlin(x,deriv=False):
    if(deriv==True):
        return nonlin(x)*(1-nonlin(x))
    return 1/(1+np.exp(-x))

def tanh(x,deriv=False):
  if(deriv==True):
    return (1 - np.tanh(x)**2)
  return np.tanh(x)

#Seleccionamos la funcion de activacion

def seleccionarFA(numero):
    if numero == 1:
        funcion = nonlin
    if numero == 2:
        funcion = tanh
    return funcion
"""
Ahora definimos quien va a ser tanto el X,Y
"""

def blob(muestras,centros,std):
    X,y = make_blobs(
        n_samples    = muestras, 
        n_features   = 2, 
        centers      = centros, 
        cluster_std  = std, 
        shuffle      = True, 
        random_state = 0
       )
    scaler = MinMaxScaler()
    y = np.reshape(y, (-1, 1))
    y_normalized = scaler.fit_transform(y)
    x_inicial = 2
    X_train, X_test, Y_train, Y_test = train_test_split(X,y_normalized , test_size=0.2, random_state=42)
    return x_inicial,X,y,X_train,Y_train

#Red neuronal con arquitectura x-10-7-1

def RN2(X,Y,x_inicial,funcion = tanh,eta=0.01,neuronas_1 = 10,neuronas_2 = 7,ite = 3000):
  np.random.seed(1)

  # Inicializar los pesos de las capas ocultas y de la capa de salida
  syn0 = 2*np.random.random((x_inicial,neuronas_1)) - 1
  syn1 = 2*np.random.random((neuronas_1,neuronas_2)) - 1
  syn2 = 2*np.random.random((neuronas_2, 1)) - 1

  # Inicializar la lista de errores
  errors = []
  iteraciones = []
  # Entrenar la red neuronal
  for iter in range(ite):
      # Propagación hacia adelante
      l0 = X
      neta1 = np.dot(l0, syn0)
      l1 = funcion(neta1)
      neta2 = np.dot(l1, syn1)
      l2 = funcion(neta2)
      neta3 = np.dot(l2, syn2)
      l3 = funcion(neta3)

      # Calcular el error
      l3_error = Y - l3

      if (iter % 100 == 0):
        iteraciones.append(iter)
        errors.append(np.mean(np.abs(l3_error), axis=0))

      # Retropropagación del error
      l3_delta = l3_error * funcion(neta3,deriv=True) * eta
      l2_error = l3_delta.dot(syn2.T)
      l2_delta = l2_error * funcion(neta2,deriv=True) * eta
      l1_error = l2_delta.dot(syn1.T)
      l1_delta = l1_error * funcion(neta1,deriv=True) * eta

      # Actualizar los pesos
      syn2 += np.dot(l2.T, l3_delta)
      syn1 += np.dot(l1.T, l2_delta)
      syn0 += np.dot(l0.T, l1_delta)

      # Almacenar el error en la lista de errores
  return (syn0,syn1,syn2,errors,iteraciones,l3_error)