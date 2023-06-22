import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools
import seaborn as sns
import pingouin as pg
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder #Integer encoding
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import randint as sp_randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from tensorflow import keras
import xgboost as xgb
import math ####
import joblib
#import pickle
#from pickle import dump
#Semilla
np.random.seed(7)

# A continuación, se definen las funciones que son transversales a todos los modelos:

# La siguiente función define el target del dataset que se podrá concatenar con la parte numérica o alfanumérica del mismo
def target(dataframe, lista, lista_nombre): # lista contiene el nombre de las columnas que son target - # lista_nombre contiene los nuevos nombres para esas columnas
  df = dataframe.dropna();
  df_target = pd.DataFrame(columns = lista_nombre)
  for i in range(0, len(lista), 1):
    n = df[lista[i]].copy();
    df_target.iloc[:, i] = n;
  return df_target

# La siguiente función permite separar la parte numérica de la parte alfanumérica del dataset
def separacion(dataframe, indicador): # Para obtener la parte númerica del dataset, se debe colocar "0" en indicador  - Para obtener la parte alfanumérica, se debe colocar "1" en indicador.
  df = dataframe.dropna();
  count_n = 0; count_a = 0;
  for i in range(0, df.shape[1], 1):
    n1 = df.iloc[:, i].copy();
    if str(df.iloc[:, i].dtypes) == 'float64' or str(df.iloc[:, i].dtypes) == 'int64': # 'int64' por pd.Int64Index
      if count_n == 0: num = n1.copy(); count_n = 1;
      elif count_n == 1: num = pd.concat([num, n1], axis = 1);
    elif str(df.iloc[:, i].dtypes) == 'object':
      if count_a == 0: alf = n1.copy(); count_a = 1;
      elif count_a == 1: alf = pd.concat([alf, n1], axis = 1);
  if indicador == 0: return num; print(num);
  elif indicador == 1: return alf; print(alf);

# La siguiente función permite aplicar retrasos (de 0 a 12 meses) en los datos para obtener la mayor correlación
def retraso(dataframe, variable_analisis, metodo): # 'variable_analisis' es la posición: 1, 2, ..., n-1 de la variable independiente sobre la que se desea realizar el análisis de correlación.
    # 'metodo' depende si se quiere hacer una evaluación de la correlación de pearson o spearman.
  n_re = 12; # N°de meses de retraso
  dataframe2 = dataframe.copy() # Se crea una copia de respaldo para no modificar el dataframe original
  # a = variable_analisis # a determina sobre qué variable se quiere obtener la correlación
  b = dataframe2.shape[1] - variable_analisis; # Esta diferencia significa que se deben retrasar todas las variables menos el target
  spear = np.empty(shape = (n_re+1, b)) # Se crea un arreglo vacío donde el número de filas es igual al número de retrasos y el número de columnas es igual al número de variables que entran en el análisis.
  for i in range(0, n_re+1, 1):
    if i > 0:
      dataframe2.iloc[:, : b] = dataframe2.iloc[:, : b:].shift(1);
      df = dataframe2.dropna()
      spear[i, :] = df.corr(metodo).iloc[b: b+1, : b] # spear es la matriz de correlaciones
    elif i == 0: # Se guardan los primeros valores de correlación mostrados en el apartado anterior
      spear[i, :] = dataframe2.corr(metodo).iloc[b: b+1, : b]
  spear = abs(spear) #OJO CON ESTO: Revisar!!! Nos interesa la mayor correlación sin importar el signo (por el momento)
  delay = np.empty(shape = b, dtype = int);
  datos = np.empty(shape = b, dtype = object);

  # Con el siguiente bucle, se obtienen los respectivos retrasos para cada variable :
  for z in range(0, b, 1):
    delay[z] = functools.reduce(lambda sub, ele: sub*10+ele, np.where(spear[:, z] == np.amax(spear[:, z])))  #OJO CON ESTO: Revisar!!! Esta función contiene todos los retrasos; hay algunos que NO usamos...
    datos[z] = str(dataframe2.columns[z])
  #print(dataframe2.columns[b])
  #print(datos)
  lista = delay.tolist(); print(lista) #lista contiene los retrasos adecuados para cada variable
  return lista

# La siguiente función permite obtener el dataframe con los retrasos respectivos
def delay(dataframe, lista): # lista contiene los retrasos para cada una de las variables
  count = 0
  dataframe2 = dataframe.copy()
  for i in lista:
    dataframe2.iloc[:, count] = dataframe2.iloc[:, count].shift(i);
    count = count+1;
  dataframe2 = dataframe2.dropna()
  #dataframe2.head(13) # Prueba: Para verificar que se hayan hecho los retrasos respectivos
  return dataframe2

# Esta función permite calcular el MAPE; se deben ingresar el valor actual (o real) y el valor predicho (o estimado)
def MAPE(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual-Y_Predicted)/Y_actual))*100
    return mape

# Se importan los datos por única vez
df = pd.read_excel('BD_Actualizada.xlsx', sheet_name='RUNT_Modelo_Yamaha', converters={'TRM': int, 'SMMLV&AUXTTE': int, 'DIAS HABILES': int, 'FESTIVOS': int, 'RUNT MERCADO': int, 'RUNT YAMAHA': int, 'RUNT NMAX': int, 'RUNT NMAX CONNECTED': int, 'RUNT CRYPTON FI': int, 'RUNT XTZ125': int, 'RUNT XTZ150': int, 'RUNT XTZ250': int, 'RUNT MT03': int, 'RUNT FZ25': int, 'RUNT FZ15': int, 'RUNT SZ15RR': int, 'RUNT YBRZ125': int, 'RUNT YCZ110': int, 'RUNT XMAX': int})

target_list = ['RUNT MERCADO', 'RUNT YAMAHA', 'RUNT NMAX', 'RUNT NMAX CONNECTED', 'RUNT CRYPTON FI', 'RUNT XTZ125', 'RUNT XTZ150', 'RUNT XTZ250', 'RUNT MT03', 'RUNT FZ25', 'RUNT FZ15', 'RUNT SZ15RR', 'RUNT YBRZ125', 'RUNT YCZ110', 'RUNT XMAX']

###########################################################################################################################################################################################################

# PARTE A. Entrenamiento con valores en tiempo t: PRONÓSTICOS PUNTUALES.

# 1. Modelo Redes Neuronales (RN)

def preprocesamientoRN(dataframe):
    df = dataframe.copy()
    df = df.values

    X = df[:, 0: df.shape[1] - 1]
    Y = df[:, df.shape[1] - 1:]

    min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
    X_scale = min_max_scaler.fit_transform(X)
    Y_scale = min_max_scaler.fit_transform(Y)

    # X_test = X_scale[X_scale.shape[0] - 4:, :]
    # Y_test = Y_scale[Y_scale.shape[0] - 4:, :]
    # X_scale = X_scale[:X_scale.shape[0] - 4, :]
    # Y_scale = Y_scale[:Y_scale.shape[0] - 4, :]

    X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y_scale, test_size=0.2, shuffle=True, random_state=8)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_scale, Y_scale, test_size=0.3, random_state=1)
    return min_max_scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def entrenamientoRN(X_train, Y_train, X_valid, Y_valid):
    model = Sequential([Dense(6, activation='relu', input_shape=(X_train.shape[1],)), Dense(4, activation='relu'), Dense(Y_train.shape[1], activation='relu')]) #tanh
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])
    hist = model.fit(X_train, Y_train, batch_size=2, epochs=300, validation_data=(X_valid, Y_valid), callbacks=[es])
    return hist, model

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRN(dataset_target)

    hist, modeloRN = entrenamientoRN(X_train, Y_train, X_valid, Y_valid)

    # Estimaciones
    Y_hat_train = modeloRN.predict(X_train);
    Y_hat_train = scaler.inverse_transform(Y_hat_train);
    Y_hat_valid = modeloRN.predict(X_valid);
    Y_hat_valid = scaler.inverse_transform(Y_hat_valid);
    Y_hat_test = modeloRN.predict(X_test);
    Y_hat_test = scaler.inverse_transform(Y_hat_test);
    # Reales
    Y_train_normal = scaler.inverse_transform(Y_train)
    Y_valid_normal = scaler.inverse_transform(Y_valid)
    Y_test_normal = scaler.inverse_transform(Y_test)

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid_normal, Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal, Y_hat_valid)]
    # errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
    np.save('error_RNN_actual_' + t[5:] + '.npy', errores) # Cambiar por t2.replace(" ", "")
    # modelo = np.load('error_RNN_actual_Yamaha.npy')

    # Se guarda el modelo en la misma carpeta del proyecto
    modeloRN.save('modeloRN_' + t[5:] + '.h5')

# 2. Modelo Random Forest (RF)

def preprocesamientoRF_XG(dataframe):
    df = dataframe.copy()
    df = df.values

    X = df[:, 0: df.shape[1] - 1]
    Y = df[:, df.shape[1] - 1:]

    # min_max_scaler = preprocessing.MinMaxScaler((-1, 1))  # OPORTUNIDAD DE MEJORA 1: Ensayar otros métodos de normalización (o estandarización): Normalizer, StandardScaler, RobustScaler, entre otros.
    # X_scale = min_max_scaler.fit_transform(X)
    # Y_scale = min_max_scaler.fit_transform(Y)

    #X_test = X[X.shape[0] - 4:, :]
    #Y_test = Y[Y.shape[0] - 4:, :].ravel()
    #X = X[: X.shape[0] - 4, :]
    #Y = Y[: Y.shape[0] - 4, :].ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=8)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=8)  # 0.25 x 0.8 = 0.2

    #X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]] # , 'RUNT' + t[6:]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRF_XG(dataset_target)

    # Búsqueda de hiperparámetros
    # Grid de hiperparámetros a evaluar
    # ==============================================================================
    param_grid = ParameterGrid({'n_estimators': range(20, 41, 1), 'max_features': ['auto', 'sqrt'], 'max_depth': range(20, 41, 1)})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:

        modeloRF = RandomForestRegressor(n_jobs=-1, random_state=1, criterion='squared_error', bootstrap=True, **params)  # Se cambió 'mse' por 'squared_error'

        modeloRF.fit(X_train, Y_train)
        y_hat_train = modeloRF.predict(X_train)
        y_hat_valid = modeloRF.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('mape_valid', ascending=True)
    best = resultados.head(1)

    # Se ajusta el modelo Random Forest para RUNT Yamaha, con los mejores hiperparámetros encontrados anteriormente:
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid({'n_estimators': best['n_estimators'].values, 'max_features': best['max_features'].values, 'max_depth': best['max_depth'].values})  # Estas son las mejores combinaciones de hiperparámetros

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:

        modeloRF = RandomForestRegressor(n_jobs=-1, random_state=1, criterion='squared_error', bootstrap=True, **params)

        modeloRF.fit(X_train, Y_train)
        y_hat_train = modeloRF.predict(X_train)
        y_hat_valid = modeloRF.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('mape_valid', ascending=True)
    resultados.head(4)

    Y_train = Y_train.reshape([-1, 1])
    y_hat_train = y_hat_train.reshape([-1, 1])
    Y_valid = Y_valid.reshape([-1, 1])
    y_hat_valid = y_hat_valid.reshape([-1, 1])
    Y_test = Y_test.reshape([-1, 1])
    y_hat_test = modeloRF.predict(X_test)
    y_hat_test = y_hat_test.reshape([-1, 1])

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid, y_hat_valid), mean_absolute_percentage_error(Y_valid, y_hat_valid)]
    # errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
    np.save('error_RF_actual_' + t[5:] + '.npy', errores)
    # modelo = np.load('error_RF_actual_Yamaha.npy')

    # Se guarda el modelo en la misma carpeta del proyecto
    joblib.dump(modeloRF, 'modeloRF_' + t[5:] + '.pkl')

# 3. Modelo XGBoost (XG)

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]] # , 'RUNT' + t[6:]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRF_XG(dataset_target)

    # Búsqueda de hiperparámetros
    # Grid de hiperparámetros a evaluar
    # ==============================================================================
    param_grid = ParameterGrid({'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8], 'subsample': [0.4, 0.5, 0.6, 0.7, 0.8], 'gamma': [5, 10], 'max_depth': range(5, 25, 5), 'n_estimators': range(10, 60, 10), 'reg_lambda': [0, 1]})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:  # Se cambió objectcive por objective

        modeloXG = xgb.XGBRegressor(objective='reg:squarederror', random_state=1, **params)

        modeloXG.fit(X_train, Y_train)
        y_hat_train = modeloXG.predict(X_train)
        y_hat_valid = modeloXG.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        y_hat_train = np.asarray(y_hat_train)  # Se cambió np.matrix por np.asarray
        # y_hat_train=y_hat_train.T
        y_hat_valid = np.asarray(y_hat_valid)
        # y_hat_valid=y_hat_valid.T
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('rmse_valid', ascending=True)
    best = resultados.head(1)

    # Se ajusta el modelo XGBoost para RUNT Yamaha, con los mejores hiperparámetros encontrados anteriormente:
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid({'colsample_bytree': best['colsample_bytree'].values, 'subsample': best['subsample'].values,
                                'gamma': best['gamma'].values, 'max_depth': [int(best['max_depth'].values)],
                                'n_estimators': [int(best['n_estimators'].values)],
                                'reg_lambda': best['reg_lambda'].values})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:

        modeloXG = xgb.XGBRegressor(objective='reg:squarederror', random_state=1, **params)

        modeloXG.fit(X_train, Y_train)
        y_hat_train = modeloXG.predict(X_train)
        y_hat_valid = modeloXG.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        y_hat_train = np.asarray(y_hat_train)
        # y_hat_train=y_hat_train.T
        y_hat_valid = np.asarray(y_hat_valid)
        # y_hat_valid=y_hat_valid.T
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('rmse_valid', ascending=True)
    resultados.head(4)

    Y_train = Y_train.reshape([-1, 1])
    y_hat_train = y_hat_train.reshape([-1, 1])
    Y_valid = Y_valid.reshape([-1, 1])
    y_hat_valid = y_hat_valid.reshape([-1, 1])
    Y_test = Y_test.reshape([-1, 1])
    y_hat_test = modeloXG.predict(X_test)
    y_hat_test = y_hat_test.reshape([-1, 1])

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid, y_hat_valid), mean_absolute_percentage_error(Y_valid, y_hat_valid)]
    # errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
    np.save('error_XG_actual_' + t[5:] + '.npy', errores)
    # modelo = np.load('error_XG_actual_Yamaha.npy')

    # Se guarda el modelo en la misma carpeta del proyecto
    joblib.dump(modeloXG, 'modeloXG_' + t[5:] + '.pkl')

#########################################################################################################################################################################################

# PARTE B. Entrenamiento con variables rezagadas (t-12): PRONÓSTICO EN LOTE.

def dataset_rezagado(df, t):
    df2 = df.copy()
    df2 = df2.drop(['IEC', 'ICE'], axis=1)
    target_sin_rezago = df2[t]
    lista = [12, 12, 12, 12, 12, 0, 12]  # Se define un rezago de 12 meses para cada una de las variables, excepto para RATIO_DH_F.
    df3 = delay(df2, lista)
    count = 0
    for i in lista:
        df3.rename(columns={df3.columns[count]: (df3.columns[count] + '_' + str(i))}, inplace=True)  # agregar indicador de retardo
        count += 1
    df3 = pd.concat([df3, target_sin_rezago], axis=1)
    df3 = df3.dropna()
    df3 = df3.reset_index(drop=True)
    return df3

# 1. Modelo Redes Neuronales (RN)

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]] # , 'RUNT' + t[6:]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    dataset_r = dataset_rezagado(dataset_target, t)
    #print(dataset_r)

    scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRN(dataset_r)
    #print(preprocesamientoRN(dataset_r))

    hist_r, modeloRN_r = entrenamientoRN(X_train, Y_train, X_valid, Y_valid)

    # Estimaciones
    Y_hat_train = modeloRN_r.predict(X_train);
    Y_hat_train = scaler.inverse_transform(Y_hat_train);
    Y_hat_valid = modeloRN_r.predict(X_valid);
    Y_hat_valid = scaler.inverse_transform(Y_hat_valid);
    Y_hat_test = modeloRN_r.predict(X_test);
    Y_hat_test = scaler.inverse_transform(Y_hat_test);
    # Reales
    Y_train_normal = scaler.inverse_transform(Y_train)
    Y_valid_normal = scaler.inverse_transform(Y_valid)
    Y_test_normal = scaler.inverse_transform(Y_test)

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid_normal, Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal, Y_hat_valid)]
    # errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
    np.save('error_RNN_rez_' + t[5:] + '.npy', errores)
    # modelo = np.load('error_RN_rez_Yamaha.npy')

    # Se guarda el modelo el modelo en la misma carpeta del proyecto
    modeloRN_r.save('modeloRN_r_' + t[5:] + '.h5')

# 2. Modelo Random Forest (RF)

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]] # 'RUNT' + t[6:]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    dataset_r = dataset_rezagado(dataset_target, t)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRF_XG(dataset_r)

    # Búsqueda de hiperparámetros
    # Grid de hiperparámetros a evaluar
    # ==============================================================================
    param_grid = ParameterGrid({'n_estimators': range(20, 41, 1), 'max_features': ['auto', 'sqrt'], 'max_depth': [20]})
    # param_grid = ParameterGrid({'n_estimators': range(20,41,1), 'max_features': ['auto','sqrt'], 'max_depth': range(20,41,1)}) # Probar con esto

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:

        modeloRF_r = RandomForestRegressor(n_jobs=-1, random_state=1, criterion='squared_error', bootstrap=True, **params)

        modeloRF_r.fit(X_train, Y_train)
        y_hat_train = modeloRF_r.predict(X_train)
        y_hat_valid = modeloRF_r.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('mape_valid', ascending=True)
    best = resultados.head(1)
    print(best)

    # Se ajusta el modelo Random Forest para RUNT Yamaha, con los mejores hiperparámetros encontrados anteriormente:
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid({'n_estimators': best['n_estimators'].values, 'max_features': best['max_features'].values, 'max_depth': best['max_depth'].values})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': []}

    for params in param_grid:

        modeloRF_r = RandomForestRegressor(n_jobs=-1, random_state=1, criterion='squared_error', bootstrap=True, **params)

        modeloRF_r.fit(X_train, Y_train)
        y_hat_train = modeloRF_r.predict(X_train)
        y_hat_valid = modeloRF_r.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('mape_valid', ascending=True)
    resultados.head(4)

    Y_train = Y_train.reshape([-1, 1])
    y_hat_train = y_hat_train.reshape([-1, 1])
    Y_valid = Y_valid.reshape([-1, 1])
    y_hat_valid = y_hat_valid.reshape([-1, 1])
    Y_test = Y_test.reshape([-1, 1])
    y_hat_test = modeloRF_r.predict(X_test)
    y_hat_test = y_hat_test.reshape([-1, 1])

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid, y_hat_valid), mean_absolute_percentage_error(Y_valid, y_hat_valid)]
    # errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
    np.save('error_RF_rez_' + t[5:] + '.npy', errores)
    #modelo = np.load('error_RF_rez_' + t[5:] + '.npy')
    # print(modelo[0])
    # print(modelo[1])

    # Se guarda el modelo en la misma carpeta del proyecto
    joblib.dump(modeloRF_r, 'modeloRF_r_' + t[5:] + '.pkl')
    # pickle.dump(modeloRF_r, 'modeloRF_r_' + t[5:] + '.pkl')

# 3. Modelo XGBoost (XG)

for t in target_list:

    df_aux = df[['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'ICC', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', t]] # 'RUNT' + t[6:]

    dataset = df_aux.dropna()

    # Se crea una nueva columna con la ponderación de días hábiles entre festivos
    dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
    dataset = dataset.drop(['FESTIVOS'], axis=1)
    dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # La variable RATIO_DH_F tiene estacionalidad ## , 'RUNT' + t[6:]: t

    dataset_target = dataset[['DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'ICE', 'PRECIO PETROLEO WTI', 'RATIO_DH_F', t]]

    dataset_r = dataset_rezagado(dataset_target, t)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocesamientoRF_XG(dataset_r)

    # Búsqueda de hiperparámetros
    # Grid de hiperparámetros a evaluar
    # ==============================================================================
    param_grid = ParameterGrid({'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8], 'subsample': [0.4, 0.5, 0.6, 0.7, 0.8], 'gamma': [5, 10], 'max_depth': range(5, 25, 5), 'n_estimators': range(10, 60, 10), 'reg_lambda': [0, 1]})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': [], 'mape_test': []}

    for params in param_grid:

        modeloXG_r = xgb.XGBRegressor(objective='reg:squarederror', random_state=1, **params)

        modeloXG_r.fit(X_train, Y_train)
        y_hat_train = modeloXG_r.predict(X_train)
        y_hat_valid = modeloXG_r.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        y_hat_train = np.asarray(y_hat_train)
        # y_hat_train=y_hat_train.T
        y_hat_valid = np.asarray(y_hat_valid)
        # y_hat_valid=y_hat_valid.T
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)
        y_hat_test = modeloXG_r.predict(X_test)
        mape_test = MAPE(Y_test, y_hat_test)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)
        resultados['mape_test'].append(mape_test)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('rmse_valid', ascending=True)
    best = resultados.head(1)

    # Se ajusta el modelo XGBoost para RUNT Yamaha, con los mejores hiperparámetros encontrados anteriormente:
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid({'colsample_bytree': best['colsample_bytree'].values, 'subsample': best['subsample'].values,
                                'gamma': best['gamma'].values, 'max_depth': [int(best['max_depth'].values)],
                                'n_estimators': [int(best['n_estimators'].values)],
                                'reg_lambda': best['reg_lambda'].values})

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train': [], 'mae_valid': [], 'mape_train': [], 'mape_valid': [], 'mape_test': []}

    for params in param_grid:

        modeloXG_r = xgb.XGBRegressor(objective='reg:squarederror', random_state=1, **params)

        modeloXG_r.fit(X_train, Y_train)
        y_hat_train = modeloXG_r.predict(X_train)
        y_hat_valid = modeloXG_r.predict(X_valid)
        mse_train = mean_squared_error(Y_train, y_hat_train)
        mse_valid = mean_squared_error(Y_valid, y_hat_valid)
        mae_train = mean_absolute_error(Y_train, y_hat_train)
        mae_valid = mean_absolute_error(Y_valid, y_hat_valid)
        y_hat_train = np.asarray(y_hat_train)
        # y_hat_train=y_hat_train.T
        y_hat_valid = np.asarray(y_hat_valid)
        # y_hat_valid=y_hat_valid.T
        mape_train = MAPE(Y_train, y_hat_train)
        mape_valid = MAPE(Y_valid, y_hat_valid)
        y_hat_test = modeloXG_r.predict(X_test)
        mape_test = MAPE(Y_test, y_hat_test)

        resultados['params'].append(params)
        resultados['rmse_train'].append(math.sqrt(mse_train))
        resultados['rmse_valid'].append(math.sqrt(mse_valid))
        resultados['mae_train'].append(mae_train)
        resultados['mae_valid'].append(mae_valid)
        resultados['mape_train'].append(mape_train)
        resultados['mape_valid'].append(mape_valid)
        resultados['mape_test'].append(mape_test)

        # print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('rmse_valid', ascending=True)
    resultados.head(4)

    Y_train = Y_train.reshape([-1, 1])
    y_hat_train = y_hat_train.reshape([-1, 1])
    Y_valid = Y_valid.reshape([-1, 1])
    y_hat_valid = y_hat_valid.reshape([-1, 1])
    Y_test = Y_test.reshape([-1, 1])
    y_hat_test = modeloXG_r.predict(X_test)
    y_hat_test = y_hat_test.reshape([-1, 1])

    #t2 = t[7:]

    errores = [mean_absolute_error(Y_valid, y_hat_valid), mean_absolute_percentage_error(Y_valid, y_hat_valid)]
    # errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
    np.save('error_XG_rez_' + t[5:] + '.npy', errores)
    # modelo = np.load('error_XG_rez_Yamaha.npy')
    # print(modelo[0])
    # print(modelo[1])

    # Se guarda el modelo en la misma carpeta del proyecto
    joblib.dump(modeloXG_r, 'modeloXG_r_' + t[5:] + '.pkl')