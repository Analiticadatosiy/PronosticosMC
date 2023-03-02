# Se importan las librerías necesarias

import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import base64
import plotly.express as px
from dateutil.relativedelta import relativedelta
import datetime
import time
import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import base64
import plotly.express as px
from dateutil.relativedelta import relativedelta
import datetime
from datetime import datetime as dt
import time

# Se configuran los atributos de estilo de la app
st.set_page_config(
  layout="wide",  # Puede tomar los valores: "centered" o "wide" (o "dashboard", en el futuro), entre otros.
  initial_sidebar_state="auto",  # Puede tomar los valores "auto", "expanded" o "collapsed".
  page_title='Pronósticos',  # Acepta los valores String o None. Los String se agregan con "• Streamlit".
  page_icon=None,  # Acepta String, cualquier cosa compatible con st.image, o None.
)

# Se importa el dataframe
df_runt_total = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name='RUNT_Total', converters={'TRM': int, 'SMMLV&AUXTTE': int, 'DIAS HABILES': int, 'FESTIVOS': int, 'RUNT MERCADO': int, 'RUNT YAMAHA': int}) #***
#df_runt_categoria_yamaha = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name='RUNT_Categoria_Yamaha') #***
df_runt_modelo_yamaha = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name='RUNT_Modelo_Yamaha', converters={'TRM': int, 'SMMLV&AUXTTE': int, 'DIAS HABILES': int, 'FESTIVOS': int, 'RUNT MERCADO': int, 'RUNT YAMAHA': int,
                                      'RUNT NMAX': int, 'RUNT NMAX CONNECTED': int, 'RUNT CRYPTON FI': int, 'RUNT XTZ125': int, 'RUNT XTZ150': int, 'RUNT XTZ250': int, 'RUNT MT03': int, 'RUNT FZ25': int, 'RUNT FZ15': int,
                                      'RUNT SZ15RR': int, 'RUNT YBRZ125': int, 'RUNT YCZ110': int, 'RUNT XMAX': int}) #***
#df_runt_marca_mercado = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name='RUNT_Marca_Mercado') #***
#df_runt_categoria_mercado = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name='RUNT_Categoria_Mercado') #***

# La siguiente función de escalamiento es transversal a todos los pronósticos que se deriven de redes neuronales:
def preprocesamientoRN(df):
  df = pd.DataFrame(df)
  X = df.iloc[:, 0: df.shape[1]-1]
  Y = df.iloc[:, df.shape[1]-1:]
  #X = df[:, 0: df.shape[1]-1]
  #Y = df[:, df.shape[1]-1:]
  min_max_scaler = preprocessing.MinMaxScaler([-1,1]) # OPORTUNIDAD DE MEJORA 1: Ensayar otros métodos de normalización (o estandarización): Normalizer, StandardScaler, RobustScaler, entre otros.
  X_scale = min_max_scaler.fit_transform(X)
  Y_scale = min_max_scaler.fit_transform(Y)
  return min_max_scaler, X_scale, Y_scale

# La siguiente función es transversal a todos los modelos; permite extraer el target de interés.
def target(dataframe, lista): # lista contiene el nombre de las columnas que son target
  df = dataframe.dropna();
  df_target = pd.DataFrame(columns=lista)
  for i in range(0, len(lista), 1):
    n = df[lista[i]].copy();
    df_target.iloc[:, i] = n;
  return df_target

###########################################################################################################################################

# PARTE 1: A continuación, se define la función que hará la limpieza del dataframe original para los siguientes casos de pronóstico:
# Demanda: Yamaha ó Mercado
# Dinámica: Suponiendo indicadores económicos (actual)
# Alcance: Predicción de un solo mes (individual) ó Predicción de varios meses (lote)
# Esta limpieza se aplica en las funciones: actual_individual (Y/M) y actual_lote (Y/M)

def limpieza_actual(df, *args):
  # Se eliminan los valores nulos correspondientes a las primeras 10 filas (enero a octubre de 2001: sin registro de ICC, IEE Y ICE)
  df = df.dropna()
  # Se reestablece el índice del dataframe a la indexación predeterminada: 0 a (No.filas-1)
  df = df.reset_index(drop=True)
  # Se devuelve una cadena de caracteres con el formato especificado: DD/MM/AAAA, para la columna FECHA.
  df.FECHA = df.FECHA.apply(lambda x: x.strftime('%d/%m/%Y'))
  # Se crea una copia de respaldo del dataframe
  dataset = df.copy()
  # Se eliminan las columnas 'CRECIMIENTO PIB','ISM','ICC','MES','TEMPORADA','VACACIONES','CLIMA','INGRESOS','PRECIOS','IMPUESTOS','WHOLESALE', del dataset original.
  # OPORTUNIDAD DE MEJORA 2: Se deberían incluir éstas u otras variables en el análisis? OJO: En el dataset, IEE corresponde a IEC.
  #dataset = dataset.drop(['CRECIMIENTO PIB','ISM','ICC','MES','TEMPORADA','VACACIONES','CLIMA','INGRESOS','PRECIOS','IMPUESTOS','WHOLESALE', *args], axis=1)
  dataset = dataset.drop(['ICC', *args], axis=1)
  # Se crea una copia de dataset, llamada dataset2
  dataset2 = dataset.copy()
  dataset2.set_index('FECHA', inplace=True) # dataset2 tiene como índice las fechas
  # Se reemplazarán las columnas DIAS HABILES y FESTIVOS por una ponderación entre los días hábiles y los festivos,
  # para no castigar tan duramente el pronóstico en los meses con mayor cantidad de días festivos (las tiendas no abren o no son tan visitadas)
  dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
  # Se elimina la columna FESTIVOS; esta columna contiene la cantidad de festivos que tiene cada mes
  dataset = dataset.drop(['FESTIVOS'], axis=1)
  dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True) #Esta variable tiene estacionalidad
  # OPORTUNIDAD DE MEJORA 3: Se puede modelar el problema de otra manera con base a este hallazgo?
  # Se crea una copia de dataset, llamada dataset1
  dataset1 = dataset.copy() # dataset1 tiene como índice números enteros
  dataset1 = dataset1.drop(['FECHA'], axis=1) # Se elimina la columna fecha
  numpy1 = dataset1.values
  return dataset1, numpy1, dataset2

# A continuación, se definen las funciones de pronóstico para RUNT Yamaha y RUNT Mercado, suponiendo indicadores económicos, con predicción de un sólo mes: actual_individual.

def actual_individual(selectbox1, selectbox4): # Predicción de un sólo mes. Esta función le solicita al usuario seleccionar la fecha a estimar (año y mes),
  # e ingresar los valores de los indicadores económicos: DESEMPLEO, INFLACION, TRM, SMMLV&AUXTTE, IEC, ICE, PRECIO PETROLEO WTI, DIAS HABILES y FESTIVOS.

  if (selectbox1 == 'Yamaha') & (selectbox4 == 'Total'): # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.
    variable_modelo = 'YAMAHA'
    variable_df = 'RUNT MERCADO'
    visualizar = 'RUNT YAMAHA'

    # Se definen el encabezado de la página y las instrucciones para el usuario.
    st.subheader('Estimar demanda a un sólo mes: ' + selectbox1 + ' - ' + selectbox1 + '.')
    st.write('Por favor, ingrese para cada variable el valor que supone tendría en el tiempo futuro en el que desea estimar la demanda.\n Tenga en cuenta que el dato de festivos y días hábiles corresponde al valor real del mes en cuestión que quiere proyectar.')
    st.write('Puede guiarse de los últimos 6 datos de la tabla para que le sirvan de ejemplo y guía de cómo debe ingresar los datos supuestos.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_actual(df_runt_total, 'RUNT MERCADO')

    # Se imprimen los últimos 6 registros de la tabla
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    col1, col2 = st.columns(2)
    ini = 2023

    # El usuario selecciona el año y el mes de pronóstico
    year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini + 11))
    month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))

    # El usuario ingresa por teclado los valores que él supone van a tener las variables económicas en el mes de pronóstico,
    # y cada uno de estos valores se asignan a la nueva variable relacionada.
    # OPORTUNIDAD DE MEJORA 4: Indicarle explícitamente al usuario, por pantalla, el tipo de formato que debe usar para ingresar
    # números enteros (sin puntos ni comas) y decimales (con comas).
    DESEMPLEO = st.number_input("Desempleo", format="%.3f")
    INFLACION = st.number_input("Inflación", format="%.3f")
    TRM = st.number_input("Tasa de cambio representativa del mercado (TRM)", format="%.2f")
    SMMLV_AUXTTE = st.number_input("Salario mínimo más auxilio de transporte (SMMLV&AUXTTE)", format="%.0f")
    IEC = st.number_input("Indice de expectativas del consumidor (IEC)", format="%.2f")
    ICE = st.number_input("Indice de condiciones económicas (ICE)", format="%.2f")
    PRECIO_PETROLEO_WTI = st.number_input("Precio del crudo WTI (en dólares)", format="%.3f")
    DIAS_HABILES = st.number_input("Dias hábiles", format="%.0f")
    FESTIVOS = st.number_input("Festivos", format="%.0f")

    # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
    # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
    if (st.button('Pronosticar')):

      modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')
      modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
      modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

      # Se preparan los datos para entrenar los modelos
      # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
      X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, IEC, ICE, PRECIO_PETROLEO_WTI, (DIAS_HABILES)/(DIAS_HABILES+FESTIVOS)]])
      # Se asigna un valor semilla (igual a 8000) a la variable RUNT YAMAHA.
      X_RN = np.concatenate([numpy1, np.reshape(np.append(X, [8000]), (1, -1))]) #AJUSTAR ESTE VALOR SEMILLA

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho (por el modelo de redes neuronales) para RUNT YAMAHA, escalado entre -1 y 1.
      y_hat_scale = modeloRN.predict(np.reshape(X_scale[-1], (1, -1)))
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
      y_hat_RF = modeloRF.predict(X)

      # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
      # XGBoost
      y_hat_XG = modeloXG.predict(X)

      # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
      # OPORTUNIDAD DE MEJORA 5: Se puede calcular un promedio ponderado? Los pesos los puede dar el experto de negocio o el mismo algoritmo?
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      index = ['1/' + str(month) + '/' + str(year)]

      # Se almacenan los resultados en un dataframe y se imprimen en pantalla
      resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
      st.write(resultados)

      # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
      # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
      # por el script generador_modelos_definitivo.py, en la misma carpeta del proyecto.
      errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
      errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
      errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

      # Se almacenan los errores en un dataframe y se imprimen en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

  elif selectbox1 == 'Mercado': # Los nombres de la variables que se recibirán por el menú: ¿Qué demanda desea estimar?.
    variable_modelo = 'MERCADO'
    variable_df = 'RUNT YAMAHA'
    visualizar = 'RUNT MERCADO'

    # Se definen el encabezado de la página y las instrucciones para el usuario.
    st.subheader('Estimar demanda a un sólo mes: ' + selectbox1 + '.')
    st.write('Por favor, ingrese para cada variable el valor que supone tendría en el tiempo futuro en el que desea estimar la demanda.\n Tenga en cuenta que el dato de festivos y días hábiles corresponde valor real del mes en cuestión que quiere proyectar.')
    st.write('Puede guiarse de los últimos 6 datos de la tabla para que le sirvan de ejemplo y guía de cómo debe ingresar los datos supuestos.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_actual(df_runt_total, 'RUNT YAMAHA')

    # Se cambian los nombres de las columnas en la tabla de datos para mejorar la presentación al usuario
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    col1, col2 = st.columns(2)
    ini = 2023

    # El usuario selecciona el año y el mes de pronóstico
    year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini + 11))
    month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))

    # El usuario ingresa por teclado los valores que él supone van a tener las variables económicas en el mes de pronóstico,
    # y cada uno de estos valores se asignan a la nueva variable relacionada.
    # OPORTUNIDAD DE MEJORA 4: Indicarle explícitamente al usuario, por pantalla, el tipo de formato que debe usar para ingresar
    # números enteros (sin puntos ni comas) y decimales (con comas).
    DESEMPLEO = st.number_input("Desempleo", format="%.3f")
    INFLACION = st.number_input("Inflación", format="%.3f")
    TRM = st.number_input("Tasa de cambio representativa del mercado (TRM)", format="%.2f")
    SMMLV_AUXTTE = st.number_input("Salario mínimo más auxilio de transporte (SMMLV&AUXTTE)", format="%.0f")
    IEC = st.number_input("Indice de expectativas del consumidor (IEC)", format="%.2f")
    ICE = st.number_input("Indice de condiciones económicas (ICE)", format="%.2f")
    PRECIO_PETROLEO_WTI = st.number_input("Precio del crudo WTI (en dólares)", format="%.3f")
    DIAS_HABILES = st.number_input("Dias hábiles", format="%.0f")
    FESTIVOS = st.number_input("Festivos", format="%.0f")

    # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
    # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
    if (st.button('Pronosticar')):
      modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')
      modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
      modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

      # Se preparan los datos para entrenar los modelos
      # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
      X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, IEC, ICE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS)]])
      # Se asigna un valor semilla (igual a 50000) a la variable RUNT MERCADO.
      X_RN = np.concatenate([numpy1, np.reshape(np.append(X, [50000]), (1, -1))])  # OJO: NO ME CUADRA ESTE DATO

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho (por el modelo de redes neuronales) para RUNT YAMAHA/MERCADO, escalado entre -1 y 1.
      y_hat_scale = modeloRN.predict(np.reshape(X_scale[-1], (1, -1)))
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA/MERCADO.
      y_hat_RF = modeloRF.predict(X)

      # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA/MERCADO.
      # XGBoost
      y_hat_XG = modeloXG.predict(X)

      # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
      # OPORTUNIDAD DE MEJORA 5: Se puede calcular un promedio ponderado? Los pesos los puede dar el experto de negocio o el mismo algoritmo?
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      index = ['1/' + str(month) + '/' + str(year)]

      # Se almacenan los resultados en un dataframe y se imprimen en pantalla
      resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
      st.write(resultados)

      # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
      # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
      # por el script generador_modelos_definitivo.py, en la misma carpeta del proyecto.
      errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
      errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
      errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

      # Se almacenan los errores en un dataframe y se imprimen en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

# A continuación, se definen las funciones de pronóstico para RUNT Yamaha y RUNT Mercado, suponiendo indicadores económicos, con predicción de varios meses: actual_lote.

def actual_lote(selectbox1, selectbox4): # Predicción de varios meses. Esta función le solicita al usuario cargar en una plantilla
  # los valores que tomarán las siguientes variables: FECHA, DESEMPLEO, INFLACION, TRM, SMMLV&AUXTTE, IEC, ICE, PRECIO PETROLEO WTI, DIAS HABILES y FESTIVOS, para los n meses de pronóstico.

  if (selectbox1 == 'Yamaha') & (selectbox4 == 'Total'): # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.
    variable_modelo = 'YAMAHA'
    variable_df = 'RUNT MERCADO'
    visualizar = 'RUNT YAMAHA'

    st.subheader('Estimar demanda para varios meses: ' + selectbox1 + ' - ' + selectbox4 + '.')
    st.write('Por favor suba un archivo con los valores que supone que tendrán las variables en el horizonte futuro, tenga en cuenta que debe tener las mismas variables y en el mismo orden de la tabla.')
    st.write('Para facilitar el cargue de los datos, utilice y descargue la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_actual(df_runt_total, 'RUNT MERCADO')

    # Se imprimen los últimos 6 registros de la tabla
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    # Se carga la plantilla para que el usuario la descargue
    with open("plantilla_lote.xlsx", "rb") as file:
      btn = st.download_button(
        label = "Descargue plantilla",
        data = file,
        file_name = "plantilla.xlsx",
        mime = "image/png")

    # Se sube la plantilla diligenciada por el usuario, y se almacena como un archivo de extensión xlsx en la variable data_file.
    st.markdown('**Subir plantilla**')
    st.write("Warning: El archivo que suba debe tener extensión 'xlsx'.")
    data_file = st.file_uploader('Archivo', type=['xlsx'])

    if data_file is not None: # Este bloque de código se ejecuta, una vez el usuario haya subido el archivo.

      # Se lee el archivo y se almacena en df_p
      df_p = pd.read_excel(data_file)

      # Se almacena la columna FECHA, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
      index_pron = df_p['FECHA']
      # Se elimina la columna Fecha
      df_p = df_p.drop(['FECHA'], axis=1)

      # Se almacena el número de columnas del dataset, en la variable auxiliar columns, para construir la columna RATIO_DH_F (registro a registro).
      columns = df_p.shape[1]
      # Se crea la columna RATIO_DH_F
      df_p.iloc[:, columns - 2] = df_p.iloc[:, columns - 2] / (df_p.iloc[:, columns - 2] + df_p.iloc[:, columns - 1])

      # Se rellena la última columna RUNT YAMAHA con el valor semilla de 8000, para los meses de pronóstico ingresados por el usuario.
      for i in range(0, df_p.shape[0], 1):
        df_p.iloc[i, columns - 1] = 8000 #AJUSTAR ESTE VALOR

      # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost
      # que fueron guardados por generador_modelos_definitivo.py en formatos h5 y pkl, respectivamente, en la misma carpeta del proyecto.
      # Aquí variable_modelo es igual a 'yamaha' o 'mercado', dependiendo del tipo de demanda que desee estimar el usuario.
      modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')  # (variable_modelo)
      modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
      modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

      # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
      X = df_p.values
      X_RN = np.concatenate([numpy1, X])

      # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 8000/50000 para RUNT YAMAHA/MERCADO
      X = X[:, 0: df_p.shape[1] - 1]

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA.
      y_hat_scale = modeloRN.predict(X_scale[(len(X_RN) - len(X)):, :])
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      y_hat_RF = modeloRF.predict(X)

      # XGBoost
      y_hat_XG = modeloXG.predict(X)

      # Promedio
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      st.markdown('**Pronóstico**')

      # Se almacenan los resultados en un dataframe
      resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
      resultados['Fecha'] = resultados.Fecha.apply(lambda x: x.strftime('%d/%m/%Y'))
      resultados.set_index('Fecha', inplace=True)

      # Se habilita la descarga de los resultados para el usuario
      resultados.to_excel('pronosticos.xlsx', index=True)
      with open("pronosticos.xlsx", "rb") as file:
        btn = st.download_button(
          label = "Descargar pronosticos",
          data = file,
          file_name = "Pronosticos.xlsx",
          mime = "image/png")
      st.write(resultados)

      # Se cargan los errores generados por el script generador_modelos_definitivo, desde la carpeta del proyecto.
      errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
      errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
      errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

      # Se imprimen los errores en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

      # Se grafican los resultados de pronóstico
      graficar=dataset2[-60:]
      total=pd.concat([graficar[visualizar], resultados])
      total.rename(columns={0: visualizar}, inplace=True) #esta variable tiene estacionalidad
      total=total.reset_index()
      df_melt = total.melt(id_vars='index', value_vars=[visualizar,'Redes Neuronales','Random Forest','XGBoost','Promedio'])
      px.defaults.width = 1100
      px.defaults.height = 500
      fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA",  "value": "RUNT"})
      st.plotly_chart(fig)

  elif selectbox1 == 'Mercado':  # El nombre de la variable que se recibirá por el menú ¿Qué demanda desea estimar? de Streamlit
    variable_modelo = 'MERCADO'
    variable_df = 'RUNT YAMAHA'
    visualizar = 'RUNT MERCADO'

    st.subheader('Estimar demanda para varios meses: ' + selectbox1 + '.')
    st.write('Por favor suba un archivo con los valores que supone que tendrán las variables en el horizonte futuro, tenga en cuenta que debe tener las mismas variables y en el mismo orden de la tabla.')
    st.write('Para facilitar el cargue de los datos, utilice y descargue la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_actual(df_runt_total, 'RUNT YAMAHA')

    # Se imprimen los últimos 6 registros de la tabla
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    # Se carga la plantilla para que el usuario la descargue
    with open("plantilla_lote.xlsx", "rb") as file:
      btn = st.download_button(
        label = "Descargue plantilla",
        data = file,
        file_name = "plantilla.xlsx",
        mime = "image/png")

    # Se sube la plantilla diligenciada por el usuario, y se almacena como un archivo de extensión xlsx en la variable data_file.
    st.markdown('**Subir plantilla**')
    st.write("Warning: El archivo que suba debe tener extensión 'xlsx'.")
    data_file = st.file_uploader('Archivo', type=['xlsx'])

    if data_file is not None:  # Después de que el usuario suba el archivo, se ejecuta este bloque de código.

      # Se lee el archivo y se almacena en df_p
      df_p = pd.read_excel(data_file)

      # Se almacena la columna FECHA, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
      index_pron = df_p['FECHA']
      # Se elimina la columna Fecha
      df_p = df_p.drop(['FECHA'], axis=1)

      # Se almacena el número de columnas del dataset, en la variable auxiliar columns, para construir la columna RATIO_DH_F (registro a registro).
      columns = df_p.shape[1]
      # Se crea la columna RATIO_DH_F
      df_p.iloc[:, columns - 2] = df_p.iloc[:, columns - 2] / (df_p.iloc[:, columns - 2] + df_p.iloc[:, columns - 1])

      # Se rellena la última columna RUNT MERCADO con el valor semilla de 50000, para los meses de pronóstico ingresados por el usuario.
      for i in range(0, df_p.shape[0], 1):
        df_p.iloc[i, columns - 1] = 50000

      # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost
      # que fueron guardados por generador_modelos_definitivo.py en formatos h5 y pkl, respectivamente, en la misma carpeta del proyecto.
      modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')
      modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
      modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

      # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
      X = df_p.values
      X_RN = np.concatenate([numpy1, X])

      # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 50000 para RUNT MERCADO
      X = X[:, 0: df_p.shape[1] - 1]

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT MERCADO.
      y_hat_scale = modeloRN.predict(X_scale[(len(X_RN) - len(X)):, :])
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      y_hat_RF = modeloRF.predict(X)

      # XGBoost
      y_hat_XG = modeloXG.predict(X)

      # Promedio
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      st.markdown('**Pronóstico**')

      # Se almacenan los resultados en un dataframe
      resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
      resultados['Fecha'] = resultados.Fecha.apply(lambda x: x.strftime('%d/%m/%Y'))
      resultados.set_index('Fecha', inplace=True)

      # Se habilita la descarga de los resultados para el usuario
      resultados.to_excel('pronosticos.xlsx', index=True)
      with open("pronosticos.xlsx", "rb") as file:
        btn = st.download_button(
          label = "Descargar pronosticos",
          data = file,
          file_name = "Pronosticos.xlsx",
          mime = "image/png")
      st.write(resultados)

      # Se cargan los errores generados por el script generador_modelos_definitivo, desde la carpeta del proyecto.
      errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
      errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
      errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

      # Se imprimen los errores en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

      # Se grafican los resultados de pronóstico
      graficar = dataset2[-60:]
      total = pd.concat([graficar[visualizar], resultados])
      total.rename(columns={0: visualizar}, inplace=True)  # Esta variable tiene estacionalidad
      total = total.reset_index()
      df_melt = total.melt(id_vars='index', value_vars=[visualizar, 'Redes Neuronales', 'Random Forest', 'XGBoost', 'Promedio'])
      px.defaults.width = 1100
      px.defaults.height = 500
      fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
      st.plotly_chart(fig)

############################################################################################################################################

# PARTE 2: A continuación, se define la función de limpieza para rezago_yamaha y rezago_yamaha_lote, y, rezago_mercado y rezago_mercado_lote,
# para los siguientes casos de pronóstico:
# Demanda: Yamaha o Mercado
# Dinámica: Con datos reales del último año
# Alcance: Predicción de un solo mes (rezago_yamaha o rezago_mercado) ó Predicción de varios meses (rezago_yamaha_lote o rezago_mercado_lote)
# Esta limpieza se aplica en las funciones: rezago_yamaha y rezago_yamaha_lote, y, rezago_mercado y rezago_mercado_lote

def limpieza_rezago(df, *args):
  df = df.reset_index(drop=True)
  dataset = df.copy()
  dataset = dataset.loc[:, [i for i in ['FECHA', 'DESEMPLEO', 'INFLACION', 'TRM', 'SMMLV&AUXTTE', 'IEC', 'PRECIO PETROLEO WTI', 'DIAS HABILES', 'FESTIVOS', *args]]]
  dataset2 = dataset.copy()
  dataset2.set_index('FECHA', inplace=True)  # dataset2 tiene como índice las fechas
  dataset['DIAS HABILES'] = dataset['DIAS HABILES'] / (dataset['DIAS HABILES'] + dataset['FESTIVOS'])
  dataset = dataset.drop(['FESTIVOS'], axis=1)
  dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # Esta variable tiene estacionalidad
  dataset1 = dataset.copy()
  dataset1 = dataset1.drop(['FECHA'], axis=1) # dataset1 tiene como índice número enteros
  dataset1 = dataset1.dropna()
  numpy1 = dataset1.values
  dataset2 = dataset2.dropna()
  return dataset1, numpy1, dataset2

# A continuación, se definen las funciones de pronóstico para RUNT Yamaha, con datos reales del último año,
# con predicción de un sólo mes: rezago_yamaha y con predicción de varios meses: rezago_yamaha_lote.

def rezago_yamaha(): # Predicción de un solo mes

  if (selectbox1 == 'Yamaha') & (selectbox4 == 'Total'): # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.

    # Se crea el subtítulo
    st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + ' - ' + selectbox4 + '.')
    # Se dan instrucciones (por pantalla) al usuario sobre los datos que debe ingresar y su formato
    st.write('Por favor, ingrese el dato de las variables un año atras del período que desea estimar, es decir, si desea estimar Junio de 2021, ingrese los datos de Junio de 2020 (para el caso de los días hábiles y festivos, sí se deben ingresar los valores reales para el mes que se desea pronosticar)')
    st.write('Tome como guía los valores y formatos de la tabla que se muestra a continuación.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_total, 'RUNT MERCADO', 'RUNT YAMAHA')

    # Se imprimen en pantalla los últimos 6 registros del dataset limpio
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    # Se capturan el año y mes de pronóstico que el usuario ingresa, y se incluye el rezago.
    col1, col2 = st.columns(2)
    ini = 2023
    year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini + 11))
    month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))
    day = 1

    date = datetime.datetime(year-1, month, day).strftime("%Y/%m/%d")

    # Se traen los datos de las variables un año atras del período que se desea estimar, y se da formato a los demás datos ingresados por el usuario.
    DESEMPLEO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'DESEMPLEO'].item()
    INFLACION = df_runt_total.loc[df_runt_total['FECHA'] == date, 'INFLACION'].item()
    TRM = df_runt_total.loc[df_runt_total['FECHA'] == date, 'TRM'].item()
    SMMLV_AUXTTE = df_runt_total.loc[df_runt_total['FECHA'] == date, 'SMMLV&AUXTTE'].item()
    PRECIO_PETROLEO_WTI = df_runt_total.loc[df_runt_total['FECHA'] == date, 'PRECIO PETROLEO WTI'].item()
    DIAS_HABILES = st.number_input("Dias hábiles (en el mes de pronóstico)", format="%.0f")
    FESTIVOS = st.number_input("Festivos (en el mes de pronóstico)", format="%.0f")
    RUNT_MERCADO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'RUNT MERCADO'].item()

    # Una vez el usuario dé clic en el botón Pronosticar, se cargan los 3 modelos generados por generador_modelos_pronosticos.py: Redes Neuronales, Random Forest y XGBoost.
    if (st.button('Pronosticar')):

      modeloRN_r_yamaha = keras.models.load_model('modeloRN_r_YAMAHA.h5')
      modeloRF_r_yamaha = joblib.load('modeloRF_r_YAMAHA.pkl')
      modeloXG_r_yamaha = joblib.load('modeloXG_r_YAMAHA.pkl')

      # Se preparan los datos para entrenar los modelos
      # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
      X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, PRECIO_PETROLEO_WTI, (DIAS_HABILES)/(DIAS_HABILES+FESTIVOS), RUNT_MERCADO]])
      # Se asigna un valor semilla (igual a 7000) a la variable RUNT YAMAHA.
      #X_RN = np.concatenate([numpy1, np.reshape(np.append(X, [7000]), (1, -1))])
      X_RN = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS), RUNT_MERCADO, np.reshape(7000, (1, -1))]])

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA
      y_hat_scale = modeloRN_r_yamaha.predict(np.reshape(X_scale[-1], (1, -1)))
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
      y_hat_RF = modeloRF_r_yamaha.predict(X)

      # XGBoost
      # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
      y_hat_XG = modeloXG_r_yamaha.predict(X)

      # Promedio
      # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      # Se almacenan los resultados en un dataframe
      index = ['1/' + str(month) + '/' + str(year)]
      resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})

      # Se imprimen los resultados en pantalla
      st.write(resultados)

      # Se cargan los errores generados por el script generador_modelos_pronosticos, desde la carpeta del proyecto.
      errores_RN = np.load('error_RNN_rez_YAMAHA.npy')
      errores_RF = np.load('error_RF_rez_YAMAHA.npy')
      errores_XG = np.load('error_XG_rez_YAMAHA.npy')

      # Se almacenan los errores en un dataframe y se imprimen en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

      # Se grafican los resultados de pronóstico
      y_hat = resultados.values
      y_hat = np.delete(y_hat, 3)
      index = ['Redes Neuronales', 'Random Forest', 'XGBoost']
      resultados2 = pd.DataFrame({'Resultados': y_hat}, index=index)

      promedio = resultados['Promedio'].values
      promedio = np.array([promedio, promedio, promedio])
      promedio = promedio.ravel()

def rezago_yamaha_lote(): # Predicción de varios meses

  if (selectbox1 == 'Yamaha') & (selectbox4 == 'Total'):  # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.

    st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + ' - ' + selectbox4 + '.')

    st.write('Si ingresa al aplicativo el 1er día del mes, realice su consulta después de las 4 PM, una vez se haya consolidado el RUNT del mes inmediatamente anterior.')

    st.write('Por favor, ingrese el año actual:')
    ANIO = st.number_input("Año actual:", value=2023)

    st.write('Por favor, ingrese el mes actual:')
    MES = st.number_input("Mes actual:", value=1)

    DIA = 1

    Nro_Meses = 12

    date_act = datetime.datetime(ANIO, MES, DIA).strftime("%Y/%m/%d")
    #st.write(date_act)
    date_ini = dt.strptime(date_act, "%Y/%m/%d") - pd.DateOffset(months=Nro_Meses)
    date_fin = dt.strptime(date_act, "%Y/%m/%d") + pd.DateOffset(months=Nro_Meses)
    date_ini = date_ini.strftime("%Y/%m/%d")
    #st.write(date_ini)
    #st.write(date_fin)

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_total, 'RUNT YAMAHA')
    #st.write(numpy1)

    # Se imprimen en pantalla los últimos 6 registros del dataset limpio
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    df_p = df_runt_total.loc[(df_runt_total['FECHA'] > date_ini) & (df_runt_total['FECHA'] <= date_act)]
    df_p = df_p.iloc[:, [0, 1, 2, 3, 4, 6, 8, 9, 10]].dropna() # No se incluye la columna 21: RUNT YAMAHA y, de los indicadores económicos, se conserva sólo IEC.

    index_pron = df_p['FECHA'] # Se almacena la columna Fecha, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
    df_p = df_p.drop(['FECHA'], axis=1) # Se elimina la columna Fecha
    df_p.iloc[:, 6] = df_p.iloc[:, 6] / (df_p.iloc[:, 6] + df_p.iloc[:, 7]) # Se crea la nueva columna RATIO_DH_F
    df_p.rename(columns = {'DIAS HABILES': 'RATIO_DH_F'}, inplace = True) # Se cambia el nombre de la nueva columna
    df_p = df_p.drop(['FESTIVOS'], axis=1) # Se elimina la columna Festivos
    #st.write(df_p)

    # Se rellena la última columna RUNT YAMAHA con el valor semilla de 7000, para los meses de rezago ingresados por el usuario.
    vector = pd.DataFrame((np.ones((df_p.shape[0], 1), dtype=int)) * (7000))
    #st.write(vector)
    df_p.insert(len(df_p.columns), 'YAMAHA SEED', vector.values)
    #st.write(df_p)

    # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
    # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
    modeloRN_r_yamaha = keras.models.load_model('modeloRN_r_YAMAHA.h5')
    modeloRF_r_yamaha = joblib.load('modeloRF_r_YAMAHA.pkl')
    modeloXG_r_yamaha = joblib.load('modeloXG_r_YAMAHA.pkl')

    # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
    X = df_p.values
    X_RN = np.concatenate([numpy1, X])  # Para RN

    # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 7000 para RUNT YAMAHA.
    X = X[:, 0: df_p.shape[1] - 1]  # Para RF y XG

    # Redes Neuronales
    # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
    scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
    # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA.
    y_hat_scale = modeloRN_r_yamaha.predict(X_scale[(len(X_RN) - len(X)):, :])
    # Se regresa a la escala original el valor predicho para RUNT YAMAHA
    y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

    # Random Forest
    # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
    y_hat_RF = modeloRF_r_yamaha.predict(X)

    # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
    # XGBoost
    y_hat_XG = modeloXG_r_yamaha.predict(X)

    # Promedio
    # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
    y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

    # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
    st.write('**Pronóstico**')
    resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
    resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
    resultados['Fecha'] = resultados['Fecha'].apply(lambda x: (x + relativedelta(months=+12)).strftime('%d/%m/%Y'))
    resultados.set_index('Fecha', inplace=True)
    resultados.to_excel('pronosticos.xlsx', index=True)

    # Se habilita al usuario la descarga de los pronósticos por pantalla
    with open("pronosticos.xlsx", "rb") as file:
      btn = st.download_button(
        label="Descargar pronosticos",
        data=file,
        file_name="Pronosticos.xlsx",
        mime="image/png")
    st.write(resultados)

    # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
    # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
    # por el script generador_modelos_pronosticos.py, en la misma carpeta del proyecto.
    errores_RN = np.load('error_RNN_rez_YAMAHA.npy')
    errores_RF = np.load('error_RF_rez_YAMAHA.npy')
    errores_XG = np.load('error_XG_rez_YAMAHA.npy')

    # Se almacenan los errores en un dataframe y se imprimen en pantalla
    errores = pd.DataFrame()
    errores['Errores'] = ['MAE', 'MAPE']
    errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
    errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
    errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
    errores.set_index('Errores', inplace=True)
    st.markdown('**Errores**')
    st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

    # Se grafican los resultados de pronóstico
    graficar = dataset2[-60:]
    total = pd.concat([graficar['RUNT YAMAHA'], resultados])
    total.rename(columns={0: 'RUNT YAMAHA'}, inplace=True)  # Esta variable tiene estacionalidad
    total = total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['RUNT YAMAHA', 'Redes Neuronales', 'Random Forest', 'XGBoost', 'Promedio'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
    st.plotly_chart(fig)

# A continuación, se definen las funciones de pronóstico para RUNT Mercado, con datos reales del último año,
# con predicción de un sólo mes: rezago_mercado y con predicción de varios meses: rezago_mercado_lote.

def rezago_mercado():

  if selectbox1 == 'Mercado': # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.

    # Se crea el subtítulo
    st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + '.')
    # Se dan instrucciones (por pantalla) al usuario sobre los datos que debe ingresar y su formato
    st.write('Por favor, ingrese el dato de las variables un año atras del período que desea estimar, es decir, si desea estimar Junio de 2021, ingrese los datos de Junio de 2020 (para el caso de los días hábiles y festivos, sí se deben ingresar los valores reales para el mes que se desea pronosticar)')
    st.write('Tome como guía los valores y formatos de la tabla que se muestra a continuación.')

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_total, 'RUNT MERCADO')

    # Se imprimen en pantalla los últimos 6 registros del dataset limpio
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    # Se capturan el año y mes de pronóstico que el usuario ingresa, y se incluye el rezago.
    col1, col2 = st.columns(2)
    ini = 2023
    year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini + 11))
    month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))
    day = 1

    date = datetime.datetime(year-1, month, day).strftime("%Y/%m/%d")

    # Se traen los datos de las variables un año atras del período que se desea estimar, y se da formato a los demás datos ingresados por el usuario.
    DESEMPLEO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'DESEMPLEO'].item()
    INFLACION = df_runt_total.loc[df_runt_total['FECHA'] == date, 'INFLACION'].item()
    TRM = df_runt_total.loc[df_runt_total['FECHA'] == date, 'TRM'].item()
    SMMLV_AUXTTE = df_runt_total.loc[df_runt_total['FECHA'] == date, 'SMMLV&AUXTTE'].item()
    PRECIO_PETROLEO_WTI = df_runt_total.loc[df_runt_total['FECHA'] == date, 'PRECIO PETROLEO WTI'].item()
    DIAS_HABILES = st.number_input("Dias hábiles (en el mes de pronóstico)", format="%.0f")
    FESTIVOS = st.number_input("Festivos (en el mes de pronóstico)", format="%.0f")
    RUNT_MERCADO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'RUNT MERCADO'].item()

    # Una vez el usuario dé clic en el botón Pronosticar, se cargan los 3 modelos generados por generador_modelos_pronosticos.py: Redes Neuronales, Random Forest y XGBoost.
    if (st.button('Pronosticar')):
      modeloRN_r_mercado = keras.models.load_model('modeloRN_r_MERCADO.h5')
      modeloRF_r_mercado = joblib.load('modeloRF_r_MERCADO.pkl')
      modeloXG_r_mercado = joblib.load('modeloXG_r_MERCADO.pkl')

      # Se preparan los datos para entrenar los modelos
      # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
      X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS), RUNT_MERCADO]])
      #st.write(X)
      # # Se asigna un valor semilla (igual a 7000) a la variable RUNT YAMAHA.
      X_RN = np.concatenate([numpy1, np.reshape(np.append(X, [7000]), (1, -1))])

      # Redes Neuronales
      # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
      scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
      # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA
      y_hat_scale = modeloRN_r_mercado.predict(np.reshape(X_scale[-1], (1, -1)))
      # Se regresa a la escala original el valor predicho para RUNT YAMAHA
      y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

      # Random Forest
      # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
      y_hat_RF = modeloRF_r_mercado.predict(X)

      # XGBoost
      # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
      y_hat_XG = modeloXG_r_mercado.predict(X)

      # Promedio
      # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
      y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

      # Se almacenan los resultados en un dataframe
      index = ['1/' + str(month) + '/' + str(year)]
      resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
      resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})

      # Se imprimen los resultados en pantalla
      st.write(resultados)

      # Se cargan los errores generados por el script generador_modelos_pronosticos, desde la carpeta del proyecto.
      errores_RN = np.load('error_RNN_rez_MERCADO.npy')
      errores_RF = np.load('error_RF_rez_MERCADO.npy')
      errores_XG = np.load('error_XG_rez_MERCADO.npy')

      # Se almacenan los errores en un dataframe y se imprimen en pantalla
      errores = pd.DataFrame()
      errores['Errores'] = ['MAE', 'MAPE']
      errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
      errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
      errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
      errores.set_index('Errores', inplace=True)
      st.markdown('**Errores**')
      st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

      # Se grafican los resultados de pronóstico
      y_hat = resultados.values
      y_hat = np.delete(y_hat, 3)
      index = ['Redes Neuronales', 'Random Forest', 'XGBoost']
      resultados2 = pd.DataFrame({'Resultados': y_hat}, index=index)

      promedio = resultados['Promedio'].values
      promedio = np.array([promedio, promedio, promedio])
      promedio = promedio.ravel()

def rezago_mercado_lote(): # Predicción de varios meses. Esta función le solicita al usuario cargar en una plantilla
  # los valores que tomaron las siguientes variables: FECHA, DESEMPLEO, INFLACION, IEC, PRECIO PETROLEO, DIAS HABILES y FESTIVOS, durante los 12 meses anteriores.

  if selectbox1 == 'Mercado':  # Los nombres de las variables que se recibirán por el menú '¿Qué demanda desea estimar?' de Streamlit.

    st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + '.')

    st.write('Por favor, ingrese el año actual:')
    ANIO = st.number_input("Año actual:", value=2023)

    st.write('Por favor, ingrese el mes actual:')
    MES = st.number_input("Mes actual:", value=1)

    DIA = 1

    Nro_Meses = 12

    date_act = datetime.datetime(ANIO, MES, DIA).strftime("%Y/%m/%d")
    #st.write(date_act)
    date_ini = dt.strptime(date_act, "%Y/%m/%d") - pd.DateOffset(months=Nro_Meses)
    date_fin = dt.strptime(date_act, "%Y/%m/%d") + pd.DateOffset(months=Nro_Meses)
    date_ini = date_ini.strftime("%Y/%m/%d")
    #st.write(date_ini)
    #st.write(date_fin)

    # Se realiza la limpieza del dataset original
    dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_total, 'RUNT MERCADO')

    # Se imprimen en pantalla los últimos 6 registros del dataset limpio
    #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
    st.write(dataset2.tail(6))

    df_p = df_runt_total.loc[(df_runt_total['FECHA'] > date_ini) & (df_runt_total['FECHA'] <= date_act)]
    df_p = df_p.iloc[:, [0, 1, 2, 3, 4, 6, 8, 9, 10]].dropna() # No se incluye la columna 20 = RUNT MERCADO y, de los indicadores económicos, se conserva sólo IEC.

    index_pron = df_p['FECHA'] # Se almacena la columna Fecha, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
    df_p = df_p.drop(['FECHA'], axis=1) # Se elimina la columna Fecha
    df_p.iloc[:, 6] = df_p.iloc[:, 6] / (df_p.iloc[:, 6] + df_p.iloc[:, 7]) # Se crea la nueva columna RATIO_DH_F
    df_p.rename(columns = {'DIAS HABILES': 'RATIO_DH_F'}, inplace = True) # Se cambia el nombre de la nueva columna
    df_p = df_p.drop(['FESTIVOS'], axis=1) # Se elimina la columna Festivos
    #st.write(df_p)

    # Se rellena la última columna RUNT MERCADO con el valor semilla de 7000, para los meses de rezago ingresados por el usuario.
    vector = pd.DataFrame((np.ones((df_p.shape[0], 1), dtype=int)) * (7000))
    df_p.insert(len(df_p.columns), 'YAMAHA SEED', vector.values)
    #st.write(df_p)

    # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
    # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
    modeloRN_r_mercado = keras.models.load_model('modeloRN_r_MERCADO.h5')
    modeloRF_r_mercado = joblib.load('modeloRF_r_MERCADO.pkl')
    modeloXG_r_mercado = joblib.load('modeloXG_r_MERCADO.pkl')

    # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
    X = df_p.values
    X_RN = np.concatenate([numpy1, X])  # Para RN

    # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 7000 para RUNT MERCADO.
    X = X[:, 0: df_p.shape[1] - 1]  # Para RF y XG

    # Redes Neuronales
    # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
    scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
    # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT MERCADO.
    y_hat_scale = modeloRN_r_mercado.predict(X_scale[(len(X_RN) - len(X)):, :])
    # Se regresa a la escala original el valor predicho para RUNT MERCADO
    y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

    # Random Forest
    # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT MERCADO.
    y_hat_RF = modeloRF_r_mercado.predict(X)

    # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT MERCADO.
    # XGBoost
    y_hat_XG = modeloXG_r_mercado.predict(X)

    # Promedio
    # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
    y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

    # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
    st.write('**Pronóstico**')
    resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
    resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
    resultados['Fecha'] = resultados['Fecha'].apply(lambda x: (x + relativedelta(months=+12)).strftime('%d/%m/%Y'))
    resultados.set_index('Fecha', inplace=True)
    resultados.to_excel('pronosticos.xlsx', index=True)

    # Se habilita al usuario la descarga de los pronósticos por pantalla
    with open("pronosticos.xlsx", "rb") as file:
      btn = st.download_button(
        label="Descargar pronosticos",
        data=file,
        file_name="Pronosticos.xlsx",
        mime="image/png")
    st.write(resultados)

    # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
    # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
    # por el script generador_modelos_pronosticos.py, en la misma carpeta del proyecto.
    errores_RN = np.load('error_RNN_rez_MERCADO.npy')
    errores_RF = np.load('error_RF_rez_MERCADO.npy')
    errores_XG = np.load('error_XG_rez_MERCADO.npy')

    # Se almacenan los errores en un dataframe y se imprimen en pantalla
    errores = pd.DataFrame()
    errores['Errores'] = ['MAE', 'MAPE']
    errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
    errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
    errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
    errores.set_index('Errores', inplace=True)
    st.markdown('**Errores**')
    st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

    # Se grafican los resultados de pronóstico
    graficar = dataset2[-60:]
    total = pd.concat([graficar['RUNT MERCADO'], resultados])
    total.rename(columns={0: 'RUNT MERCADO'}, inplace=True)  # Esta variable tiene estacionalidad
    total = total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['RUNT MERCADO', 'Redes Neuronales', 'Random Forest', 'XGBoost', 'Promedio'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
    st.plotly_chart(fig)

###################################################################################################################################

# PARTE 3: A partir de aquí, se consideran los siguientes casos de pronóstico:
# Demanda: Yamaha, Mercado u Otra demanda
# Dinámica: Sólo con el histórico de ventas
# Para el cálculo de estos pronósticos, se usará el método de suavizamiento exponencial de Holt-Winters;
# se definirán dos funciones para predecir estas demandas: una para 'Yamaha'/'Mercado' y otra para 'Otras Demandas'.

# Se configuran los parámetros de suavizamiento:
def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # Se define la lista de configuraciones
    t_params = ['add', 'mul', None] # Componente de la tendencia (add: aditivo - mul: multiplicativo)
    d_params = [True, False]
    s_params = ['add', 'mul', None] # Componente estacional (add: aditivo - mul: multiplicativo)
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # Se crean las instancias de las configuraciones
    for t in t_params:
      for d in d_params:
        for s in s_params:
          for p in p_params:
            for b in b_params:
              for r in r_params:
                cfg = [t, d, s, p, b, r]
                models.append(cfg)
    return models

# Función 1: Se define el método Holt-Winters para demanda YAMAHA y demanda MERCADO:
def HoltWinters(variable):

  if selectbox1 == 'Yamaha': # El nombre de la variable que se recibirá por el menú ¿Qué demanda desea estimar? de Streamlit

    data = 'RUNT YAMAHA' # Los datos que se van a usar para hacer el pronóstico de RUNT YAMAHA, serán el histórico de ventas YAMAHA.

    st.write('Por favor, ingrese cuántos meses hacia adelante desea estimar la demanda por modelo Yamaha:')
    MES = st.number_input("Meses", value=12) # El número de meses se establece en 12, por defecto;
    # sin embargo, el usuario puede modificarlo a voluntad desde la app.
    MES = int(MES)

    # Se carga el dataset original y se realizan sobre él algunas funciones básicas de limpieza
    df = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name="RUNT_Total")
    df3 = df.copy()
    df3 = df3.reset_index(drop=True)
    df3.set_index('FECHA', inplace=True)
    df3.index.freq = 'MS'
    df3 = df3[data]
    df3 = df3.dropna()

  elif selectbox1 == 'Mercado':

    data = 'RUNT MERCADO'  # Los datos que se van a usar para hacer el pronóstico de RUNT MERCADO, serán el histórico de ventas MERCADO.

    st.write('Por favor, ingrese cuántos meses hacia adelante desea estimar la Demanda ' + variable + ' del Mercado:')
    MES = st.number_input("Meses", value=12)  # El número de meses se establece en 12, por defecto;
    # sin embargo, el usuario puede modificarlo a voluntad desde la app.
    MES = int(MES)

    # Se carga el dataset original y se realizan sobre él algunas funciones básicas de limpieza
    df = pd.read_excel('BD_Actualizada_Ene2023.xlsx', sheet_name="RUNT_Total")
    df3 = df.copy()
    df3 = df3.reset_index(drop=True)
    df3.set_index('FECHA', inplace=True)
    df3.index.freq = 'MS'
    df3 = df3[data]
    df3 = df3.dropna()

    df3 = df3[48:] # Se ignoran los primeros 48 datos de RUNT Mercado correspondientes al rango de fechas 11/2001-10/2005;
    # son valores muy bajitos que generan errores muy altos al tratar de ajustar el modelo, de ser tenidos en cuenta.
    # OPORTUNIDAD DE MEJORA: Probar el desempeño del modelo si se incluyen estos datos.

  cfg_list = exp_smoothing_configs(seasonal=[12])  # Se puede probar con [0,6,12]

  if (st.button('Pronosticar')):
    train_size = int(len(df3) * 0.85) # Se define el tamaño del conjunto de entrenamiento: el 85%  de los datos.
    test_size = len(df3) - train_size # Se calcula el tamaño del conjunto de prueba: los datos restantes.
    ts = df3.iloc[0:train_size].copy() # Se define el conjunto de entrenamiento
    ts_v = df3.iloc[train_size:len(df3)].copy() # Se define el conjunto de prueba
    ind = df3.index[-test_size:] # Se seleccionan los índices de los últimos 12 meses

    best_RMSE = np.inf
    best_config = []
    t1 = d1 = s1 = p1 = b1 = r1 = None
    mape = []
    y_forecast = []
    model = ()

    my_bar = st.progress(0) # Barra de progreso en Streamlit
    status_text = st.empty()

    for j in range(len(cfg_list)):
      try:
        cg = cfg_list[j]
        t, d, s, p, b, r = cg
        # Se define el modelo HoltWinters
        if (t == None):
          model = HWES(ts, trend=t, seasonal=s, seasonal_periods=p)
        else:
          model = HWES(ts, trend=t, damped=d, seasonal=s, seasonal_periods=p)
        # Se entrena el modelo
        model_fit = model.fit(optimized=True, remove_bias=r)
        y_forecast = model_fit.forecast(test_size)
        pred_ = pd.Series(data=y_forecast, index=ind)
        # Se imprimen las predicciones pasadas
        # df_pass_pred = pd.concat([ts_v, pred_.rename('pred_HW')], axis=1)
        # st.write(df_pass_pred)
        # Se calcula el error
        mape = mean_absolute_percentage_error(ts_v, y_forecast)
        # rmse = np.sqrt(mean_squared_error(ts_v,y_forecast))
        if mape < best_RMSE: # Cambiar mape por RMSE
          best_RMSE = mape
          best_config = cfg_list[j]
      except Exception as e:
        print(e)
        continue

      time.sleep(0.1)
      status_text.warning('Calculando')
      if j == (len(cfg_list) - 1):
        j = 100
      my_bar.progress(j)

    status_text.success('Listo!')
    #st.write(best_config)
    #status_text.success(best_config)
    t1, d1, s1, p1, b1, r1 = best_config

    # Se entrenará el modelo con los parametros hallados (uno entrenará con el conjunto de entrenamiento -hw_model1- para obtener errores,
    # y el otro entrenará con el dataset completo -hw-)
    if t1 == None:
      hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1)
      hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1)
    else:
      hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
      hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)

    fit2 = hw_model1.fit(optimized=True,  remove_bias=r1)
    pred_HW = fit2.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
    pred_HW = pd.Series(data=pred_HW, index=ind)

    fitted = hw.fit(optimized=True,  remove_bias=r1)
    y_hat = fitted.forecast(steps=MES)

    modelo = HWES(ts, seasonal_periods=12, trend='add', seasonal='add')
    fitted_wo = modelo.fit(optimized=True, use_brute=True)
    pred = fitted_wo.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
    pred = pd.Series(data=pred, index=ind)

    model = HWES(df3, seasonal_periods=12, trend='add', seasonal='add')
    fit = model.fit(optimized=True, remove_bias=True)
    y_hat2 = fit.forecast(steps=MES)

    tiempo = []
    nuevo_index = []
    for i in range(0, MES, 1):
      a = df3.index[len(df3) - 1] + relativedelta(months=+(1 + i))
      b = a.strftime('%d/%m/%Y')
      nuevo_index.append(a)
      tiempo.append(b)

    # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
    #st.markdown('**Pronósticos:**')
    resultados = pd.DataFrame({'Resultados optimizados': np.around(y_hat).ravel(), 'Resultados sin optimizar': np.around(y_hat2).ravel()}, index=tiempo)
    resultados.to_excel('pronosticos.xlsx', index=True)

    # Se habilita al usuario la descarga de los pronósticos por pantalla, en formato xlsx.
    with open("pronosticos.xlsx", "rb") as file:
      btn = st.download_button(
        label="Descargar pronosticos",
        data=file,
        file_name="Pronosticos.xlsx",
        mime="image/png")

    # Se imprimen los resultados de los pronósticos en pantalla
    resultados['Resultados optimizados'] = resultados['Resultados optimizados'].astype(int)
    resultados['Resultados sin optimizar'] = resultados['Resultados sin optimizar'].astype(int)
    st.dataframe(resultados)

    # Se calculan los errores
    st.write('**Errores**')
    MAE_Opt = "{:.0f}".format(mean_absolute_error(ts_v, pred_HW))
    MAPE_Opt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred_HW))
    MAE_SinOpt = "{:.0f}".format(mean_absolute_error(ts_v, pred))
    MAPE_SinOpt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred))

    # Se imprimen los errores en pantalla
    errores = pd.DataFrame()
    errores['Errores'] = ['MAE', 'MAPE']
    errores['Optimizado'] = [MAE_Opt, MAPE_Opt]
    errores['Sin optimizar'] = [MAE_SinOpt, MAPE_SinOpt]
    errores.set_index('Errores', inplace=True)
    st.write(errores.T)

    # Gráfica 1: Se grafican los pronósticos optimizados y sin optimizar
    anio = '2015' # Parámetro de control: Para determinar desde que año se va a graficar.
    agrupados = pd.DataFrame({'Optimizado': np.around(y_hat).ravel(), 'Sin optimizar': np.around(y_hat2).ravel()}, index=nuevo_index)
    total = pd.concat([df3[anio:], agrupados])
    total.rename(columns={0: 'RUNT REAL'}, inplace=True)  # Esta variable tiene estacionalidad
    total = total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['RUNT REAL', 'Optimizado', 'Sin optimizar'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
    st.plotly_chart(fig)

    # Gráfica 2: Se grafican los pronósticos ajustados y optimizados, y, ajustados sin optimizar
    ajustados = pd.DataFrame({'Ajustado optimizado': np.around(fitted.fittedvalues).ravel(), 'Ajustado sin optimizar': np.around(fit.fittedvalues).ravel()}, index=df3.index)
    ajustados_total = pd.concat([df3[anio:], ajustados[anio:]], axis=1)
    ajustados_total = ajustados_total.reset_index()
    df_melt_fitted = ajustados_total.melt(id_vars='FECHA', value_vars=[data, 'Ajustado optimizado', 'Ajustado sin optimizar'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt_fitted, x='FECHA', y='value', color='variable', labels={"value": "RUNT"})
    st.plotly_chart(fig)

    st.write('Aunque en las gráficas se observa el Runt desde ' + anio + ', los modelos de predicción están construidos con datos desde 2001 en el caso de Yamaha, y 2005 en el caso de Mercado.')

# Función 2: Se define el método HoltWinters para 'Otras demandas':
def Holt_Winters():

  st.subheader("**Otra demanda a pronosticar**")
  st.write('En esta opción puede pronosticar demanda de categorías, modelos, marcas, wholesale, zonas, distribuidores, referencias de productos, etc., dependiendo de su necesidad; sin embargo, debe tener en cuenta:')
  text3 = """
            * Los datos deben ser mensuales y debe tener mínimo 25 meses de información.
            * Los datos se deben subir como se muestra en la plantilla. **No cambie el formato de fechas ni el nombre de las columnas (FECHA, DEMANDA)**.
            * El pronóstico se calcula para una variable; si desea calcular multiples variables, debe repetir el proceso para cada una de ellas.
            **Ejemplo:** Si desea pronosticar categoría ON/OFF y SCOOTER, debe diligenciar la plantilla para ON/OFF, ejecutar el proceso y luego repetirlo para SCOOTER.
            * Tenga en cuenta que si usted tiene datos atípicos dentro del histórico, es probable que obtenga modelos con errores muy altos y además que reflejen poco la realidad; considere siempre analizar si hubo eventos que condicionaron la demanda y modifique ese valor para llevarlo a algo más "normal".
            """
  st.markdown(text3)

  # Se carga la plantilla para que el usuario la descargue, la diligencie y la vuelva a cargar
  with open("plantilla_otros.xlsx", "rb") as file:
    btn = st.download_button(
      label="Descargue plantilla",
      data=file,
      file_name="plantilla.xlsx",
      mime="image/png")

  st.markdown('**Subir plantilla**')
  st.write("Warning: Una vez diligenciada la plantilla vuelva a subir a la aplicación el archivo con formato 'xlsx'.")
  data_file = st.file_uploader('Archivo', type=['xlsx'])

  if data_file is not None: # Después de que el usuario suba el archivo, se ejecuta este bloque de código.
    df_p = pd.read_excel(data_file)
    if len(df_p) >= 25: # El archivo que suba el usuario debe tener mínimo 25 registros
      st.write('Por favor, ingrese cuántos meses hacia adelante desea estimar la demanda:')
      MES = st.number_input("Meses", value=12) #El número de meses se establece en 12, por defecto;
      # sin embargo, el usuario puede modificarlo a voluntad desde la app.
      MES = int(MES)

      df_p.set_index('FECHA', inplace=True)
      df_p.index.freq = 'MS'
      df_p = df_p.dropna()

      cfg_list = exp_smoothing_configs(seasonal=[12]) # Se puede probar con [0,6,12]

      if (st.button('Pronosticar')):
        train_size = int(len(df_p) * 0.85) # Se define el tamaño del conjunto de entrenamiento: el 85%  de los datos. # 4000
        test_size = len(df_p) - train_size # Se calcula el tamaño del conjunto de prueba: los datos restantes. # 1000
        ts = df_p.iloc[0:train_size].copy() # Se define el conjunto de entrenamiento
        ts_v = df_p.iloc[train_size:len(df_p)].copy() # Se define el conjunto de prueba
        ind = df_p.index[-test_size:] # Se seleccionan los índices de los últimos 12 meses

        best_RMSE = np.inf
        best_config = []
        t1 = d1 = s1 = p1 = b1 = r1 = None
        mape = []
        y_forecast = []
        model = ()

        my_bar = st.progress(0)
        status_text = st.empty()
        for j in range(len(cfg_list)):
          try:
            cg = cfg_list[j]
            t, d, s, p, b, r = cg
            # Se define el modelo HoltWinters
            if (t == None):
              model = HWES(ts, trend=t, seasonal=s, seasonal_periods=p)
            else:
              model = HWES(ts, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            # Se entrena el modelo
            model_fit = model.fit(optimized=True, remove_bias=r) #use_boxcox=b,
            y_forecast = model_fit.forecast(test_size)
            pred_ = pd.Series(data=y_forecast, index=ind)
            # Se imprimen las predicciones pasadas
            # df_pass_pred = pd.concat([ts_v, pred_.rename('pred_HW')], axis=1)
            # st.write(df_pass_pred)
            # Se calcula el error
            mape = mean_absolute_percentage_error(ts_v, y_forecast)
            # rmse = np.sqrt(mean_squared_error(ts_v,y_forecast))

            if mape < best_RMSE: # Cambiar MAPE por RMSE
              best_RMSE = mape
              best_config = cfg_list[j]
          except:
            continue
          time.sleep(0.1)
          status_text.warning('Calculando') # Barra de progreso
          if j == (len(cfg_list) - 1):
            j = 100
          my_bar.progress(j)

        status_text.success('Listo!')
        #st.write(best_config)
        t1, d1, s1, p1, b1, r1 = best_config

        # Se entrenará el modelo con los parametros hallados (uno entrenará con el conjunto de entrenamiento -hw_model1- para obtener errores,
        # y el otro entrenará con el dataset completo -hw-)
        if t1 == None:
          hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1)
          hw = HWES(df_p, trend=t1, seasonal=s1, seasonal_periods=p1)
        else:
          hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
          hw = HWES(df_p, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)

        fit2 = hw_model1.fit(optimized=True, remove_bias=r1) #use_boxcox=b1,
        pred_HW = fit2.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
        pred_HW = pd.Series(data=pred_HW, index=ind)

        fitted = hw.fit(optimized=True, remove_bias=r1) #use_boxcox=b1,
        y_hat = fitted.forecast(steps=MES)

        modelo = HWES(ts, seasonal_periods=12, trend='add', seasonal='add')
        fitted_wo = modelo.fit(optimized=True, use_brute=True)
        pred = fitted_wo.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
        pred = pd.Series(data=pred, index=ind)

        model = HWES(df_p, seasonal_periods=12, trend='add', seasonal='add')
        fit = model.fit(optimized=True, remove_bias=True) #use_boxcox=True,
        y_hat2 = fit.forecast(steps=MES)

        tiempo = []
        nuevo_index = []
        for i in range(0, MES, 1):
          a = df_p.index[len(df_p) - 1] + relativedelta(months=+(1 + i))
          b = a.strftime('%d/%m/%Y')
          nuevo_index.append(a)
          tiempo.append(b)

        # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
        st.markdown('**Pronósticos:**')
        resultados = pd.DataFrame({'Resultados optimizados': np.around(y_hat).ravel(), 'Resultados sin optimizar': np.around(y_hat2).ravel()}, index=tiempo)
        resultados.to_excel('pronosticos.xlsx', index=True)

        # Se habilita al usuario la descarga de los pronósticos por pantalla, en formato xlsx.
        with open("pronosticos.xlsx", "rb") as file:
          btn = st.download_button(
            label="Descargar pronosticos",
            data=file,
            file_name="Pronosticos.xlsx",
            mime="image/png")

        # Se calculan los errores
        st.write('**Errores**')
        MAE_Opt = "{:.0f}".format(mean_absolute_error(ts_v, pred_HW))
        MAPE_Opt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred_HW))
        MAE_SinOpt = "{:.0f}".format(mean_absolute_error(ts_v, pred))
        MAPE_SinOpt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred))

        # Se imprimen los errores en pantalla
        errores = pd.DataFrame()
        errores['Errores'] = ['MAE', 'MAPE']
        errores['Optimizado'] = [MAE_Opt, MAPE_Opt]
        errores['Sin optimizar'] = [MAE_SinOpt, MAPE_SinOpt]
        errores.set_index('Errores', inplace=True)
        st.write(errores.T)

        # Gráfica 1: Se grafican los pronósticos optimizados y sin optimizar
        anio = '2015' # Parámetro de control: Para determinar desde que año se va a graficar.
        agrupados = pd.DataFrame({'Optimizado': np.around(y_hat).ravel(), 'Sin_optimizar': np.around(y_hat2).ravel()}, index=nuevo_index)
        total = pd.concat([df_p[anio:], agrupados])
        total.rename(columns={0: 'DEMANDA'}, inplace=True)  # esta variable tiene estacionalidad
        total = total.reset_index()
        df_melt = total.melt(id_vars='index', value_vars=['DEMANDA', 'Optimizado', 'Sin_optimizar'])
        px.defaults.width = 1100
        px.defaults.height = 500
        fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "DEMANDA"})
        st.plotly_chart(fig)

        # Gráfica 2: Se grafican los pronósticos ajustados y optimizados, y, ajustados sin optimizar# Gráfica 2
        ajustados = pd.DataFrame({'Ajustado_optimizado': np.around(fitted.fittedvalues).ravel(), 'Ajustado_sin_optimizar': np.around(fit.fittedvalues).ravel()}, index=df_p.index)
        ajustados_total = pd.concat([df_p[anio:], ajustados[anio:]], axis=1)
        ajustados_total = ajustados_total.reset_index()
        df_melt_fitted = ajustados_total.melt(id_vars='FECHA', value_vars=['DEMANDA', 'Ajustado_optimizado', 'Ajustado_sin_optimizar'])
        px.defaults.width = 1100
        px.defaults.height = 500
        fig = px.line(df_melt_fitted, x='FECHA', y='value', color='variable', labels={"value": "DEMANDA"})
        st.plotly_chart(fig)
    else:
      st.error('La demanda a pronosticar debe tener como mínimo 25 meses de información histórica.')

# PARTE 5: Se construye el frontend de la app:

st.title("Pronósticos Motocicletas - Incolmotos Yamaha")

st.sidebar.image("https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/YAMAHA.PNG?token=ATEVFY6JYIBS3BZZKNKD5ADAWVFBY", width=250)
# img = Image.open(get_file_content_as_string("YAMAHA.PNG"))
# st.sidebar.image(img, width=250)

status = st.sidebar.radio("¿Cuál es su objetivo?", ("Informarse", "Pronosticar"))

if status == "Informarse":
  st.markdown("---")
  st.write('Esta aplicación se construye con el propósito de soportar las decisiones relacionadas con las proyecciones de motocicletas (MTP - Medium Term Plan). A continuación, se detalla la metodología y los datos asociados a ésta:')

  st.markdown("\n ## Datos")
  st.write('Estos pronósticos se han construido con datos históricos del RUNT desde 2001, además de otras variables macroeconómicas (en algunas metodologías):')

  text3 = """
            * Desempleo
            * Inflación (% anual calculado por el [banco de la república](https://totoro.banrep.gov.co/analytics/saw.dll?Download&Format=excel2007&Extension=.xls&BypassCache=true&lang=es&NQUser=publico&NQPassword=publico123&path=%2Fshared%2FSeries%20Estad%C3%ADsticas_T%2F1.%20IPC%20base%202018%2F1.2.%20Por%20a%C3%B1o%2F1.2.5.IPC_Serie_variaciones)) 
            * Salario mínimo (SMMLV) más auxilio de transporte (AUX&TTE)
            * Indice de expectativas del consumidor (IEC) *
            * Indice de condiciones económicas (ICE) *
            * Precio promedio mensual del petroleo tipo WTI
            * Tasa representativa del mercado (TRM)
            * Proporción de días hábiles y festivos (en el mes)

            *Estos índices se utilizan para el cálculo del índice de confianza del consumidor (ICC) de Fedesarrollo
            """
  st.markdown(text3, unsafe_allow_html=True)
  # link = '[Banrep](https://totoro.banrep.gov.co/analytics/saw.dll?Download&Format=excel2007&Extension=.xls&BypassCache=true&lang=es&NQUser=publico&NQPassword=publico123&path=%2Fshared%2FSeries%20Estad%C3%ADsticas_T%2F1.%20IPC%20base%202018%2F1.2.%20Por%20a%C3%B1o%2F1.2.5.IPC_Serie_variaciones)'
  # st.markdown(link, unsafe_allow_html=True)

  st.markdown("\n ## Recomendaciones de uso")
  st.write('Dependiendo de las necesidades del usuario y de los datos que éste tenga disponibles, existen tres opciones para la creación de los pronósticos:')
  text2 = """
            * **Trabajando con los últimos 12 meses (t-12):** Se utilizan los datos tanto de ventas como de las variables macroeconómicas de los últimos 12 meses, 
              para predecir los próximos 12 meses hacia adelante.
            * **Suponiendo el comportamiento futuro de las variables macroeconómicas:** En este caso, se debe ingresar a la aplicación el valor que tomarán las variables
              macroeconomicas en los n meses hacia adelante que se quieran pronosticar (incluyendo el mes en que se realiza la consulta).
            * **Sólo con el histórico de ventas:** El usuario sólo debe ingresar cuántos meses hacia adelante quiere pronosticar, pero dichos pronósticos sólo dependerán del
              histórico.
            """
  st.markdown(text2)

  st.markdown("\n ## Metodologías")
  st.write('Se construyeron estos pronósticos a partir de cuatro metodologías:')
  text = """
            * Redes neuronales artificiales (RNN)
            * XGBoost (XGB)
            * Random Forest (RF)
            * Holt Winters (HW) \n
            De éstas, las tres primeras metodologías tienen asociadas variables externas como: TRM, desempleo, inflación, etc., 
            mientras que la última (HW) sólo pronostica basándose en el comportamiento histórico de la demanda.
            """
  st.markdown(text)

  st.markdown("\n ## Errores")
  st.write('Para entender qué tan acertado es un método de pronóstico se utilizan, en este caso, dos medidas de error (MAE y MAPE):')

  # Explicación del MAE
  text = """
            * **Error medio absoluto (MAE)**: Es el promedio de las diferencias absolutas entre los valores reales y los valores pronosticados.
            """
  st.markdown(text)
  # Cálculo del MAE (Imagen)
  st.image("https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/MAE.JPG?token=ATEVFY3U2WJFFMHXGNGTU73AWVFF4", width=200)
  # img_MAE = Image.open("MAE.jpg")

  # Explicación del MAPE
  text = """
            * **Error medio absoluto porcentual (MAPE)**: Es el porcentaje promedio de desviación respecto al valor real.\n
            """
  st.markdown(text)
  # Cálculo del MAPE (Imagen)
  st.image("https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/MAPE.JPG?token=ATEVFYZAOBQ5DZK5LAYIKKTAWVFII", width=200)
  # img_MAPE= Image.open("MAPE.jpg")

  st.markdown('Sin embargo, es importante entender que muchas veces el error puede estar inducido por factores externos que condicionan el valor real; por ejemplo, si en un mes se pronostica vender 3.000 motocicletas pero no tenemos inventario y sólo vendemos 1.500, esto impactará mucho al error porque el pronostico se alejó mucho de la realidad. Por tanto, se sugiere realizar un análisis de los valores pronosticados, así como de los errores calculados, de cara a los datos usados para hacer el pronóstico.')

else:
  st.markdown("---")

  opciones1 = ['Seleccione demanda', 'Yamaha', 'Mercado', 'Otra demanda']

  opciones4 = ['Seleccione segmento', 'Total', 'Por modelo']

  opciones2 = ['Seleccione dinámica', 'Suponiendo indicadores económicos', 'Con datos reales del último año',
               'Sólo con el histórico de ventas']
  opciones3 = ['Seleccione alcance', 'Predicción de un sólo mes', 'Predicción de varios meses']

  selectbox1 = st.sidebar.selectbox('¿Qué demanda desea estimar?', opciones1)

  if selectbox1 == 'Yamaha':

    selectbox4 = st.sidebar.selectbox('¿Para qué segmento?', opciones4)

    # Yamaha - Total
    if selectbox4 == 'Total':
      selectbox2 = st.sidebar.selectbox('¿Cómo desea hacer la estimación?', opciones2)
      if selectbox2 == 'Suponiendo indicadores económicos':
        selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)
        if selectbox3 == 'Predicción de un sólo mes':
          actual_individual('Yamaha', 'Total')
        elif selectbox3 == 'Predicción de varios meses':
          actual_lote('Yamaha', 'Total')
      elif selectbox2 == 'Con datos reales del último año':
        selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)
        if selectbox3 == 'Predicción de un sólo mes':
          rezago_yamaha()
        elif selectbox3 == 'Predicción de varios meses':
          rezago_yamaha_lote()
      elif selectbox2 == 'Sólo con el histórico de ventas':
        HoltWinters('Total')

    # Yamaha - Por modelo
    elif selectbox4 == 'Por modelo':

      modelos_yamaha = ['NMAX', 'NMAX CONNECTED', 'CRYPTON FI', 'XTZ125', 'XTZ150', 'XTZ250', 'MT03',
                        'FZ25', 'FZ15', 'SZ15RR', 'YBRZ125', 'YCZ110', 'XMAX']

      label1 = "NMAX"
      m1 = st.sidebar.checkbox(label1, key="1")
      label2 = "NMAX CONNECTED"
      m2 = st.sidebar.checkbox(label2, key="2")
      label3 = "CRYPTON FI"
      m3 = st.sidebar.checkbox(label3, key="3")
      label4 = "XTZ125"
      m4 = st.sidebar.checkbox(label4, key="4")
      label5 = "XTZ150"
      m5 = st.sidebar.checkbox(label5, key="5")
      label6 = "XTZ250"
      m6 = st.sidebar.checkbox(label6, key="6")
      label7 = "MT03"
      m7 = st.sidebar.checkbox(label7, key="7")
      label8 = "FZ25"
      m8 = st.sidebar.checkbox(label8, key="8")
      label9 = "FZ15"
      m9 = st.sidebar.checkbox(label9, key="9")
      label10 = "SZ15RR"
      m10 = st.sidebar.checkbox(label10, key="10")
      label11 = "YBRZ125"
      m11 = st.sidebar.checkbox(label11, key="11")
      label12 = "YCZ110"
      m12 = st.sidebar.checkbox(label12, key="12")
      label13 = "XMAX"
      m13 = st.sidebar.checkbox(label13, key="13")

      modelos_seleccionados = []

      if m1 == True:
        modelos_seleccionados.append(label1)
      if m2 == True:
        modelos_seleccionados.append(label2)
      if m3 == True:
        modelos_seleccionados.append(label3)
      if m4 == True:
        modelos_seleccionados.append(label4)
      if m5 == True:
        modelos_seleccionados.append(label5)
      if m6 == True:
        modelos_seleccionados.append(label6)
      if m7 == True:
        modelos_seleccionados.append(label7)
      if m8 == True:
        modelos_seleccionados.append(label8)
      if m9 == True:
        modelos_seleccionados.append(label9)
      if m10 == True:
        modelos_seleccionados.append(label10)
      if m11 == True:
        modelos_seleccionados.append(label11)
      if m12 == True:
        modelos_seleccionados.append(label12)
      if m13 == True:
        modelos_seleccionados.append(label13)

      #st.write(modelos_seleccionados)

      selectbox2 = st.sidebar.selectbox('¿Cómo desea hacer la estimación?', opciones2)

      if selectbox2 == 'Suponiendo indicadores económicos':

        selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)

        if selectbox3 == 'Predicción de un sólo mes':

          # Se definen el encabezado de la página y las instrucciones para el usuario.
          st.subheader('Demanda a un sólo mes: ' + selectbox1 + ' - ' + selectbox4 + '.')
          st.write('Por favor, ingrese para cada variable el valor que supone tendría en el tiempo futuro en el que desea estimar la demanda.\n Tenga en cuenta que el dato de festivos y días hábiles corresponde valor real del mes en cuestión que quiere proyectar.')
          st.write('Puede guiarse de los últimos 6 datos de la tabla para que le sirvan de ejemplo y guía de cómo debe ingresar los datos supuestos.')

          # Se imprimen los últimos 6 registros de la tabla
          dataset1, numpy1, dataset2 = limpieza_actual(df_runt_modelo_yamaha, 'RUNT MERCADO', 'RUNT YAMAHA')
          #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
          st.write(dataset2.tail(6))

          col1, col2 = st.columns(2)
          ini = 2023

          # El usuario selecciona el año y el mes de pronóstico
          year = col1.selectbox('Año a estimar (Año)', range(ini, ini + 11))
          month = col2.selectbox('Mes a estimar (Mes)', range(1, 13))

          # El usuario ingresa por teclado los valores que él supone van a tener las variables económicas en el mes de pronóstico,
          # y cada uno de estos valores se asignan a la nueva variable relacionada.
          # OPORTUNIDAD DE MEJORA 4: Indicarle explícitamente al usuario, por pantalla, el tipo de formato que debe usar para ingresar
          # números enteros (sin puntos ni comas) y decimales (con comas).
          DESEMPLEO = st.number_input("Desempleo", format="%.3f")
          INFLACION = st.number_input("Inflación", format="%.3f")
          TRM = st.number_input("Tasa de cambio representativa del mercado (TRM)", format="%.2f")
          SMMLV_AUXTTE = st.number_input("Salario mínimo más auxilio de transporte (SMMLV&AUXTTE)", format="%.0f")
          IEC = st.number_input("Indice de expectativas del consumidor (IEC)", format="%.2f")
          ICE = st.number_input("Indice de condiciones económicas (ICE)", format="%.2f")
          PRECIO_PETROLEO_WTI = st.number_input("Precio del crudo WTI (en dólares)", format="%.3f")
          DIAS_HABILES = st.number_input("Dias hábiles", format="%.0f")
          FESTIVOS = st.number_input("Festivos", format="%.0f")

          for modelo_MC in modelos_seleccionados:

            lista_targets = ['RUNT MERCADO', 'RUNT YAMAHA', 'RUNT NMAX', 'RUNT NMAX CONNECTED', 'RUNT CRYPTON FI',
                             'RUNT XTZ125', 'RUNT XTZ150', 'RUNT XTZ250', 'RUNT MT03', 'RUNT FZ25', 'RUNT FZ15',
                             'RUNT SZ15RR', 'RUNT YBRZ125', 'RUNT YCZ110', 'RUNT XMAX']

            variable_modelo = modelo_MC
            variable_df = [x for x in lista_targets if x != 'RUNT ' + str(modelo_MC)]
            visualizar = str(modelo_MC)

            # Se realiza la limpieza del dataset original
            dataset1, numpy1, dataset2 = limpieza_actual(df_runt_modelo_yamaha, *variable_df)

            # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
            # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
            #f (st.button('Pronosticar', key=modelo_MC)):
            st.subheader(visualizar)

            modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')
            modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
            modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

            # Se preparan los datos para entrenar los modelos
            # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
            X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, IEC, ICE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS)]])
            # Se asigna un valor semilla (igual a 10) a la variable modelo; este valor se puede cambiar por el mínimo (o a la media) del Runt de/para cada modelo.
            # X_RN = np.concatenate([numpy1, np.reshape(np.append(X, [10]), (1, -1))])
            X_RN = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, IEC, ICE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS), np.reshape(10, (1, -1))]])

            # Redes Neuronales
            # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
            scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
            # Se almacena, en la nueva variable y_hat_scale, el valor predicho (por el modelo de redes neuronales) para RUNT YAMAHA, escalado entre -1 y 1.
            y_hat_scale = modeloRN.predict(np.reshape(X_scale[-1], (1, -1)))
            # Se regresa a la escala original el valor predicho para RUNT YAMAHA
            y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

            # Random Forest
            # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
            y_hat_RF = modeloRF.predict(X)

            # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
            # XGBoost
            y_hat_XG = modeloXG.predict(X)

            # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
            # OPORTUNIDAD DE MEJORA 5: Se puede calcular un promedio ponderado? Los pesos los puede dar el experto de negocio o el mismo algoritmo?
            y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

            # st.write(str(modelo_MC))
            #st.markdown(str(modelo_MC))

            index = ['1/' + str(month) + '/' + str(year)]

            # Se almacenan los resultados en un dataframe y se imprimen en pantalla
            resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
            resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
            st.write(resultados)

            # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
            # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
            # por el script generador_modelos_definitivo.py, en la misma carpeta del proyecto.
            errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
            errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
            errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

            # Se almacenan los errores en un dataframe y se imprimen en pantalla
            errores = pd.DataFrame()
            errores['Errores'] = ['MAE', 'MAPE']
            errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
            errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
            errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
            errores.set_index('Errores', inplace=True)
            st.markdown('**Errores**')
            st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

        elif selectbox3 == 'Predicción de varios meses':

          st.subheader('Estimar demanda para varios meses: ' + selectbox1 + ' - ' + selectbox4 + '.')
          st.write('Por favor suba un archivo con los valores que supone que tendrán las variables en el horizonte futuro, tenga en cuenta que debe tener las mismas variables y en el mismo orden de la tabla.')
          st.write('Para facilitar el cargue de los datos, utilice y descargue la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla.')

          # Se imprimen los últimos 6 registros de la tabla
          dataset1, numpy1, dataset2 = limpieza_actual(df_runt_modelo_yamaha, 'RUNT MERCADO', 'RUNT YAMAHA')
          #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
          st.write(dataset2.tail(6))

          # Se carga la plantilla para que el usuario la descargue
          with open("plantilla_lote.xlsx", "rb") as file:
            btn = st.download_button(
              label="Descargue plantilla",
              data=file,
              file_name="plantilla.xlsx",
              mime="image/png")

          # Se sube la plantilla diligenciada por el usuario, y se almacena como un archivo de extensión xlsx en la variable data_file.
          st.markdown('**Subir plantilla**')
          st.write("Warning: El archivo que suba debe tener extensión 'xlsx'.")
          data_file = st.file_uploader('Archivo', type=['xlsx'])

          if data_file is not None:  # Después de que el usuario suba el archivo, se ejecuta este bloque de código.

            # Se lee el archivo y se almacena en df_p
            df_p = pd.read_excel(data_file)

            # Se almacena la columna Fecha, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
            index_pron = df_p['FECHA']
            # Se elimina la columna Fecha
            df_p = df_p.drop(['FECHA'], axis=1)

            # Se almacena el número de columnas del dataset, en la variable auxiliar columns, para construir la columna RATIO_DH_F (registro a registro).
            columns = df_p.shape[1]
            # Se crea la columna RATIO_DH_F
            df_p.iloc[:, columns - 2] = df_p.iloc[:, columns - 2] / (df_p.iloc[:, columns - 2] + df_p.iloc[:, columns - 1])

            # Se rellena la última columna RUNT YAMAHA con el valor semilla de 8000, para los meses de pronóstico ingresados por el usuario.
            for i in range(0, df_p.shape[0], 1):
              df_p.iloc[i, columns - 1] = 8000

            lista_targets = ['RUNT MERCADO', 'RUNT YAMAHA', 'RUNT NMAX', 'RUNT NMAX CONNECTED', 'RUNT CRYPTON FI',
                             'RUNT XTZ125', 'RUNT XTZ150', 'RUNT XTZ250', 'RUNT MT03', 'RUNT FZ25', 'RUNT FZ15',
                             'RUNT SZ15RR', 'RUNT YBRZ125', 'RUNT YCZ110', 'RUNT XMAX']

            for modelo_MC in modelos_seleccionados:

              variable_modelo = modelo_MC
              variable_df = [x for x in lista_targets if x != 'RUNT ' + str(modelo_MC)]
              visualizar = 'RUNT ' + str(modelo_MC)

              # st.write(str(modelo_MC))
              #st.markdown(str(modelo_MC))
              st.subheader('Pronósticos ' + modelo_MC + ':')

              # Se realiza la limpieza del dataset original
              dataset1, numpy1, dataset2 = limpieza_actual(df_runt_modelo_yamaha, *variable_df)

              # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost
              # que fueron guardados por generador_modelos_definitivo.py en formatos h5 y pkl, respectivamente, en la misma carpeta del proyecto.
              modeloRN = keras.models.load_model('modeloRN_' + variable_modelo + '.h5')
              modeloRF = joblib.load('modeloRF_' + variable_modelo + '.pkl')
              modeloXG = joblib.load('modeloXG_' + variable_modelo + '.pkl')

              # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
              X = df_p.values
              X_RN = np.concatenate([numpy1, X])

              # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 8000 para RUNT YAMAHA
              X = X[:, 0: df_p.shape[1] - 1]

              # Redes Neuronales
              # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
              scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
              # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA/MERCADO.
              y_hat_scale = modeloRN.predict(X_scale[(len(X_RN) - len(X)):, :])
              # Se regresa a la escala original el valor predicho para RUNT YAMAHA
              y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

              # Random Forest
              y_hat_RF = modeloRF.predict(X)

              # XGBoost
              y_hat_XG = modeloXG.predict(X)

              # Promedio
              y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

              #st.markdown('**Pronóstico**')

              # st.write(str(modelo))
              #st.markdown(str(modelo))

              # Se almacenan los resultados en un dataframe
              resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
              resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
              resultados['Fecha'] = resultados.Fecha.apply(lambda x: x.strftime('%d/%m/%Y'))
              resultados.set_index('Fecha', inplace=True)

              # Se habilita la descarga de los resultados para el usuario
              resultados.to_excel('pronosticos.xlsx', index=True)
              with open("pronosticos.xlsx", "rb") as file:
                btn = st.download_button(
                  label="Descargar pronosticos",
                  data=file,
                  file_name="Pronosticos.xlsx",
                  mime="image/png")
              st.write(resultados)

              # Se cargan los errores generados por el script generador_modelos_definitivo, desde la carpeta del proyecto.
              errores_RN = np.load('error_RNN_actual_' + variable_modelo + '.npy')
              errores_RF = np.load('error_RF_actual_' + variable_modelo + '.npy')
              errores_XG = np.load('error_XG_actual_' + variable_modelo + '.npy')

              # Se imprimen los errores en pantalla
              errores = pd.DataFrame()
              errores['Errores'] = ['MAE', 'MAPE']
              errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
              errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
              errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
              errores.set_index('Errores', inplace=True)
              st.markdown('**Errores**')
              st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

              # Se grafican los resultados de pronóstico
              graficar = dataset2[-60:]
              total = pd.concat([graficar[visualizar], resultados])
              total.rename(columns={0: visualizar}, inplace=True)  # esta variable tiene estacionalidad
              total = total.reset_index()
              df_melt = total.melt(id_vars='index', value_vars=[visualizar, 'Redes Neuronales', 'Random Forest', 'XGBoost', 'Promedio'])
              px.defaults.width = 1100
              px.defaults.height = 500
              fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
              st.plotly_chart(fig)

      elif selectbox2 == 'Con datos reales del último año':

        selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)

        if selectbox3 == 'Predicción de un sólo mes':

          # Se crea el subtítulo
          st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + ' - ' + selectbox4 + '.')
          # Se dan instrucciones (por pantalla) al usuario sobre los datos que debe ingresar y su formato
          st.write('Por favor, ingrese el dato de las variables un año atras del período que desea estimar, es decir, si desea estimar Junio de 2021, ingrese los datos de Junio de 2020 (para el caso de los días hábiles y festivos, sí se deben ingresar los valores reales para el mes que se desea pronosticar)')
          st.write('Tome como guía los valores y formatos de la tabla que se muestra a continuación.')

          # Se imprimen los últimos 6 registros de la tabla
          dataset1, numpy1, dataset2 = limpieza_actual(df_runt_modelo_yamaha, 'RUNT MERCADO', 'RUNT YAMAHA')
          #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
          st.write(dataset2.tail(6))

          # Se capturan el año y mes de pronóstico que el usuario ingresa, y se incluye el rezago.
          col1, col2 = st.columns(2)
          ini = 2023
          year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini + 11))
          month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))
          day = 1

          date = datetime.datetime(year-1, month, day).strftime("%Y/%m/%d")

          # Se traen los datos de las variables un año atras del período que se desea estimar, y se da formato a los demás datos ingresados por el usuario.
          DESEMPLEO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'DESEMPLEO'].item()
          INFLACION = df_runt_total.loc[df_runt_total['FECHA'] == date, 'INFLACION'].item()
          TRM = df_runt_total.loc[df_runt_total['FECHA'] == date, 'TRM'].item()
          SMMLV_AUXTTE = df_runt_total.loc[df_runt_total['FECHA'] == date, 'SMMLV&AUXTTE'].item()
          PRECIO_PETROLEO_WTI = df_runt_total.loc[df_runt_total['FECHA'] == date, 'PRECIO PETROLEO WTI'].item()
          DIAS_HABILES = st.number_input("Dias hábiles (en el mes de pronóstico)", format="%.0f")
          FESTIVOS = st.number_input("Festivos (en el mes de pronóstico)", format="%.0f")
          RUNT_MERCADO = df_runt_total.loc[df_runt_total['FECHA'] == date, 'RUNT MERCADO'].item()  # Se debe incluir?
          RUNT_YAMAHA = df_runt_total.loc[df_runt_total['FECHA'] == date, 'RUNT YAMAHA'].item()  # Se debe incluir?

          lista_targets = ['RUNT MERCADO', 'RUNT YAMAHA', 'RUNT NMAX', 'RUNT NMAX CONNECTED', 'RUNT CRYPTON FI',
                           'RUNT XTZ125', 'RUNT XTZ150', 'RUNT XTZ250', 'RUNT MT03', 'RUNT FZ25', 'RUNT FZ15',
                           'RUNT SZ15RR', 'RUNT YBRZ125', 'RUNT YCZ110', 'RUNT XMAX']

          for modelo_MC in modelos_seleccionados:

            # Se trae el valor puntual del RUNT para el modelo seleccionado, un año atrás del período que se desea estimar.
            #RUNT_modelo = df_runt_modelo_yamaha.loc[df_runt_modelo_yamaha['FECHA'] == date, str('RUNT ' + modelo)].item() Es correcto esto? Nooo!!! :/

            # st.write(str(modelo_MC))
            #st.markdown(str(modelo_MC))
            st.subheader('Pronósticos ' + modelo_MC + ':')

            # Se realiza la limpieza del dataset original
            dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_modelo_yamaha, str('RUNT ' + modelo_MC))

            modeloRN_r_yamaha = keras.models.load_model('modeloRN_r_' + modelo_MC + '.h5')  # Cambiar por modelo.replace(" ", "")
            modeloRF_r_yamaha = joblib.load('modeloRF_r_' + modelo_MC + '.pkl')  # Cambiar por modelo.replace(" ", "")
            modeloXG_r_yamaha = joblib.load('modeloXG_r_' + modelo_MC + '.pkl')  # Cambiar por modelo.replace(" ", "")

            # Se preparan los datos para entrenar los modelos
            # Se agrega una nueva fila al arreglo numpy1 con los valores de las variables que ingresó el usuario por teclado.
            X = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS), RUNT_MERCADO]])
            # Se asigna un valor semilla (igual a 10) para cada uno de los modelos YAMAHA.
            X_RN = np.array([[DESEMPLEO, INFLACION, TRM, SMMLV_AUXTTE, PRECIO_PETROLEO_WTI, (DIAS_HABILES) / (DIAS_HABILES + FESTIVOS), RUNT_MERCADO, np.reshape(10, (1, -1))]])

            # Redes Neuronales
            # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
            scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
            # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA
            y_hat_scale = modeloRN_r_yamaha.predict(np.reshape(X_scale[-1], (1, -1)))
            # Se regresa a la escala original el valor predicho para RUNT YAMAHA
            y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

            # Random Forest
            # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
            y_hat_RF = modeloRF_r_yamaha.predict(X)

            # XGBoost
            # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
            y_hat_XG = modeloXG_r_yamaha.predict(X)

            # Promedio
            # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
            y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

            # Se almacenan los resultados en un dataframe
            index = ['1/' + str(month) + '/' + str(year)]
            resultados = pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)}, index=index)
            resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})

            # Se imprimen los resultados en pantalla
            st.write(resultados)

            # Se cargan los errores generados por el script generador_modelos_pronosticos, desde la carpeta del proyecto.
            errores_RN = np.load('error_RNN_rez_' + modelo_MC + '.npy')  # Cambiar por modelo.replace(" ", "")
            errores_RF = np.load('error_RF_rez_' + modelo_MC + '.npy')  # Cambiar por modelo.replace(" ", "")
            errores_XG = np.load('error_XG_rez_' + modelo_MC + '.npy')  # Cambiar por modelo.replace(" ", "")

            # Se almacenan los errores en un dataframe y se imprimen en pantalla
            errores = pd.DataFrame()
            errores['Errores'] = ['MAE', 'MAPE']
            errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
            errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
            errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
            errores.set_index('Errores', inplace=True)
            st.markdown('**Errores**')
            st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

            # Se grafican los resultados de pronóstico
            y_hat = resultados.values
            y_hat = np.delete(y_hat, 3)
            index = ['Redes Neuronales', 'Random Forest', 'XGBoost']
            resultados2 = pd.DataFrame({'Resultados': y_hat}, index=index)

            promedio = resultados['Promedio'].values
            promedio = np.array([promedio, promedio, promedio])
            promedio = promedio.ravel()

        elif selectbox3 == 'Predicción de varios meses':

          st.subheader('Estimar demanda con datos reales del último año: ' + selectbox1 + ' - ' + selectbox4 + '.')

          lista_targets = ['RUNT MERCADO', 'RUNT YAMAHA', 'RUNT NMAX', 'RUNT NMAX CONNECTED', 'RUNT CRYPTON FI',
                           'RUNT XTZ125', 'RUNT XTZ150', 'RUNT XTZ250', 'RUNT MT03', 'RUNT FZ25', 'RUNT FZ15',
                           'RUNT SZ15RR', 'RUNT YBRZ125', 'RUNT YCZ110', 'RUNT XMAX']

          # # Se imprimen los últimos 6 registros de la tabla
          # dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_modelo_yamaha, ", ".join(map(str, 'RUNT ' + modelos_seleccionados)))
          # dataset2.index = dataset2.index.strftime('%d/%m/%Y')
          # st.write(dataset2.tail(6))

          st.write('Por favor, ingrese el año actual:')
          ANIO = st.number_input("Año actual:", value=2023)

          st.write('Por favor, ingrese el mes actual:')
          MES = st.number_input("Mes actual:", value=1)

          DIA = 1

          Nro_Meses = 12

          date_act = datetime.datetime(ANIO, MES, DIA).strftime("%Y/%m/%d")
          # st.write(date_act)
          date_ini = dt.strptime(date_act, "%Y/%m/%d") - pd.DateOffset(months=Nro_Meses)
          date_fin = dt.strptime(date_act, "%Y/%m/%d") + pd.DateOffset(months=Nro_Meses)
          date_ini = date_ini.strftime("%Y/%m/%d")
          # st.write(date_ini)
          # st.write(date_fin)

          for modelo_MC in modelos_seleccionados:

            st.subheader(modelo_MC)

            # Se realiza la limpieza del dataset original
            dataset1, numpy1, dataset2 = limpieza_rezago(df_runt_modelo_yamaha, 'RUNT ' + modelo_MC)

            st.write('**Históricos**')
            # Se imprimen los últimos 6 registros de la tabla
            #dataset2.index = dataset2.index.strftime('%d/%m/%Y')
            st.write(dataset2.tail(6))

            df_p = df_runt_total.loc[(df_runt_total['FECHA'] > date_ini) & (df_runt_total['FECHA'] <= date_act)]
            df_p = df_p.iloc[:, [0, 1, 2, 3, 4, 6, 8, 9, 10]].dropna()

            index_pron = df_p['FECHA']  # Se almacena la columna Fecha, en la nueva variable index_pron, con la fecha correspondiente a cada fila de pronóstico.
            df_p = df_p.drop(['FECHA'], axis=1)  # Se elimina la columna Fecha
            df_p.iloc[:, 6] = df_p.iloc[:, 6] / (df_p.iloc[:, 6] + df_p.iloc[:, 7])  # Se crea la nueva columna RATIO_DH_F
            df_p.rename(columns={'DIAS HABILES': 'RATIO_DH_F'}, inplace=True)  # Se cambia el nombre de la nueva columna
            df_p = df_p.drop(['FESTIVOS'], axis=1)  # Se elimina la columna Festivos
            # st.write(df_p)

            # Se rellena la última columna RUNT YAMAHA con el valor semilla de 10, para los meses de rezago ingresados por el usuario.
            vector = pd.DataFrame((np.ones((df_p.shape[0], 1), dtype=int)) * (10))
            # st.write(vector)
            df_p.insert(len(df_p.columns), 'YAMAHA SEED', vector.values)
            # st.write(df_p)

            # Para el modelo de RN: El dataset df_p se convierte en arreglo, para poder concatenarlo con numpy1.
            X = df_p.values
            X_RN = np.concatenate([numpy1, X])  # Para RN

            # Se cargan los 3 modelos de pronóstico: Redes Neuronales, Random Forest y XGBoost que fueron guardados
            # por generador_modelos_pronosticos.py, en formatos h5 y pkl, en la misma carpeta del proyecto.
            modeloRN_r_yamaha = keras.models.load_model('modeloRN_r_' + modelo_MC + '.h5')
            modeloRF_r_yamaha = joblib.load('modeloRF_r_' + modelo_MC + '.pkl')
            modeloXG_r_yamaha = joblib.load('modeloXG_r_' + modelo_MC + '.pkl')

            # Para los modelos RF y XG: Se elimina la última columna de df_p, con los valores imputados en 7000 para RUNT YAMAHA.
            X = X[:, 0: df_p.shape[1] - 1]  # Para RF y XG

            # Redes Neuronales
            # Se escalan los datos con la función preprocesamientoRN que fue definida en las primeras líneas de código
            scaler, X_scale, Y_scale = preprocesamientoRN(X_RN)
            # Se almacena, en la nueva variable y_hat_scale, el valor predicho para RUNT YAMAHA.
            y_hat_scale = modeloRN_r_yamaha.predict(X_scale[(len(X_RN) - len(X)):, :])
            # Se regresa a la escala original el valor predicho para RUNT YAMAHA
            y_hat_RN = scaler.inverse_transform(y_hat_scale).ravel()

            # Random Forest
            # Se almacena, en la variable y_hat_RF, el valor predicho (por el modelo Random Forest) para RUNT YAMAHA.
            y_hat_RF = modeloRF_r_yamaha.predict(X)

            # Se almacena, en la variable y_hat_XG, el valor predicho (por el modelo XGBoost) para RUNT YAMAHA.
            # XGBoost
            y_hat_XG = modeloXG_r_yamaha.predict(X)

            # Promedio
            # Se calcula un promedio aritmético de los 3 pronósticos (uno por cada modelo)
            y_hat_prom = (y_hat_RN + y_hat_RF + y_hat_XG) / 3

            # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
            st.write('**Pronósticos**')
            resultados = pd.DataFrame({'Fecha': index_pron, 'Redes Neuronales': np.around(y_hat_RN), 'Random Forest': np.around(y_hat_RF), 'XGBoost': np.around(y_hat_XG), 'Promedio': np.around(y_hat_prom)})
            resultados = resultados.astype({'Redes Neuronales': int, 'Random Forest': int, 'XGBoost': int, 'Promedio': int})
            resultados['Fecha'] = resultados['Fecha'].apply(lambda x: (x + relativedelta(months=+12)).strftime('%d/%m/%Y'))
            resultados.set_index('Fecha', inplace=True)
            resultados.to_excel('pronosticos.xlsx', index=True)

            # Se habilita al usuario la descarga de los pronósticos por pantalla
            with open("pronosticos.xlsx", "rb") as file:
              btn = st.download_button(
                label="Descargar pronosticos",
                data=file,
                file_name="Pronosticos.xlsx",
                mime="image/png")
            st.write(resultados)

            # Se cargan los errores arrojados por los 3 modelos desde la carpeta del proyecto.
            # Estos errores fueron generados y almacenados en 3 archivos con extensión .npy,
            # por el script generador_modelos_pronosticos.py, en la misma carpeta del proyecto.
            errores_RN = np.load('error_RNN_rez_' + modelo_MC + '.npy')
            errores_RF = np.load('error_RF_rez_' + modelo_MC + '.npy')
            errores_XG = np.load('error_XG_rez_' + modelo_MC + '.npy')

            # Se almacenan los errores en un dataframe y se imprimen en pantalla
            errores = pd.DataFrame()
            errores['Errores'] = ['MAE', 'MAPE']
            errores['Redes Neuronales'] = [errores_RN[0], (errores_RN[1])]
            errores['Random Forest'] = [int(errores_RF[0]), errores_RF[1]]
            errores['XGBoost'] = [int(errores_XG[0]), errores_XG[1]]
            errores.set_index('Errores', inplace=True)
            st.markdown('**Errores**')
            st.write(errores.T.style.format({'MAE': '{:.0f}', 'MAPE': "{:.0%}"}))

            # Se grafican los resultados de pronóstico
            graficar = dataset2[-60:]
            #graficar.index = graficar.index.strftime('%d/%m/%Y')
            total = pd.concat([graficar['RUNT ' + modelo_MC], resultados])
            #st.write(total)
            total.rename(columns={0: 'RUNT ' + modelo_MC}, inplace=True)  # Esta variable tiene estacionalidad
            total = total.reset_index()
            df_melt = total.melt(id_vars='index', value_vars=['RUNT ' + modelo_MC, 'Redes Neuronales', 'Random Forest', 'XGBoost', 'Promedio'])
            px.defaults.width = 1100
            px.defaults.height = 500
            fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
            st.plotly_chart(fig)

      elif selectbox2 == 'Sólo con el histórico de ventas':

        st.write('Por favor, ingrese cuántos meses hacia adelante desea estimar la demanda por modelo Yamaha:')
        MES = st.number_input("Meses", value=12)  # El número de meses se establece en 12, por defecto; sin embargo, el usuario puede modificarlo a voluntad desde la app.
        MES = int(MES)

        for modelo_MC in modelos_seleccionados:

          data = 'RUNT ' + modelo_MC # Los datos que se van a usar para hacer el pronóstico de RUNT Modelo, serán el histórico de ventas Modelo.

          # Se carga el dataset original y se realizan sobre él algunas funciones básicas de limpieza
          df = df_runt_modelo_yamaha
          df3 = df.copy()
          df3 = df3.reset_index(drop=True)
          df3.set_index('FECHA', inplace=True)
          df3.index.freq = 'MS'
          df3 = df3[data]
          df3 = df3.dropna()

          cfg_list = exp_smoothing_configs(seasonal=[12])  # Se puede probar con [0,6,12]

          train_size = int(len(df3) * 0.85)  # Se define el tamaño del conjunto de entrenamiento: el 85%  de los datos.
          test_size = len(df3) - train_size  # Se calcula el tamaño del conjunto de prueba: los datos restantes.
          ts = df3.iloc[0:train_size].copy()  # Se define el conjunto de entrenamiento
          ts_v = df3.iloc[train_size:len(df3)].copy()  # Se define el conjunto de prueba
          ind = df3.index[-test_size:]  # Se seleccionan los índices de los últimos 12 meses

          best_RMSE = np.inf
          best_config = []
          t1 = d1 = s1 = p1 = b1 = r1 = None
          mape = []
          y_forecast = []
          model = ()

          my_bar = st.progress(0)  # Barra de progreso en Streamlit
          status_text = st.empty()

          for j in range(len(cfg_list)):
            try:
              cg = cfg_list[j]
              t, d, s, p, b, r = cg
              # Se define el modelo HoltWinters
              if (t == None):
                model = HWES(ts, trend=t, seasonal=s, seasonal_periods=p)
              else:
                model = HWES(ts, trend=t, damped=d, seasonal=s, seasonal_periods=p)
              # Se entrena el modelo
              model_fit = model.fit(optimized=True, remove_bias=r)
              y_forecast = model_fit.forecast(test_size)
              pred_ = pd.Series(data=y_forecast, index=ind)
              # Se imprimen las predicciones pasadas
              # df_pass_pred = pd.concat([ts_v, pred_.rename('pred_HW')], axis=1)
              # st.write(df_pass_pred)
              # Se calcula el error
              mape = mean_absolute_percentage_error(ts_v, y_forecast)
              # rmse = np.sqrt(mean_squared_error(ts_v,y_forecast))
              if mape < best_RMSE:  # Cambiar mape por RMSE
                best_RMSE = mape
                best_config = cfg_list[j]
            except Exception as e:
              print(e)
              continue

            time.sleep(0.1)
            status_text.warning('Calculando')
            if j == (len(cfg_list) - 1):
              j = 100
            my_bar.progress(j)

          st.subheader(modelo_MC)

          status_text.success('Listo!')
          # st.write(best_config)
          # status_text.success(best_config)
          t1, d1, s1, p1, b1, r1 = best_config

          # Se entrenará el modelo con los parametros hallados (uno entrenará con el conjunto de entrenamiento -hw_model1- para obtener errores, y el otro entrenará con el dataset completo -hw-)
          if t1 == None:
            hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1)
            hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1)
          else:
            hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
            hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)

          fit2 = hw_model1.fit(optimized=True, remove_bias=r1)
          pred_HW = fit2.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
          pred_HW = pd.Series(data=pred_HW, index=ind)

          fitted = hw.fit(optimized=True, remove_bias=r1)
          y_hat = fitted.forecast(steps=MES)

          modelo = HWES(ts, seasonal_periods=12, trend='add', seasonal='add')
          fitted_wo = modelo.fit(optimized=True, use_brute=True)
          pred = fitted_wo.predict(start=pd.to_datetime(ts_v.index[0]), end=pd.to_datetime(ts_v.index[len(ts_v) - 1]))
          pred = pd.Series(data=pred, index=ind)

          model = HWES(df3, seasonal_periods=12, trend='add', seasonal='add')
          fit = model.fit(optimized=True, remove_bias=True)
          y_hat2 = fit.forecast(steps=MES)

          tiempo = []
          nuevo_index = []
          for i in range(0, MES, 1):
            a = df3.index[len(df3) - 1] + relativedelta(months=+(1 + i))
            b = a.strftime('%d/%m/%Y')
            nuevo_index.append(a)
            tiempo.append(b)

          # Se almacenan los resultados en un dataframe y se exportan a Excel, para que el usuario pueda descargarlos.
          # st.markdown('**Pronósticos:**')
          resultados = pd.DataFrame({'Resultados optimizados': np.around(y_hat).ravel(), 'Resultados sin optimizar': np.around(y_hat2).ravel()}, index=tiempo)
          resultados.to_excel('pronosticos.xlsx', index=True)

          # Se habilita al usuario la descarga de los pronósticos por pantalla, en formato xlsx.
          with open("pronosticos.xlsx", "rb") as file:
            btn = st.download_button(
              label="Descargar pronosticos",
              data=file,
              file_name="Pronosticos.xlsx",
              mime="image/png")

          # Se imprimen los resultados de los pronósticos en pantalla
          resultados['Resultados optimizados'] = resultados['Resultados optimizados'].astype(int)
          resultados['Resultados sin optimizar'] = resultados['Resultados sin optimizar'].astype(int)
          st.dataframe(resultados)

          # Se calculan los errores
          st.write('**Errores**')
          MAE_Opt = "{:.0f}".format(mean_absolute_error(ts_v, pred_HW))
          MAPE_Opt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred_HW))
          MAE_SinOpt = "{:.0f}".format(mean_absolute_error(ts_v, pred))
          MAPE_SinOpt = "{:.2%}".format(mean_absolute_percentage_error(ts_v, pred))

          # Se imprimen los errores en pantalla
          errores = pd.DataFrame()
          errores['Errores'] = ['MAE', 'MAPE']
          errores['Optimizado'] = [MAE_Opt, MAPE_Opt]
          errores['Sin optimizar'] = [MAE_SinOpt, MAPE_SinOpt]
          errores.set_index('Errores', inplace=True)
          st.write(errores.T)

          # Gráfica 1: Se grafican los pronósticos optimizados y sin optimizar
          anio = '2015'  # Parámetro de control: Para determinar desde que año se va a graficar.
          agrupados = pd.DataFrame({'Optimizado': np.around(y_hat).ravel(), 'Sin optimizar': np.around(y_hat2).ravel()}, index=nuevo_index)
          total = pd.concat([df3[anio:], agrupados])
          total.rename(columns={0: 'RUNT ' + modelo_MC}, inplace=True)  # Esta variable tiene estacionalidad
          total = total.reset_index()
          df_melt = total.melt(id_vars='index', value_vars=['RUNT ' + modelo_MC, 'Optimizado', 'Sin optimizar'])
          px.defaults.width = 1100
          px.defaults.height = 500
          fig = px.line(df_melt, x='index', y='value', color='variable', labels={"index": "FECHA", "value": "RUNT"})
          st.plotly_chart(fig)

          # Gráfica 2: Se grafican los pronósticos ajustados y optimizados, y, ajustados sin optimizar
          ajustados = pd.DataFrame({'Ajustado optimizado': np.around(fitted.fittedvalues).ravel(), 'Ajustado sin optimizar': np.around(fit.fittedvalues).ravel()}, index=df3.index)
          ajustados_total = pd.concat([df3[anio:], ajustados[anio:]], axis=1)
          ajustados_total = ajustados_total.reset_index()
          df_melt_fitted = ajustados_total.melt(id_vars='FECHA', value_vars=[data, 'Ajustado optimizado', 'Ajustado sin optimizar'])
          px.defaults.width = 1100
          px.defaults.height = 500
          fig = px.line(df_melt_fitted, x='FECHA', y='value', color='variable', labels={"value": "RUNT"})
          st.plotly_chart(fig)

          st.write('Aunque en las gráficas se observa el Runt desde ' + anio + ', los modelos de predicción están construidos con datos desde 2001 en el caso de Yamaha, y 2005 en el caso de Mercado.')

  elif selectbox1 == 'Mercado':
    selectbox2 = st.sidebar.selectbox('¿Cómo desea hacer la estimación?: ', opciones2)
    if selectbox2 == 'Suponiendo indicadores económicos':
      selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        actual_individual('Mercado', 'Total')
      elif selectbox3 == 'Predicción de varios meses':
        actual_lote('Mercado', 'Total')
    elif selectbox2 == 'Con datos reales del último año':
      selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        rezago_mercado()
      elif selectbox3 == 'Predicción de varios meses':
        rezago_mercado_lote()
    elif selectbox2 == 'Sólo con el histórico de ventas':
      HoltWinters('Total')

  elif selectbox1 == 'Otra demanda':
    Holt_Winters()

st.sidebar.subheader('Creado por:')
st.sidebar.write('Analítica de Datos')  # :)