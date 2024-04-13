# Prediccion_de_Demanda_Energetica
Código del  sistema

import tensorflow as tf
import numpy as np
dias = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=float)
kWh = np.array([9,7,10,12,6,9,10,15,8,7,6,8,9,9,7,8.5,9.5,10.5,8.6,7.8], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'

)
print("Entrenando modelo")
historial = modelo.fit(dias, kWh, epochs = 1000, verbose = False)
print("Modelo entrenado")


print("Probando la predicción con Regresion lineal")
resultado = modelo.predict([31])
res = round((resultado[0][0]),0)
print("La cantidad aprox que se usara en ese día: " + str(res) + " KWh.")
print("Variables internas del modelo")
print(capa.get_weights())
