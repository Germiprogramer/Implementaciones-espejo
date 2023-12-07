import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_pruebas = datos["train"], datos["test"]
nombres_clases = metadatos.features["label"].names

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 
  return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache() 

for imagen, etiqueta in datos_entrenamiento.take(1):
  break
imagen = imagen.numpy().reshape((28, 28))

'''
plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
'''

modelo = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Para clasificación
])

#Compilación del modelo
modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

TAMANO_LOTE = 32

num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

print(num_ej_entrenamiento)
print(num_ej_pruebas)

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

historial = modelo.fit(datos_entrenamiento, epochs=10, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))
'''
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
'''
#Pintar una cuadricula con varias predicciones, y marcar si fue correcta (azul) o incorrecta (roja)
import numpy as np

for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)
  
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_clases[etiqueta_real]),
                                color=color)
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0, 1]) 
  etiqueta_prediccion = np.argmax(arr_predicciones)
  
  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')
  
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()


ruta_imagen = 'imagenes/pantalones_persona.jpeg'

# Cargar la imagen y redimensionarla a (28, 28) si es necesario
imagen = Image.open(ruta_imagen).convert('L').resize((28, 28))

# Convertir la imagen a un arreglo numpy
imagen_np = np.array(imagen)

# Normalizar la imagen (si tu modelo espera imágenes normalizadas)
imagen_normalizada = imagen_np / 255.0

# Añadir una dimensión adicional para que tenga la forma correcta para tu modelo
imagen_final = np.expand_dims(imagen_normalizada, axis=0)

# Convertir la imagen a tipo float32
imagen_final = imagen_final.astype('float32')

plt.imshow(imagen_np, cmap='gray')
plt.title("Imagen de entrada")
plt.show()

# Realizar la predicción
prediccion = modelo.predict(imagen_final)

# Obtener la etiqueta predicha
etiqueta_predicha = nombres_clases[np.argmax(prediccion)]
print(etiqueta_predicha)