import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import json

# Define las dimensiones de las imágenes de entrada
imagen_alto = 480
imagen_ancho = 640

# Define el número de personas en tus datos de entrenamiento
num_personas = 19  # Ajusta este valor al número real de personas en tus datos

personas = {}

# Función para cargar las imágenes y etiquetas de las personas
def cargar_imagenes_y_etiquetas(directorio):
    imagenes = []
    etiquetas = []

    for etiqueta, nombre_persona in enumerate(os.listdir(directorio)):
        carpeta_persona = os.path.join(directorio, nombre_persona)
        if os.path.isdir(carpeta_persona):
            for imagen_nombre in os.listdir(carpeta_persona):
                imagen_path = os.path.join(carpeta_persona, imagen_nombre)
                imagen = cv2.imread(imagen_path)
                imagenes.append(imagen)
                etiquetas.append(etiqueta)
                personas[etiqueta] = nombre_persona
    
    #Write the person names to a json file
    with open('personas.json', 'w') as file:
        json.dump(personas, file)

    imagenes_entrenamiento, etiquetas_entrenamiento = np.array(imagenes), np.array(etiquetas)
    
    # Normalizar imágenes
    imagenes_entrenamiento = imagenes_entrenamiento / 255.0

    # Crear el modelo de reconocimiento facial
    modelo = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imagen_alto, imagen_ancho, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_personas, activation='softmax')
    ])

    # Compilar el modelo
    modelo.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar el modelo
    modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=150)

    # Guardar el modelo entrenado
    modelo.save("modelo_reconocimiento_facial.h5")

def load_models():

    # Cargar el modelo entrenado
    modelo_cargado = models.load_model("modelo_reconocimiento_facial.h5")
    
    #Load the person names from the json file
    with open('personas.json', 'r') as file:
        personas = json.load(file)

    # Ruta de la imagen de entrada a identificar
    imagen_a_identificar = "01.png"

    # Cargar la imagen de entrada
    imagen_entrada = cv2.imread(imagen_a_identificar)
    imagen_entrada = cv2.resize(imagen_entrada, (imagen_ancho, imagen_alto))
    imagen_entrada = np.expand_dims(imagen_entrada, axis=0)

    # Realizar la identificación de la persona en la imagen de entrada
    prediccion = modelo_cargado.predict(imagen_entrada)
    etiqueta_identificada = np.argmax(prediccion)

    print("La persona identificada es:", personas[str(etiqueta_identificada)])

#Function to reverse all the images in every data folder and save as new images
def reverse_images():
    
    directorio_entrenamiento = "data"
    
    
    for etiqueta, nombre_persona in enumerate(os.listdir(directorio_entrenamiento)):
        carpeta_persona = os.path.join(directorio_entrenamiento, nombre_persona)
        if os.path.isdir(carpeta_persona):
            for imagen_nombre in os.listdir(carpeta_persona):
                imagen_path = os.path.join(carpeta_persona, imagen_nombre)
                imagen = cv2.imread(imagen_path)
                imagen = cv2.flip(imagen, 1)
                # Agregar sufijo al nombre del archivo antes de guardarlo
                imagen_path_flipped = os.path.join(carpeta_persona, "flipped_" + imagen_nombre)
                cv2.imwrite(imagen_path_flipped, imagen)

# Directorio que contiene las imágenes de entrenamiento
# directorio_entrenamiento = "data"
# cargar_imagenes_y_etiquetas(directorio_entrenamiento)

# load_models()

reverse_images()
