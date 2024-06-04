import tensorflow as tf
import numpy as np
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from app.database import df

try:
    model = tf.keras.models.load_model("./models/best_model.h5")
    print("Modelo cargado exitosamente.")
except OSError as e:
    print(f"Error al cargar el modelo: {e}")

def proc_prices():
    # prices_path = 'dataset/verduras_normalizadas.csv'
    prices_df = pd.read_csv('dataset/VerdurasNormalizadas.csv')
    prices_df['Precio'] = prices_df['Precio'].str.replace(r'[^0-9]', '', regex=True)
    prices_df['Precio'] = pd.to_numeric(prices_df['Precio'], errors='coerce') / 100
    return prices_df


def train():
    # Directorios de datos
    base_dir = 'dataset/reorganized_dataset'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Parámetros
    img_height, img_width = 150, 150
    batch_size = 32
    epochs = 25

    # Generadores de datos con aumentación
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Utilizamos el 20% de los datos para validación
    )

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
 
    return train_generator



def supermercado_mas_barato(producto, prices_df):
    precios_producto = prices_df[prices_df['Producto'] == producto]
    supermercado_mas_barato = precios_producto.loc[precios_producto['Precio'].idxmin(), 'Supermercado']
    precio_mas_bajo = precios_producto['Precio'].min()
    return supermercado_mas_barato, precio_mas_bajo


def procesar_imagen(imagen_path):
    prices_df = proc_prices()
    img = load_img(imagen_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array /= 255.0

    prediction = model.predict(img_array)
    clase_predicha = np.argmax(prediction)



    # Obtener clases
    train_generator = train()
    if train_generator:
        clases = train_generator.class_indices
        frutas_verduras = {v: k for k, v in clases.items()}
        producto = frutas_verduras[clase_predicha]

        # Encontrar el supermercado más barato para el producto dado
        supermercado, precio = supermercado_mas_barato(producto, prices_df)
        return producto, supermercado, precio
    else:
        return None, None, None


