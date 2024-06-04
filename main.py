import os
import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
import gdown

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.responses import JSONResponse

app = FastAPI()

# Directorio donde se guardarán las imágenes subidas
UPLOAD_DIR = Path("./public/static/uploads/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Montar el directorio estático
app.mount("/static",StaticFiles(directory="./public/static"),name="static")

# Configuración de las plantillas
templates = Jinja2Templates(directory="./public/templates")

# Configurar CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500"],  # Lista de orígenes permitidos
#     allow_credentials=True,
#     allow_methods=["*"],  # Métodos permitidos
#     allow_headers=["*"],  # Cabeceras permitidas
# )

# Leer el archivo CSV
df = pd.read_csv('recomendacion_supermercado_id.csv')



# ------------------------------FUNCIONES------------------------------

# Carga la matriz de recomendaciones 
matriz_recomendaciones_long = pd.read_pickle("matriz_recomendaciones_long.pkl")


# Devolver n productos unicos
def get_products():    
    products = df.head(100)
    df_unique  = products.drop_duplicates(subset='ProductID')
    return df_unique 

# Devolver todos los usuarios
def load_users():
    users = df['UserID'].drop_duplicates() 
    return users.tolist()



# # -------Procesamiento de imagenes-------

# Nombre del archivo a descargar y cargar
archivo_modelo = 'best_model.h5'

# Verificar si el archivo ya existe en el sistema
if not os.path.exists(archivo_modelo):
    # URL del archivo en Google Drive
    url = 'https://drive.google.com/uc?id=1dqMwAW4ZTlZgs04rEVcZ7454ZthHta7u'
    
    # Descargar el archivo
    gdown.download(url, archivo_modelo, quiet=False)
else:
    print(f"El archivo '{archivo_modelo}' ya existe en el sistema.")

# # Cargar el modelo
# if os.path.exists(archivo_modelo):
#     model = tf.keras.models.load_model("./best_model.h5")
#     print("Modelo cargado exitosamente.")
#     print(model.summary())
# else:
#     print("No se pudo cargar el modelo porque el archivo no existe.")



# # Limpiar y convertir los precios a valores numéricos
# def proc_prices():
#     # Ruta del archivo CSV
#     prices_path = './VerdurasNormalizadas.csv'

#     # Cargar el archivo CSV
#     prices_df = pd.read_csv(prices_path)

#     # Limpiar y convertir los precios a valores numéricos
#     prices_df['Precio'] = prices_df['Precio'].str.replace(r'[^0-9]', '', regex=True)
#     prices_df['Precio'] = pd.to_numeric(prices_df['Precio'], errors='coerce') / 100

#     # Imprimir el DataFrame resultante
#     # print(prices_df)

#     return prices_df



# def gen_data():
#     # Directorio base actual donde se encuentran las imágenes aumentadas
#     current_augmented_data_dir = './dataset/augmented_data'

#     # Nuevo directorio base donde se reorganizarán las imágenes aumentadas
#     new_augmented_data_dir = './reorganized_dataset'
#     os.makedirs(new_augmented_data_dir, exist_ok=True)  # Crear el directorio si no existe

#     # Recorrer cada supermercado y cada fruta/verdura en el directorio actual
#     for supermarket in os.listdir(current_augmented_data_dir):
#         supermarket_path = os.path.join(current_augmented_data_dir, supermarket)
#         if os.path.isdir(supermarket_path):
#             for fruit in os.listdir(supermarket_path):
#                 fruit_path = os.path.join(supermarket_path, fruit)
#                 if os.path.isdir(fruit_path):
#                     for img_name in os.listdir(fruit_path):
#                         img_path = os.path.join(fruit_path, img_name)
#                         try:
#                             # Crear la nueva ruta para la imagen
#                             new_dir_path = os.path.join(new_augmented_data_dir, fruit, supermarket)
#                             os.makedirs(new_dir_path, exist_ok=True)  # Crear el directorio si no existe

#                             # Mover la imagen a la nueva ubicación
#                             new_img_path = os.path.join(new_dir_path, img_name)
#                             shutil.move(img_path, new_img_path)
#                         except Exception as e:
#                             print(f"Error moviendo la imagen {img_path}: {e}")

#     print("Reorganización completa.")

# def train():
#     # Directorios de datos
#     base_dir = './reorganized_dataset'
#     train_dir = os.path.join(base_dir, 'train')
#     validation_dir = os.path.join(base_dir, 'validation')

#     # Parámetros
#     img_height, img_width = 150, 150
#     batch_size = 32
#     epochs = 25

#     # Generadores de datos con aumentación
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest',
#         validation_split=0.2 # Utilizamos el 20% de los datos para validación
#     )

#     train_generator = train_datagen.flow_from_directory(
#         base_dir,
#         target_size=(img_height, img_width),
#         batch_size=batch_size,
#         class_mode='categorical',
#         subset='training'
#     )
 
#     return train_generator


# # Función para encontrar el supermercado más barato para un producto dado
# def supermercado_mas_barato(producto, prices_df):
#     precios_producto = prices_df[prices_df['Producto'] == producto]
#     supermercado_mas_barato = precios_producto.loc[precios_producto['Precio'].idxmin(), 'Supermercado']
#     precio_mas_bajo = precios_producto['Precio'].min()
#     return supermercado_mas_barato, precio_mas_bajo


# # Función para clasificar una imagen y encontrar el supermercado más barato para esa categoría
# def clasificar_y_encontrar_supermercado(imagen_path, prices_df):
#     # Cargar la imagen y preprocesarla
#     img = load_img(imagen_path, target_size=(150, 150))
#     img_array = img_to_array(img)
#     img_array = img_array.reshape((1,) + img_array.shape)
#     img_array /= 255.0 # Normalizar los valores de píxeles

#     # Clasificar la imagen utilizando el modelo
#     prediction = model.predict(img_array)
#     clase_predicha = np.argmax(prediction)

#     # Mapear la clase predicha al nombre de la fruta o verdura
#     # gen_data()
#     train_generator = train()
#     print("train",train_generator)
#     if train_generator:
#         clases = train_generator.class_indices
#         frutas_verduras = {v: k for k, v in clases.items()}
#         producto = frutas_verduras[clase_predicha]

#         # Encontrar el supermercado más barato para el producto dado
#         supermercado, precio = supermercado_mas_barato(producto, prices_df)
#         return producto, supermercado, precio
#     else:
#         return None, None, None



# ------------------------------RUTAS------------------------------

@app.get("/")
async def read_root(request: Request):
    user_ids = load_users()
    return templates.TemplateResponse("index.html", {"request": request, "user_ids": user_ids})

@app.get("/search-by-text")
async def read_root(request: Request):
    return templates.TemplateResponse("search-by-text.html", {"request": request})

@app.get("/upload-image", response_class=HTMLResponse)
async def cargar_img_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.get("/search-by-image")
async def read_root(request: Request):
    # Ruta de la imagen que deseas clasificar y comparar precios
    imagen_path = './public/static/uploads/imagen.jpg' 

    # prices_df = proc_prices()

    # Clasificar la imagen y encontrar el supermercado más barato
    # if imagen_path:
    #     producto, supermercado, precio = clasificar_y_encontrar_supermercado(imagen_path, prices_df)

    # Imprimir el resultado
    # if producto:
    #     return templates.TemplateResponse("search-by-image.html", {"request": request, "supermercado": supermercado, "precio": precio, "producto": producto})
    # else:
        # return templates.TemplateResponse("search-by-image.html", {"request": request})
    return templates.TemplateResponse("search-by-image.html", {"request": request})



@app.get("/product/{productId}/{userId}",response_class=HTMLResponse)
def product(request: Request, productId:int, userId:str):
   
    producto =  df.loc[df['ProductID'] == productId].to_dict(orient='records')
    print(producto)
    return templates.TemplateResponse("product.html",{"request":request, "productId": productId, "userId": userId, "producto": producto[0]})




# ------------------------------END POINTS------------------------------

# Devuelve todos los productos
@app.get("/products")
async def all_products():
    products = get_products()
    return products.to_dict(orient='records')

# Devuelve un producto
@app.get("/product/{productId}")
async def ver_producto(productId: int):

    producto =  df.loc[df['ProductID'] == productId].to_dict(orient='records')

    if producto:
        return producto[0]
    return {"error": "Producto no encontrado"}

# Devuelve las recomendaciones del usuario logueado
@app.get("/recomendaciones/{user_id}")
async def hacer_recomendacion(user_id: int, n: int = 3):
    # Verifica que el usuario exista en la matriz
    if user_id in matriz_recomendaciones_long['id1'].unique():
        # Filtra donde 'id1' sea igual al id proporcionado
        recomendaciones = matriz_recomendaciones_long[matriz_recomendaciones_long['id1'] == user_id]
        
        # Ordena por similitud de manera descendente y selecciona los primeros n resultados
        recomendaciones = recomendaciones.sort_values(by='similitud', ascending=False).head(n)
        
        return recomendaciones.to_dict(orient="records")
    else:
        raise HTTPException(status_code=404, detail=f"Error: El ID {user_id} no se encuentra en las columnas del DataFrame.")


# Cargar imagenes
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Ruta donde se guardará el archivo subido
    file_path = UPLOAD_DIR / "imagen.jpg"
    
    # Guardar el archivo en el sistema de archivos
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Devolver una respuesta con la información del archivo subido
    return RedirectResponse(url="/search-by-image", status_code=303)

# Obtener imagen
@app.get("/get-image")
def getImage():
    # URL de la imagen guardada
    image_url = f"./public/static/uploads/imagen.jpg"
    
    # Devolver una respuesta con la URL de la imagen
    return JSONResponse(content={"image_url": image_url})




