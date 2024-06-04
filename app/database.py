import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('dataset/recomendacion_supermercado_id.csv')

# Cargar la matriz de recomendaciones
matriz_recomendaciones_long = pd.read_pickle("dataset/matriz_recomendaciones_long.pkl")