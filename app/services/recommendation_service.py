from app.database import matriz_recomendaciones_long
from fastapi import HTTPException

def hacer_recomendacion(user_id: int, n: int = 3):
    if user_id in matriz_recomendaciones_long['id1'].unique():
        recomendaciones = matriz_recomendaciones_long[matriz_recomendaciones_long['id1'] == user_id]
        recomendaciones = recomendaciones.sort_values(by='similitud', ascending=False).head(n)
        return recomendaciones.to_dict(orient="records")
    else:
        raise HTTPException(status_code=404, detail=f"Error: El ID {user_id} no se encuentra en las columnas del DataFrame.")