from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import hacer_recomendacion

router = APIRouter()

@router.get("/recomendaciones/{user_id}")
async def recomendaciones(user_id: int, n: int = 3):
    return hacer_recomendacion(user_id, n)