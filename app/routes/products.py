from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from app.services.product_service import get_products, get_product_by_id
from app.database import df
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Configuración de las plantillas
templates = Jinja2Templates(directory="app/templates")

@router.get("/products")
async def all_products():
    products = get_products()
    return products.to_dict(orient='records')

@router.get("/product/{productId}")
async def ver_producto(productId: int):
    producto = get_product_by_id(productId)
    if producto:
        return producto
    raise HTTPException(status_code=404, detail="Producto no encontrado")

@router.get("/product/{productId}/{userId}", response_class=HTMLResponse)
def product(request: Request, productId: int, userId: str):
    producto = df.loc[df['ProductID'] == productId].to_dict(orient='records')
    return templates.TemplateResponse("product.html", {"request": request, "productId": productId, "userId": userId, "producto": producto[0]})