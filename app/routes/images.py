from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from app.config import UPLOAD_DIR
from app.services.image_service import procesar_imagen
import shutil
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Configuración de las plantillas
templates = Jinja2Templates(directory="app/templates")

@router.get("/upload-image", response_class=HTMLResponse)
async def cargar_img_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / "imagen.jpg"
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/search-by-image", status_code=303)

@router.get("/search-by-image")
async def search_by_image(request: Request):
    imagen_path = './public/static/uploads/imagen.jpg'
    result = procesar_imagen(imagen_path)
    if result:
        producto, supermercado, precio = result
        return templates.TemplateResponse("search-by-image.html", {"request": request, "supermercado": supermercado, "precio": precio, "producto": producto})
    return templates.TemplateResponse("search-by-image.html", {"request": request})