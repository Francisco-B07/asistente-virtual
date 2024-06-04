from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import products, users, recommendations, images
from app.config import UPLOAD_DIR

from app.database import df

app = FastAPI()

# Montar el directorio estático
app.mount("/static", StaticFiles(directory="./public/static"), name="static")

# Configuración de las plantillas
templates = Jinja2Templates(directory="app/templates")

# Crear el directorio de subida si no existe
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Incluir rutas
app.include_router(products.router)
app.include_router(users.router)
app.include_router(recommendations.router)
app.include_router(images.router)

# Devolver todos los usuarios
def load_users():
    users = df['UserID'].drop_duplicates() 
    return users.tolist()

@app.get("/")
async def read_root(request: Request):
    user_ids = load_users()
    return templates.TemplateResponse("index.html", {"request": request, "user_ids": user_ids})