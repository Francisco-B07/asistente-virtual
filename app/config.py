from pathlib import Path

# Directorio donde se guardarán las imágenes subidas
UPLOAD_DIR = Path("./public/static/uploads/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)