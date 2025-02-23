from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
from fastapi import Request

# Initialize FastAPI
app = FastAPI()

# Load Model
MODEL_PATH = "models/plant_model.h5"
model = load_model(MODEL_PATH)

# Define Classes
CLASS_NAMES = ["Healthy", "Diseased", "Powdery"]

# Set up Templates (for Frontend)
templates = Jinja2Templates(directory="templates")

# Serve Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root Route - Show HTML Form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction Route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load Image
        img = image.load_img(file_path, target_size=(150, 150))  # Adjust as per model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if required

        # Predict
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        return {"filename": file.filename, "prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}

# Run the API Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
