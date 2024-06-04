# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import PlantDiseaseModel, prompt_disease
import tempfile
import shutil
import os

app = FastAPI()

# Initialize the model with the path to the TFLite model file
model_path = "plant_model.tflite"  # Replace with your model path
model = PlantDiseaseModel(model_path=model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Prediction API"}

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Make a prediction using the saved file
        predicted_class = model.predict_image(temp_file_path)
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/disease-info/{disease}")
def get_disease_info(disease: str):
    try:
        info = prompt_disease(disease)
        return JSONResponse(content={"disease_info": info})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# http://127.0.0.1:8000/docs