# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from model import plant_disease_model
from pydantic import BaseModel

app = FastAPI()

# Initialize the model with the path to the TFLite model file
model_path = "model_fix2"  # Replace with your model path
model = plant_disease_model(model_path=model_path)

class ImageData(BaseModel):
    base64_encoded: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Prediction API"}

@app.post("/predict")
async def predict_plant_disease(image_data: ImageData):
    try:
        # Make a prediction using the base64 encoded image data
        base64_encoded = image_data.base64_encoded
        predicted_class = model.predict_tf(base64_encoded)
        
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/disease-info/{disease}")
def get_disease_info(disease: str):
    try:
        info = plant_disease_model.prompt_disease(disease)
        return JSONResponse(content={"disease_info": info})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

# http://localhost:8000/docs