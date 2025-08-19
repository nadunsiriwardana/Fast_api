

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Iris Classification API",
    description="API to predict Iris flower species"
)


# Create FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="API to predict Iris flower species"
)

# Load trained model at startup
model = joblib.load("model.pkl")

# Input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float = None

# 1️⃣ Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

# 2️⃣ Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        
        # Make prediction
        pred_class_index = model.predict(features)[0]
        pred_class_name = ["setosa", "versicolor", "virginica"][pred_class_index]
        
        # Prediction probability
        confidence = float(np.max(model.predict_proba(features)))
        
        return PredictionOutput(prediction=pred_class_name, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 3️⃣ Model info endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"]
    }
