from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load both models
ridge_model = joblib.load("model_1.pkl")  # Replace with actual path if different
rf_model = joblib.load("model_2.pkl")        # Replace with actual path if different

# Define input schema
class HousePredictionRequest(BaseModel):
    town: str
    list_year: int
    assessed_value: float
    property_type: str
    residential_type: str
    model_type: str  # "ridge" or "random_forest"

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: HousePredictionRequest):
    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'Town': [request.town],
        'List Year': [request.list_year],
        'Assessed Value': [request.assessed_value],
        'Property Type': [request.property_type],
        'Residential Type': [request.residential_type]
    })

    # Select model based on model_type
    if request.model_type == "ridge":
        model = ridge_model
    elif request.model_type == "random_forest":
        model = rf_model
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'ridge' or 'random_forest'.")

    # Make prediction
    try:
        predicted_price_log = model.predict(input_data)
        predicted_price = np.expm1(predicted_price_log)[0]  # Convert back from log scale if necessary
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API! Use /predict to get a prediction."}
