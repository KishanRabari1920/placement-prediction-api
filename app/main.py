from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# CREATE FASTAPI APP
app = FastAPI()

# LOAD TRAINED MODEL
print("Loading Trained Model...")

with open("../model/placement_model.pkl", "rb") as file:
    model = pickle.load(file)
print("Model Loaded Successfully")

# REQUEST BODY
class StudentData(BaseModel):
    cgpa: float
    iq: int
    communication: int
    internship: int

# HOME ROUTE
@app.get("/")
def home():
    return {
        "message": "Placement Prediction API Running Successfully"
    }

# PREDICTION ROUTE
@app.post("/predict")
def predict(data: StudentData):

    # convert input into numpy array
    input_data = np.array([
        [
            data.cgpa,
            data.iq,
            data.communication,
            data.internship
        ]
    ])

    # prediction
    prediction = model.predict(input_data)

    # probability (optional)
    try:
        probability = model.predict_proba(input_data)
        confidence = round(
            max(probability[0]) * 100,
            2
        )
    except:
        confidence = None

    # result
    if prediction[0] == 1:
        result = "Placed"
    else:
        result = "Not Placed"

    # return response
    return {
        "prediction": result,
        "confidence_percentage": confidence,
        "input_data": {
            "cgpa": data.cgpa,
            "iq": data.iq,
            "communication": data.communication,
            "internship": data.internship
        }
    }

# HEALTH CHECK ROUTE
@app.get("/health")
def health_check():
    return {
        "status": "API is healthy",
        "model_loaded": True
    }