from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os
from model.food_model import predict_diseases  # Import your disease prediction function

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()

# Serve static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")



templates = Jinja2Templates(directory="app/templates")

# Load the machine learning model
model = joblib.load('model/xgboost_best_model.pkl')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, age: int = Form(...), gender: int = Form(...), 
                  bp: float = Form(...), cholesterol: float = Form(...), heart_rate: float = Form(...),
                  glucose: float = Form(...), insulin: float = Form(...), bmi: float = Form(...)):
    # Prepare user input as a dictionary
    user_input = {
        'Age': age,
        'Gender': gender,
        'BP': bp,
        'Cholesterol': cholesterol,
        'Heart Rate': heart_rate,
        'Glucose': glucose,
        'Insulin': insulin,
        'BMI': bmi
    }

    # Predict diseases and get food recommendations
    predicted_diseases, combined_food_recommendations = predict_diseases(user_input)

    # Render the results in the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_diseases": predicted_diseases,
        "food_recommendations": combined_food_recommendations
    })
