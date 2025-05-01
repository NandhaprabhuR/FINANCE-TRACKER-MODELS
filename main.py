from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load your trained model
model = joblib.load("expense_classifier.pkl")

# Define request schema
class InputText(BaseModel):
    text: str

# Create FastAPI app
app = FastAPI()

# Enable CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origin like ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction route
@app.post("/predict")
async def predict_category(input: InputText):
    prediction = model.predict([input.text])[0]
    category = prediction  # Direct use if model returns label string
    return {"category": category}
