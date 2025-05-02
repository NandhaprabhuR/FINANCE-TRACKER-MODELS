from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load your trained model
try:
    model = joblib.load("expense_classifier.pkl")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


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
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")

    try:
        # Perform prediction
        prediction = model.predict([input.text])[0]
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
