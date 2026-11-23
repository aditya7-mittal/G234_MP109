
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

# Configure JSON-compatible structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","message":"%(message)s"}'
)

logger = logging.getLogger(__name__)

# Load trained champion model
model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(
    title="MP109 Used Two-Wheeler Price Prediction API",
    description="Predicts used two-wheeler resale prices in thousand rupees.",
    version="1.0.0"
)


# Request schema
class BikeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float
    km_driven: float
    engine_cc: int = Field(le=800)
    owners: int
    service_records: int = Field(ge=0)
    accidents: int
    brand_tier: str
    city_zone: str


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(request: BikeRequest):

    input_data = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_data)[0]

    logger.info("Prediction request processed successfully")

    return {
        "prediction": round(float(prediction), 2)
    }
