"""Minimal churn-scoring API. Run with: uvicorn app:app --reload
Expects churn_pipeline.joblib and churn_model_metadata.json in the same directory."""
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Telco Churn Scoring API")
pipeline = joblib.load("churn_pipeline.joblib")
with open("churn_model_metadata.json") as f:
    METADATA = json.load(f)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": METADATA["model_version"]}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    proba = float(pipeline.predict_proba(row)[0, 1])
    threshold = METADATA["business_threshold"]
    tier = "High" if proba >= 0.5 else ("Elevated" if proba >= threshold else "Low")
    return {"churn_probability": round(proba, 4), "risk_tier": tier,
            "flagged_at_business_threshold": proba >= threshold}
