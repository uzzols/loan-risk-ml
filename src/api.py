from pathlib import Path
from typing import Optional, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_loan_risk_model.joblib"

app = FastAPI(title="Loan Risk Prediction API")

model = None


class LoanApplication(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        print(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")

    return model


def get_risk_level(probability: Optional[float]) -> str:
    if probability is None:
        return "Unknown Risk"

    if probability >= 0.90:
        return "Very High Risk"
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.50:
        return "Moderate Risk"
    return "Low Risk"


def get_risk_drivers(data: LoanApplication) -> List[str]:
    drivers = []

    if data.DTIRatio >= 35:
        drivers.append("High debt-to-income ratio")

    if data.InterestRate >= 10:
        drivers.append("High interest rate")

    if data.CreditScore < 620:
        drivers.append("Lower credit score")

    if data.Income < 40000:
        drivers.append("Lower annual income")

    if data.Income > 0 and data.LoanAmount > data.Income * 4:
        drivers.append("Loan amount is high compared with income")

    if data.MonthsEmployed < 12:
        drivers.append("Short employment history")

    return drivers


def build_explanation(probability: Optional[float], drivers: List[str]) -> str:
    if drivers:
        return "Risk is elevated due to " + ", ".join(drivers).lower() + "."

    if probability is not None and probability >= 0.85:
        return (
            "The model identifies this application as high risk based on the overall "
            "borrower and loan profile, even though no single rule-based driver stands out."
        )

    if probability is not None and probability < 0.50:
        return (
            "This application appears lower risk based on the provided borrower, income, "
            "credit, employment, and loan details."
        )

    return (
        "The application shows moderate risk. Review the full borrower profile before "
        "making a decision."
    )


@app.get("/")
def root():
    return {"message": "Loan Risk Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_loan_risk(application: LoanApplication):
    try:
        loaded_model = get_model()

        input_df = pd.DataFrame([application.model_dump()])

        prediction = loaded_model.predict(input_df)[0]

        probability = None
        if hasattr(loaded_model, "predict_proba"):
            probability = float(loaded_model.predict_proba(input_df)[0][1])

        risk_level = get_risk_level(probability)
        top_risk_drivers = get_risk_drivers(application)
        explanation = build_explanation(probability, top_risk_drivers)

        return {
            "prediction": int(prediction),
            "default_risk_probability": probability,
            "risk_level": risk_level,
            "top_risk_drivers": top_risk_drivers,
            "explanation": explanation,
        }

    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")