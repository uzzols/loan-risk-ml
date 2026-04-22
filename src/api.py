from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_loan_risk_model.joblib"

app = FastAPI(title="Loan Risk Prediction API")

# Load trained model when app starts
model = joblib.load(MODEL_PATH)


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


@app.get("/")
def root():
    return {"message": "Loan Risk Prediction API is running"}


@app.post("/predict")
def predict_loan_risk(application: LoanApplication):
    input_df = pd.DataFrame([application.model_dump()])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    else:
        probability = None

    return {
        "prediction": int(prediction),
        "default_risk_probability": probability
    }