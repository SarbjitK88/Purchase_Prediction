from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import joblib, json
import pandas as pd
import numpy as np

app = FastAPI(title="Purchase Amount Predictor", version="1.0.0")

# ---------- Load model + columns at startup (singleton) ----------
MODEL_PATH = "xgb_purchase_model.joblib"
COLUMNS_PATH = "xgb_purchase_model.columns.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, "r") as f:
        MODEL_COLUMNS = json.load(f)  # list[str] in the exact training order
except Exception as e:
    # Fail fast if artefacts are missing
    raise RuntimeError(f"Failed to load artefacts: {e}")

# ---------- Schemas ----------
class Row(BaseModel):
    # Flexible: accept any feature dict so you don't need to hard-code schema
    # Example keys you likely have: Gender, Age, Marital_Status, Occupation,
    # Stay_In_Current_City_Years, B, C, Product_Category_1/2/3
    data: Dict[str, Any]

class Batch(BaseModel):
    rows: List[Row]

# Helpers 
def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Reindex to training columns (missing -> 0, extras -> dropped)
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # Make sure everything is numeric; coerce errors to NaN then fill with 0
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Safety: ensure column order matches training exactly
    assert list(df.columns) == list(MODEL_COLUMNS)
    return df

# --- Routes ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Purchase prediction API",
        "model": "XGBoost",
        "features_expected": MODEL_COLUMNS,
    }

@app.post("/predict")
def predict_one(row: Row):
    try:
        df = to_dataframe([row.data])
        pred = model.predict(df)
    
        return {"prediction": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(payload: Batch):
    try:
        rows = [r.data for r in payload.rows]
        df = to_dataframe(rows)
        preds = model.predict(df)
        return {"predictions": [float(x) for x in preds]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "up"}