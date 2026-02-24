from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model_over = joblib.load(os.path.join(BASE_DIR, "model_overtrading.pkl"))
model_panic = joblib.load(os.path.join(BASE_DIR, "model_panic.pkl"))
model_short = joblib.load(os.path.join(BASE_DIR, "model_short.pkl"))

class Transaction(BaseModel):
    investor_id: str
    date: str
    action: str
    total_value: float
    units: float

class TransactionList(BaseModel):
    transactions: List[Transaction]

@app.get("/")
def home():
    return {"status": "Investor Bias Detection API running"}

@app.post("/predict")
def predict(payload: TransactionList):
    df = pd.DataFrame([t.dict() for t in payload.transactions])

    # Feature engineering — exact same as your app.py
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["investor_id", "date"])

    trade_frequency = df.groupby("investor_id").size()

    df["prev_date"] = df.groupby("investor_id")["date"].shift(1)
    df["holding_days"] = (df["date"] - df["prev_date"]).dt.days
    holding_duration = df.groupby("investor_id")["holding_days"].mean()

    buy_count = df[df["action"] == "Buy"].groupby("investor_id").size()
    sell_count = df[df["action"] == "Sell"].groupby("investor_id").size()
    sell_buy_ratio = (sell_count / buy_count).fillna(0)

    avg_trade_value = df.groupby("investor_id")["total_value"].mean()

    features = pd.concat([
        trade_frequency.rename("trade_frequency"),
        holding_duration.rename("avg_holding_duration"),
        sell_buy_ratio.rename("sell_buy_ratio"),
        avg_trade_value.rename("avg_trade_value")
    ], axis=1).fillna(0)

    # Predictions — exact same as your app.py
    features["overtrading_prob"] = model_over.predict_proba(
        features[["avg_holding_duration", "sell_buy_ratio", "avg_trade_value"]]
    )[:, 1]

    features["panic_prob"] = model_panic.predict_proba(
        features[["avg_holding_duration", "trade_frequency", "avg_trade_value"]]
    )[:, 1]

    features["short_prob"] = model_short.predict_proba(
        features[["trade_frequency", "sell_buy_ratio", "avg_trade_value"]]
    )[:, 1]

    # Risk score — exact same as your app.py
    features["risk_score"] = (
        features["overtrading_prob"] * 0.5 +
        features["panic_prob"] * 0.3 +
        features["short_prob"] * 0.2
    ) * 100
    features["risk_score"] = features["risk_score"].round(1)

    # Dominant bias — exact same as your app.py
    def detect_dominant_bias(row):
        max_prob = max(
            row["overtrading_prob"],
            row["panic_prob"],
            row["short_prob"]
        )
        if max_prob < 0.2:
            return "No Significant Bias"
        elif row["overtrading_prob"] == max_prob:
            return "Overtrading Bias"
        elif row["panic_prob"] == max_prob:
            return "Panic Selling Bias"
        else:
            return "Short-Term Bias"

    features["dominant_bias"] = features.apply(detect_dominant_bias, axis=1)

    # Risk category — exact same as your app.py
    def risk_category(score):
        if score >= 70:
            return "High Risk"
        elif score >= 40:
            return "Moderate Risk"
        else:
            return "Low Risk"

    features["risk_category"] = features["risk_score"].apply(risk_category)

    return features.reset_index().to_dict(orient="records")