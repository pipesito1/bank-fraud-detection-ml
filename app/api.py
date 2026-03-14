"""
Fraud Detection API
Author: Felipe
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# =============================
# LOAD MODEL
# =============================

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

print("✅ Model loaded")

# =============================
# APP
# =============================

app = Flask(__name__)
CORS(app)

# =============================
# PREPROCESS
# =============================

def preprocess(df):

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    df = df[cols]

    df[["Amount", "Time"]] = scaler.transform(
        df[["Amount", "Time"]]
    )

    return df


# =============================
# ROUTES
# =============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fraud Detection API Running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    df = pd.DataFrame([data])

    df = preprocess(df)

    prob = model.predict_proba(df)[0][1]

    pred = 1 if prob > 0.3 else 0

    result = "FRAUD" if pred == 1 else "NORMAL"

    return jsonify({
        "prediction": result,
        "probability": round(float(prob), 4)
    })


# =============================
# RUN
# =============================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

