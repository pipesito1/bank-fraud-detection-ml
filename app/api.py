"""
Fraud Detection API
Author: Felipe
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# =============================
# LOAD MODEL
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


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
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )