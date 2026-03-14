"""
Bank Fraud Detection System
Console application for simulating transactions
and detecting fraud using Machine Learning.

Author: Felipe
"""

import joblib
import pandas as pd
import logging
import sys


# =============================
# LOGGING CONFIG
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =============================
# LOAD MODEL
# =============================

try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    logging.info("Model and scaler loaded successfully")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit()


# =============================
# SAFE INPUT
# =============================

def get_float_input(name):
    """Safely ask for numeric input"""

    while True:
        try:
            value = float(input(f"{name}: "))
            return value
        except ValueError:
            print("❌ Please enter a valid number.")


# =============================
# SIMULATE TRANSACTION
# =============================

def simulate_transaction():
    """Ask user for transaction data"""

    logging.info("Starting new transaction input")

    data = {}

    data["Time"] = get_float_input("Time")
    data["Amount"] = get_float_input("Amount")

    for i in range(1, 29):
        data[f"V{i}"] = get_float_input(f"V{i}")

    return pd.DataFrame([data])


# =============================
# PREPROCESS
# =============================

def preprocess(df):
    """Prepare data for prediction"""

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    df = df[cols]

    df[["Amount", "Time"]] = scaler.transform(
        df[["Amount", "Time"]]
    )

    logging.info("Data preprocessed")

    return df


# =============================
# PREDICT
# =============================

def predict(transaction):
    """Predict fraud probability"""

    prob = model.predict_proba(transaction)[0][1]

    pred = 1 if prob > 0.3 else 0

    logging.info(f"Prediction done | Probability: {prob:.4f}")

    return pred, prob


# =============================
# MAIN SYSTEM
# =============================

def main():
    """Main loop"""

    print("\n💳 BANK FRAUD DETECTOR")
    print("======================\n")

    logging.info("System started")

    while True:

        print("Enter transaction data:\n")

        df = simulate_transaction()

        df = preprocess(df)

        pred, prob = predict(df)

        print("\n------------------")

        if pred == 1:
            print("⚠️ FRAUD DETECTED")
        else:
            print("✅ NORMAL PAYMENT")

        print(f"Probability: {prob:.4f}")

        print("------------------\n")

        again = input("Another test? (y/n): ")

        if again.lower() != "y":
            logging.info("System closed by user")
            print("\n👋 Goodbye!")
            break


# =============================
# RUN
# =============================

if __name__ == "__main__":
    main()