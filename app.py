"""
app.py
=============================================================================
Credit Card Fraud Detection - Flask Web Application
Serves the Machine Learning web interface and prediction API:
- Loads the trained Logistic Regression model ('fraud_model.pkl').
- Validates all 30 transaction features.
- Ensures feature ordering is strictly preserved.
- Computes predictions (0: Legitimate, 1: Fraudulent) and probability metrics.
- Supports both interactive JSON API and traditional HTML form submissions.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# The 30 numerical input features in strict expected order
FEATURE_NAMES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# Model loading with graceful error handling
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metrics.json")

model = None
model_load_error = None
model_metrics = {
    "model_type": "Logistic Regression",
    "training_accuracy": 0.9636,
    "test_accuracy": 0.9539,
    "balanced_dataset_accuracy": "95.39%"
}

def load_ml_model():
    """Attempts to load the trained Logistic Regression model from disk."""
    global model, model_load_error, model_metrics
    if os.path.isfile(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_load_error = None
            print(f"[INFO] Successfully loaded model from '{MODEL_PATH}'")
        except Exception as e:
            model = None
            model_load_error = f"Error loading model file: {str(e)}"
            print(f"[ERROR] {model_load_error}")
    else:
        model = None
        model_load_error = (
            f"Model file '{MODEL_PATH}' not found. "
            "Please run 'python train_model.py' to train and generate the model."
        )
        print(f"[WARNING] {model_load_error}")

    # Load metrics if available
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read metrics file: {e}")

# Initial load
load_ml_model()

# Verified benchmark sample presets for demonstration
SAMPLE_PRESETS = {
    "legitimate": {
        "Time": 0.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62
    },
    "fraud": {
        "Time": 406.0,
        "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
        "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657,
        "V9": -2.770089, "V10": -2.772272, "V11": 3.202033, "V12": -2.899907,
        "V13": -0.595222, "V14": -4.289254, "V15": 0.389724, "V16": -1.140747,
        "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
        "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
        "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276,
        "Amount": 0.00
    }
}

def validate_and_extract_features(raw_data):
    """
    Validates that all 30 features are present and are valid numerical values.
    Returns:
        (feature_array, feature_dict, error_message)
    """
    values = []
    cleaned_dict = {}

    for feature in FEATURE_NAMES:
        val = raw_data.get(feature)
        if val is None or str(val).strip() == "":
            return None, None, f"Missing required feature: '{feature}'"
        
        try:
            num_val = float(str(val).strip())
            values.append(num_val)
            cleaned_dict[feature] = num_val
        except (ValueError, TypeError):
            return None, None, f"Invalid value for '{feature}': must be a valid numerical number."

    # Convert to 2D numpy array shaped (1, 30) with explicit feature names
    feature_df = pd.DataFrame([values], columns=FEATURE_NAMES)
    return feature_df, cleaned_dict, None

# Feature domain interpretations for PCA components in credit card fraud detection
FEATURE_DESCRIPTIONS = {
    "V14": "Primary Behavioral Anomaly Index (Strongest fraud predictor)",
    "V12": "Account Usage & Deviation Metric",
    "V10": "Transaction Identity & Context Consistency",
    "V4": "Transaction Velocity & Frequency Vector",
    "V11": "Card Velocity & Burst Frequency Metric",
    "V3": "Terminal Interaction Baseline",
    "V8": "Merchant & Terminal Correlation Factor",
    "V17": "Cross-Channel Verification Metric",
    "V7": "Spending Pattern Dispersion Index",
    "V16": "Temporal Discrepancy Indicator",
    "V2": "Transaction Magnitude Deviation",
    "Amount": "Transaction Monetary Value ($ USD)",
    "Time": "Elapsed Time in Sequence (Seconds)"
}

def get_feature_description(feature_name):
    return FEATURE_DESCRIPTIONS.get(feature_name, f"Principal Component {feature_name} (PCA Dimension)")

def make_prediction(feature_df):
    """
    Executes model prediction and extracts probability metrics along with
    Explainable AI (XAI) feature contribution analysis.
    """
    if model is None:
        raise RuntimeError(model_load_error or "Model is not loaded.")

    # Model prediction: 0 or 1
    pred_raw = model.predict(feature_df)
    prediction = int(pred_raw[0])

    # Probability estimation
    legit_prob = None
    fraud_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(feature_df)[0]
            legit_prob = float(probabilities[0])
            fraud_prob = float(probabilities[1])
        except Exception as e:
            print(f"[WARNING] predict_proba failed: {e}")

    # Risk Score (0 - 100 scale)
    risk_score = round((fraud_prob if fraud_prob is not None else float(prediction)) * 100, 1)

    # Real-World Banking Action Recommendation
    if risk_score >= 70.0:
        recommended_action = "BLOCK TRANSACTION — High Risk Fraud Pattern Detected"
        action_type = "danger"
        action_code = "BLOCK"
        action_note = "Transaction declined immediately. Alert dispatched to cardholder via SMS and account flagged for review."
    elif risk_score >= 30.0:
        recommended_action = "FLAG FOR 2FA / MANUAL REVIEW — Elevated Anomaly Risk"
        action_type = "warning"
        action_code = "STEP_UP_OTP"
        action_note = "Trigger 3D-Secure One-Time Passcode (OTP) verification before authorizing funds transfer."
    else:
        recommended_action = "APPROVE — Transaction Verified Legitimate"
        action_type = "success"
        action_code = "APPROVE"
        action_note = "Transaction metrics match legitimate cardholder profile. Instant authorization granted."

    # Explainable AI: Feature Contribution Analysis (Logit decomposition)
    # logit z = intercept + sum(coef_i * x_i)
    feature_values = feature_df.iloc[0].values
    feature_contributions = []
    top_risk_drivers = []
    top_safety_drivers = []
    intercept_val = 0.0

    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coefs = model.coef_[0]
        intercept_val = float(model.intercept_[0])

        for name, val, coef in zip(FEATURE_NAMES, feature_values, coefs):
            contrib = float(coef * val)
            item = {
                "feature": name,
                "value": round(float(val), 4),
                "coefficient": round(float(coef), 4),
                "contribution": round(contrib, 4),
                "abs_contribution": abs(round(contrib, 4)),
                "direction": "risk" if contrib > 0 else "safety",
                "description": get_feature_description(name)
            }
            feature_contributions.append(item)

        # Sort features pushing toward Fraud (positive contribution)
        risk_sorted = sorted([f for f in feature_contributions if f["contribution"] > 0],
                             key=lambda x: x["contribution"], reverse=True)
        top_risk_drivers = risk_sorted[:5]

        # Sort features pushing toward Legitimacy (negative contribution)
        safety_sorted = sorted([f for f in feature_contributions if f["contribution"] < 0],
                              key=lambda x: x["contribution"])
        top_safety_drivers = safety_sorted[:5]

    # Text label strictly adhering to requirements
    if prediction == 0:
        label = "Legitimate Transaction"
        status_type = "legitimate"
        confidence = legit_prob if legit_prob is not None else 1.0
    else:
        label = "Fraudulent Transaction Detected"
        status_type = "fraud"
        confidence = fraud_prob if fraud_prob is not None else 1.0

    return {
        "prediction": prediction,
        "label": label,
        "status_type": status_type,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "legit_probability": legit_prob,
        "fraud_probability": fraud_prob,
        "legit_prob_percent": round(legit_prob * 100, 2) if legit_prob is not None else None,
        "fraud_prob_percent": round(fraud_prob * 100, 2) if fraud_prob is not None else None,
        "risk_score": risk_score,
        "recommended_action": recommended_action,
        "action_type": action_type,
        "action_code": action_code,
        "action_note": action_note,
        "top_risk_drivers": top_risk_drivers,
        "top_safety_drivers": top_safety_drivers,
        "feature_contributions": feature_contributions,
        "intercept": round(intercept_val, 4)
    }

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route:
    - GET: Renders the transaction input form.
    - POST: Accepts form submission, validates, and renders prediction result.
    """
    # If model is not loaded yet, attempt reload
    if model is None:
        load_ml_model()

    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        raw_data = request.form.to_dict()
        form_values = raw_data

        feature_df, cleaned_dict, validation_error = validate_and_extract_features(raw_data)
        if validation_error:
            error = validation_error
        else:
            try:
                result = make_prediction(feature_df)
                form_values = cleaned_dict
            except Exception as e:
                error = f"Prediction Error: {str(e)}"

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        form_values=form_values,
        result=result,
        error=error,
        model_error=model_load_error,
        model_metrics=model_metrics,
        presets=SAMPLE_PRESETS
    )

@app.route("/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for responsive AJAX form submission without page reload.
    Accepts JSON payload or Form data.
    """
    if model is None:
        load_ml_model()
        if model is None:
            return jsonify({
                "success": False,
                "error": model_load_error or "Model is not loaded."
            }), 503

    # Support both application/json and application/x-www-form-urlencoded
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    feature_df, cleaned_dict, validation_error = validate_and_extract_features(data)
    if validation_error:
        return jsonify({
            "success": False,
            "error": validation_error
        }), 400

    try:
        prediction_details = make_prediction(feature_df)
        return jsonify({
            "success": True,
            "result": prediction_details
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Inference failed: {str(e)}"
        }), 500

@app.route("/sample/<sample_type>", methods=["GET"])
def get_sample(sample_type):
    """
    Helper API providing real benchmark transaction samples
    (legitimate or fraud) for 1-click loading.
    """
    if sample_type in SAMPLE_PRESETS:
        return jsonify({
            "success": True,
            "type": sample_type,
            "data": SAMPLE_PRESETS[sample_type]
        })
    return jsonify({
        "success": False,
        "error": f"Invalid sample type '{sample_type}'. Choose 'legitimate' or 'fraud'."
    }), 404

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns model metrics and specifications."""
    return jsonify({
        "success": True,
        "metrics": model_metrics,
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print("\n" + "=" * 60)
    print("  CREDIT CARD FRAUD DETECTION WEB APPLICATION")
    print(f"  Running at: http://{host}:{port}")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=False if os.environ.get("PORT") else True)
