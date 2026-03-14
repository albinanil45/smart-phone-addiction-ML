from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load trained models
# -----------------------------
addiction_model = joblib.load("addiction_model.pkl")
severity_model = joblib.load("severity_model.pkl")
encoder = joblib.load("severity_encoder.pkl")

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        daily_screen_time = float(data["daily_screen_time"])
        social_media_hours = float(data["social_media_hours"])
        gaming_hours = float(data["gaming_hours"])
        work_study_hours = float(data["work_study_hours"])
        sleep_hours = float(data["sleep_hours"])
        weekend_screen_time = float(data["weekend_screen_time"])

    except KeyError:
        return jsonify({
            "error": "Missing required input fields"
        }), 400

    # create input array
    X = np.array([[
        daily_screen_time,
        social_media_hours,
        gaming_hours,
        work_study_hours,
        sleep_hours,
        weekend_screen_time
    ]])

    # predict addiction
    addicted = int(addiction_model.predict(X)[0])

    result = {
        "addicted": bool(addicted),
        "severity": None
    }

    # predict severity only if addicted
    if addicted == 1:
        severity_encoded = severity_model.predict(X)[0]
        severity = encoder.inverse_transform([severity_encoded])[0]
        result["severity"] = severity

    return jsonify(result)


# -----------------------------
# Health check route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Smartphone Addiction Prediction API running"
    })


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)