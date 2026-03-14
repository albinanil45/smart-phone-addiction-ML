import joblib
import numpy as np

# Load models
addiction_model = joblib.load("addiction_model.pkl")
severity_model = joblib.load("severity_model.pkl")
encoder = joblib.load("severity_encoder.pkl")

# Example user input
daily_screen_time = 12
social_media_hours = 6
gaming_hours = 6
work_study_hours = 1
sleep_hours = 3
weekend_screen_time = 14

# Create input array
X = np.array([[

    daily_screen_time,
    social_media_hours,
    gaming_hours,
    work_study_hours,
    sleep_hours,
    weekend_screen_time

]])

# Predict addiction
addicted = addiction_model.predict(X)[0]

# Predict severity
severity_encoded = severity_model.predict(X)[0]
severity = encoder.inverse_transform([severity_encoded])[0]

print("Addicted:", addicted)
print("Addiction Severity:", severity)