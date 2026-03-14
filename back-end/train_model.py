import pandas as pd
import kagglehub
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Download dataset
# -----------------------------
path = kagglehub.dataset_download(
    "jayjoshi37/smartphone-usage-and-addiction-prediction"
)

print("Dataset downloaded to:", path)

# -----------------------------
# Find CSV file
# -----------------------------
files = os.listdir(path)
print("Files found:", files)

csv_file = [f for f in files if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(path, csv_file))

print(df.head())

# -----------------------------
# Select features
# -----------------------------
features = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time"
]

X = df[features]

# -----------------------------
# Target for addiction model
# -----------------------------
y_addicted = df["addicted_label"]

# -----------------------------
# Train/Test split for addiction model
# -----------------------------
X_train, X_test, y_train_add, y_test_add = train_test_split(
    X, y_addicted, test_size=0.2, random_state=42
)

# -----------------------------
# Train addiction model
# -----------------------------
addiction_model = RandomForestClassifier(n_estimators=100, random_state=42)

addiction_model.fit(X_train, y_train_add)

# -----------------------------
# Evaluate addiction model
# -----------------------------
pred_add = addiction_model.predict(X_test)

print("\nAddiction Prediction Accuracy:", accuracy_score(y_test_add, pred_add))
print(classification_report(y_test_add, pred_add))

# =====================================================
# Train Severity Model ONLY for addicted users
# =====================================================

severity_df = df[df["addicted_label"] == 1]

X_sev = severity_df[features]
y_sev = severity_df["addiction_level"]

# encode severity labels
severity_encoder = LabelEncoder()
y_sev_encoded = severity_encoder.fit_transform(y_sev)

# split
X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_sev, y_sev_encoded, test_size=0.2, random_state=42
)

# train model
severity_model = RandomForestClassifier(n_estimators=100, random_state=42)

severity_model.fit(X_train_sev, y_train_sev)

# evaluate
pred_sev = severity_model.predict(X_test_sev)

print("\nSeverity Prediction Accuracy:", accuracy_score(y_test_sev, pred_sev))

# -----------------------------
# Save models
# -----------------------------
joblib.dump(addiction_model, "addiction_model.pkl")
joblib.dump(severity_model, "severity_model.pkl")
joblib.dump(severity_encoder, "severity_encoder.pkl")

print("\nModels saved successfully!")