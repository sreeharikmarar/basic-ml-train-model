import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Dataset: hours studied vs exam score
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score":         [35, 45, 50, 60, 65, 70, 78, 82, 88, 95],
}

df = pd.DataFrame(data)
print(f"{len(df)} samples loaded")

X = df[["hours_studied"]]
y = df["score"]

# Train
model = LinearRegression()
model.fit(X, y)

print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept:           {model.intercept_:.2f}")
print(f"Formula:             score = {model.coef_[0]:.2f} * hours + {model.intercept_:.2f}")

# Evaluate
y_pred = model.predict(X)
print(f"MAE:  {mean_absolute_error(y, y_pred):.2f}")
print(f"R²:   {r2_score(y, y_pred):.4f}")

# Save model
MODEL_PATH = "score_predictor.joblib"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Verify
loaded_model = joblib.load(MODEL_PATH)
test_hours = 5.0
predicted_score = loaded_model.predict(pd.DataFrame([{"hours_studied": test_hours}]))[0]
print(f"Verification: {test_hours} hours -> predicted score: {predicted_score:.1f}")
