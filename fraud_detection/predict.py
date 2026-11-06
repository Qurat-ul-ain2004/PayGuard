import joblib 
import pandas as pd
from pathlib import Path



model_path = Path(__file__).resolve().parent / "models" / "fraud_model.joblib"
saved_data = joblib.load(model_path)
model = saved_data["model"]
threshold = saved_data["threshold"]
features = saved_data["feature_columns"]

data_path = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
df = pd.read_csv(data_path)

sample = df[features].head(30)

fraud_scores = model.predict_proba(sample)[:, 1]
predictions = (fraud_scores >= threshold).astype(int)
results = pd.DataFrame({
    "Fraud Score": fraud_scores,
    "Prediction": predictions
})
results["Prediction"] = results["Prediction"].map(
{0: "Genuine",
  1: "Suspicious"}
)
print("Fraud Detection Results for Sample Transactions:")
print(results)
