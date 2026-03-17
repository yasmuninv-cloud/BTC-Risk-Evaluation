import pandas as pd
import numpy as np
import json

# Load saved model parameters
with open("final_linear_params.json", "r") as f:
    saved_params = json.load(f)

# Choose which model to load
MODEL_NAME = "Linear"  # or "Ridge" or "Lasso"
params = saved_params[MODEL_NAME]
coef = np.array(params["coef"])
intercept = np.array(params["intercept"])
print("=== Loaded model params ===")
print("Model:", MODEL_NAME)

# Manual prediction function with debugging
def predict_from_params(features, coef, intercept):
    out = features @ coef.T + intercept
    return out

# Interactive prediction engine
def predict_future_envelope():
    newest_date = BTC_data["date"].max()
    print("date in data base up to:", newest_date)
    date_str = input("Enter date (YYYY-MM-DD): ")
    current_date = pd.to_datetime(date_str).date()
    if current_date not in BTC_data["date"].values:
        print("Date not found in dataset.")
        return
    # Extract features for that date
    feature_cols = ['ols_slope350','ols_slope128','ols_slope14','price_trend','volatility']
    features = BTC_data.loc[ BTC_data["date"] == current_date, feature_cols].values.reshape(1, -1)

    # Predict using saved params
    pred = predict_from_params(features, coef, intercept)
    pred = np.array(pred)
    if pred.ndim == 0:
        # scalar
        gain_pred = float(pred)
        loss_pred = float(pred)
    elif pred.ndim == 1:
        if pred.shape[0] == 2:
            gain_pred, loss_pred = float(pred[0]), float(pred[1])
        elif pred.shape[0] == 1:
            gain_pred = float(pred[0])
            loss_pred = float(pred[0])
        else:
            raise ValueError(f"Unexpected 1D pred shape: {pred.shape}")
    elif pred.ndim == 2:
        if pred.shape[1] == 2:
            gain_pred, loss_pred = float(pred[0, 0]), float(pred[0, 1])
        elif pred.shape[1] == 1:
            gain_pred = float(pred[0, 0])
            loss_pred = float(pred[0, 0])
        else:
            raise ValueError(f"Unexpected 2D pred shape: {pred.shape}")
    else:
        raise ValueError(f"Unexpected pred ndim: {pred.ndim}, shape: {pred.shape}")

    # Current price
    current_price = BTC_data.loc[BTC_data["date"] == current_date, "hloc_avg"].iloc[0]
    predicted_high = current_price * (1 + gain_pred)
    predicted_low  = current_price * (1 + loss_pred)

    print(f"\nPrediction for {current_date}:")
    print(f"  Current price: {current_price:.2f}")
    print(f"  Predicted 21-day gain: {gain_pred:.4f} → High ≈ {predicted_high:.2f}")
    print(f"  Predicted 21-day loss: {loss_pred:.4f} → Low  ≈ {predicted_low:.2f}")

# ---------------------------------------------------------
# Run the interactive program
# ---------------------------------------------------------
if __name__ == "__main__":
    BTC_data = pd.read_csv("AddedFeatureData.csv")
    BTC_data["date"] = pd.to_datetime(BTC_data["date"]).dt.date
    predict_future_envelope() 