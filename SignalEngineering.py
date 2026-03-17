import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import plotly.express as px
BTC_data = pd.read_csv("BTC_daily_data.csv")
#---------
def plot_full_period_predictions_from_memory( BTC_data, params_file="final_linear_params.json", model_name="Linear"):
    """
    Uses the in-memory BTC_data DataFrame (already containing all engineered features and targets) to compute and plot real vs predicted gain/loss.
    """
    
    feature_cols = [   "ols_slope350",   "ols_slope128",   "ols_slope14",   "price_trend",   "volatility"]
    X = BTC_data[feature_cols].values
    Y = BTC_data[["gain3w", "loss3w"]].values
    with open(params_file, "r") as f:
        saved = json.load(f)
    params = saved[model_name]
    coef = np.array(params["coef"])        # shape (2, 5)
    intercept = np.array(params["intercept"])  # shape (2,)
    preds = X @ coef.T + intercept
    pred_gain = preds[:, 0]
    pred_loss = preds[:, 1]
    real_gain = Y[:, 0]
    real_loss = Y[:, 1]
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter( x=BTC_data["date"], y=real_gain, mode="lines", name="Real Gain", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=BTC_data["date"], y=pred_gain,mode="lines", name="Predicted Gain", line=dict(color="lightgreen", dash="dash")))
    fig.add_trace(go.Scatter( x=BTC_data["date"], y=real_loss, mode="lines", name="Real Loss", line=dict(color="red")))
    fig.add_trace(go.Scatter( x=BTC_data["date"], y=pred_loss, mode="lines", name="Predicted Loss", line=dict(color="pink", dash="dash")))
    fig.update_layout( title="Real vs Predicted Gain/Loss (Full Period)", xaxis_title="Date", yaxis_title="Value", template="plotly_white", legend=dict(x=0.01, y=0.99))
    fig.show()
####

def plot_candy_bars_two(BTC_data, params_file="final_linear_params.json", model_name="Linear"):
    """
    Computes predictions (gain + loss), removes NaNs safely, prints key statistics, and plots two boxplots.
    """
    with open(params_file, "r") as f:
        saved = json.load(f)
    params = saved[model_name]
    coef = np.array(params["coef"])
    intercept = np.array(params["intercept"])
    feature_cols = [  "ols_slope350",  "ols_slope128",  "ols_slope14",  "price_trend",  "volatility"]
    X = BTC_data[feature_cols].values
    # Predictions
    preds = X @ coef.T + intercept
    pred_gain = preds[:, 0]
    pred_loss = preds[:, 1]
    # Errors
    gain_error = np.abs(BTC_data["gain3w"].values - pred_gain)
    loss_error = np.abs(BTC_data["loss3w"].values - pred_loss)
    # Remove NaNs
    gain_nan_count = np.isnan(gain_error).sum()
    loss_nan_count = np.isnan(loss_error).sum()
    print(f"\nRemoved {gain_nan_count} NaNs from gain_error")
    print(f"Removed {loss_nan_count} NaNs from loss_error")
    gain_error = gain_error[~np.isnan(gain_error)]
    loss_error = loss_error[~np.isnan(loss_error)]
    # Print statistics
    def print_stats(name, arr):
        print(f"\n=== {name} Error Statistics ===")
        print(f"Count:                    {len(arr)}")
        print(f"Mean Absolute Error:      {np.mean(arr):.4f}")
        print(f"Median (Q2):              {np.median(arr):.4f}")
        print(f"Q1 (25%):                 {np.percentile(arr, 25):.4f}")
        print(f"Q3 (75%):                 {np.percentile(arr, 75):.4f}")
        print(f"IQR:                      {np.percentile(arr, 75) - np.percentile(arr, 25):.4f}")
        print(f"Min:                      {np.min(arr):.4f}")
        print(f"Max:                      {np.max(arr):.4f}")
        # Outliers using 1.5×IQR
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        outliers = np.sum(arr > Q3 + 1.5 * IQR)
        print(f"Outliers (1.5×IQR rule):  {outliers}")

    print_stats("Gain", gain_error)
    print_stats("Loss", loss_error)
    # Prepare for boxplot
    df = pd.DataFrame({   "Error": np.concatenate([gain_error, loss_error]),   "Type":  (["Gain Error"] * len(gain_error)) + (["Loss Error"] * len(loss_error))})
    # Plot
    fig = px.box(  df,  x="Type", y="Error", color="Type", title="Overall Prediction Error (Two Candy Bars)", points="outliers", template="plotly_white")
    fig.update_layout( height=800, yaxis=dict( tickmode="linear", tick0=0, dtick=0.02, range=[0, max(np.max(gain_error), np.max(loss_error)) * 1.1], title="Absolute Error" ))
    fig.show()
    #---------
# 1. Rolling Feature Functions 
def rolling_ols_slope(series, window):
    return series.rolling(window).apply(lambda x: np.polyfit(np.arange(window), x, 1)[0],raw=False)

def rolling_vol(series, window):
    return series.pct_change().rolling(window).std()
def rolling_ols_slope(series, window):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            beta, _ = np.polyfit(x, y, 1)  # slope, intercept
            slopes.append(beta)
    return pd.Series(slopes, index=series.index)
BTC_data['ols_slope350'] =rolling_ols_slope(BTC_data['log_return'], 350)
BTC_data['ols_slope128'] =rolling_ols_slope(BTC_data['log_return'], 128)
BTC_data['ols_slope14']  = rolling_ols_slope(BTC_data['log_return'], 14)
# --- Targets: 3weeks gain/loss buckets ---
h = 21
future_max = BTC_data['hloc_avg'].shift(-1).rolling(h).max()
future_min = BTC_data['hloc_avg'].shift(-1).rolling(h).min()
BTC_data['gain3w'] = future_max / BTC_data['hloc_avg'] - 1
BTC_data['loss3w'] = future_min / BTC_data['hloc_avg'] - 1
# 3. Regime Features (Trend + Volatility)
BTC_data['price_trend'] = BTC_data['hloc_avg'].rolling(100).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],raw=False)
BTC_data['volatility'] = rolling_vol(BTC_data['hloc_avg'], 21)

feature_cols = ['ols_slope350','ols_slope128','ols_slope14','price_trend','volatility']
target_cols  = ['gain3w','loss3w']

# rows where all features exist
feature_mask = BTC_data[feature_cols].notna().all(axis=1)
# rows where all targets exist
target_mask = BTC_data[target_cols].notna().all(axis=1)
# final usable region
valid_idx = BTC_data.index[feature_mask & target_mask]
X = BTC_data.loc[valid_idx, feature_cols]
Y = BTC_data.loc[valid_idx, target_cols]
BTC_data.to_csv("AddedFeatureData.csv", index=False)

models = { 'Linear': LinearRegression(),'Ridge':  Ridge(alpha=1.0), 'Lasso':  Lasso(alpha=0.001)}
window  = 500
horizon = 50
results = []
pred = {}
final_params = {}
n = len(X)
for start in range(0, n - window - horizon, horizon):
    end_train = start + window
    end_test  = end_train + horizon
    X_train = X.iloc[start:end_train]
    Y_train = Y.iloc[start:end_train]
    X_test  = X.iloc[end_train:end_test]
    Y_test  = Y.iloc[end_train:end_test]
    fold_res = {'start': start}
    is_last_fold = (start + horizon >= n - window - horizon)
    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        fold_res[f'{name}_MSE'] = mse
        if is_last_fold:
            final_params[name] = { 'coef': model.coef_.tolist(),
                'intercept': (model.intercept_.tolist()if hasattr(model.intercept_, 'tolist')
                    else model.intercept_)}
    results.append(fold_res)  
res_df = pd.DataFrame(results)
with open('final_linear_params.json', 'w') as f:
    json.dump(final_params, f, indent=2)

print(res_df)
plot_full_period_predictions_from_memory(BTC_data)
plot_candy_bars_two(BTC_data)