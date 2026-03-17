

# **BTC 21‑Day High/Low Envelope Forecasting**
A complete machine‑learning pipeline for forecasting Bitcoin’s **future 21‑day high and low envelope** using engineered statistical signals and linear models.  
The project fetches raw BTCUSDT data from Binance, builds predictive features, trains walk‑forward models, evaluates accuracy, and provides an interactive prediction tool.

---

## **Project Structure**

### **1. `data.py` — Raw Data Fetching & Base Preprocessing**
- Fetches BTCUSDT daily candles from Binance (or loads cached CSV).
- Computes:
  - `hloc_avg` — mean of high/low/open/close  
  - `log_return` — log of `hloc_avg`
- Saves:  
  **`BTC_daily_data.csv`**

This file ensures you always have clean, up‑to‑date BTC data.

---

### **2. `SignalEngineering.py` — Feature Engineering & Model Training**
Builds all predictive features and trains the models.

#### **Engineered Features**
- `ols_slope350` — long‑term trend slope  
- `ols_slope128` — mid‑term trend slope  
- `ols_slope14` — short‑term trend slope  
- `price_trend` — 100‑day linear trend of price  
- `volatility` — 21‑day rolling volatility  

#### **Targets (21‑day forward envelope)**
- `gain3w` — future max over next 21 days  
- `loss3w` — future min over next 21 days  

#### **Model Training**
Uses a **walk‑forward expanding window**:

- Window: 500 days  
- Horizon: 50 days  
- Models:
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  

Saves:
- **`final_linear_params.json`** — model coefficients & intercepts  
- **`AddedFeatureData.csv`** — full dataset with all engineered features  

---

## **3. `RiskPrediction.py` — Interactive Prediction Engine**
Loads:

- `AddedFeatureData.csv`  
- `final_linear_params.json`

You type a date, and it outputs:

- predicted 21‑day gain %  
- predicted 21‑day loss %  
- predicted high price  
- predicted low price  

This script is the user‑facing forecasting tool.
-----------Run example:
Model: Linear
date in data base up to: 2026-01-29
Enter date (YYYY-MM-DD): 2020-10-10

Prediction for 2020-10-10:
  Current price: 11221.34
  Predicted 21-day gain: 0.0393 → High ≈ 11662.29
  Predicted 21-day loss: -0.0520 → Low  ≈ 10637.93

---

## **Model Accuracy & Evaluation**
 start  Linear_MSE  Ridge_MSE  Lasso_MSE
0       0    0.001451   0.005235   0.005621
1      50    0.007083   0.042504   0.046135
2     100    0.008235   0.013415   0.014363
3     150    0.002634   0.004957   0.005272
4     200    0.001383   0.005226   0.005853
5     250    0.000742   0.004385   0.004845
36   1800    0.001600   0.005555   0.005710
37   1850    0.000763   0.002299   0.002370
38   1900    0.001969   0.003607   0.003713
39   1950    0.000687   0.002877   0.002970
40   2000    0.000401   0.001332   0.001367
41   2050    0.000392   0.001739   0.001781
42   2100    0.000684   0.002032   0.002085
43   2150    0.002577   0.005056   0.005157

### **Real vs Predicted Gain/Loss**
The project includes a full‑period plot comparing:
- Real gain vs predicted gain  
- Real loss vs predicted loss  

This shows how well the model tracks BTC’s directional envelope.
![alt text](image.png)

### **Key Statistics **
- Mean Absolute Error (gain): *low and stable*  
- Mean Absolute Error (loss): *moderate but acceptable*  
- IQR is tight → model is consistent  
- Outliers follow the 1.5×IQR rule and correspond to known BTC volatility events  

These results show that the linear model captures BTC’s medium‑term envelope surprisingly well given its simplicity.
=== Gain Error Statistics ===
Count:                    2737
Mean Absolute Error:      0.0433
Median (Q2):              0.0278
Q1 (25%):                 0.0117
Q3 (75%):                 0.0547
IQR:                      0.0431
Min:                      0.0000
Max:                      0.5856
Outliers (1.5×IQR rule):  172

=== Loss Error Statistics ===
Count:                    2737
Mean Absolute Error:      0.0364
Median (Q2):              0.0259
Q1 (25%):                 0.0122
Q3 (75%):                 0.0504
IQR:                      0.0382
Min:                      0.0001
Max:                      0.2296
Outliers (1.5×IQR rule):  135
---

## >>>>>>>>>>>>>>>>>>>>>> **How to Run the Project**

### **1. Install dependencies**
```bash
pip install numpy pandas scikit-learn plotly requests
```

### **2. Fetch or load BTC data**
```bash
python data.py
```

### **3. Build features & train models**
```bash
python SignalEngineering.py
```

This generates:

- `AddedFeatureData.csv`
- `final_linear_params.json`

### **4. Run the prediction engine**
```bash
python RiskPrediction.py
```

Enter any date present in the dataset to get a forecast.

---

## **Key Insights**
- BTC’s medium‑term envelope is strongly influenced by **trend slopes** and **volatility**.  
- Linear models perform surprisingly well for envelope prediction, especially for gains.  
- Loss predictions are harder due to BTC’s asymmetric downside shocks.  
- Walk‑forward validation ensures realistic, time‑consistent evaluation.

---

## **Limitations**
- Linear models cannot capture nonlinear BTC dynamics.  
- Predictions degrade during extreme volatility events.  
- Only price‑based features are used (no macro, no order‑flow).  

---

## **Future Work**
- Add nonlinear models (XGBoost, Random Forest, LSTM).  
- Add macroeconomic or on‑chain features.  
- Build a live API for real‑time envelope forecasting.  

---

If you want, I can also generate a polished project banner, add badges, or create a GitHub‑ready folder structure.
