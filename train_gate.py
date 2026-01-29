import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

CSV = "XAU_ADX_ATR_Slope.csv"
SEP = ";"

# Choose these based on your timeframe:
# - If you export M1: LOOKAHEAD=30 means 30 minutes
# - If you export H1: LOOKAHEAD=24 means ~1 day
LOOKAHEAD = 30
BAD_MOVE_POINTS = 500

# XAUUSD point is usually 0.01. Adjust if your broker differs.
POINT = 0.01

df = pd.read_csv(CSV, sep=SEP)

# Normalize column names from your exporter
rename_map = {
    "ADX(14)": "adx",
    "ATR(14)": "atr",
    "StdDev(14)": "stddev",
    "EMA(50)": "ema",
    "EMA_Slope_points": "ema_slope_points",
    "Close": "close",
    "Low": "low",
}
df = df.rename(columns=rename_map)

needed = ["adx", "atr", "stddev", "ema", "ema_slope_points", "close", "low"]
for c in needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().reset_index(drop=True)

# Label: worst adverse down move (buy-danger proxy) over next LOOKAHEAD bars
future_min_low = (
    df["low"]
    .rolling(window=LOOKAHEAD, min_periods=LOOKAHEAD)
    .min()
    .shift(-LOOKAHEAD)
)
adverse_down_points = (df["close"] - future_min_low) / POINT

df["allow_trade"] = (adverse_down_points <= BAD_MOVE_POINTS).astype(int)

# Remove tail rows without future window
df = df.iloc[:-LOOKAHEAD].copy()

features = ["adx", "atr", "stddev", "ema_slope_points"]

X = df[features]
y = df["allow_trade"]

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for tr, te in tscv.split(X):
    pipe.fit(X.iloc[tr], y.iloc[tr])
    scores.append(pipe.score(X.iloc[te], y.iloc[te]))

print("CV accuracy mean:", float(np.mean(scores)))
print("CV scores:", scores)
print("Class balance allow_trade=1:", float(y.mean()))

pipe.fit(X, y)

bundle = {
    "model": pipe,
    "features": features,
    "point": POINT,
    "lookahead": LOOKAHEAD,
    "bad_move_points": BAD_MOVE_POINTS,
    "threshold": 0.5,
}
joblib.dump(bundle, "gate_model.joblib")
print("Saved gate_model.joblib")