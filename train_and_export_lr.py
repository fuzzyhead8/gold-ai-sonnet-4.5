import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CSV = "XAU_ADX_ATR_Slope.csv"
SEP = ";"

# For M1 export: 30 = 30 minutes; for H1 export: 24 = ~1 day
LOOKAHEAD = 30
BAD_MOVE_POINTS = 500
POINT = 0.01

FEATURES = ["adx", "atr", "stddev", "ema_slope_points"]

# ---- Load CSV and rename columns from your exporter ----
df = pd.read_csv(CSV, sep=SEP).rename(
    columns={
        "ADX(14)": "adx",
        "ATR(14)": "atr",
        "StdDev(14)": "stddev",
        "EMA_Slope_points": "ema_slope_points",
        "Close": "close",
        "Low": "low",
    }
)

for c in FEATURES + ["close", "low"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().reset_index(drop=True)

# ---- Label: worst adverse down move from close over next LOOKAHEAD bars ----
future_min_low = (
    df["low"]
    .rolling(window=LOOKAHEAD, min_periods=LOOKAHEAD)
    .min()
    .shift(-LOOKAHEAD)
)
adverse_down_points = (df["close"] - future_min_low) / POINT
df["allow_trade"] = (adverse_down_points <= BAD_MOVE_POINTS).astype(int)

df = df.iloc[:-LOOKAHEAD].copy()

X = df[FEATURES].to_numpy(dtype=float)
y = df["allow_trade"].to_numpy(dtype=int)

print("Rows:", len(df), "allow_trade=1 rate:", float(y.mean()))

# ---- Time series CV sanity check ----
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for tr, te in tscv.split(X):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    clf = LogisticRegression(max_iter=4000)
    clf.fit(Xtr, y[tr])
    cv_scores.append(clf.score(Xte, y[te]))

print("CV accuracy mean:", float(np.mean(cv_scores)), "scores:", cv_scores)

# ---- Fit final model on all data ----
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=4000)
clf.fit(Xs, y)

means = scaler.mean_.tolist()
stds = scaler.scale_.tolist()
weights = clf.coef_[0].tolist()
bias = float(clf.intercept_[0])

bundle = {
    "features": FEATURES,
    "means": means,
    "stds": stds,
    "weights": weights,
    "bias": bias,
    "threshold": 0.5,
    "lookahead": LOOKAHEAD,
    "bad_move_points": BAD_MOVE_POINTS,
    "point": POINT,
}

# Save both: joblib (optional) + json constants for MQL5
joblib.dump({"scaler": scaler, "model": clf, **bundle}, "gate_lr.joblib")
with open("gate_lr_constants.json", "w", encoding="utf-8") as f:
    json.dump(bundle, f, indent=2)

print("Saved gate_lr.joblib and gate_lr_constants.json")
print("Paste these into MQL5:")
print("FEATURES:", FEATURES)
print("means:", means)
print("stds:", stds)
print("weights:", weights)
print("bias:", bias)