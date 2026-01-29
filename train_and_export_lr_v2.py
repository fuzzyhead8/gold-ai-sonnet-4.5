import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


CSV = "XAU_M1_Features_v2.csv"
SEP = ";"  # MT5 exporter uses ';'

# Label parameters (match your grid pain)
LOOKAHEAD = 30          # 30 minutes for M1
BAD_MOVE_POINTS = 500   # tune to your grid step / max levels
POINT = 0.01            # XAUUSD point (adjust if needed)

# Features used for LR gate (must match EA later)
FEATURES = [
    "adx",
    "atr_points",
    "stddev_points",
    "ema_slope_points",
    "ret1_points",
    "range_points",
    "minute_of_day",
    "dow",
]


def to_num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    df = pd.read_csv(CSV, sep=SEP)

    required = (
        ["time", "close", "low"] + FEATURES
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}\nFound: {df.columns.tolist()}")

    # Parse time for correct sorting (exported as "YYYY.MM.DD HH:MM")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    df = to_num(df, ["close", "low"] + FEATURES)
    df = df.dropna().reset_index(drop=True)

    # Label: worst adverse down move from close over next LOOKAHEAD bars
    future_min_low = (
        df["low"]
        .rolling(window=LOOKAHEAD, min_periods=LOOKAHEAD)
        .min()
        .shift(-LOOKAHEAD)
    )
    adverse_down_points = (df["close"] - future_min_low) / POINT
    df["allow_trade"] = (adverse_down_points <= BAD_MOVE_POINTS).astype(int)

    # Remove tail rows with no future
    df = df.iloc[:-LOOKAHEAD].copy()

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["allow_trade"].to_numpy(dtype=int)

    print("Rows:", len(df))
    print("allow_trade=1 rate:", float(y.mean()))

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in tscv.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=5000)
        clf.fit(Xtr, y[tr])
        scores.append(clf.score(Xte, y[te]))

    print("CV accuracy mean:", float(np.mean(scores)))
    print("CV scores:", scores)

    # Fit final model on all data
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=5000)
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

    with open("gate_lr_constants_v2.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print("Saved gate_lr_constants_v2.json")
    print("Next: paste constants into the EA (arrays + bias).")


if __name__ == "__main__":
    main()