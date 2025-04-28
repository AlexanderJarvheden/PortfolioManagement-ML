"""
XGBoost model for crypto Fear‑and‑Greed Index (FGI)
==================================================
Trains two models:
  1. XGBoostClassifier -> predicts action class (Buy/Sell/Hold)
  2. XGBoostRegressor  -> predicts *amount* (position sizing in % of equity)

Both models rely on engineered trend features taken from recent history of
FGI and price data.  Final `predict()` returns:
   action (str), amount (float 0‑1), certainty (float 0‑1)

Assumptions
-----------
* Input dataframe contains columns [`Date`, `Crypto_FGI`, `Close`, `Change %`].
* `Date` is parseable to pandas.Timestamp.
* `Change %` is daily percentage change of price.  If not present, you can
  compute it as `pct_change()` of `Close`.
* Target labels are **derived** from the NEXT‑day percentage move (`fwd_change`).
    * fwd_change > +1%  ⇒  **Buy**
    * fwd_change < –1%  ⇒  **Sell**
    * otherwise         ⇒  **Hold**
* Amount target is `abs(fwd_change).clip(0, 10) / 10` (→ 0‑1 scale).

Edit the thresholds in `label_targets()` if you prefer different rules.

-----------------------------------------------------------------------
Usage (example)
-----------------------------------------------------------------------
import pandas as pd
from xgboost_fgi_model import Trainer, load_data

df = load_data("crypto_data.csv")
trainer = Trainer()
trainer.fit(df)
trainer.save("models/")

#--- later / inference ---
trainer = Trainer.load("models/")
latest_row = df.iloc[-1:]
print(trainer.predict(latest_row))  # ("buy", 0.35, 0.84)
"""
from __future__ import annotations
import json
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load

ACTION_MAP = {-1: "sell", 0: "hold", 1: "buy"}
REV_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def load_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Load csv or xlsx/ipynb exported to csv. Automatically parses Date."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["Date"])
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, parse_dates=["Date"])
    else:
        raise ValueError("Unsupported file type. Convert to CSV or XLSX.")
    return df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    df = df.copy()
    
    # Säkerställ att 'Date' är datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # errors="coerce" sätter ogiltiga till NaT
        print("hello man")
    # Base numerical columns
    df["Close_lag1"] = df["Close"].shift(1)
    df["FGI_lag1"] = df["Crypto_FGI"].shift(1)
    df["Change_pct"] = (
        df["Close"].pct_change() * 100 if "Change %" not in df else df["Change %"].astype(float)
    )
    
    # Rolling statistics
    df["FGI_roll_mean"] = df["Crypto_FGI"].rolling(lookback).mean()
    df["FGI_roll_std"] = df["Crypto_FGI"].rolling(lookback).std()
    df["Price_roll_mean"] = df["Close"].rolling(lookback).mean()
    df["Price_roll_std"] = df["Close"].rolling(lookback).std()
    
    # Momentum features
    df["FGI_mom"] = df["Crypto_FGI"].diff(lookback)
    df["Price_mom"] = df["Close"].diff(lookback)
    
    # Date‑time features
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna().reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------

def label_targets(df: pd.DataFrame, threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Returnerar (action_labels, amount_targets) baserat på nästa dags procentuella prisförändring.
    
    - action_labels: 0 = Sell, 1 = Hold, 2 = Buy
    - amount_targets: Skalad rörelse mellan 0 och 1
    """
    # Nästa dags procentuella förändring
    fwd_change = df["Change_pct"].shift(-1)
    
    # Bestäm action: 1 = köp, -1 = sälj, 0 = neutral
    action = np.where(
        fwd_change > threshold, 1,
        np.where(fwd_change < -threshold, -1, 0)
    )
    
    # Skala rörelse (magnitude) till intervallet [0, 1]
    amount = np.clip(np.abs(fwd_change), 0, 10) / 10.0

    # Ta bort sista raden (ingen nästa dags förändring)
    valid = ~fwd_change.isna()
    
    # Skifta action så att -1 ➔ 0, 0 ➔ 1, 1 ➔ 2 (som XGBClassifier kräver)
    action_shifted = (action[valid].astype(int) + 1)
    
    # Returnera
    return action_shifted, amount[valid].to_numpy()

# ---------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------

class Trainer:
    def __init__(self, clf: XGBClassifier | None = None, reg: XGBRegressor | None = None):
        self.clf = clf or XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.reg = reg or XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        self.scaler = None  # will be a fitted StandardScaler
        self.feature_names: list[str] | None = None

    # -------------------------------------------------------------
    def fit(self, raw_df: pd.DataFrame):
        df = engineer_features(raw_df)
        y_action, y_amount = label_targets(df)
    # Align X with y
        X = df.iloc[:-1]
        X = X.drop(columns=["Date"])
    
    # Spara feature-namn här!
        self.feature_names = X.columns.tolist()
    
    # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

    # Fit models
        self.clf.fit(X_scaled, y_action)
        self.reg.fit(X_scaled, y_amount)
        return self


    # -------------------------------------------------------------
    def predict(self, latest_rows: pd.DataFrame) -> Tuple[str, float, float]:
        if self.scaler is None or self.clf is None:
            raise RuntimeError("Model not fitted or loaded.")
        feats = engineer_features(latest_rows)
        X = feats[self.feature_names].tail(1)  # use last engineered row
        X_scaled = self.scaler.transform(X)
        proba = self.clf.predict_proba(X_scaled)[0]
        action_idx = int(np.argmax(proba))
        certainty = float(np.max(proba))
        amount = float(np.clip(self.reg.predict(X_scaled)[0], 0, 1))
        # Map action index (0,1,2) to {-1,0,1}
        mapped = {-1: 0, 0: 1, 1: 2}  # REV mapping of ACTION_MAP ordering
        inverse_map = {v: k for k, v in mapped.items()}
        action_num = inverse_map[action_idx]
        return ACTION_MAP[action_num], amount, certainty

    # -------------------------------------------------------------
    def save(self, directory: str | pathlib.Path):
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "clf": self.clf,
                "reg": self.reg,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            },
            directory / "xgb_fgi.joblib",
        )

    # -------------------------------------------------------------
    @classmethod
    def load(cls, directory: str | pathlib.Path) -> "Trainer":
        obj = load(pathlib.Path(directory) / "xgb_fgi.joblib")
        t = cls(obj["clf"], obj["reg"])
        t.scaler = obj["scaler"]
        t.feature_names = obj["feature_names"]
        return t

# ---------------------------------------------------------------------
# Script entry‑point (optional CLI)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Train XGBoost FGI model")
    p.add_argument("data", help="CSV/XLSX file containing Date,Crypto_FGI,Close,Change %")
    p.add_argument("--out", default="models", help="Directory to save model artefacts")
    args = p.parse_args()

    df = load_data(args.data)
    trainer = Trainer()
    trainer.fit(df)
    trainer.save(args.out)
    print(f"Model saved to {args.out}/xgb_fgi.joblib", file=sys.stderr)