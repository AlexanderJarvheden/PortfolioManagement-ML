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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

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
    # Base numerical columns
    df["Close_lag1"] = df["Close"].shift(1)
    df["FGI_lag1"] = df["FGI"].shift(1)
    df["Change_pct"] = (
        df["Close"].pct_change() * 100 if "Change %" not in df else df["Change %"].astype(float)
    )
    
    # Rolling statistics
    df["FGI_roll_mean"] = df["FGI"].rolling(lookback).mean()
    df["FGI_roll_std"] = df["FGI"].rolling(lookback).std()
    df["Price_roll_mean"] = df["Close"].rolling(lookback).mean()
    df["Price_roll_std"] = df["Close"].rolling(lookback).std()
    
    # Momentum features
    df["FGI_mom"] = df["FGI"].diff(lookback)
    df["Price_mom"] = df["Close"].diff(lookback)
    
    # # Date‑time features
    # df["dayofweek"] = df["Date"].dt.dayofweek
    # df["month"] = df["Date"].dt.month

    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna().reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------

def label_targets(df: pd.DataFrame,
                  thr: float = 1.0,
                  clip_pct: float = 10.0):
    """
    Returnerar:
        action_labels  (np.ndarray)  — 0=Sell, 1=Hold, 2=Buy   (för classifiern)
        amount_targets (np.ndarray)  — framtida %-rörelse       (för regressorn)

    • action bestäms av threshold 'thr' (1 %=default)
    • amount är faktiska fwd-change i procent, klippt till ±clip_pct
      och omräknat till decimal (+7 % → 0.07).
    """
    fwd = df["Change_pct"].shift(-1)                 # nästa dags %-rörelse

    action_raw = np.where(fwd >  thr,  1,            # Buy
                 np.where(fwd < -thr, -1, 0))        # Sell / Hold

    amount = fwd.clip(-clip_pct, clip_pct) / 100.0   # +7%→0.07,  -4%→-0.04

    valid = ~fwd.isna()                              # sista raden saknar target
    action_labels  = (action_raw[valid] + 1)         # -1→0, 0→1, 1→2
    amount_targets = amount[valid].to_numpy()

    return action_labels, amount_targets

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
            n_estimators=391,
            learning_rate=0.17325459975947805,
            max_depth=10,
            subsample=0.6621349590183688,
            colsample_bytree=0.748811322879415,
            gamma=0.21536167152019156,
            reg_alpha=0.06750949051842556,
            reg_lambda=0.4791100358107637,
            objective="reg:squarederror",
            random_state=42,
            
            # n_estimators=400,
            # learning_rate=0.03,
            # max_depth=6,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # objective="reg:squarederror",
            # random_state=42,
        )
        self.scaler = None
        self.feature_names = None
        self.regression_error = None

    def fit(self, raw_df: pd.DataFrame):
        df = engineer_features(raw_df)
        y_action, y_amount = label_targets(df)

        X = df.iloc[:-1]
        X = X.drop(columns=["Date"])

        self.feature_names = X.columns.tolist()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

    # ➤ Träna på 80%, spara 20% till riktig testning
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_amount, test_size=0.2, random_state=42
    )

        self.reg.fit(X_train, y_train)

    # ➤ Beräkna regressionens fel på TESTSET, inte träning
        y_pred_test = self.reg.predict(X_test)
        self.regression_error = mean_squared_error(y_test, y_pred_test)

        print(f"[INFO] Trained regressor — Test MSE: {self.regression_error:.6f}")

        return self



    # -------------------------------------------------------------
    def predict(self, latest_rows: pd.DataFrame):
        if self.scaler is None or self.clf is None:
            raise RuntimeError("Model not fitted or loaded.")

        feats = engineer_features(latest_rows)
        X     = feats[self.feature_names].tail(1)
        X_s   = self.scaler.transform(X)

    # --- Regressor ---
        amount_pct = float(self.reg.predict(X_s)[0])

    # Gör om till log-return vy
        if amount_pct >= 0:
            mu_view = np.log(1 + amount_pct)
        else:
            mu_view = -np.log(1 + abs(amount_pct))

    # --- Confidence ---
    # ➔ Ny formel: |prediction| / (|prediction| + sqrt(mse))
        if self.regression_error is None:
            raise RuntimeError("Model has no regression error recorded (fit() not run?).")
        mse = self.regression_error
        rmse = np.sqrt(mse)
        confidence = abs(amount_pct) / (abs(amount_pct) + rmse)

        # print(f"DEBUG — amount_pct: {amount_pct} | mu_view: {mu_view} | confidence: {confidence}")

        return mu_view, confidence



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
    p.add_argument("data", help="CSV/XLSX file containing Date,FGI,Close,Change %")
    p.add_argument("--out", default="models", help="Directory to save model artefacts")
    args = p.parse_args()

    df = load_data(args.data)
    trainer = Trainer()
    trainer.fit(df)
    trainer.save(args.out)
    print(f"Model saved to {args.out}/xgb_fgi.joblib", file=sys.stderr)