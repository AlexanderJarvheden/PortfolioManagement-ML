import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load your data
from xgb import load_data, engineer_features, label_targets

df = load_data("df_spy_ml.csv")
df = engineer_features(df)
y_action, y_amount = label_targets(df)

X = df.iloc[:-1].drop(columns=["Date"])
X = X.values
y = y_amount

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
    }

    model = XGBRegressor(**params)

    # --- kör vanlig fit utan early stopping ---
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=600)  # kör i 10 minuter

    print("\nBästa hyperparametrar:")
    print(study.best_params)

    print("\nMSE på test-set:")
    print(study.best_value)
