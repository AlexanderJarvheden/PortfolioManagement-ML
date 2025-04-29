from __future__ import annotations
import argparse
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from xgb import Trainer, load_data

# ————————————————————————————————————————————————————————————————
# Helper ── single‑asset backtest
# ————————————————————————————————————————————————————————————————

def backtest_single_asset(csv_path: str | pathlib.Path,
                          model_dir: str | pathlib.Path,
                          lookback: int = 30,
                          start: str | None = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Walk‑forward backtest for one asset."""

    csv_path = pathlib.Path(csv_path)
    model_dir = pathlib.Path(model_dir)

    df = load_data(csv_path)
    if start:
        df = df[df["Date"] >= start].reset_index(drop=True)

    trainer = Trainer()

    # --- först: utvärdera på 20%-split
    split_test_results = trainer.backtest_on_internal_split(df)
    split_metrics = quality_metrics(split_test_results)

    # --- sedan: riktig walk-forward backtest
    dates: List[pd.Timestamp] = []
    preds: List[float] = []
    actual: List[float] = []

    print("\n" + "="*20 + " Walkforward Predictions " + "="*20)
    
    for i in range(lookback, len(df) - 1):
        hist = df.iloc[:i+1]
        train_data = hist.iloc[:-1]
        test_data = hist.tail(lookback + 1)

        trainer = Trainer().fit_for_backtest(train_data)
        mu, _ = trainer.predict(test_data)
        preds.append(float(mu))

        p_t = df.iloc[i]["Close"]
        p_t1 = df.iloc[i+1]["Close"]
        actual_ret = np.log(p_t1 / p_t)
        actual.append(actual_ret)

        dates.append(pd.to_datetime(df.iloc[i+1]["Date"]))

        # --- PRINT predicted vs actual ---
        print(f"Date: {df.iloc[i+1]['Date']} | Predicted: {mu:.6f} | Actual: {actual_ret:.6f} | Error: {(mu - actual_ret):.6f}")

    walkforward_df = pd.DataFrame({
        "Date": dates,
        "pred": preds,
        "actual": actual
    })
    walkforward_df["error"] = walkforward_df["pred"] - walkforward_df["actual"]

    # --- Visa topp 5 bästa och sämsta dagar ---
    print("\n" + "="*20 + " Best 5 days (smallest error) " + "="*20)
    print(walkforward_df.reindex(walkforward_df['error'].abs().sort_values().index).head(5))

    print("\n" + "="*20 + " Worst 5 days (largest error) " + "="*20)
    print(walkforward_df.reindex(walkforward_df['error'].abs().sort_values(ascending=False).index).head(5))

    return split_test_results, split_metrics, walkforward_df


# ————————————————————————————————————————————————————————————————
# Metrics
# ————————————————————————————————————————————————————————————————

def quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute standard regression error metrics and sign hit‑rate."""
    mse = float((df["error"] ** 2).mean())
    mae = float(df["error"].abs().mean())
    r2 = 1.0 - (df["error"] ** 2).sum() / ((df["actual"] - df["actual"].mean()) ** 2).sum()
    hitrate = float((np.sign(df["pred"]) == np.sign(df["actual"]).astype(int)).mean())
    return {"mse": mse, "mae": mae, "r2": r2, "hitrate": hitrate}

# ————————————————————————————————————————————————————————————————
# CLI
# ————————————————————————————————————————————————————————————————

def _cli():
    p = argparse.ArgumentParser(description="Walk‑forward backtest for XGB‑FGI models")
    sub = p.add_subparsers(dest="mode", required=True)

    # ── single asset (WALKFORWARD)
    p1 = sub.add_parser("single", help="Walk-forward backtest")
    p1.add_argument("--csv", required=True, help="ML‑ready csv file")
    p1.add_argument("--model_dir", required=True, help="Directory with saved model")
    p1.add_argument("--lookback", type=int, default=30, help="Lookback window (days)")
    p1.add_argument("--start", help="First date (yyyy‑mm‑dd) to include")
    p1.add_argument("--out", help="Where to save detailed walkforward results (.csv)")

    # ── quick split-test mode
    p2 = sub.add_parser("split", help="Quick test on 20% holdout set")
    p2.add_argument("--csv", required=True, help="ML‑ready csv file")

    args = p.parse_args()

    if args.mode == "single":
        split_df, split_metrics, walkforward_df = backtest_single_asset(
            args.csv, args.model_dir,
            lookback=args.lookback, start=args.start
        )

        print("\n" + "=" * 20 + " Internal 20% Split Evaluation " + "=" * 20)
        for k, v in split_metrics.items():
            print(f"{k.upper():8s}: {v:.6f}")

        print("\n" + "=" * 20 + " Walk-Forward Real Backtest " + "=" * 20)
        walkforward_metrics = quality_metrics(walkforward_df)
        for k, v in walkforward_metrics.items():
            print(f"{k.upper():8s}: {v:.6f}")

        if args.out:
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            walkforward_df.to_csv(args.out, index=False)
            print(f"\nDetailed walkforward results saved → {args.out}")

    elif args.mode == "split":
        # Bara snabbtest på 20%
        df = load_data(args.csv)
        trainer = Trainer()
        split_df = trainer.backtest_on_internal_split(df)
        metrics = quality_metrics(split_df)
        print("\n" + "=" * 20 + " Quick 20% Split Test " + "=" * 20)
        for k, v in metrics.items():
            print(f"{k.upper():8s}: {v:.6f}")


if __name__ == "__main__":
    _cli()
