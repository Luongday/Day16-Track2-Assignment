"""LightGBM benchmark on the Kaggle Credit Card Fraud dataset (Lab 16, Step 4.4).

Usage (on the compute node):
    python3 benchmark.py [--data ~/ml-benchmark/creditcard.csv] [--out benchmark_result.json]
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    args = parser.parse_args()

    # 1. Load data
    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(df):,} rows in {load_time:.2f}s")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # 2. Train
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        metric="auc",
        # Class imbalance (~0.17% fraud): without these, leaves with a handful of
        # positives get huge outputs and validation AUC collapses after iteration 1.
        min_sum_hessian_in_leaf=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)],
    )
    train_time = time.perf_counter() - t0
    best_iter = int(model.best_iteration_ or model.n_estimators)
    print(f"Trained in {train_time:.2f}s (best iteration {best_iter})")

    # 3. Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
    }

    # 4. Inference latency (1 row, averaged) and throughput (1000 rows)
    one = X_test.iloc[[0]]
    model.predict(one)  # warm-up
    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(one)
    latency_ms = (time.perf_counter() - t0) / runs * 1000

    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    model.predict(batch)
    batch_time = time.perf_counter() - t0
    throughput = len(batch) / batch_time

    result = {
        "dataset_rows": int(len(df)),
        "load_data_seconds": round(load_time, 4),
        "training_seconds": round(train_time, 4),
        "best_iteration": best_iter,
        **{k: round(float(v), 6) for k, v in metrics.items()},
        "inference_latency_1row_ms": round(latency_ms, 4),
        "inference_1000rows_seconds": round(batch_time, 4),
        "inference_throughput_rows_per_sec": round(throughput, 1),
        "numpy_version": np.__version__,
        "lightgbm_version": lgb.__version__,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
