# src/train_baseline.py

from pathlib import Path
import argparse
import json
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_preprocessing import (
    preprocess_and_split,
    RANDOM_STATE,
)


DATASETS = {
    "uci": {
        "path": "data/uci_phishing.csv",
        "target": None,
    },
    "web_page": {
        "path": "data/web_page_phishing.csv",
        "target": None,
    },
    "phiusil": {
        "path": "data/phiusil.csv",
        "target": None,
    },
    "zenodo": {
        "path": "data/zenodo_phishing.csv",
        "target": None,
    },
}


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_random_forest(X_train, y_train):
    """
    E1: Random Forest using library defaults.
    """
    model = RandomForestClassifier()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def train_xgboost(X_train, y_train):
    """
    E4: XGBoost using library defaults.
    """
    model = XGBClassifier()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def run_dataset(dataset_name):
    config = DATASETS[dataset_name]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata,
    ) = preprocess_and_split(
        path=config["path"],
        dataset_name=dataset_name,
        target_column=config["target"],
        random_state=RANDOM_STATE,
    )

    print(
        f"\nDataset: {dataset_name}"
        f"\nTraining samples: {len(X_train)}"
        f"\nTest samples: {len(X_test)}"
        f"\nFeatures: {X_train.shape[1]}"
    )

    # ------------------------------------------------------------
    # E1 - Random Forest
    # ------------------------------------------------------------
    rf_model, rf_training_time = train_random_forest(
        X_train,
        y_train
    )

    rf_path = MODEL_DIR / f"{dataset_name}_E1_RF.joblib"

    joblib.dump(
        rf_model,
        rf_path
    )

    # ------------------------------------------------------------
    # E4 - XGBoost
    # ------------------------------------------------------------
    xgb_model, xgb_training_time = train_xgboost(
        X_train,
        y_train
    )

    xgb_path = MODEL_DIR / f"{dataset_name}_E4_XGB.joblib"

    joblib.dump(
        xgb_model,
        xgb_path
    )

    # Save the test partition so E1-E6 can be evaluated on exactly
    # the same held-out observations.
    split_path = MODEL_DIR / f"{dataset_name}_split.joblib"

    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": list(X_train.columns),
            "metadata": metadata,
        },
        split_path
    )

    results = {
        "dataset": dataset_name,
        "E1": {
            "model": str(rf_path),
            "training_time_seconds": rf_training_time,
        },
        "E4": {
            "model": str(xgb_path),
            "training_time_seconds": xgb_training_time,
        },
        "preprocessing": metadata,
    }

    result_path = (
        MODEL_DIR /
        f"{dataset_name}_baseline_metadata.json"
    )

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print(f"E1 saved to: {rf_path}")
    print(f"E4 saved to: {xgb_path}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to train."
    )

    args = parser.parse_args()

    run_dataset(args.dataset)


if __name__ == "__main__":
    main()