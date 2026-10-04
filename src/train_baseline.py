import json

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from common import (
    DATASETS,
    MODEL_DIR,
    RESULT_DIR,
    RANDOM_STATE,
    ensure_directories,
    save_model,
    save_json,
)

from data_preprocessing import get_dataset_split


# ============================================================
# BASELINE TRAINING
# ============================================================

def train_random_forest(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E1_random_forest.joblib"
    )

    save_model(
        model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E1",
        "classifier": "Random Forest",
        "optimizer": "Default",
        "parameters": model.get_params(),
    }

    save_json(
        metadata,
        RESULT_DIR
        / f"{dataset_name}_E1_training.json",
    )


def train_xgboost(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E4_xgboost.joblib"
    )

    save_model(
        model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E4",
        "classifier": "XGBoost",
        "optimizer": "Default",
        "parameters": model.get_params(),
    }

    save_json(
        metadata,
        RESULT_DIR
        / f"{dataset_name}_E4_training.json",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_directories()

    for dataset_name in DATASETS:
        print(f"\nTraining E1: {dataset_name}")
        train_random_forest(dataset_name)

        print(f"Training E4: {dataset_name}")
        train_xgboost(dataset_name)

    print("\nBaseline training complete.")