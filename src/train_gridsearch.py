# src/train_gridsearch.py

from pathlib import Path
import argparse
import json
import time

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
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


CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.10, 0.20],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def rf_complexity(params):
    """
    Thesis tie-breaking proxy:

        n_estimators * max_depth

    None is treated as infinity because it represents an
    unbounded tree depth.
    """
    depth = params["max_depth"]

    if depth is None:
        return float("inf")

    return (
        params["n_estimators"] *
        depth
    )


def xgb_complexity(params):
    """
    Complexity proxy for XGBoost.

    n_estimators * max_depth
    """
    return (
        params["n_estimators"] *
        params["max_depth"]
    )


def select_best_configuration(
    cv_results,
    complexity_function
):
    """
    Select the highest-scoring configuration.

    If mean validation scores tie, choose the configuration with
    lower expected inference complexity.
    """
    results = []

    for index, score in enumerate(
        cv_results["mean_test_score"]
    ):
        params = cv_results["params"][index]

        if np.isnan(score):
            continue

        complexity = complexity_function(params)

        results.append(
            (
                float(score),
                complexity,
                params
            )
        )

    if not results:
        raise RuntimeError(
            "GridSearchCV did not produce any valid results."
        )

    # Highest score first.
    # For equal scores, lower complexity first.
    results.sort(
        key=lambda item: (
            -item[0],
            item[1]
        )
    )

    best_score, best_complexity, best_params = results[0]

    return (
        best_params,
        best_score,
        best_complexity
    )


def optimize_random_forest(
    X_train,
    y_train
):
    """
    E2: Random Forest GridSearchCV.
    """
    estimator = RandomForestClassifier()

    search = GridSearchCV(
        estimator=estimator,
        param_grid=RF_GRID,
        scoring="accuracy",
        cv=CV,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
    )

    start = time.perf_counter()

    search.fit(
        X_train,
        y_train
    )

    optimization_time = (
        time.perf_counter() - start
    )

    (
        best_params,
        best_score,
        complexity,
    ) = select_best_configuration(
        search.cv_results_,
        rf_complexity
    )

    # Retrain on complete training partition.
    model = RandomForestClassifier(
        **best_params
    )

    model.fit(
        X_train,
        y_train
    )

    return (
        model,
        best_params,
        best_score,
        complexity,
        optimization_time,
    )


def optimize_xgboost(
    X_train,
    y_train
):
    """
    E5: XGBoost GridSearchCV.
    """
    estimator = XGBClassifier()

    search = GridSearchCV(
        estimator=estimator,
        param_grid=XGB_GRID,
        scoring="accuracy",
        cv=CV,
        n_jobs=-1,
        refit=False,
        return_train_score=False,
    )

    start = time.perf_counter()

    search.fit(
        X_train,
        y_train
    )

    optimization_time = (
        time.perf_counter() - start
    )

    (
        best_params,
        best_score,
        complexity,
    ) = select_best_configuration(
        search.cv_results_,
        xgb_complexity
    )

    # Retrain on complete training partition.
    model = XGBClassifier(
        **best_params
    )

    model.fit(
        X_train,
        y_train
    )

    return (
        model,
        best_params,
        best_score,
        complexity,
        optimization_time,
    )


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

    # ------------------------------------------------------------
    # E2 - Random Forest GridSearchCV
    # ------------------------------------------------------------
    (
        rf_model,
        rf_params,
        rf_score,
        rf_complexity_value,
        rf_search_time,
    ) = optimize_random_forest(
        X_train,
        y_train
    )

    rf_path = (
        MODEL_DIR /
        f"{dataset_name}_E2_RF_GridSearch.joblib"
    )

    joblib.dump(
        rf_model,
        rf_path
    )

    # ------------------------------------------------------------
    # E5 - XGBoost GridSearchCV
    # ------------------------------------------------------------
    (
        xgb_model,
        xgb_params,
        xgb_score,
        xgb_complexity_value,
        xgb_search_time,
    ) = optimize_xgboost(
        X_train,
        y_train
    )

    xgb_path = (
        MODEL_DIR /
        f"{dataset_name}_E5_XGB_GridSearch.joblib"
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    results = {
        "dataset": dataset_name,

        "E2": {
            "model": str(rf_path),
            "best_parameters": rf_params,
            "mean_cv_accuracy": rf_score,
            "complexity_proxy": rf_complexity_value,
            "optimization_wall_clock_seconds": rf_search_time,
        },

        "E5": {
            "model": str(xgb_path),
            "best_parameters": xgb_params,
            "mean_cv_accuracy": xgb_score,
            "complexity_proxy": xgb_complexity_value,
            "optimization_wall_clock_seconds": xgb_search_time,
        },

        "preprocessing": metadata,
    }

    result_path = (
        MODEL_DIR /
        f"{dataset_name}_gridsearch_metadata.json"
    )

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print("\nE2 Random Forest")
    print("Best parameters:", rf_params)
    print("CV accuracy:", rf_score)
    print(
        "Optimization time:",
        rf_search_time,
        "seconds"
    )

    print("\nE5 XGBoost")
    print("Best parameters:", xgb_params)
    print("CV accuracy:", xgb_score)
    print(
        "Optimization time:",
        xgb_search_time,
        "seconds"
    )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True
    )

    args = parser.parse_args()

    run_dataset(args.dataset)


if __name__ == "__main__":
    main()