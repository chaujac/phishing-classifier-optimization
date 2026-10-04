# src/train_optuna.py

from pathlib import Path
import argparse
import json
import time

import joblib
import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
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


N_TRIALS = 50


CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


def rf_complexity(params):
    return (
        params["n_estimators"] *
        params["max_depth"]
    )


def xgb_complexity(params):
    return (
        params["n_estimators"] *
        params["max_depth"]
    )


def optimize_random_forest(
    X_train,
    y_train
):
    """
    E3: Random Forest optimized using Optuna TPE.

    Thesis-defined Optuna search space:
        n_estimators: 100-300
        max_depth: 10-30
        min_samples_split: 2-10
        min_samples_leaf: 1-4
        max_features: sqrt/log2
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                300
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                10,
                30
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                10
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                4
            ),

            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            ),
        }

        model = RandomForestClassifier(
            **params
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="accuracy",
            cv=CV,
            n_jobs=-1,
        )

        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    start = time.perf_counter()

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    optimization_time = (
        time.perf_counter() - start
    )

    # ------------------------------------------------------------
    # Thesis tie-breaking rule
    # ------------------------------------------------------------
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise RuntimeError(
            "Optuna completed no valid trials."
        )

    completed_trials.sort(
        key=lambda trial: (
            -trial.value,
            rf_complexity(trial.params)
        )
    )

    best_trial = completed_trials[0]

    best_params = best_trial.params

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
        float(best_trial.value),
        rf_complexity(best_params),
        optimization_time,
    )


def optimize_xgboost(
    X_train,
    y_train
):
    """
    E6: XGBoost optimized using Optuna TPE.

    Thesis-defined Optuna search space:
        n_estimators: 100-300
        learning_rate: 0.01-0.20
        max_depth: 3-7
        subsample: 0.8-1.0
        colsample_bytree: 0.8-1.0
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                300
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.20
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                7
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.8,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.8,
                1.0
            ),
        }

        model = XGBClassifier(
            **params
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="accuracy",
            cv=CV,
            n_jobs=-1,
        )

        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    start = time.perf_counter()

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    optimization_time = (
        time.perf_counter() - start
    )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise RuntimeError(
            "Optuna completed no valid trials."
        )

    completed_trials.sort(
        key=lambda trial: (
            -trial.value,
            xgb_complexity(trial.params)
        )
    )

    best_trial = completed_trials[0]

    best_params = best_trial.params

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
        float(best_trial.value),
        xgb_complexity(best_params),
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
    # E3 - Random Forest Optuna
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
        f"{dataset_name}_E3_RF_Optuna.joblib"
    )

    joblib.dump(
        rf_model,
        rf_path
    )

    # ------------------------------------------------------------
    # E6 - XGBoost Optuna
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
        f"{dataset_name}_E6_XGB_Optuna.joblib"
    )

    joblib.dump(
        xgb_model,
        xgb_path
    )

    results = {
        "dataset": dataset_name,

        "E3": {
            "model": str(rf_path),
            "trials": N_TRIALS,
            "best_parameters": rf_params,
            "mean_cv_accuracy": rf_score,
            "complexity_proxy": rf_complexity_value,
            "optimization_wall_clock_seconds": rf_search_time,
        },

        "E6": {
            "model": str(xgb_path),
            "trials": N_TRIALS,
            "best_parameters": xgb_params,
            "mean_cv_accuracy": xgb_score,
            "complexity_proxy": xgb_complexity_value,
            "optimization_wall_clock_seconds": xgb_search_time,
        },

        "preprocessing": metadata,
    }

    result_path = (
        MODEL_DIR /
        f"{dataset_name}_optuna_metadata.json"
    )

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print("\nE3 Random Forest")
    print("Best parameters:", rf_params)
    print("CV accuracy:", rf_score)
    print(
        "Optimization time:",
        rf_search_time,
        "seconds"
    )

    print("\nE6 XGBoost")
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