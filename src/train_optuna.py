import time
import json

import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from xgboost import XGBClassifier

from common import (
    DATASETS,
    MODEL_DIR,
    TRAINING_RESULT_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    N_OPTUNA_TRIALS,
    RF_OPTUNA,
    XGB_OPTUNA,
    ensure_directories,
    save_model,
    save_json,
)
from data_preprocessing import get_dataset_split


# ============================================================
# CROSS-VALIDATION
# ============================================================

CV = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# RANDOM FOREST OPTUNA
# ============================================================

def optimize_random_forest(
    X_train,
    y_train,
):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                RF_OPTUNA["n_estimators"][0],
                RF_OPTUNA["n_estimators"][1],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                RF_OPTUNA["max_depth"][0],
                RF_OPTUNA["max_depth"][1],
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                RF_OPTUNA["min_samples_split"][0],
                RF_OPTUNA["min_samples_split"][1],
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                RF_OPTUNA["min_samples_leaf"][0],
                RF_OPTUNA["min_samples_leaf"][1],
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                RF_OPTUNA["max_features"],
            ),
        }

        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            **params,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="accuracy",
            cv=CV,
            n_jobs=-1,
        )

        return scores.mean()

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    start_time = time.perf_counter()

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
    )

    optimization_time = (
        time.perf_counter()
        - start_time
    )

    return (
        study,
        optimization_time,
    )


# ============================================================
# XGBOOST OPTUNA
# ============================================================

def optimize_xgboost(
    X_train,
    y_train,
):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                XGB_OPTUNA["n_estimators"][0],
                XGB_OPTUNA["n_estimators"][1],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                XGB_OPTUNA["learning_rate"][0],
                XGB_OPTUNA["learning_rate"][1],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                XGB_OPTUNA["max_depth"][0],
                XGB_OPTUNA["max_depth"][1],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                XGB_OPTUNA["subsample"][0],
                XGB_OPTUNA["subsample"][1],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                XGB_OPTUNA["colsample_bytree"][0],
                XGB_OPTUNA["colsample_bytree"][1],
            ),
        }

        model = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            **params,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="accuracy",
            cv=CV,
            n_jobs=-1,
        )

        return scores.mean()

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    start_time = time.perf_counter()

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
    )

    optimization_time = (
        time.perf_counter()
        - start_time
    )

    return (
        study,
        optimization_time,
    )


# ============================================================
# RANDOM FOREST TRAINING
# ============================================================

def train_random_forest(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    study, optimization_time = (
        optimize_random_forest(
            X_train,
            y_train,
        )
    )

    best_params = study.best_params
    best_score = study.best_value

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        **best_params,
    )

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E3_random_forest.joblib"
    )

    save_model(
        model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E3",
        "classifier": "Random Forest",
        "optimizer": "Optuna",
        "sampler": "TPE",
        "trials": N_OPTUNA_TRIALS,
        "cv_folds": CV_FOLDS,
        "best_cv_accuracy": float(best_score),
        "best_parameters": best_params,
        "optimization_time_seconds": optimization_time,
    }

    save_json(
        metadata,
        TRAINING_RESULT_DIR
        / f"{dataset_name}_E3_training.json",
    )


# ============================================================
# XGBOOST TRAINING
# ============================================================

def train_xgboost(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    study, optimization_time = (
        optimize_xgboost(
            X_train,
            y_train,
        )
    )

    best_params = study.best_params
    best_score = study.best_value

    model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        **best_params,
    )

    model.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E6_xgboost.joblib"
    )

    save_model(
        model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E6",
        "classifier": "XGBoost",
        "optimizer": "Optuna",
        "sampler": "TPE",
        "trials": N_OPTUNA_TRIALS,
        "cv_folds": CV_FOLDS,
        "best_cv_accuracy": float(best_score),
        "best_parameters": best_params,
        "optimization_time_seconds": optimization_time,
    }

    save_json(
        metadata,
        TRAINING_RESULT_DIR
        / f"{dataset_name}_E6_training.json",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_directories()

    for dataset_name in DATASETS:
        print(f"\nTraining E3: {dataset_name}")
        train_random_forest(dataset_name)

        print(f"Training E6: {dataset_name}")
        train_xgboost(dataset_name)

    print("\nOptuna training complete.")