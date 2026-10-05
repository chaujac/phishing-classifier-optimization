import time
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from common import (
    DATASETS,
    MODEL_DIR,
    TRAINING_RESULT_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    RF_GRID,
    XGB_GRID,
    ensure_directories,
    save_model,
    save_json,
)
from data_preprocessing import get_dataset_split


# ============================================================
# CROSS-VALIDATION STRATEGY
# ============================================================

CV = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# RANDOM FOREST GRIDSEARCH
# ============================================================

def train_random_forest(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_GRID,
        scoring="accuracy",
        cv=CV,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    grid_search.fit(X_train, y_train)
    optimization_time = time.perf_counter() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E2_random_forest.joblib"
    )

    save_model(
        best_model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E2",
        "classifier": "Random Forest",
        "optimizer": "GridSearchCV",
        "cv_folds": CV_FOLDS,
        "best_cv_accuracy": float(best_score),
        "best_parameters": best_params,
        "optimization_time_seconds": optimization_time,
    }

    save_json(
        metadata,
        TRAINING_RESULT_DIR
        / f"{dataset_name}_E2_training.json",
    )


# ============================================================
# XGBOOST GRIDSEARCH
# ============================================================

def train_xgboost(dataset_name):
    split = get_dataset_split(dataset_name)

    X_train = split["X_train"]
    y_train = split["y_train"]

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=XGB_GRID,
        scoring="accuracy",
        cv=CV,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    grid_search.fit(X_train, y_train)
    optimization_time = time.perf_counter() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_E5_xgboost.joblib"
    )

    save_model(
        best_model,
        model_path,
    )

    metadata = {
        "dataset": dataset_name,
        "experiment": "E5",
        "classifier": "XGBoost",
        "optimizer": "GridSearchCV",
        "cv_folds": CV_FOLDS,
        "best_cv_accuracy": float(best_score),
        "best_parameters": best_params,
        "optimization_time_seconds": optimization_time,
    }

    save_json(
        metadata,
        TRAINING_RESULT_DIR
        / f"{dataset_name}_E5_training.json",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_directories()

    for dataset_name in DATASETS:
        print(f"\nTraining E2 (GridSearch RF): {dataset_name}")
        train_random_forest(dataset_name)

        print(f"Training E5 (GridSearch XGB): {dataset_name}")
        train_xgboost(dataset_name)

    print("\nGridSearchCV training complete.")