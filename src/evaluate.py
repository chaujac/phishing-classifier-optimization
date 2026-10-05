import argparse
import json
import time
import tracemalloc

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from common import (
    DATASETS,
    MODEL_DIR,
    RESULT_DIR,
    CLASSIFICATION_THRESHOLD,
    LATENCY_WARMUP,
    LATENCY_SAMPLES,
    ensure_directories,
    load_model,
    load_split,
    save_json,
)


# ============================================================
# PREDICTION
# ============================================================

def predict_with_threshold(
    model,
    X,
):
    probabilities = model.predict_proba(X)[:, 1]

    return (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)


# ============================================================
# LATENCY MEASUREMENT
# ============================================================

def measure_latency(
    model,
    X_test,
):
    warmup_count = min(
        LATENCY_WARMUP,
        len(X_test),
    )

    for i in range(warmup_count):
        model.predict_proba(
            X_test.iloc[[i]]
        )

    sample_count = min(
        LATENCY_SAMPLES,
        len(X_test),
    )

    latencies = []

    tracemalloc.start()

    for i in range(sample_count):
        start = time.perf_counter()

        model.predict_proba(
            X_test.iloc[[i]]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed * 1000
        )

    _, peak_memory = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    return {
        "sample_count": sample_count,
        "warmup_count": warmup_count,
        "mean_ms": float(
            np.mean(latencies)
        ),
        "std_ms": float(
            np.std(latencies)
        ),
        "peak_memory_bytes": int(
            peak_memory
        ),
        "peak_memory_mb": float(
            peak_memory / (1024 ** 2)
        ),
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    dataset_name,
    experiment,
    model_filename,
):
    split = load_split(
        dataset_name
    )

    X_test = split["X_test"]
    y_test = split["y_test"]

    model_path = (
        MODEL_DIR / model_filename
    )

    model = load_model(
        model_path
    )

    y_pred = predict_with_threshold(
        model,
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Legitimate",
            "Phishing",
        ],
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    latency = measure_latency(
        model,
        X_test,
    )

    results = {
        "dataset": dataset_name,
        "experiment": experiment,
        "model": model_filename,

        "classification": {
            "threshold": CLASSIFICATION_THRESHOLD,
            "accuracy": float(accuracy),
            "macro_precision": float(
                macro_precision
            ),
            "macro_recall": float(
                macro_recall
            ),
            "macro_f1": float(
                macro_f1
            ),
            "per_class": report,
            "confusion_matrix": confusion.tolist(),
        },

        "inference": latency,
    }

    output_path = (
        EVALUATION_RESULT_DIR
        / f"{dataset_name}_{experiment}_evaluation.json"
    )

    save_json(
        results,
        output_path,
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
            "E6",
        ],
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    args = parser.parse_args()

    ensure_directories()

    results = evaluate_model(
        args.dataset,
        args.experiment,
        args.model,
    )

    print(
        json.dumps(
            results,
            indent=4,
        )
    )