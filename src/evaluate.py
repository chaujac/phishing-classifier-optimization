# src/evaluate.py

from pathlib import Path
import argparse
import json
import time
import tracemalloc

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


MODEL_DIR = Path("models")

DECISION_THRESHOLD = 0.50

WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 1000


def load_experiment_data(dataset_name):
    """
    Load the held-out test partition created by the preprocessing
    pipeline.
    """
    split_path = (
        MODEL_DIR /
        f"{dataset_name}_split.joblib"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {split_path}. "
            "Run train_baseline.py first."
        )

    return joblib.load(split_path)


def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_with_threshold(model, X):
    """
    Generate probability predictions and apply the strict 0.50
    decision threshold.

    1 = phishing
    0 = legitimate
    """
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    return probabilities, predictions


def calculate_predictive_metrics(y_true, y_pred):
    """
    Calculate aggregate and per-class predictive metrics.
    """
    labels = [0, 1]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision_macro": float(
            precision_macro
        ),

        "recall_macro": float(
            recall_macro
        ),

        "f1_macro": float(
            f1_macro
        ),

        "per_class": {
            "legitimate_0": {
                "precision": float(
                    precision_per_class[0]
                ),
                "recall": float(
                    recall_per_class[0]
                ),
                "f1_score": float(
                    f1_per_class[0]
                ),
            },

            "phishing_1": {
                "precision": float(
                    precision_per_class[1]
                ),
                "recall": float(
                    recall_per_class[1]
                ),
                "f1_score": float(
                    f1_per_class[1]
                ),
            },
        },

        "confusion_matrix": {
            "labels": [
                "legitimate_0",
                "phishing_1"
            ],
            "matrix": cm.tolist(),
        },
    }


def measure_inference_latency(
    model,
    X_test
):
    """
    Measure single-instance prediction latency.

    Protocol:
        1. Warm up with 10 predictions.
        2. Discard warm-up measurements.
        3. Measure 1,000 single-instance predictions.
        4. Use time.perf_counter().
        5. Measure peak Python memory allocation with tracemalloc.

    The input is already prepared, consistent with the thesis definition
    of model-level inference latency.
    """
    if len(X_test) == 0:
        raise ValueError(
            "Test set is empty."
        )

    # ------------------------------------------------------------
    # Warm-up phase
    # ------------------------------------------------------------
    warmup_count = min(
        WARMUP_ITERATIONS,
        len(X_test)
    )

    for i in range(warmup_count):
        sample = X_test.iloc[[i]]

        model.predict_proba(sample)

    # ------------------------------------------------------------
    # Measurement samples
    # ------------------------------------------------------------
    measurement_count = MEASUREMENT_ITERATIONS

    samples = [
        X_test.iloc[
            [i % len(X_test)]
        ]
        for i in range(measurement_count)
    ]

    # Start memory tracking only for the actual measurement phase.
    tracemalloc.start()

    latency_values = []

    for sample in samples:
        start = time.perf_counter()

        model.predict_proba(sample)

        end = time.perf_counter()

        latency_values.append(
            (end - start) * 1000.0
        )

    current_memory, peak_memory = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    latency_values = np.asarray(
        latency_values,
        dtype=np.float64
    )

    return {
        "iterations": measurement_count,

        "warmup_iterations": warmup_count,

        "mean_latency_ms": float(
            latency_values.mean()
        ),

        "std_latency_ms": float(
            latency_values.std(
                ddof=1
            )
        ),

        "min_latency_ms": float(
            latency_values.min()
        ),

        "max_latency_ms": float(
            latency_values.max()
        ),

        "peak_memory_mb": float(
            peak_memory /
            (1024 ** 2)
        ),

        "current_memory_mb": float(
            current_memory /
            (1024 ** 2)
        ),
    }


def evaluate_model(
    model_path,
    dataset_name
):
    """
    Evaluate one E1-E6 model.
    """
    experiment_data = load_experiment_data(
        dataset_name
    )

    model = load_model(
        model_path
    )

    X_test = experiment_data["X_test"]
    y_test = experiment_data["y_test"]

    # ------------------------------------------------------------
    # Predictive evaluation
    # ------------------------------------------------------------
    probabilities, predictions = (
        predict_with_threshold(
            model,
            X_test
        )
    )

    predictive_metrics = (
        calculate_predictive_metrics(
            y_test,
            predictions
        )
    )

    # ------------------------------------------------------------
    # Inference latency + memory
    # ------------------------------------------------------------
    computational_metrics = (
        measure_inference_latency(
            model,
            X_test
        )
    )

    result = {
        "dataset": dataset_name,
        "model": str(model_path),

        "decision_threshold": DECISION_THRESHOLD,

        "predictive_metrics": predictive_metrics,

        "computational_metrics": computational_metrics,
    }

    return result


def save_results(result):
    dataset_name = result["dataset"]
    model_path = Path(result["model"])

    experiment_name = model_path.stem

    output_path = (
        MODEL_DIR /
        f"{dataset_name}_{experiment_name}_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4
        )

    return output_path


def print_results(result):
    predictive = result[
        "predictive_metrics"
    ]

    computational = result[
        "computational_metrics"
    ]

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    print(
        f"Dataset: {result['dataset']}"
    )

    print(
        f"Model: {result['model']}"
    )

    print(
        f"Decision threshold: "
        f"{result['decision_threshold']:.2f}"
    )

    print("\nPredictive Metrics")
    print(
        f"Accuracy: "
        f"{predictive['accuracy']:.6f}"
    )

    print(
        f"Macro Precision: "
        f"{predictive['precision_macro']:.6f}"
    )

    print(
        f"Macro Recall: "
        f"{predictive['recall_macro']:.6f}"
    )

    print(
        f"Macro F1: "
        f"{predictive['f1_macro']:.6f}"
    )

    print("\nLegitimate (0)")
    print(
        f"Precision: "
        f"{predictive['per_class']['legitimate_0']['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{predictive['per_class']['legitimate_0']['recall']:.6f}"
    )

    print(
        f"F1: "
        f"{predictive['per_class']['legitimate_0']['f1_score']:.6f}"
    )

    print("\nPhishing (1)")
    print(
        f"Precision: "
        f"{predictive['per_class']['phishing_1']['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{predictive['per_class']['phishing_1']['recall']:.6f}"
    )

    print(
        f"F1: "
        f"{predictive['per_class']['phishing_1']['f1_score']:.6f}"
    )

    print("\nConfusion Matrix")
    print(
        np.array(
            predictive[
                "confusion_matrix"
            ]["matrix"]
        )
    )

    print("\nInference")
    print(
        f"Iterations: "
        f"{computational['iterations']}"
    )

    print(
        f"Warm-up: "
        f"{computational['warmup_iterations']}"
    )

    print(
        f"Mean latency: "
        f"{computational['mean_latency_ms']:.6f} ms"
    )

    print(
        f"Std latency: "
        f"{computational['std_latency_ms']:.6f} ms"
    )

    print(
        f"Peak memory: "
        f"{computational['peak_memory_mb']:.6f} MB"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "uci",
            "web_page",
            "phiusil",
            "zenodo",
        ]
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained .joblib model."
    )

    args = parser.parse_args()

    result = evaluate_model(
        model_path=args.model,
        dataset_name=args.dataset
    )

    output_path = save_results(
        result
    )

    print_results(
        result
    )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()