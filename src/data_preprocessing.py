# src/data_preprocessing.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


DATASET_NAMES = {
    "uci": "UCI",
    "web_page": "Web Page Phishing",
    "phiusil": "PhiUSIIL",
    "zenodo": "Zenodo",
}


# Common target-column names encountered in the selected datasets.
TARGET_CANDIDATES = {
    "uci": [
        "Result",
        "result",
        "label",
        "Label",
        "target",
        "Target",
    ],
    "web_page": [
        "status",
        "Status",
        "label",
        "Label",
        "target",
        "Target",
        "class",
        "Class",
    ],
    "phiusil": [
        "label",
        "Label",
        "target",
        "Target",
        "status",
        "Status",
        "class",
        "Class",
    ],
    "zenodo": [
        "label",
        "Label",
        "target",
        "Target",
        "status",
        "Status",
        "class",
        "Class",
    ],
}


def load_dataset(path):
    """
    Load a CSV or Excel dataset.

    Parameters
    ----------
    path : str or Path
        Dataset file path.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file format: {suffix}. "
        "Use CSV or Excel."
    )


def find_target_column(df, dataset_name, target_column=None):
    """
    Identify the target column.

    An explicit target_column should be supplied whenever the dataset
    schema is known. Otherwise, known dataset-specific candidates are
    checked.
    """
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"Specified target column '{target_column}' "
                f"does not exist. Available columns: {list(df.columns)}"
            )

        return target_column

    candidates = TARGET_CANDIDATES[dataset_name]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Case-insensitive fallback.
    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized_columns:
            return normalized_columns[key]

    raise ValueError(
        f"Could not automatically identify the target column for "
        f"{DATASET_NAMES[dataset_name]}. "
        f"Specify target_column explicitly."
    )


def normalize_target(series, dataset_name):
    """
    Normalize target labels to:

        1 = phishing
        0 = legitimate

    The PhiUSIIL dataset specifically uses:
        1 = legitimate
        0 = phishing

    Other common textual/numeric representations are also handled.
    """
    values = series.copy()

    # Handle textual labels.
    if values.dtype == "object" or pd.api.types.is_string_dtype(values):
        normalized = (
            values.astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "phishing": 1,
            "phish": 1,
            "malicious": 1,
            "malware": 1,
            "fraud": 1,
            "bad": 1,
            "legitimate": 0,
            "legit": 0,
            "benign": 0,
            "safe": 0,
            "good": 0,
        }

        mapped = normalized.map(mapping)

        # If the values are numeric strings, convert them.
        numeric = pd.to_numeric(normalized, errors="coerce")
        mapped = mapped.where(mapped.notna(), numeric)

        values = mapped

    else:
        values = pd.to_numeric(values, errors="coerce")

    values = values.replace(
        {
            True: 1,
            False: 0,
        }
    )

    # Dataset-specific PhiUSIIL convention:
    # 1 = legitimate
    # 0 = phishing
    if dataset_name == "phiusil":
        unique_values = set(values.dropna().unique())

        if unique_values.issubset({0, 1}):
            values = values.map({
                0: 1,
                1: 0,
            })

    # Standard convention for the other datasets.
    else:
        unique_values = set(values.dropna().unique())

        if unique_values.issubset({0, 1}):
            # Most selected datasets already use:
            # 1 = phishing
            # 0 = legitimate.
            values = values.astype(int)

    unique_values = set(values.dropna().unique())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Unable to normalize target labels for "
            f"{DATASET_NAMES[dataset_name]}. "
            f"Observed values: {sorted(unique_values)}"
        )

    return values.astype(int)


def remove_identifier_columns(df, dataset_name, target_column):
    """
    Remove columns that are clearly identifiers or raw non-predictive
    website representations.

    This follows the thesis methodology:
    - UCI: provided numerical predictors are retained.
    - Web Page: non-predictive identifiers are excluded.
    - PhiUSIIL: raw/identifier-like string variables are excluded.
    - Zenodo: screenshots and raw heterogeneous resources are not
      directly used as tree-model inputs.
    """
    df = df.copy()

    explicit_identifier_names = {
        "id",
        "ID",
        "index",
        "Index",
        "Unnamed: 0",
        "url_id",
        "URL_ID",
        "website_id",
        "Website_ID",
    }

    columns_to_remove = []

    for column in df.columns:
        if column == target_column:
            continue

        column_name = str(column).strip()

        if column_name in explicit_identifier_names:
            columns_to_remove.append(column)
            continue

        lower_name = column_name.lower()

        # Remove obvious screenshot/image/resource fields.
        if dataset_name == "zenodo":
            image_terms = [
                "screenshot",
                "image",
                "thumbnail",
                "html",
                "css",
                "javascript",
                "source_code",
                "page_source",
            ]

            if any(term in lower_name for term in image_terms):
                columns_to_remove.append(column)
                continue

        # Remove raw URL/string representations from datasets where
        # the extracted numerical features are the intended predictors.
        if dataset_name in {"web_page", "phiusil"}:
            if lower_name in {
                "url",
                "website",
                "domain",
                "hostname",
                "uri",
            }:
                columns_to_remove.append(column)

    return df.drop(columns=columns_to_remove, errors="ignore")


def convert_predictors_to_numeric(X):
    """
    Keep numerical predictors suitable for Random Forest/XGBoost.

    Non-numeric raw fields are excluded because the thesis specifies
    structured/scalar tabular variables rather than raw string or
    screenshot inputs.
    """
    X = X.copy()

    numeric_columns = X.select_dtypes(
        include=[np.number, "bool"]
    ).columns

    X = X.loc[:, numeric_columns].copy()

    # Convert boolean values to integers.
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    return X


def handle_missing_values(X):
    """
    Handle missing predictor values using median imputation.

    Median imputation is performed independently for each numeric
    predictor. Columns that contain no usable numeric value are removed.
    """
    X = X.copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns containing no usable observations.
    X = X.dropna(
        axis=1,
        how="all"
    )

    for column in X.columns:
        median = X[column].median()

        if pd.isna(median):
            X = X.drop(columns=[column])
        else:
            X[column] = X[column].fillna(median)

    return X


def preprocess_dataset(
    path,
    dataset_name,
    target_column=None
):
    """
    Complete preprocessing for one dataset.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        Normalized binary target.

    metadata : dict
        Preprocessing information useful for experiment records.
    """
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Use one of: {list(DATASET_NAMES)}"
        )

    df = load_dataset(path)

    original_rows = len(df)
    original_columns = len(df.columns)

    # Remove exact duplicate records.
    df = df.drop_duplicates().reset_index(drop=True)

    duplicate_rows_removed = original_rows - len(df)

    target_column = find_target_column(
        df,
        dataset_name,
        target_column
    )

    # Normalize target first so invalid/missing target rows can be removed.
    y = normalize_target(
        df[target_column],
        dataset_name
    )

    valid_target_mask = y.notna()

    df = df.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].copy()

    # Remove target from predictors.
    X = df.drop(columns=[target_column])

    # Dataset-specific identifier/raw-resource removal.
    X = remove_identifier_columns(
        X,
        dataset_name,
        target_column
    )

    # Keep structured numerical predictors.
    X = convert_predictors_to_numeric(X)

    # Missing-value handling.
    missing_before = int(X.isna().sum().sum())

    X = handle_missing_values(X)

    missing_after = int(X.isna().sum().sum())

    # Make sure indices line up.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Final safety checks.
    if len(X) != len(y):
        raise ValueError(
            "Predictor and target lengths do not match."
        )

    if len(X) == 0:
        raise ValueError(
            "No usable observations remain after preprocessing."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "No usable numerical predictor columns remain."
        )

    unique_labels = set(y.unique())

    if unique_labels != {0, 1}:
        raise ValueError(
            f"Final target labels must be {{0, 1}}, "
            f"but got {unique_labels}."
        )

    metadata = {
        "dataset": DATASET_NAMES[dataset_name],
        "original_rows": original_rows,
        "original_columns": original_columns,
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "target_column": target_column,
        "final_rows": len(X),
        "final_features": X.shape[1],
        "phishing_count": int((y == 1).sum()),
        "legitimate_count": int((y == 0).sum()),
    }

    return X, y, metadata


def stratified_train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
):
    """
    Perform the common 80/20 stratified train-test split.

    This split must be generated once per dataset and reused by E1-E6.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def preprocess_and_split(
    path,
    dataset_name,
    target_column=None,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
):
    """
    Convenience function that preprocesses one dataset and immediately
    performs the common stratified 80/20 split.
    """
    X, y, metadata = preprocess_dataset(
        path=path,
        dataset_name=dataset_name,
        target_column=target_column
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = stratified_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    metadata["train_rows"] = len(X_train)
    metadata["test_rows"] = len(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata
    )