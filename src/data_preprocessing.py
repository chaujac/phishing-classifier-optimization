import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from common import (
    DATASETS,
    RANDOM_STATE,
    TEST_SIZE,
    save_split,
    load_split,
)


# ============================================================
# TARGET COLUMNS
# ============================================================

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


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    paths = DATASETS[dataset_name]

    # Handle datasets split across multiple files (e.g., Zenodo)
    if isinstance(paths, list):
        dataframes = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            
            df = pd.read_csv(path)
            
            # Inject labels for Zenodo based on the filename before concatenating
            if "not-phishing" in path.name.lower():
                df["label"] = 0
            elif "phishing" in path.name.lower():
                df["label"] = 1
                
            dataframes.append(df)
            
        return pd.concat(dataframes, ignore_index=True)

    # Handle single file datasets
    if not paths.exists():
        raise FileNotFoundError(f"Dataset not found: {paths}")

    if paths.suffix.lower() == ".csv":
        return pd.read_csv(paths)

    if paths.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(paths)

    raise ValueError(f"Unsupported file format: {paths.suffix}")


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

def find_target_column(df, dataset_name):
    candidates = TARGET_CANDIDATES[dataset_name]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify target column for {dataset_name}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series, dataset_name):
    target = series.copy()

    # Handle string labels
    if target.dtype == "object" or str(target.dtype).startswith("string"):
        normalized = (
            target.astype(str)
            .str.strip()
            .str.lower()
        )

        phishing_values = {
            "phishing",
            "phish",
            "malicious",
            "1",
            "true",
        }

        legitimate_values = {
            "legitimate",
            "legit",
            "benign",
            "0",
            "false",
        }

        result = pd.Series(
            np.nan,
            index=target.index,
            dtype="float64",
        )

        result[normalized.isin(phishing_values)] = 1
        result[normalized.isin(legitimate_values)] = 0

        if result.isna().any():
            unknown = normalized[result.isna()].unique()

            raise ValueError(
                f"Unknown target labels in {dataset_name}: {unknown}"
            )

        return result.astype(int)

    numeric = pd.to_numeric(target, errors="coerce")

    if numeric.isna().any():
        raise ValueError(
            f"Target contains non-numeric values that could not "
            f"be normalized in {dataset_name}."
        )

    unique_values = set(numeric.unique())

    # UCI Phishing Websites Dataset:
    # -1 = phishing
    #  1 = legitimate
    if dataset_name == "uci" and unique_values == {-1, 1}:
        return numeric.map({
            -1: 1,
            1: 0,
        }).astype(int)

    # PhiUSIIL:
    # 1 = legitimate
    # 0 = phishing
    if dataset_name == "phiusil" and unique_values == {0, 1}:
        return numeric.map({
            0: 1,
            1: 0,
        }).astype(int)

    # Already normalized
    if unique_values == {0, 1}:
        return numeric.astype(int)

    raise ValueError(
        f"Unsupported target encoding for {dataset_name}: "
        f"{sorted(unique_values)}"
    )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


# ============================================================
# IDENTIFIER / RAW STRING REMOVAL
# ============================================================

def remove_non_predictive_columns(df, dataset_name):
    columns_to_remove = set()

    common_identifiers = {
        "id",
        "ID",
        "index",
        "Index",
        "url_id",
        "URL_ID",
        "website_id",
        "Website_ID",
    }

    columns_to_remove.update(
        column for column in df.columns
        if column in common_identifiers
    )

    if dataset_name == "web_page":
        web_identifiers = {
            "url",
            "URL",
            "domain",
            "Domain",
            "hostname",
            "Hostname",
            "uri",
            "URI",
        }

        columns_to_remove.update(
            column for column in df.columns
            if column in web_identifiers
        )

    elif dataset_name == "phiusil":
        raw_string_fields = {
            "url",
            "URL",
            "domain",
            "Domain",
            "hostname",
            "Hostname",
            "uri",
            "URI",
        }

        columns_to_remove.update(
            column for column in df.columns
            if column in raw_string_fields
        )

    elif dataset_name == "zenodo":
        resource_fields = {
            "screenshot",
            "Screenshot",
            "image",
            "Image",
            "thumbnail",
            "Thumbnail",
            "html",
            "HTML",
            "css",
            "CSS",
            "javascript",
            "JavaScript",
            "source_code",
            "Source_Code",
            "page_source",
            "Page_Source",
        }

        columns_to_remove.update(
            column for column in df.columns
            if column in resource_fields
        )

    return df.drop(
        columns=list(columns_to_remove),
        errors="ignore",
    )


# ============================================================
# NUMERICAL PREDICTORS
# ============================================================

def convert_predictors_to_numeric(X):
    # Include both numbers and booleans
    numeric_columns = X.select_dtypes(
        include=[np.number, bool]
    ).columns

    X = X[numeric_columns].copy()
    
    # Convert any boolean columns to integers (0 and 1) for XGBoost
    bool_cols = X.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def handle_missing_values(X):
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.dropna(
        axis=1,
        how="all",
    )

    for column in X.columns:
        if X[column].isna().any():
            X[column] = X[column].fillna(
                X[column].median()
            )

    return X


# ============================================================
# FULL DATASET PREPROCESSING
# ============================================================

def preprocess_dataset(dataset_name):
    df = load_dataset(dataset_name)

    original_rows = len(df)

    target_column = find_target_column(
        df,
        dataset_name,
    )

    df = remove_duplicates(df)

    y = normalize_target(
        df[target_column],
        dataset_name,
    )

    X = df.drop(
        columns=[target_column]
    )

    X = remove_non_predictive_columns(
        X,
        dataset_name,
    )

    X = convert_predictors_to_numeric(X)

    X = handle_missing_values(X)

    if len(X) != len(y):
        raise ValueError(
            "Feature and target lengths do not match."
        )

    if y.nunique() != 2:
        raise ValueError(
            f"Target must contain exactly two classes. "
            f"Found: {y.unique()}"
        )

    metadata = {
        "dataset": dataset_name,
        "original_rows": original_rows,
        "processed_rows": len(X),
        "feature_count": X.shape[1],
        "target_column": target_column,
        "class_distribution": {
            "phishing": int((y == 1).sum()),
            "legitimate": int((y == 0).sum()),
        },
    }

    return X, y, metadata


# ============================================================
# CANONICAL 80/20 SPLIT
# ============================================================

def create_and_save_split(dataset_name):
    X, y, metadata = preprocess_dataset(
        dataset_name
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata,
    }

    save_split(
        split_data,
        dataset_name,
    )

    return split_data


# ============================================================
# LOAD OR CREATE CANONICAL SPLIT
# ============================================================

def get_dataset_split(dataset_name):
    try:
        return load_split(dataset_name)
    except FileNotFoundError:
        return create_and_save_split(dataset_name)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for dataset_name in DATASETS:
        split = create_and_save_split(
            dataset_name
        )

        print(
            f"{dataset_name}: "
            f"{len(split['X_train'])} training / "
            f"{len(split['X_test'])} testing"
        )