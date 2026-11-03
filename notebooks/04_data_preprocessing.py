
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_ames_dataset,
    separate_features_target,
    build_preprocessor
)

DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 65)
print("             HOUSE PRICE DATA PREPROCESSING")
print("=" * 65)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = load_ames_dataset(DATA_PATH)

print("\n[1] DATASET LOADING")
print("-" * 65)

print(f"Total records: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print("Dataset loaded successfully.")


# ============================================================
# 3. INITIAL DATA QUALITY CHECK
# ============================================================

print("\n[2] INITIAL DATA QUALITY CHECK")
print("-" * 65)

duplicate_count = df.duplicated().sum()
missing_target = df["SalePrice"].isna().sum()

print(f"Duplicate records: {duplicate_count}")
print(f"Missing SalePrice values: {missing_target}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate records detected. Investigate before training."
    )

if missing_target > 0:
    raise ValueError(
        "Target variable contains missing values."
    )

masonry_conflicts = (
    df["MasVnrType"].eq("None")
    & df["MasVnrArea"].gt(0)
).sum()

print(
    f"Masonry records requiring investigation: {masonry_conflicts}"
)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X, y = separate_features_target(
    df,
    target_column="SalePrice",
    id_columns=("Id",)
)

print("\n[3] FEATURE AND TARGET SEPARATION")
print("-" * 65)

print(f"Input feature count: {X.shape[1]}")
print("Target variable: SalePrice")
print(f"Target records: {len(y)}")
print("Identifier column 'Id' excluded from model features.")


# ============================================================
# 5. TRAINING AND VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n[4] TRAINING AND VALIDATION SPLIT")
print("-" * 65)

print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_valid)}")
print("Validation percentage: 20%")
print("Random state: 42")


# ============================================================
# 6. BUILD AND FIT PREPROCESSING PIPELINE
# ============================================================

print("\n[5] PREPROCESSING PIPELINE")
print("-" * 65)

preprocessor = build_preprocessor(X_train)

# Learn preprocessing values from training data only.
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the fitted transformations to validation data.
X_valid_processed = preprocessor.transform(X_valid)

print("Preprocessing pipeline fitted successfully.")
print("Validation data transformed successfully.")


# ============================================================
# 7. VALIDATE TRANSFORMED DATA
# ============================================================

print("\n[6] TRANSFORMED DATA VALIDATION")
print("-" * 65)

print(f"Original training shape: {X_train.shape}")
print(f"Processed training shape: {X_train_processed.shape}")

print(f"Original validation shape: {X_valid.shape}")
print(f"Processed validation shape: {X_valid_processed.shape}")

feature_names = (
    preprocessor
    .named_steps["column_preprocessor"]
    .get_feature_names_out()
)

print(f"Total encoded features: {len(feature_names)}")


def check_finite_values(matrix):
    """Check that a transformed matrix has no NaN or infinity."""

    values = (
        matrix.data
        if sparse.issparse(matrix)
        else np.asarray(matrix)
    )

    return bool(np.isfinite(values).all())


train_is_valid = check_finite_values(X_train_processed)
validation_is_valid = check_finite_values(X_valid_processed)

print(
    "\nAll training values finite:",
    "YES" if train_is_valid else "NO"
)

print(
    "All validation values finite:",
    "YES" if validation_is_valid else "NO"
)

if not train_is_valid or not validation_is_valid:
    raise ValueError(
        "Non-finite values detected after preprocessing."
    )

if X_train_processed.shape[1] != X_valid_processed.shape[1]:
    raise ValueError(
        "Training and validation feature counts do not match."
    )


# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================

print("\n[7] SAVING PROCESSED DATA")
print("-" * 65)

if sparse.issparse(X_train_processed):

    sparse.save_npz(
        OUTPUT_DIR / "X_train_processed.npz",
        X_train_processed
    )

    sparse.save_npz(
        OUTPUT_DIR / "X_valid_processed.npz",
        X_valid_processed
    )

    saved_format = "NPZ"

else:

    np.save(
        OUTPUT_DIR / "X_train_processed.npy",
        X_train_processed
    )

    np.save(
        OUTPUT_DIR / "X_valid_processed.npy",
        X_valid_processed
    )

    saved_format = "NPY"


# Save target variables.
y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_valid.to_csv(
    OUTPUT_DIR / "y_valid.csv",
    index=False
)


# Save feature names.
with open(
    OUTPUT_DIR / "preprocessed_feature_names.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(feature_names.astype(str)))

# Save the fitted preprocessing pipeline
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    preprocessor,
    MODEL_DIR / "ames_preprocessor.joblib"
)

print("Fitted preprocessing pipeline saved successfully.")

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("              PREPROCESSING COMPLETED")
print("=" * 65)

print(f"Training records processed: {X_train_processed.shape[0]}")
print(f"Validation records processed: {X_valid_processed.shape[0]}")
print(f"Final model features: {X_train_processed.shape[1]}")
print(f"Storage format: {saved_format}")

print("\nGenerated files:")
print(f"1. X_train_processed.{saved_format.lower()}")
print(f"2. X_valid_processed.{saved_format.lower()}")
print("3. y_train.csv")
print("4. y_valid.csv")
print("5. preprocessed_feature_names.txt")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nAll preprocessing checks passed!")
print("=" * 65)
