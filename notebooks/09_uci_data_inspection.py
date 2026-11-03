
from pathlib import Path
import pandas as pd

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "uci_real_estate"
    / "Real estate valuation data set.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 65)
print("           UCI REAL ESTATE DATA INSPECTION")
print("=" * 65)

# =====================================================
# 2. LOAD EXCEL DATASET
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

excel_file = pd.ExcelFile(DATA_PATH)

print("\nAvailable Excel sheets:")
print(excel_file.sheet_names)

df = pd.read_excel(
    DATA_PATH,
    sheet_name=excel_file.sheet_names[0],
    engine="openpyxl"
)

print("\nDataset loaded successfully.")


# =====================================================
# 3. DATASET OVERVIEW
# =====================================================

print("\n[1] DATASET OVERVIEW")
print("-" * 65)

print(f"Total records: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string(index=False))


# =====================================================
# 4. DATA TYPES
# =====================================================

print("\n[2] DATASET INFORMATION")
print("-" * 65)

df.info()

print("\nData type distribution:")
print(df.dtypes.value_counts())


# =====================================================
# 5. STATISTICAL SUMMARY
# =====================================================

print("\n[3] STATISTICAL SUMMARY")
print("-" * 65)

print(df.describe(include="all").to_string())


# =====================================================
# 6. MISSING VALUE ANALYSIS
# =====================================================

print("\n[4] MISSING VALUE ANALYSIS")
print("-" * 65)

missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
].sort_values(
    by="Missing_Percentage",
    ascending=False
)

if missing_report.empty:
    print("No missing values detected.")
else:
    print(missing_report.to_string())

missing_report.to_csv(
    OUTPUT_DIR / "uci_missing_value_report.csv"
)


# =====================================================
# 7. DUPLICATE RECORD ANALYSIS
# =====================================================

print("\n[5] DUPLICATE RECORD ANALYSIS")
print("-" * 65)

print("Duplicate rows:", df.duplicated().sum())

if "No" in df.columns:
    print("Unique record IDs:", df["No"].nunique())
    print("Duplicate record IDs:",
          df["No"].duplicated().sum())


# =====================================================
# 8. FEATURE ANALYSIS
# =====================================================

print("\n[6] FEATURE ANALYSIS")
print("-" * 65)

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()

print("Numerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nUnique values in each column:")

for column in df.columns:
    print(
        f"{column}: {df[column].nunique()} unique values"
    )

# =====================================================
# 9. TARGET VARIABLE ANALYSIS
# =====================================================

print("\n[7] TARGET VARIABLE ANALYSIS")
print("-" * 65)

TARGET_COLUMN = "Y house price of unit area"

if TARGET_COLUMN in df.columns:

    print("Target variable:", TARGET_COLUMN)

    print("\nTarget statistical summary:")
    print(df[TARGET_COLUMN].describe())

    print("\nMissing target values:",
          df[TARGET_COLUMN].isnull().sum())

    print("\nTarget variable successfully identified.")

else:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

# =====================================================
# 10. SAVE INSPECTION SUMMARY
# =====================================================

print("\n[8] SAVE INSPECTION SUMMARY")
print("-" * 65)

summary = df.describe().T

summary.to_csv(
    OUTPUT_DIR / "uci_summary_statistics.csv"
)

print("Missing-value report saved.")
print("Summary statistics saved.")


# =====================================================
# 11. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("          UCI DATA INSPECTION COMPLETED")
print("=" * 65)

print(f"Records examined: {df.shape[0]}")
print(f"Features examined: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing cells: {df.isnull().sum().sum()}")

print("\nTarget variable: Y")
print("\nInspection completed successfully!")
print("=" * 65)
