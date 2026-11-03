
from pathlib import Path
import pandas as pd

# =====================================================
# 1. LOAD DATASET AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Preserve meaningful labels such as "None" and "NA".
# Only empty cells are initially treated as missing.
df = pd.read_csv(
    DATA_PATH,
    keep_default_na=False,
    na_values=[""]
)

# Columns where "NA" means that the house does not
# have the corresponding feature.
absence_columns = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature"
]

# Convert absence labels into an explicit category.
for column in absence_columns:
    df[column] = df[column].replace("NA", "None")

# In MasVnrType, "NA" is not a documented category.
# Treat it as genuinely missing information.
df["MasVnrType"] = df["MasVnrType"].replace("NA", pd.NA)

# Ensure these columns have numerical data types.
numeric_columns = [
    "MasVnrArea",
    "GarageYrBlt",
    "LotFrontage"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


print("\n" + "=" * 65)
print("          HOUSING DATA - MISSING VALUE INVESTIGATION")
print("=" * 65)

print(f"\nTotal houses: {len(df)}")
print(f"Total features: {len(df.columns)}")


# =====================================================
# 2. MASONRY VENEER ANALYSIS
# =====================================================

print("\n[1] MASONRY VENEER ANALYSIS")
print("-" * 65)

type_missing = df["MasVnrType"].isna()
area_missing = df["MasVnrArea"].isna()

# Explicitly recorded houses without masonry veneer.
no_veneer = df["MasVnrType"].eq("None")

print(f"Houses with no masonry veneer: {no_veneer.sum()}")
print(f"Houses with unknown masonry type: {type_missing.sum()}")
print(f"Houses with missing masonry area: {area_missing.sum()}")

print("\nMasonry type distribution:")
print(df["MasVnrType"].value_counts(dropna=False).to_string())

# Investigate houses explicitly marked as having no veneer.
no_veneer_area = df.loc[no_veneer, "MasVnrArea"]

print("\nMasonry area among houses with no veneer:")
print(f"Area equal to zero: {(no_veneer_area == 0).sum()}")
print(f"Area greater than zero: {(no_veneer_area > 0).sum()}")
print(f"Area missing: {no_veneer_area.isna().sum()}")

# Investigate genuinely missing masonry types.
unknown_type_area = df.loc[type_missing, "MasVnrArea"]

print("\nAmong houses with unknown masonry type:")
print(f"Area available: {unknown_type_area.notna().sum()}")
print(f"Area also missing: {unknown_type_area.isna().sum()}")

print("\nInterpretation:")
print("'None' means no masonry veneer.")
print("A missing type means the masonry information is unknown.")
print("These two situations must be handled differently.")


# =====================================================
# 3. GARAGE ANALYSIS
# =====================================================

print("\n[2] GARAGE INFORMATION ANALYSIS")
print("-" * 65)

# "None" represents houses without garages.
no_garage = df["GarageType"].eq("None")

# Missing construction year is checked separately.
garage_year_missing = df["GarageYrBlt"].isna()

print(f"Houses without garages: {no_garage.sum()}")
print(f"Houses with missing garage construction year: "
      f"{garage_year_missing.sum()}")

print("\nGarage type distribution:")
print(df["GarageType"].value_counts(dropna=False).to_string())

# Check whether no-garage records match missing garage years.
matching_records = (
    no_garage == garage_year_missing
).all()

print("\nDo houses without garages have missing garage years?")

if matching_records:
    print("YES - The records match exactly.")
else:
    print("NO - Further investigation is required.")

print("\nInterpretation:")
print("Houses without garages should retain an explicit")
print("'None' category rather than an ordinary missing value.")


# =====================================================
# 4. LOT FRONTAGE ANALYSIS
# =====================================================

print("\n[3] LOT FRONTAGE ANALYSIS")
print("-" * 65)

missing_frontage = df["LotFrontage"].isna().sum()

print(f"Missing LotFrontage records: {missing_frontage}")
print(f"Available LotFrontage records: "
      f"{df['LotFrontage'].notna().sum()}")

# Calculate frontage statistics for each neighborhood.
frontage_summary = (
    df.groupby("Neighborhood")["LotFrontage"]
    .agg(
        Available_Records="count",
        Median_Frontage="median",
        Average_Frontage="mean"
    )
    .round(2)
    .sort_values("Median_Frontage")
)

print("\nNeighborhood-wise frontage statistics:")
print(frontage_summary.to_string())

print("\nInterpretation:")
print("Lot frontage varies across neighborhoods.")
print("Neighborhood-wise median imputation is worth considering.")


# =====================================================
# 5. OVERALL MISSING VALUE REPORT
# =====================================================

print("\n[4] OVERALL MISSING VALUE REPORT")
print("-" * 65)

# This report counts genuinely missing values only.
# Explicit absence categories such as "None" are preserved.

missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2)
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
].sort_values(
    by="Missing_Percentage",
    ascending=False
)

if missing_report.empty:
    print("No missing values found.")
else:
    print(missing_report.to_string())


# =====================================================
# 6. INVESTIGATE UNUSUAL MASONRY RECORDS
# =====================================================

print("\n[5] UNUSUAL MASONRY RECORDS")
print("-" * 65)

# These houses are labelled "None" but have a positive
# masonry area. We flag them without modifying their values.

unusual_masonry = df[
    no_veneer &
    df["MasVnrArea"].notna() &
    (df["MasVnrArea"] > 0)
]

print(f"Number of unusual records: {len(unusual_masonry)}")

if not unusual_masonry.empty:

    columns_to_display = [
        "Id",
        "MasVnrType",
        "MasVnrArea",
        "Exterior1st",
        "Exterior2nd",
        "Neighborhood",
        "SalePrice"
    ]

    print("\nDetails of unusual houses:")
    print(
        unusual_masonry[columns_to_display]
        .to_string(index=False)
    )

    print("\nInterpretation:")
    print("These records contain conflicting masonry information.")
    print("We will investigate before deciding whether to correct them.")

else:
    print("No unusual masonry records found.")


# =====================================================
# 7. FINAL INVESTIGATION SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("                 INVESTIGATION SUMMARY")
print("=" * 65)

print(f"Total houses examined: {len(df)}")

print("\nMASONRY")
print(f"No masonry veneer: {no_veneer.sum()}")
print(f"Unknown masonry type: {type_missing.sum()}")
print(f"Missing masonry area: {area_missing.sum()}")
print(f"Unusual masonry records: {len(unusual_masonry)}")

print("\nGARAGE")
print(f"Houses without garages: {no_garage.sum()}")
print(f"Missing garage construction years: "
      f"{garage_year_missing.sum()}")

print("\nLOT FRONTAGE")
print(f"Missing frontage values: {missing_frontage}")

print("\nTOTAL GENUINELY MISSING VALUES:",
      int(df.isna().sum().sum()))

print("\nInvestigation completed successfully!")
print("=" * 65)
