
import pandas as pd

# Load the original housing dataset
df = pd.read_csv("../dataset/train.csv")

# 1. Display first five records
print("\n--- FIRST 5 RECORDS ---")
print(df.head())

# 2. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 3. Column information
print("\n--- DATASET INFORMATION ---")
df.info()

# 4. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 5. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 6. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())


# Missing value analysis
missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})

missing = missing[missing["Missing_Count"] > 0]
missing = missing.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print("\n--- MISSING VALUE REPORT ---")
print(missing)

# Unique values in categorical columns
categorical_cols = df.select_dtypes(
    include=["object", "str", "category"]
).columns

print("\n--- CATEGORICAL FEATURE COUNTS ---")
print(df[categorical_cols].nunique().sort_values(ascending=False))

# Target variable distribution
print("\n--- SALE PRICE DISTRIBUTION ---")
print(df["SalePrice"].describe())

# 1. Check column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 2. Check numerical and categorical feature counts
print("\n--- FEATURE TYPE COUNTS ---")
print(df.dtypes.value_counts())

# 3. Check target variable for missing values
print("\n--- TARGET VARIABLE CHECK ---")
print("Missing SalePrice:", df["SalePrice"].isnull().sum())

# 4. Check unique values in target variable
print("Unique SalePrice values:", df["SalePrice"].nunique())