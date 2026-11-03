
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Sale Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="SalePrice", kde=True)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "saleprice_distribution.png")
plt.show()

# 2. Sale Price Outlier Analysis
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["SalePrice"])
plt.title("Sale Price Outlier Analysis")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "saleprice_boxplot.png")
plt.show()

# 3. Living Area vs Sale Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Living Area vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "living_area_vs_price.png")
plt.show()

# 4. Correlation with SalePrice
correlations = (
    df.select_dtypes(include="number")
    .corr(numeric_only=True)["SalePrice"]
    .sort_values(ascending=False)
)

print("\n--- TOP 15 FEATURES CORRELATED WITH SALEPRICE ---")
print(correlations.head(15))

# 5. Correlation Heatmap
top_features = correlations.head(11).index

plt.figure(figsize=(12, 8))
sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", OUTPUT_DIR)
