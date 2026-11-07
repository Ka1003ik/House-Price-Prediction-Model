# UCI Real Estate Valuation — Feature Ablation Experiment

## Objective

Evaluate whether excluding `Feature_12` changes the performance of three regression models on the UCI Real Estate Valuation dataset.

## Dataset and target

- Records: 506
- Target column: `Feature_14`
- Target display: Median House Value (Thousand USD)
- Target validation: 506 valid numeric targets; 0 missing, non-numeric, or infinite targets
- Target mean: 22.53 Thousand USD
- Target median: 21.20 Thousand USD
- Target minimum / maximum: 5.00 / 50.00 Thousand USD

## Experiment configurations

| Setting | Experiment 1 | Experiment 2 |
|---|---|---|
| Feature_12 | Included | Excluded |
| Configured input features | 13 | 12 |
| Regression models | Linear Regression, Decision Tree, Random Forest | Linear Regression, Decision Tree, Random Forest |
| Dataset records | 506 | 506 |

Other configuration and validation settings should be kept identical when reproducing this comparison.

## Experiment 1 — Feature_12 included

| Model | CV RMSE Mean | Training RMSE | Validation RMSE | Validation MAE | Validation R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 4.7597 | 4.5392 | 4.9818 | 3.3071 | 0.6616 |
| Decision Tree | 5.3616 | 0.0000 | 3.4220 | 2.4843 | 0.8403 |
| Random Forest | 3.8542 | 1.3806 | 2.9671 | 2.0674 | 0.8799 |

All error values are in Thousand USD, except R².

## Experiment 2 — Feature_12 excluded

| Model | CV RMSE Mean | CV RMSE Std | Training RMSE | Validation RMSE | Validation MAE | Validation R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 4.8527 | 0.5236 | 4.6469 | 4.8144 | 3.2112 | 0.6839 |
| Decision Tree | 5.2873 | 0.7660 | 0.0000 | 4.6083 | 2.6578 | 0.7104 |
| Random Forest | 3.8647 | 0.5452 | 1.3827 | 2.9495 | 2.0244 | 0.8814 |

All error values are in Thousand USD, except R².

## Best-model comparison

| Metric | Experiment 1 | Experiment 2 | Change |
|---|---:|---:|---:|
| Best model | Random Forest | Random Forest | No change |
| CV RMSE Mean | 3.8542 | 3.8647 | +0.0105 (worse) |
| Validation RMSE | 2.9671 | 2.9495 | -0.0176 (better) |
| Validation MAE | 2.0674 | 2.0244 | -0.0430 (better) |
| Validation R² | 0.8799 | 0.8814 | +0.0015 (better) |

## Interpretation

Random Forest remained the strongest model in both runs. With `Feature_12` excluded, its validation RMSE and MAE decreased slightly, while validation R² increased slightly. Its cross-validation RMSE increased by 0.0105 Thousand USD, indicating a small deterioration in that measure.

The Decision Tree's validation RMSE increased from 3.4220 to 4.6083 Thousand USD and its validation R² decreased from 0.8403 to 0.7104. Linear Regression improved slightly on validation RMSE, MAE, and R².

The effect of removing `Feature_12` is therefore model-dependent. The Random Forest changes are marginal and should not be described as a major or statistically significant improvement without repeated runs or statistical testing.

## Conclusion

For this dataset and these reported runs, Random Forest is the preferred model in both configurations. Excluding `Feature_12` produced a small improvement in its validation metrics but a small deterioration in cross-validation RMSE. Retain both experiment results and report the feature-ablation outcome as a marginal, model-dependent change rather than a definitive optimization.

## Individual prediction example

The Experiment 2 application generated an individual predicted median house value of **21.50 Thousand USD** for the input record shown during testing. This is an example prediction for that particular input, not a general model-accuracy metric.
