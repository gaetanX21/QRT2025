# Challenge Data QRT 2025: Overall Survival Prediction for Patients Diagnosed with Myeloid Leukemia

[Official challenge page](https://challengedata.ens.fr/participants/challenges/162/)

## Overview
The goal of this challenge is to predict the overall survival (risk score) of patients diagnosed with myeloid leukemia based on their clinical and genetic data. More information can be found in the [challenge page](https://challengedata.ens.fr/participants/challenges/162/) or in my [report](report/report.pdf).

The metric was the C-IPCW index, which is a variant of the C-index that accounts for the presence of competing risks. Our private score is 0.7076 vs 0.6411 for the benchmark, putting us in the 13th place.

Our approach is based on the following steps:
1. **Data Preprocessing**: Transformation & Imputation.
2. **Feature Engineering**: Creation of new features, in particular by leveraging the medical literature on myeloid leukemia.
3. **Modeling**: We tested several survival models (Penalized Cox, Random Survival Forest, Survival XGBoost) and used `optuna` for Bayesian hyperparameter tuning. We used cross-validation to evaluate the models and select the best one, which was XGBoost.
4. **Analysis of feature importances and output quality**: More information can be found in the report.

## File Structure
This repository is structured as follows:
- `data/`: Directory containing the data files.
- `src/`: Directory containing the source code.
- `report/`: Directory containing the report.
- `img/`: Directory containing figures.

The `src/` directory contains the following files:
- `data.py`: Utils for loading and preprocessing data.
- `examine.py`: Utils for performing cross-validation, computing feature importance, and building metamodels.
- `optuna_rsf.py`: Script for hyperparameter tuning of Random Survival Forest using Optuna.
- `optuna_xgb.py`: Script for hyperparameter tuning of Survival XGBoost using Optuna.
- `Benchmark.ipynb`: Benchmark provided by QRT. (0.6541 public score and 0.6411 private score)
- `final.py`: Notebook for final submission. (0.7268 public score and 0.7076 private score)
