# OPTUNA
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sksurv.util import Surv
import datetime
import argparse

import data
import examine


# Set up argument parser
parser = argparse.ArgumentParser(description="XGBoost Hyperparameter Optimization with Optuna")
parser.add_argument("--transform", action="store_true", help="Apply Power Transformation to numerical features")
parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for Optuna")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--tree_method", type=str, default="hist", help="Tree method for XGBoost")
args = parser.parse_args()
    

# 1. Load data
X_train, _ = data.build_X("train", transform=args.transform)
y_train, y_xgb_train = data.load_y_train()


def objective(trial):
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "subsample": trial.suggest_float("subsample", 0.4, 0.7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        # new hyperparameters
        "tree_method": args.tree_method,
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 0.9),
    }
        
    mean_c_index_train, mean_c_index_test = examine.cross_validate(
        xgb.XGBRegressor(**params, random_state=args.seed),
        X_train,
        y_train,
        y_xgb_train,
        transform=args.transform,
        n_splits=5,
        seed=args.seed
    )
    trial.set_user_attr("c_index_train", mean_c_index_train)
    trial.set_user_attr("c_index_val", mean_c_index_test)
    
    return mean_c_index_test  # We want to maximize the test c-index

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.n_trials, n_jobs=20, show_progress_bar=True)

# Best parameters
print("Best Parameters:", study.best_params)

# Save the study
now = datetime.datetime.now()
fname = f"results/optuna_xgb_{now.strftime('%m-%d_%H:%M')}.csv"
df_results = study.trials_dataframe()
df_results["power_transform"] = args.transform
df_results.to_csv(fname, index=False)
print(f"Study saved to {fname}")