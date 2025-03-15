# OPTUNA
import optuna
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
import datetime
import argparse

import data
import examine


# Set up argument parser
parser = argparse.ArgumentParser(description="Random Survival Forest Hyperparameter Optimization with Optuna")
parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for Optuna")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()
    

# 1. Load data
X_train, _ = data.build_X("train")
y_train, y_xgb_train = data.load_y_train()


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 10),
        "max_features": trial.suggest_float("max_features", 0.2, 0.6),
        "max_depth": trial.suggest_int("max_depth", 8, 16),
        "max_samples": trial.suggest_float("max_samples", 0.3, 0.6),

    }
        
    mean_c_index_train, mean_c_index_test = examine.cross_validate(
        RandomSurvivalForest(**params, random_state=args.seed),
        X_train,
        y_train,
        y_xgb_train,
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
fname = f"results/optuna_rsf_{now.strftime('%m-%d_%H:%M')}.csv"
df_results = study.trials_dataframe()
df_results.to_csv(fname, index=False)
print(f"Study saved to {fname}")