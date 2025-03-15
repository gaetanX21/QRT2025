import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sksurv.metrics import concordance_index_ipcw
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import datetime
from sklearn.metrics import make_scorer

class MetaModel:
    def __init__(self, models: list, alphas: list):
        self.models = models
        self.alphas = alphas

    def fit(self, X, y, y_xgb):
        for model in self.models:
            if isinstance(model, XGBRegressor):
                model.fit(X, y_xgb)
            else:
                model.fit(X, y)

    def predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:,i] = model.predict(X_test)
        predictions = np.log(predictions)
        predictions = MinMaxScaler().fit_transform(predictions)
        predictions = np.dot(predictions, self.alphas)
        return predictions


class XGB_RSF:
    def __init__(self, xgb, rsf, alpha):
        self.xgb = xgb
        self.rsf = rsf
        self.alpha = alpha

    def fit(self, X, y, y_xgb):
        self.xgb.fit(X, y_xgb)
        self.rsf.fit(X, y)

    def predict(self, X_test):
        xgb_pred = self.xgb.predict(X_test)
        rsf_pred = self.rsf.predict(X_test)
        predictions = np.column_stack((xgb_pred, rsf_pred))
        predictions = np.log(predictions)
        predictions = MinMaxScaler().fit_transform(predictions)
        predictions = self.alpha * predictions[:, 0] + (1 - self.alpha) * predictions[:, 1]
        return predictions


def cross_validate(model, X: pd.DataFrame, y: np.ndarray, y_xgb: np.ndarray, n_splits: int=5, seed: int=42):

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    c_index_train_scores = []
    c_index_test_scores = []

    X = X.copy() # to avoid modifying the original dataframe
    for train_idx, test_idx in folds.split(X):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute missing values AFTER splitting to avoid data leakage
        # a) fill missing nonnegative integer values with -1
        int_imputer = SimpleImputer(strategy='constant', fill_value=-1)
        int_cols = X_train.select_dtypes(include=[np.int64]).columns
        X_train[int_cols] = int_imputer.fit_transform(X_train[int_cols])
        X_test[int_cols] = int_imputer.transform(X_test[int_cols])
        # b) fill missing float values with mode
        float_imputer = SimpleImputer(strategy='most_frequent')
        float_cols = X_train.select_dtypes(include=[np.float64]).columns
        X_train[float_cols] = float_imputer.fit_transform(X_train[float_cols])
        X_test[float_cols] = float_imputer.transform(X_test[float_cols])

        # Standardize the data
        # if transform:
        #     power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        #     X_train[float_cols] = power_transformer.fit_transform(X_train[float_cols])
        #     X_test[float_cols] = power_transformer.transform(X_test[float_cols])

        # Fit the model
        if isinstance(model, XGBRegressor):
            # XGBoost requires the target to be in a different format
            y_train_xgb = y_xgb[train_idx]
            model.fit(X_train, y_train_xgb)
        elif isinstance(model, MetaModel) or isinstance(model, XGB_RSF):
            # Fit the meta model
            y_train_xgb = y_xgb[train_idx]
            model.fit(X_train, y_train, y_train_xgb)
        else:
            model.fit(X_train, y_train)
        
        # Compute Concordance Index IPCW
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        c_index_train = concordance_index_ipcw(y_train, y_train, y_pred_train, tau=7)[0]
        c_index_test = concordance_index_ipcw(y_train, y_test, y_pred_test, tau=7)[0]

        # Append the scores
        c_index_train_scores.append(c_index_train)
        c_index_test_scores.append(c_index_test)

    # Compute the mean c-index for this trial
    mean_c_index_train = np.mean(c_index_train_scores)
    mean_c_index_test = np.mean(c_index_test_scores)

    return mean_c_index_train, mean_c_index_test


def plot_PI_xgb(model, X: pd.DataFrame, y: np.ndarray, y_xgb: np.ndarray, n_repeats: int=10, random_state: int=42, top_k: int=10):
    """
    Compute the permutation importance of the model.
    """
    # Fit the model first
    model.fit(X, y_xgb)

    # Compute the permutation importance
    # define scoring using c-index IPCW
    def scoring(y_true, y_pred):
        # don't use y_true because it's the y_xgb so it has negative values for right censored
        return concordance_index_ipcw(y, y, y_pred, tau=7)[0]
    scoring = make_scorer(scoring, greater_is_better=True)
    result = permutation_importance(model, X, y_xgb, n_repeats=n_repeats, random_state=random_state, scoring=scoring)

    # Create a DataFrame with the results
    results_df = pd.DataFrame(index=X.columns)
    results_df['importances_mean'] = np.abs(result.importances_mean)
    results_df['importances_std'] = result.importances_std
    results_df = results_df.sort_values(by="importances_mean", ascending=False)
    results_df = results_df.head(top_k) # keep only the top k features
    results_df = results_df.sort_values(by="importances_mean", ascending=True) # sort by importance

    # plot the importances
    fig, ax = plt.subplots(figsize=(6,4), dpi=200)
    results_df['importances_mean'].plot.barh(yerr=results_df['importances_std'], ax=ax)
    model_name = model.__class__.__name__
    ax.set_title(f"Permutation Importances ({model_name})")
    ax.set_xlabel("Mean Importance")
    ax.set_ylabel("Feature")
    time = datetime.datetime.now()
    fname = f"{model_name}_{time.strftime('%m-%d_%H:%M')}"
    plt.savefig(f"img/permutation_importance_xgb_{fname}.png", dpi=200, bbox_inches='tight')
    plt.close()
