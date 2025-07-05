import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"Best Params: {model.best_params_}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    return {
        "Model": name,
        "BestParams": model.best_params_,
        "MSE": round(mse, 4),
        "R2": round(r2, 4)
    }

def main():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # 1. Ridge Regression
    ridge = Ridge(max_iter=10000)
    ridge_params = {
    "alpha": [0.1, 1.0, 10.0],
    "fit_intercept": [True, False],
    "solver": ['auto', 'svd'], 
    }
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring="neg_mean_squared_error")
    ridge_grid.fit(X_train, y_train)
    results.append(evaluate_model("Ridge Regression", ridge_grid, X_test, y_test))

    # 2. Decision Tree Regressor
    dtree = DecisionTreeRegressor(random_state=42)
    dtree_params = {
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    dtree_grid = GridSearchCV(dtree, dtree_params, cv=5, scoring="neg_mean_squared_error")
    dtree_grid.fit(X_train, y_train)
    results.append(evaluate_model("Decision Tree", dtree_grid, X_test, y_test))

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8, None],
        "max_features": ['sqrt', 'log2', None]  # ⬅️ VALID options only
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf_grid, X_test, y_test))
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("\nSaved hyperparameter tuning results to hyperparameter_tuning_results.csv")

if __name__ == "__main__":
    main()
