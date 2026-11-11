"""Isolated bounded tabular model comparison. Only the parent invokes this module."""
from __future__ import annotations

from io import StringIO
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def train(frame: pd.DataFrame, target: str, problem: str) -> dict:
    if target not in frame or problem not in {"regression", "classification"}:
        raise ValueError("Select a valid target and problem type")
    data = frame.dropna(subset=[target]).copy()
    if not 30 <= len(data) <= 20000 or not 2 <= data.shape[1] <= 100:
        raise ValueError("Training requires 30–20000 rows and 2–100 columns")
    y = data.pop(target)
    dropped = [col for col in data if col.casefold() in {"id", "uuid", "row_id"} or col.casefold().endswith("_id")
               or data[col].isna().all() or (not pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() > 100)]
    x = data.drop(columns=dropped)
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in x if col not in numeric]
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        x[col] = x[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    if x.empty or len(numeric) + sum(x[col].nunique() for col in categorical) > 500:
        raise ValueError("No usable features or encoded feature budget exceeded; remove IDs/free text")
    stratify = None
    if problem == "classification":
        y = y.astype(str)
        if not 2 <= y.nunique() <= 30 or y.value_counts().min() < 3:
            raise ValueError("Classification requires 2–30 classes with at least 3 examples each")
        stratify = y
    else:
        y = pd.to_numeric(y, errors="raise")
        if not np.isfinite(y).all() or y.nunique() < 2:
            raise ValueError("Regression needs finite, nonconstant numeric targets")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=stratify)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                  ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    preprocess = ColumnTransformer(transformers)
    if problem == "classification":
        estimators = {"Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                      "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=1),
                      "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
                      "KNN": KNeighborsClassifier(n_neighbors=3),
                      "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42)}
        folds = min(3, int(y_train.value_counts().min()))
        if folds < 2:
            raise ValueError("Too few training examples per class for cross-validation")
        cv = StratifiedKFold(folds, shuffle=True, random_state=42)
        scoring, baseline = "f1_weighted", DummyClassifier(strategy="prior")
    else:
        estimators = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=1),
                      "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=1),
                      "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
                      "KNN": KNeighborsRegressor(n_neighbors=3),
                      "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42)}
        folds = 3
        cv, scoring, baseline = KFold(3, shuffle=True, random_state=42), "r2", DummyRegressor(strategy="mean")
    results = []
    warnings.filterwarnings("ignore", category=UserWarning)
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        results.append({"model": name, "validation_mean": float(scores.mean()), "validation_std": float(scores.std())})
    results.sort(key=lambda row: row["validation_mean"], reverse=True)
    winner_name = results[0]["model"]
    winner = Pipeline([("preprocess", preprocess), ("model", estimators[winner_name])]).fit(x_train, y_train)
    predictions = winner.predict(x_test)
    baseline_scores = cross_val_score(baseline, x_train, y_train, cv=cv, scoring=scoring)
    if problem == "classification":
        test = {"accuracy": float(accuracy_score(y_test, predictions)),
                "f1_weighted": float(f1_score(y_test, predictions, average="weighted"))}
    else:
        test = {"r2": float(r2_score(y_test, predictions)), "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(mean_squared_error(y_test, predictions)**.5)}
    return {"target": target, "problem_type": problem, "selection_metric": scoring,
            "selection_method": f"{folds}-fold cross-validation on training data only",
            "leaderboard": results, "winner": winner_name, "held_out_test_metrics": test,
            "baseline_validation_mean": float(baseline_scores.mean()), "train_rows": len(x_train),
            "test_rows": len(x_test), "dropped_columns": dropped, "features": list(x.columns),
            "random_seed": 42,
            "limitations": ["Single bounded split/CV experiment; not deployment or general performance certification",
                            "Possible semantic leakage columns and temporal dependencies require domain review"]}


def main():
    try:
        raw = sys.stdin.read(20*1024*1024 + 1)
        if len(raw) > 20*1024*1024:
            raise ValueError("Worker payload too large")
        payload = json.loads(raw)
        result = train(pd.read_json(StringIO(payload["frame"]), orient="split"), payload["target"], payload["problem"])
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": str(error)[:500]}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
