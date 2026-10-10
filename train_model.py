"""
Run this once (and again whenever the data changes):

    python train_model.py

It cleans the recipe data, fits the baseline (Logistic Regression) and the
comparison model (Random Forest), saves the model the app uses, and saves the
numbers the app shows. It also writes requirements.txt with the exact package
versions installed in your environment, so the deployed app matches.
"""
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (cross_validate, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = "data/recipe_site_traffic_2212.csv"
CLEAN_PATH = "data/recipes_clean.csv"
MODEL_PATH = "model/recipe_model.joblib"
METRICS_PATH = "model/metrics.json"

NUM_COLS = ["calories", "carbohydrate", "sugar", "protein"]
NUM_FEATURES = NUM_COLS + ["servings"]
CAT_FEATURES = ["category"]

# ---------------------------------------------------------------- cleaning
df = pd.read_csv(RAW_PATH)
print("Rows in raw file:", len(df))

df = df.set_index("recipe")
df["category"] = df["category"].replace({"Chicken Breast": "Chicken"})
df["servings"] = df["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)

for c in NUM_COLS:
    df[c] = df.groupby("category")[c].transform(lambda x: x.fillna(x.median()))

df["high_traffic"] = (df["high_traffic"] == "High").astype(int)

os.makedirs("data", exist_ok=True)
df.reset_index().to_csv(CLEAN_PATH, index=False)

X = df.drop(columns=["high_traffic"])
y = df["high_traffic"]


# ---------------------------------------------------------------- models
def make_pipeline(clf):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

metrics = {
    "n_recipes": int(len(df)),
    "base_rate": float(y.mean()),
    "sklearn_version": version("scikit-learn"),
    "test": {},
    "cv": {},
}

scoring = ["accuracy", "precision", "recall", "f1"]
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics["test"][name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    cv = cross_validate(pipe, X, y, cv=5, scoring=scoring)
    metrics["cv"][name] = {
        m: {"mean": float(cv[f"test_{m}"].mean()), "std": float(cv[f"test_{m}"].std())}
        for m in scoring
    }
    print(name, "test precision:", round(metrics["test"][name]["precision"], 3),
          "| CV precision:", round(metrics["cv"][name]["precision"]["mean"], 3))

# ------------------------------------------- precision/recall by threshold
# Uses out-of-fold probabilities, so every recipe is scored by a model that
# never saw it. This is the honest way to estimate how the app will behave.
oof = cross_val_predict(models["Logistic Regression"], X, y, cv=5,
                        method="predict_proba")[:, 1]
curve = []
for t in np.round(np.arange(0.30, 0.951, 0.05), 2):
    flagged = oof >= t
    n_flagged = int(flagged.sum())
    tp = int((flagged & (y.values == 1)).sum())
    curve.append({
        "threshold": float(t),
        "precision": float(tp / n_flagged) if n_flagged else None,
        "recall": float(tp / y.sum()),
        "share_flagged": float(n_flagged / len(y)),
    })
metrics["threshold_curve"] = curve

# ------------------------------------------- final model on all the data
final_model = models["Logistic Regression"]
final_model.fit(X, y)

names = list(final_model.named_steps["prep"].get_feature_names_out())
names = [n.replace("num__", "").replace("cat__category_", "Category: ") for n in names]
coefs = final_model.named_steps["clf"].coef_[0]
metrics["coefficients"] = [{"feature": n, "coef": float(c)} for n, c in zip(names, coefs)]

os.makedirs("model", exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved", MODEL_PATH, "and", METRICS_PATH)
print("Categories:", sorted(X["category"].unique()))

# ------------------------------------------------ requirements.txt
pkgs = ["streamlit", "pandas", "scikit-learn", "joblib", "altair", "numpy"]
lines = []
for p in pkgs:
    try:
        lines.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print("Not installed, skipped in requirements.txt:", p)
Path("requirements.txt").write_text("\n".join(lines) + "\n")
print("Wrote requirements.txt")