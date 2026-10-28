
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_FOLDER = Path(__file__).parent

df = pd.read_csv(DATA_FOLDER / "prepared_dataset.csv")
df = df.dropna(subset=["label", "uniquepid"]).reset_index(drop=True)

# Nine physiological summary features.
# Deliberately exclude the three reading-count features.
feature_columns = [
    "heartrate_mean",
    "heartrate_min",
    "heartrate_max",
    "heartrate_std",
    "sao2_mean",
    "sao2_min",
    "sao2_max",
    "sao2_std",
    "respiration_mean",
    "respiration_min",
    "respiration_max",
    "respiration_std",
]

X = df[feature_columns]
y = df["label"].astype(int)
groups = df["uniquepid"]

print("=" * 60)
print("COMPARISON: PHYSIOLOGICAL FEATURES WITHOUT READING COUNTS")
print("=" * 60)
print("ICU stays:", len(df))
print("Unique patients:", groups.nunique())
print("Features used:", len(feature_columns))
print("Reading-count features excluded: 3")

# --------------------------------------------------
# 2. GROUPED CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])

    overlap = train_patients.intersection(test_patients)

    if overlap:
        raise RuntimeError(
            f"Patient overlap detected in fold {fold}."
        )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    results = {
        "Fold": fold,
        "ROC-AUC": roc_auc_score(y_test, scores),
        "Average precision": average_precision_score(
            y_test, scores
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    fold_results.append(results)

    print(
        f"\nFold {fold}: "
        f"ROC-AUC={results['ROC-AUC']:.3f}, "
        f"AP={results['Average precision']:.3f}, "
        f"Precision={results['Precision']:.3f}, "
        f"Recall={results['Recall']:.3f}, "
        f"F1={results['F1']:.3f}"
    )
    print("Patients shared across sets:", len(overlap))

# --------------------------------------------------
# 3. SUMMARIZE THE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(fold_results)

metrics = [
    "ROC-AUC",
    "Average precision",
    "Precision",
    "Recall",
    "F1",
]

print("\n" + "=" * 60)
print("MEAN AND STANDARD DEVIATION ACROSS FIVE FOLDS")
print("=" * 60)

summary = results_df[metrics].agg(["mean", "std"]).round(3)
print(summary.to_string())

print("\nComparison complete.")
print("This is an exploratory experiment, not clinical validation.")
