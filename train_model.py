
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. LOAD THE PREPARED DATASET
# --------------------------------------------------

DATA_FOLDER = Path(__file__).parent
df = pd.read_csv(DATA_FOLDER / "prepared_dataset.csv")

df = df.dropna(subset=["label", "uniquepid"]).reset_index(drop=True)

feature_columns = [
    "heartrate_mean",
    "heartrate_min",
    "heartrate_max",
    "heartrate_std",
    "heartrate_count",
    "sao2_mean",
    "sao2_min",
    "sao2_max",
    "sao2_std",
    "sao2_count",
    "respiration_mean",
    "respiration_min",
    "respiration_max",
    "respiration_std",
    "respiration_count",
]

X = df[feature_columns]
y = df["label"].astype(int)
groups = df["uniquepid"]

print("=" * 55)
print("GROUPED 5-FOLD CROSS-VALIDATION")
print("=" * 55)
print("Total ICU stays:", len(df))
print("Unique patients:", groups.nunique())
print("Positive cases:", int(y.sum()))
print("Negative cases:", int((y == 0).sum()))
print("Positive case proportion:", round(y.mean(), 3))

# --------------------------------------------------
# 2. DEFINE TWO MODELS
# --------------------------------------------------

models = {
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
        ),
    ),
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),
}

# --------------------------------------------------
# 3. CREATE FIVE PATIENT-GROUPED FOLDS
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Each stay receives a prediction when it is held out.
oof_scores = {
    name: np.full(len(df), np.nan)
    for name in models
}

fold_results = []

# --------------------------------------------------
# 4. TRAIN AND TEST EACH FOLD
# --------------------------------------------------

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

    print(f"\n--- Fold {fold} ---")
    print("Training stays:", len(train_idx))
    print("Testing stays:", len(test_idx))
    print("Positive test cases:", int(y_test.sum()))
    print("Patients shared across sets:", len(overlap))

    if overlap:
        raise RuntimeError(
            "Patient leakage detected: a patient appears "
            "in both training and testing data."
        )

    if y_test.nunique() < 2:
        raise RuntimeError(
            f"Fold {fold} contains only one test class. "
            "Review the fold setup before interpreting results."
        )

    for name, model in models.items():
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_test)[:, 1]
        oof_scores[name][test_idx] = scores

        predictions = (scores >= 0.5).astype(int)

        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        fold_results.append({
            "Fold": fold,
            "Model": name,
            "ROC-AUC": auc,
            "Average precision": ap,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        print(
            f"{name}: "
            f"ROC-AUC={auc:.3f}, "
            f"AP={ap:.3f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}"
        )

# --------------------------------------------------
# 5. SUMMARIZE RESULTS ACROSS THE FIVE FOLDS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

metric_columns = [
    "ROC-AUC",
    "Average precision",
    "Precision",
    "Recall",
    "F1",
]

summary = fold_df.groupby("Model")[metric_columns].agg(
    ["mean", "std"]
)

print("\n" + "=" * 65)
print("FOLD RESULTS: MEAN AND STANDARD DEVIATION")
print("=" * 65)
print(summary.round(3).to_string())

# --------------------------------------------------
# 6. OVERALL OUT-OF-FOLD RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("OVERALL OUT-OF-FOLD RESULTS")
print("=" * 65)

for name, scores in oof_scores.items():
    if np.isnan(scores).any():
        raise RuntimeError(
            f"Some stays did not receive predictions for {name}."
        )

    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("ROC-AUC:", round(roc_auc_score(y, scores), 3))
    print(
        "Average precision:",
        round(average_precision_score(y, scores), 3),
    )
    print(
        "Precision at threshold 0.5:",
        round(precision_score(y, predictions, zero_division=0), 3),
    )
    print(
        "Recall at threshold 0.5:",
        round(recall_score(y, predictions, zero_division=0), 3),
    )
    print(
        "F1 at threshold 0.5:",
        round(f1_score(y, predictions, zero_division=0), 3),
    )
    print("True positives:", int(tp))
    print("Missed positive cases:", int(fn))
    print("False alarms:", int(fp))
    print("True negatives:", int(tn))

# --------------------------------------------------
# 7. EXPLORE RANDOM FOREST THRESHOLDS
# --------------------------------------------------

rf_scores = oof_scores["Random Forest"]

print("\n" + "=" * 65)
print("RANDOM FOREST: EXPLORATORY THRESHOLD COMPARISON")
print("=" * 65)

for threshold in [0.3, 0.4, 0.5]:
    predictions = (rf_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"\nThreshold: {threshold:.1f} | "
        f"Precision: "
        f"{precision_score(y, predictions, zero_division=0):.3f} | "
        f"Recall: "
        f"{recall_score(y, predictions, zero_division=0):.3f} | "
        f"F1: {f1_score(y, predictions, zero_division=0):.3f}"
    )
    print(
        f"True positives: {tp} | Missed positives: {fn} | "
        f"False alarms: {fp} | True negatives: {tn}"
    )

print("\nCross-validation experiment complete.")
print(
    "These results are exploratory. The prediction labels "
    "and thresholds are not clinically validated."
)
