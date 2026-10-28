
from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# 1. Load the prepared dataset
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

# --------------------------------------------------
# 2. Create patient-grouped folds
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, test_idx = next(cv.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training stays:", len(X_train))
print("Held-out test stays:", len(X_test))
print("Training positive cases:", int(y_train.sum()))
print("Held-out positive cases:", int(y_test.sum()))

# --------------------------------------------------
# 3. Train Logistic Regression on the training fold
# --------------------------------------------------

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

# --------------------------------------------------
# 4. Explain the model using SHAP
# --------------------------------------------------

# Use the fitted preprocessing steps so SHAP receives
# the same scaled features that Logistic Regression sees.
imputer = model.named_steps["simpleimputer"]
scaler = model.named_steps["standardscaler"]
classifier = model.named_steps["logisticregression"]

X_train_scaled = scaler.transform(imputer.transform(X_train))
X_test_scaled = scaler.transform(imputer.transform(X_test))

background = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns,
)

test_sample = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns,
)

explainer = shap.LinearExplainer(
    classifier,
    background,
)

shap_values = explainer(test_sample)

# --------------------------------------------------
# 5. Save the global feature-importance plot
# --------------------------------------------------

shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(DATA_FOLDER / "shap_feature_importance.png", dpi=200)
plt.close()

print("\nSaved: shap_feature_importance.png")

# --------------------------------------------------
# 6. Save a detailed explanation for one test patient
# --------------------------------------------------

# Explain the first held-out test stay.
single_explanation = shap_values[0]

shap.plots.waterfall(single_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(DATA_FOLDER / "shap_patient_explanation.png", dpi=200)
plt.close()

print("Explained ICU stay ID:", int(df.iloc[test_idx[0]]["patientunitstayid"]))
print("Actual experimental label:", int(y_test.iloc[0]))
print(
    "Predicted probability:",
    round(float(model.predict_proba(X_test.iloc[[0]])[0, 1]), 4),
)
print("Saved: shap_patient_explanation.png")
print("\nSHAP explanations complete.")
print("These explain model behaviour, not clinical causation.")
