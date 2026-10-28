
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_FOLDER = Path(__file__).parent

VITALS_FILE = DATA_FOLDER / "vitalPeriodic.csv" / "vitalPeriodic.csv"
PATIENT_FILE = DATA_FOLDER / "patient.csv" / "patient.csv"

OUTPUT_FILE = DATA_FOLDER / "prepared_dataset.csv"

# --------------------------------------------------
# 2. Load only the columns we need
# --------------------------------------------------

print("Loading ICU data...")

vitals = pd.read_csv(
    VITALS_FILE,
    usecols=[
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "sao2",
        "respiration",
    ],
)

patients = pd.read_csv(
    PATIENT_FILE,
    usecols=[
        "patientunitstayid",
        "uniquepid",
        "age",
        "gender",
    ],
)

print("Vital-sign records:", len(vitals))
print("Patient stays:", len(patients))

# --------------------------------------------------
# 3. Clean vital-sign values
# --------------------------------------------------

vital_columns = ["heartrate", "sao2", "respiration"]

vitals["observationoffset"] = pd.to_numeric(
    vitals["observationoffset"], errors="coerce"
)

for col in vital_columns:
    vitals[col] = pd.to_numeric(vitals[col], errors="coerce")

# Keep records from the first 12 hours only.
vitals = vitals[
    vitals["observationoffset"].between(0, 720, inclusive="both")
].copy()

# Treat zero and negative values as missing for this
# initial experiment, not as valid physiological readings.
for col in vital_columns:
    vitals.loc[vitals[col] <= 0, col] = np.nan

# --------------------------------------------------
# 4. Define the exploratory abnormality rules
# --------------------------------------------------

def abnormal_reading(column, values):
    if column == "heartrate":
        return (values <= 40) | (values >= 131)

    if column == "sao2":
        return values <= 91

    if column == "respiration":
        return (values <= 8) | (values > 25)

    return pd.Series(False, index=values.index)


def has_sustained_abnormality(group, column):
    """Three or more abnormal readings within 30 minutes."""
    values = group[column]
    abnormal = abnormal_reading(column, values)

    offsets = (
        group.loc[abnormal, "observationoffset"]
        .dropna()
        .sort_values()
        .to_numpy()
    )

    if len(offsets) < 3:
        return False

    for i in range(len(offsets) - 2):
        if offsets[i + 2] - offsets[i] <= 30:
            return True

    return False


# --------------------------------------------------
# 5. Separate the early and future windows
# --------------------------------------------------

early = vitals[
    vitals["observationoffset"] < 360
].copy()

future = vitals[
    vitals["observationoffset"] >= 360
].copy()

print("\nChecking early-window readings...")
early_events = {}

for stay_id, group in early.groupby("patientunitstayid"):
    early_events[stay_id] = any(
        has_sustained_abnormality(group, col)
        for col in vital_columns
    )

print("Checking future-window readings...")
future_events = {}

for stay_id, group in future.groupby("patientunitstayid"):
    future_events[stay_id] = any(
        has_sustained_abnormality(group, col)
        for col in vital_columns
    )

# --------------------------------------------------
# 6. Create features using ONLY the first six hours
# --------------------------------------------------

print("\nCreating early-window features...")

feature_rows = []

for stay_id, group in early.groupby("patientunitstayid"):
    row = {"patientunitstayid": stay_id}

    for col in vital_columns:
        series = group[col].dropna()

        row[f"{col}_mean"] = series.mean()
        row[f"{col}_min"] = series.min()
        row[f"{col}_max"] = series.max()
        row[f"{col}_std"] = series.std()
        row[f"{col}_count"] = series.count()

    feature_rows.append(row)

features = pd.DataFrame(feature_rows)

# --------------------------------------------------
# 7. Assign labels using hours 6–12
# --------------------------------------------------

# Exclude stays with sustained abnormalities in the
# first six hours for this particular experiment.
features["early_abnormal"] = (
    features["patientunitstayid"]
    .map(early_events)
    .fillna(False)
)

features["future_abnormal"] = (
    features["patientunitstayid"]
    .map(future_events)
    .fillna(False)
)

# Count valid readings in the future window.
future_counts = (
    future.groupby("patientunitstayid")[vital_columns]
    .count()
)

for col in vital_columns:
    future_counts = future_counts.rename(
        columns={col: f"future_{col}_count"}
    )

features = features.merge(
    future_counts,
    left_on="patientunitstayid",
    right_index=True,
    how="left",
)

future_count_columns = [
    f"future_{col}_count" for col in vital_columns
]

features[future_count_columns] = (
    features[future_count_columns].fillna(0)
)

# A negative label requires at least six valid readings
# of EACH vital during hours 6–12. This is a provisional
# data-coverage filter, not a clinical standard.
adequate_future_coverage = (
    features[future_count_columns] >= 6
).all(axis=1)

eligible = features[
    (~features["early_abnormal"])
    & (
        features["future_abnormal"]
        | adequate_future_coverage
    )
].copy()

eligible["label"] = eligible["future_abnormal"].astype(int)

# --------------------------------------------------
# 8. Add basic patient information
# --------------------------------------------------

eligible = eligible.merge(
    patients.drop_duplicates("patientunitstayid"),
    on="patientunitstayid",
    how="left",
)

# --------------------------------------------------
# 9. Save the prepared dataset
# --------------------------------------------------

# Keep future-window counts for auditing, but these
# MUST NOT be used as model input features.
eligible.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 10. Report results
# --------------------------------------------------

print("\n========== DATASET REPORT ==========")
print("Stays with early features:", len(features))
print("Excluded: early sustained abnormality:",
      int(features["early_abnormal"].sum()))
print("Excluded: insufficient future coverage and no event:",
      int((
          (~features["early_abnormal"])
          & (~features["future_abnormal"])
          & (~adequate_future_coverage)
      ).sum()))
print("Eligible stays:", len(eligible))
print("Positive labels (future abnormality):",
      int((eligible["label"] == 1).sum()))
print("Negative labels (no detected event with coverage):",
      int((eligible["label"] == 0).sum()))
print("Positive proportion:",
      round(eligible["label"].mean() * 100, 2)
      if len(eligible) else "N/A", "%")

print("\nSaved to:", OUTPUT_FILE)
print("\nImportant: this is an exploratory label, not a")
print("clinically validated diagnosis or deterioration label.")


# --------------------------------------------------
# 11. Check dataset quality before model training
# --------------------------------------------------

print("\n========== QUALITY CHECK ==========")

print("\nDataset shape (rows, columns):")
print(eligible.shape)

print("\nMissing values in model feature columns:")
feature_columns = [
    "heartrate_mean", "heartrate_min", "heartrate_max",
    "heartrate_std", "heartrate_count",
    "sao2_mean", "sao2_min", "sao2_max",
    "sao2_std", "sao2_count",
    "respiration_mean", "respiration_min",
    "respiration_max", "respiration_std",
    "respiration_count",
]

print(eligible[feature_columns].isna().sum())

print("\nUnique ICU stays:")
print(eligible["patientunitstayid"].nunique())

print("\nUnique patients:")
print(eligible["uniquepid"].nunique())

print("\nLabel counts:")
print(eligible["label"].value_counts().sort_index())
