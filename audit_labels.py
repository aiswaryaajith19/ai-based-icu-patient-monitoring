
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path(__file__).parent
df = pd.read_csv(DATA_FOLDER / "prepared_dataset.csv")

future_columns = [
    "future_heartrate_count",
    "future_sao2_count",
    "future_respiration_count",
]

print("=" * 60)
print("LABEL AND FUTURE-DATA COVERAGE AUDIT")
print("=" * 60)

print("\n1. LABEL COUNTS")
print(df["label"].value_counts().sort_index().to_string())

print("\n2. FUTURE READING COUNTS BY LABEL")
for column in future_columns:
    print(f"\n{column}")
    print(
        df.groupby("label")[column]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )

print("\n3. CASES WITH FEWER THAN 6 FUTURE READINGS")
for column in future_columns:
    print(f"\n{column}")
    print(
        pd.crosstab(
            df["label"],
            df[column] < 6,
            rownames=["Label"],
            colnames=["Fewer than 6 readings"],
        ).to_string()
    )

print("\n4. EARLY ABNORMALITY FLAG")
print(
    pd.crosstab(
        df["label"],
        df["early_abnormal"],
        rownames=["Future label"],
        colnames=["Early abnormal flag"],
    ).to_string()
)

print("\n5. MISSING EARLY FEATURES")
feature_columns = [
    "heartrate_mean", "heartrate_min", "heartrate_max",
    "heartrate_std", "heartrate_count",
    "sao2_mean", "sao2_min", "sao2_max",
    "sao2_std", "sao2_count",
    "respiration_mean", "respiration_min",
    "respiration_max", "respiration_std",
    "respiration_count",
]
print(df[feature_columns].isna().sum().to_string())

print("\n6. REPEATED PATIENTS")
print("Total ICU stays:", len(df))
print("Unique patients:", df["uniquepid"].nunique())
print("Patients with multiple stays:", int(
    (df["uniquepid"].value_counts() > 1).sum()
))

print("\nAudit complete. No files were changed.")
