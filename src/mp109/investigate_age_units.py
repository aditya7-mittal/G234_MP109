import pandas as pd

df = pd.read_csv("data/raw/train.csv")

suspect = df[df["age_years"] > 25].copy()
suspect["converted_age"] = suspect["age_years"] / 12

# Check whether suspicious values are consistent with month-based recording
remainder = suspect["age_years"] % 1.2
consistent = remainder.abs() < 0.000001

print("Suspicious records:", len(suspect))
print("Values consistent with 1.2-month increments:", consistent.sum())
print("Values not consistent:", (~consistent).sum())

print("\nConverted age range:")
print("Minimum:", suspect["converted_age"].min())
print("Maximum:", suspect["converted_age"].max())

print("\nConverted age distribution:")
print(suspect["converted_age"].describe())

print("\nExamples not matching the pattern:")
print(
    suspect.loc[~consistent, ["record_id", "age_years"]]
    .head(20)
    .to_string(index=False)
)