
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

# Check values that may have been recorded in months
suspect = df[df["age_years"] > 25].copy()

suspect["age_if_months"] = suspect["age_years"] / 12

print("Total suspicious age records:", len(suspect))

print("\nFirst 30 suspicious records:")
print(suspect[
    ["record_id", "age_years", "age_if_months"]
].head(30).to_string(index=False))

print("\nAge distribution:")
print(df["age_years"].describe(percentiles=[
    .1, .25, .5, .75, .9, .95, .99
]))

print("\nCount of ages above selected thresholds:")
for threshold in [20, 25, 50, 100, 150]:
    print(
        f"Above {threshold} years:",
        (df["age_years"] > threshold).sum()
    )

print("\nInvestigation completed.")
