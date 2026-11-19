import pandas as pd

df = pd.read_csv("data/raw/train.csv")

df["mileage_per_year"] = (
    df["km_driven"] / df["age_years"].replace(0, pd.NA)
)

suspect = df[df["age_years"] > 25]

print("Overall mileage per year:")
print(df["mileage_per_year"].describe())

print("\nMileage per year for suspicious ages:")
print(suspect["mileage_per_year"].describe())

print("\nMileage per year for normal ages:")
print(df[df["age_years"] <= 25]["mileage_per_year"].describe())

print("\nSample suspicious records:")
print(
    suspect[
        ["record_id", "age_years", "km_driven", "mileage_per_year"]
    ].head(25).to_string(index=False)
)

print("\nInvestigation completed.")