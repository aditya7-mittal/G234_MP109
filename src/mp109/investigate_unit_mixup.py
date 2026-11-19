import pandas as pd

df = pd.read_csv("data/raw/train.csv")

numeric_columns = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "resale_thousand_rs"
]

for column in numeric_columns:
    print(f"\n{'=' * 60}")
    print(f"COLUMN: {column}")
    print("=" * 60)

    result = df.groupby("city_zone")[column].agg(
        ["min", "mean", "max"]
    )

    print(result.round(2))

    print("\nMaximum / minimum group means:")

    means = result["mean"]
    print(round(means.max() / means.min(), 3))

print("\nUNIT MIXUP INVESTIGATION COMPLETED")