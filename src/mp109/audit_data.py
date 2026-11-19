
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

print("=" * 60)
print("1. DUPLICATE RECORD INVESTIGATION")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate record IDs:", df["record_id"].duplicated().sum())

duplicate_ids = df[df.duplicated("record_id", keep=False)]

print("\nRecords with repeated IDs:")
print(duplicate_ids.sort_values("record_id").to_string(index=False))

print("\n" + "=" * 60)
print("2. CATEGORICAL VALUE INVESTIGATION")
print("=" * 60)

print("\nBrand tier categories:")
for value in sorted(df["brand_tier"].unique()):
    print(repr(value))

print("\nCity zone categories:")
for value in sorted(df["city_zone"].unique()):
    print(repr(value))

print("\n" + "=" * 60)
print("3. AGE INVESTIGATION")
print("=" * 60)

print("Age below 0:", (df["age_years"] < 0).sum())
print("Age above 25:", (df["age_years"] > 25).sum())
print("Age above 100:", (df["age_years"] > 100).sum())

print(df.loc[df["age_years"] > 25,
             ["record_id", "age_years"]].to_string(index=False))

print("\n" + "=" * 60)
print("4. SERVICE RECORD INVESTIGATION")
print("=" * 60)

print("Negative service records:",
      (df["service_records"] < 0).sum())

print(df.loc[df["service_records"] < 0,
             ["record_id", "service_records"]].to_string(index=False))

print("\n" + "=" * 60)
print("5. NUMERICAL RANGE INVESTIGATION")
print("=" * 60)

checks = {
    "owners outside 1-4": ~df["owners"].between(1, 4),
    "accidents outside 0-3": ~df["accidents"].between(0, 3),
    "engine_cc outside 100-500": ~df["engine_cc"].between(100, 500),
    "negative km_driven": df["km_driven"] < 0,
    "non-positive resale price": df["resale_thousand_rs"] <= 0
}

for name, condition in checks.items():
    print(f"{name}: {condition.sum()}")

print("\n" + "=" * 60)
print("6. DATE INVESTIGATION")
print("=" * 60)

dates = pd.to_datetime(df["date"], errors="coerce")

print("Invalid dates:", dates.isna().sum())
print("Earliest date:", dates.min())
print("Latest date:", dates.max())

print("\n" + "=" * 60)
print("7. TARGET VARIABLE INVESTIGATION")
print("=" * 60)

print("Target missing:", df["resale_thousand_rs"].isna().sum())
print("Target minimum:", df["resale_thousand_rs"].min())
print("Target maximum:", df["resale_thousand_rs"].max())
print("Target mean:", df["resale_thousand_rs"].mean())

print("\nAUDIT COMPLETED")
