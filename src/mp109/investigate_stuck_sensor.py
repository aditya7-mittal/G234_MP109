import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

columns = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone"
]

# Summarize each column by date.
daily = (
    df.sort_values("date")
    .groupby("date")[columns]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

for col in columns:
    print(f"\n{'=' * 50}")
    print("COLUMN:", col)

    daily["new_run"] = (
        daily[col].ne(daily[col].shift())
        | daily["date"].diff().dt.days.ne(1)
    )

    daily["run_id"] = daily["new_run"].cumsum()

    runs = daily.groupby("run_id").agg(
        value=(col, "first"),
        days=(col, "size"),
        first_date=("date", "min"),
        last_date=("date", "max")
    )

    suspicious = runs[runs["days"] >= 5].sort_values(
        "days", ascending=False
    )

    print(suspicious.to_string(index=False))

print("\nCALENDAR-DAY INVESTIGATION COMPLETED")