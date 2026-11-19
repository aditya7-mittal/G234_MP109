import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")

print("AGE UNIT MIXUP VERIFICATION")

for zone in ["Central", "North", "South"]:
    ages = train.loc[train["city_zone"] == zone, "age_years"]

    if zone == "Central":
        ages = ages / 12

    print(
        f"{zone}: "
        f"min={ages.min():.2f}, "
        f"mean={ages.mean():.2f}, "
        f"max={ages.max():.2f}"
    )

print("\nBRAND TIER VERIFICATION")

print("Training unique categories:")
print(sorted(train["brand_tier"].unique()))

print("\nClean categories in later.csv:")
print(sorted(later["brand_tier"].unique()))

print("\nSENTINEL VERIFICATION")
print("Number of -999 values:",
      (train["service_records"] == -999).sum())

print("\nDUPLICATE VERIFICATION")
print("Extra duplicate rows:", train.duplicated().sum())

print("\nVERIFICATION COMPLETED")