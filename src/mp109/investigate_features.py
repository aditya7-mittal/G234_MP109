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

print("UNIQUE VALUE COUNTS")
for col in numeric_columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(10))

print("\nCORRELATION MATRIX")
print(df[numeric_columns].corr().round(2))

print("\nZERO VALUE COUNTS")
for col in numeric_columns:
    print(f"{col}: {(df[col] == 0).sum()}")

print("\nINVESTIGATION COMPLETED")