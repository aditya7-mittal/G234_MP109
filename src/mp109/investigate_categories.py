import pandas as pd

df = pd.read_csv("data/raw/train.csv")

categorical_columns = ["brand_tier", "city_zone"]

for col in categorical_columns:
    print(f"\n--- {col} ---")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts(dropna=False).to_string())

print("\nWHITESPACE CHECK")
for col in categorical_columns:
    values = df[col].dropna().astype(str)
    leading = values.str.startswith(" ").sum()
    trailing = values.str.endswith(" ").sum()
    print(f"{col}: leading spaces={leading}, trailing spaces={trailing}")

print("\nMISSING CATEGORICAL VALUES")
print(df[categorical_columns].isna().sum())

print("\nINVESTIGATION COMPLETED")