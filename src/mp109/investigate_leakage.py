import pandas as pd

train = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

target = "resale_thousand_rs"

print("TRAIN COLUMNS:")
print(train.columns.tolist())

print("\nHOLDOUT COLUMNS:")
print(holdout.columns.tolist())

print("\nCOLUMNS MISSING FROM HOLDOUT:")
missing = set(train.columns) - set(holdout.columns)
print(missing)

print("\nCORRELATION WITH TARGET:")
correlations = train.corr(numeric_only=True)[target].sort_values(
    ascending=False
)
print(correlations)

print("\nINVESTIGATION COMPLETED")