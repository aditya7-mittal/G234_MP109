import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("DATE RANGE")
print("Minimum:", df["date"].min())
print("Maximum:", df["date"].max())

print("\nINVALID DATES:", df["date"].isna().sum())

print("\nRECORDS PER MONTH")
print(df["date"].dt.to_period("M").value_counts().sort_index())

print("\nMOST COMMON DATES")
print(df["date"].value_counts().head(15))

print("\nFUTURE DATES")
print((df["date"] > pd.Timestamp.today()).sum())

print("\nINVESTIGATION COMPLETED")