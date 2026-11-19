import pandas as pd

df = pd.read_csv("data/raw/train.csv")

target = "resale_thousand_rs"

# Sort target values
sorted_prices = df[target].sort_values().reset_index(drop=True)

print("TARGET DISTRIBUTION")
print(df[target].describe())

print("\nHIGHEST 30 TARGET VALUES")
print(
    df.nlargest(30, target)[
        ["record_id", target, "engine_cc", "age_years", "brand_tier"]
    ].to_string(index=False)
)

# Check unusually large gaps between consecutive target values
gaps = sorted_prices.diff().sort_values(ascending=False)

print("\nLARGEST 15 GAPS IN TARGET VALUES")
for index in gaps.head(15).index:
    print(
        f"Gap: {gaps[index]:.2f} | "
        f"Lower: {sorted_prices[index - 1]:.2f} | "
        f"Upper: {sorted_prices[index]:.2f}"
    )

# Check whether unusually high values are approximately 10x
# another value in the dataset.
print("\nPOSSIBLE 10X TARGET VALUES")

for value in df[target].nlargest(30):
    divided = value / 10
    nearest = (df[target] - divided).abs().min()

    print(
        f"Value: {value:.2f} | "
        f"Value / 10: {divided:.2f} | "
        f"Nearest target difference: {nearest:.2f}"
    )

print("\nTARGET TYPO INVESTIGATION COMPLETED")