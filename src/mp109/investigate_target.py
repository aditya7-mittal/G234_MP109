
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

target = "resale_thousand_rs"

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)

outliers = df[
    (df[target] < lower) |
    (df[target] > upper)
]

print("\nPotential target outliers:", len(outliers))
print(outliers[
    ["record_id", "engine_cc", "age_years",
     "km_driven", "brand_tier", target]
].to_string(index=False))
