import pandas as pd

# Import the source data
df = pd.read_csv(
    "DataSet/newcastle_sausage_roll_pricing_challenge_20261020(data)(in).csv",
    encoding="cp1252"
)

if df.empty:
    raise ValueError("Error: Source dataset is empty.")

print("Dataset imported successfully.")
print(df.head())