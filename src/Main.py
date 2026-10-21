
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

SOURCE = Path("DataSet/newcastle_sausage_roll_pricing_challenge_20261020(data)(in).csv")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

PROMOTIONS = ["20_percent_off", "two_for_2", "one_pound_week"]

# ==================================================
# 2. IMPORT AND VALIDATION
# ==================================================

df = pd.read_csv(SOURCE, encoding="cp1252")

if df.empty:
    raise ValueError("Source dataset is empty.")

df.columns = df.columns.str.replace("ï»¿", "", regex=False).str.strip()
df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")

for col in ["sausage_roll_takings_gbp", "total_shop_takings_gbp"]:
    df[col] = pd.to_numeric(
        df[col].astype("string")
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("Â", "", regex=False),
        errors="coerce"
    ).astype(float)

print(f"Imported {len(df)} records across {df['store_id'].nunique()} stores.")

# ==================================================
# 3. DATA CLEANING
# ==================================================

clean = df.drop_duplicates().copy()

# Flag invalid observations rather than silently deleting them.
bad_sales = (
    (clean["sausage_rolls_sold"] < 0) |
    (clean["sausage_rolls_sold"] > 1000)
)

bad_transactions = (
    (clean["sausage_roll_transactions"] < 0) |
    (clean["total_shop_transactions"] < 0) |
    (clean["sausage_roll_transactions"] > clean["total_shop_transactions"]) |
    (
        (clean["sausage_rolls_sold"] == 0) &
        (clean["sausage_roll_transactions"] > 0)
    )
)

bad_revenue = (
    (clean["sausage_roll_takings_gbp"] < 0) |
    (clean["sausage_roll_takings_gbp"] > clean["total_shop_takings_gbp"]) |
    (clean["sausage_roll_takings_gbp"] > 10000)
)

clean["invalid_sales"] = bad_sales | bad_transactions
clean["invalid_revenue"] = bad_revenue | bad_sales | bad_transactions

clean.loc[clean["invalid_sales"], "sausage_rolls_sold"] = np.nan
clean.loc[clean["invalid_revenue"], "sausage_roll_takings_gbp"] = np.nan

# Handle impossible temperature readings.
bad_temperature = ~clean["temperature_c"].between(-20, 45)
clean.loc[bad_temperature, "temperature_c"] = np.nan
clean["temperature_c"] = clean["temperature_c"].fillna(
    clean.groupby("date")["temperature_c"].transform("median")
).fillna(clean["temperature_c"].median())

print("\n=== DATA QUALITY ===")
print(f"Duplicates removed: {len(df) - len(clean)}")
print(f"Invalid sales records: {bad_sales.sum() + bad_transactions.sum() - (bad_sales & bad_transactions).sum()}")
print(f"Invalid revenue records: {clean['invalid_revenue'].sum()}")
print(f"Invalid temperatures: {bad_temperature.sum()}")
print("\nRemaining missing values:")
print(clean.isna().sum()[clean.isna().sum() > 0])

clean.to_csv(OUTPUT / "cleaned_data.csv", index=False)

# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

clean = clean.sort_values(["store_id", "date"])

clean["day_name"] = clean["date"].dt.day_name()
clean["trend"] = (clean["date"] - clean["date"].min()).dt.days

# Smooth annual seasonality.
day = clean["date"].dt.dayofyear
clean["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
clean["annual_cos"] = np.cos(2 * np.pi * day / 365.25)

# Flag seven days following each promotional run.
# Consecutive promotion dates form one campaign period.
for promo in PROMOTIONS:
    column = "post_" + promo
    clean[column] = 0

    dates = sorted(clean.loc[clean["promotion"] == promo, "date"].unique())

    if not dates:
        continue

    dates = pd.DatetimeIndex(dates)
    ends = dates[
        (dates.to_series().shift(-1) - dates.to_series()).dt.days.ne(1).to_numpy()
    ]

    for end in ends:
        mask = (
            (clean["date"] > end) &
            (clean["date"] <= end + pd.Timedelta(days=7)) &
            (clean["promotion"] == "none")
        )
        clean.loc[mask, column] = 1

# ==================================================
# 5. EXPLORATORY ANALYSIS
# ==================================================

print("\n=== PROMOTION OVERVIEW ===")

summary = clean.groupby("promotion").agg(
    records=("date", "size"),
    average_units=("sausage_rolls_sold", "mean"),
    average_revenue=("sausage_roll_takings_gbp", "mean")
)

print(summary.round(2).to_string())

daily = clean.groupby("date")["sausage_rolls_sold"].sum(min_count=1)

plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily.values, linewidth=0.8)
plt.title("Daily sausage roll sales — all stores")
plt.xlabel("Date")
plt.ylabel("Units sold")
plt.tight_layout()
plt.savefig(OUTPUT / "sales_timeline.png", dpi=200)
plt.close()

# ==================================================
# 6. BASELINE REGRESSION
# ==================================================

# Reference category is explicitly 'none'.
# This makes promotion coefficients relative to no promotion.

formula = (
    "TARGET ~ C(promotion, Treatment(reference='none'))"
    " + C(other_promotion, Treatment(reference='none'))"
    " + C(store_id) + C(day_name)"
    " + trend + annual_sin + annual_cos"
    " + temperature_c + rain_mm"
    " + home_match_day + school_holidays"
    " + post_20_percent_off + post_two_for_2"
    " + post_one_pound_week"
)

def fit_model(target):
    data = clean.dropna(subset=[target]).copy()
    model = smf.ols(
        formula.replace("TARGET", target),
        data=data
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["date"]}
    )
    return model, data

sales_model, sales_data = fit_model("sausage_rolls_sold")
revenue_model, revenue_data = fit_model("sausage_roll_takings_gbp")

with open(OUTPUT / "model_summary.txt", "w") as file:
    file.write("SALES MODEL\n")
    file.write(sales_model.summary().as_text())
    file.write("\n\nREVENUE MODEL\n")
    file.write(revenue_model.summary().as_text())

# ==================================================
# 7. PROMOTION EFFECTIVENESS
# ==================================================

results = []

for promo in PROMOTIONS:
    term = f"C(promotion, Treatment(reference='none'))[T.{promo}]"
    post_term = "post_" + promo

    sales_effect = sales_model.params[term]
    revenue_effect = revenue_model.params[term]

    sales_ci = sales_model.conf_int().loc[term]
    revenue_ci = revenue_model.conf_int().loc[term]

    n_sales = (sales_data["promotion"] == promo).sum()
    n_revenue = (revenue_data["promotion"] == promo).sum()

    # Counterfactual baseline for promotional observations.
    subset = sales_data[sales_data["promotion"] == promo].copy()
    baseline = subset.copy()
    baseline["promotion"] = "none"

    predicted_baseline = sales_model.predict(baseline).sum()
    incremental_units = sales_effect * n_sales

    results.append({
        "promotion": promo,
        "store_days": n_sales,
        "baseline_units": predicted_baseline,
        "incremental_units": incremental_units,
        "uplift_pct": (
            incremental_units / predicted_baseline * 100
            if predicted_baseline > 0 else np.nan
        ),
        "sales_effect_per_store_day": sales_effect,
        "sales_ci_low": sales_ci.iloc[0],
        "sales_ci_high": sales_ci.iloc[1],
        "sales_p_value": sales_model.pvalues[term],
        "incremental_revenue_gbp": revenue_effect * n_revenue,
        "revenue_ci_low": revenue_ci.iloc[0],
        "revenue_ci_high": revenue_ci.iloc[1],
        "post_effect_per_store_day": sales_model.params[post_term],
        "post_p_value": sales_model.pvalues[post_term]
    })

results = pd.DataFrame(results)
results = results.sort_values("incremental_units", ascending=False)

results.to_csv(OUTPUT / "promotion_results.csv", index=False)

print("\n=== PROMOTION RESULTS ===")
print(results.round(2).to_string(index=False))

# ==================================================
# 8. VISUALISATION
# ==================================================

fig, ax = plt.subplots(figsize=(9, 5))

labels = results["promotion"].str.replace("_", " ")
values = results["sales_effect_per_store_day"]

errors = np.array([
    values - results["sales_ci_low"],
    results["sales_ci_high"] - values
])

ax.bar(labels, values, color=["#2563eb", "#0d9488", "#d97706"])
ax.errorbar(
    labels, values, yerr=errors,
    fmt="none", ecolor="black", capsize=5
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Estimated promotion effect on daily sales")
ax.set_ylabel("Additional rolls per store-day")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT / "promotion_comparison.png", dpi=200)
plt.close()

print("\n=== ANALYSIS COMPLETE ===")
print(f"Results saved to: {OUTPUT.resolve()}")
