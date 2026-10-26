import pandas as pd
from src.shipment_prep import clean_shipments, build_features
from src.shipment_prep import FEATURES, clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
dedup = raw.drop_duplicates().copy()
missing_distance_km_before = int(dedup["distance_km"].isna().sum())
missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
missing_declared_value_before = int(dedup["declared_value"].isna().sum())
missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())
print("missing distance:", missing_distance_km_before)
print("missing weight:", missing_parcel_weight_before)
print("sentinel weights:", sentinel_weight_rows)
print("missing declared value:", missing_declared_value_before)
print("missing payment mode:", missing_payment_mode_before)
raw_rows = len(raw)
duplicate_rows_removed = len(raw) - len(dedup)
rows_after_dedup = len(dedup)
print(raw_rows, duplicate_rows_removed, rows_after_dedup)
hubs = pd.read_csv("data/hubs.csv")

print("raw rows:", len(raw))
print("hub rows:", len(hubs))


clean, medians = clean_shipments(raw)
service_type_spellings_before = int(dedup["service_type"].nunique())
service_type_spellings_after = int(clean["service_type"].nunique())
print("service spellings before:", service_type_spellings_before)
print("service spellings after:", service_type_spellings_after)
featured = build_features(clean, hubs)

featured["is_delayed"] = (featured["delivery_hours"] > featured["promised_hours"]).astype(int)
correlations = featured.select_dtypes(include="number").corr()
correlations.round(4).to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    index_label="column",
)

nulls_in_features_and_target = int(
    featured[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
)
print("nulls in features and target:", nulls_in_features_and_target)

declared_value_mean_inr = featured["declared_value_inr"].mean()
declared_value_median_inr = featured["declared_value_inr"].median()
log_declared_value_mean = featured["log_declared_value"].mean()
log_declared_value_median = featured["log_declared_value"].median()

print("declared value mean/median:", declared_value_mean_inr, declared_value_median_inr)
print("log value mean/median:", log_declared_value_mean, log_declared_value_median)

delayed_count = int(featured["is_delayed"].sum())
delay_rate = float(featured["is_delayed"].mean())

hours_q1 = featured["delivery_hours"].quantile(0.25)
hours_q3 = featured["delivery_hours"].quantile(0.75)
hours_iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr
hours_outlier_count = int(
    ((featured["delivery_hours"] < hours_lower_bound) |
     (featured["delivery_hours"] > hours_upper_bound)).sum()
)
print("hours quartiles:", hours_q1, hours_q3)
print("outlier bounds and count:", hours_lower_bound, hours_upper_bound, hours_outlier_count)
print("delayed count and rate:", delayed_count, delay_rate)
featured.to_csv("data/modelling_table.csv", index=False)
print("delayed parcels:", int(featured["is_delayed"].sum()))

print("rows after cleaning and hub merge:", len(featured))
print("medians:", medians)


checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
]

checks.extend([
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(featured)),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
])

checks_df = pd.DataFrame(checks, columns=["metric", "value"]).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)


delay_rows = []

for group in ["Economy", "Express", "Standard"]:
    group_data = featured[featured["service_type"] == group]
    delay_rows.append({
        "dimension": "service_type",
        "group": group,
        "shipments": len(group_data),
        "delayed": int(group_data["is_delayed"].sum()),
        "delay_rate": float(group_data["is_delayed"].mean()),
        "mean_hours": float(group_data["delivery_hours"].mean()),
    })

for group in [1, 2, 3]:
    group_data = featured[featured["hub_tier"] == group]
    delay_rows.append({
        "dimension": "hub_tier",
        "group": group,
        "shipments": len(group_data),
        "delayed": int(group_data["is_delayed"].sum()),
        "delay_rate": float(group_data["is_delayed"].mean()),
        "mean_hours": float(group_data["delivery_hours"].mean()),
    })

distance_band = pd.cut(
    featured["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

for group in distance_band.cat.categories:
    group_data = featured[distance_band == group]
    delay_rows.append({
        "dimension": "distance_band",
        "group": group,
        "shipments": len(group_data),
        "delayed": int(group_data["is_delayed"].sum()),
        "delay_rate": float(group_data["is_delayed"].mean()),
        "mean_hours": float(group_data["delivery_hours"].mean()),
    })

delay_df = pd.DataFrame(delay_rows)
delay_df.round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)
print(pd.DataFrame(delay_rows))