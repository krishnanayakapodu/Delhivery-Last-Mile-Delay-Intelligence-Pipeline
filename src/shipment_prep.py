import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no",
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")


    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    clean = clean.drop(columns=["declared_value"])

    clean["dispatch_time"] = pd.to_datetime(
        clean["dispatch_time"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    
    clean["parcel_weight_kg"] = pd.to_numeric(clean["parcel_weight_kg"], errors="coerce")
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    median_columns = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {
            column: float(clean[column].median())
            for column in median_columns
        }    
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    for column in median_columns:
      clean[column] = clean[column].fillna(medians[column])
    return clean, medians

def build_features(clean, hubs):
    hub_columns = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    result = clean.merge(hubs[hub_columns], on="hub_id", how="left")
    result["promised_hours"] = result["service_type"].map(PROMISED_HOURS).astype(int)
    result["is_cod"] = (result["payment_mode"] == "COD").astype(int)
    result["is_weekend"] = (result["dispatch_time"].dt.dayofweek >= 5).astype(int)
    result["dispatch_hour"] = result["dispatch_time"].dt.hour.astype(int)
    result["log_declared_value"] = np.log(result["declared_value_inr"])
    return result