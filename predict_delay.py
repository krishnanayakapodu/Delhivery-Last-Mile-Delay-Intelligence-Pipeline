import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import build_features, clean_shipments


# Load the saved model, threshold, feature order, imputation values, and segments
model = joblib.load("models/delay_model.pkl")
threshold = float(
    pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"]
)
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()

impute_table = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_table["column"], impute_table["median"]))

hub_segments = pd.read_csv("outputs/hub_segments.csv")[
    ["hub_id", "segment"]
]


# Read new parcels and prepare them with Part A's shared functions
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean_new, _ = clean_shipments(new_raw, medians=medians)
featured_new = build_features(clean_new, hubs)

# Keep the original order of rows in new_shipments.csv
featured_new["_file_order"] = np.arange(len(featured_new))

# Arrange inputs exactly as the packaged model expects
X_new = featured_new[feature_order]
raw_probabilities = model.predict_proba(X_new)[:, 1]

# Round probability first; the flag rule uses the displayed probability
featured_new["delay_probability"] = np.round(raw_probabilities, 4)
featured_new["flagged"] = (
    featured_new["delay_probability"] >= threshold
).astype(int)

# Attach the segment and choose the operations action
actions = featured_new.merge(
    hub_segments,
    on="hub_id",
    how="left",
    sort=False,
)
actions = actions.sort_values("_file_order").reset_index(drop=True)

actions["action"] = np.where(
    actions["flagged"].eq(0),
    "No action",
    np.where(
        actions["segment"].eq("Escalate"),
        "Escalate to hub manager",
        "Proactive customer call",
    ),
)

actions = actions[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]

actions.to_csv("outputs/new_shipment_actions.csv", index=False)

print("new parcels scored:", len(actions))
print("flagged parcels:", int(actions["flagged"].sum()))
print(
    "flagged for hub-manager escalation:",
    int(
        (
            actions["flagged"].eq(1)
            & actions["action"].eq("Escalate to hub manager")
        ).sum()
    ),
)
print(actions.head())