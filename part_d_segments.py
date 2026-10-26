import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load the historical modelling table created in Part A
data = pd.read_csv("data/modelling_table.csv")

# Calculate one row of performance measures for each hub
hub_profile = (
    data.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

# Round decimal results to four places before saving
decimal_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]
hub_profile[decimal_columns] = hub_profile[decimal_columns].round(4)

hub_profile.to_csv("outputs/hub_profile.csv", index=False)

print("hub rows:", len(hub_profile))
print("shipments per hub:", sorted(hub_profile["shipments"].unique()))
print(hub_profile.head())


# Scale the four hub-level measures for distance-based clustering
profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]
scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(hub_profile[profile_columns])

# Hub pair indexes, used to calculate within- and across-cluster distances
hub_pairs = list(combinations(range(len(hub_profile)), 2))

separation_rows = []
cluster_labels_3 = None

for k in [2, 3, 4, 5]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(scaled_profiles)

    if k == 3:
        cluster_labels_3 = labels

    within_distances = []
    across_distances = []

    for first, second in hub_pairs:
        distance = np.linalg.norm(
            scaled_profiles[first] - scaled_profiles[second]
        )
        if labels[first] == labels[second]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True,
    )

    separation_rows.append(
        {
            "K": k,
            "across_over_within": (
                np.mean(across_distances) / np.mean(within_distances)
            ),
            "cluster_sizes": "-".join(
                str(size) for size in cluster_sizes
            ),
        }
    )

separation_table = pd.DataFrame(separation_rows)
separation_table["across_over_within"] = separation_table[
    "across_over_within"
].round(4)
separation_table.to_csv("outputs/part_d_separation.csv", index=False)

print(separation_table)


# Attach the K=3 cluster labels to the sorted hub profile
profile_with_clusters = hub_profile.copy()
profile_with_clusters["cluster_id"] = cluster_labels_3

# Name clusters in order of their average hub delay rate
cluster_delay_rates = (
    profile_with_clusters.groupby("cluster_id")["delay_rate"].mean()
)
ordered_clusters = cluster_delay_rates.sort_values().index.tolist()
segment_names = ["Standard", "Watch", "Escalate"]
cluster_to_segment = dict(zip(ordered_clusters, segment_names))

profile_with_clusters["segment"] = profile_with_clusters[
    "cluster_id"
].map(cluster_to_segment)

# Add city and tier information from the hub file
hubs = pd.read_csv("data/hubs.csv")
hub_segments = profile_with_clusters[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left",
)
hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id").reset_index(drop=True)
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# Summarize the mean hub-level values in each segment
profile_for_segments = profile_with_clusters.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    suffixes=("", "_joined"),
)
segment_profile = (
    profile_for_segments.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "nunique"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)

segment_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
segment_profile["_order"] = segment_profile["segment"].map(segment_order)
segment_profile = segment_profile.sort_values("_order").drop(
    columns="_order"
)
decimal_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]
segment_profile[decimal_columns] = segment_profile[decimal_columns].round(4)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

# Find hubs where the observed segment differs from the tier's expected segment
tier_to_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}
hub_segments["expected_segment"] = hub_segments["hub_tier"].map(
    tier_to_segment
)
mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
][["hub_id", "segment", "hub_city", "hub_tier"]].copy()
mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(segment_profile)
print(mismatches)