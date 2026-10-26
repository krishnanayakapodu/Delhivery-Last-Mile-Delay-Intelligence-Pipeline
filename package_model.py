import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# Read the final classifier settings chosen in Part C
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

if str(config["max_depth"]).lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(config["max_depth"])


# Calculate the same imputation medians using Part A's shared cleaning function
raw_shipments = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw_shipments)


# Rebuild the stratified Part C train/validation/test split
data = pd.read_csv("data/modelling_table.csv")
X = data[FEATURES]
y = data["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)


# Rebuild and fit Part C's final model on training rows
final_model = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            ),
        ),
    ]
)
final_model.fit(X_train, y_train)


# Save the model and its supporting files
joblib.dump(final_model, "models/delay_model.pkl")

pd.DataFrame({"threshold": [threshold]}).round(4).to_csv(
    "models/delay_threshold.csv",
    index=False,
)

pd.DataFrame({"feature": FEATURES}).to_csv(
    "models/feature_order.csv",
    index=False,
)

impute_table = pd.DataFrame(
    {
        "column": list(medians.keys()),
        "median": list(medians.values()),
    }
)
impute_table["median"] = impute_table["median"].round(4)
impute_table.to_csv("models/impute_values.csv", index=False)


# Reload the saved model and compare its test probabilities with the in-memory model
reloaded_model = joblib.load("models/delay_model.pkl")
memory_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

reload_check = pd.DataFrame(
    {
        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold",
        ],
        "value": [
            len(X_test),
            np.max(
                np.abs(memory_probabilities - reloaded_probabilities)
            ),
            threshold,
        ],
    }
)
reload_check["value"] = reload_check["value"].round(4)
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

print(reload_check)