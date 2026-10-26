import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES


# Load Part A's modelling table and select the required inputs and target
data = pd.read_csv("data/modelling_table.csv")
X = data[FEATURES]
y = data["delivery_hours"]

# Split into training, validation, and test rows
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("train rows:", len(X_train))
print("validation rows:", len(X_val))
print("test rows:", len(X_test))


# Fit the scaler on training rows only; reuse it to transform test rows
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

checks = pd.DataFrame(
    {
        "metric": [
            "train_rows",
            "val_rows",
            "test_rows",
            "train_scaled_max_abs_mean",
            "test_scaled_max_abs_mean",
        ],
        "value": [
            len(X_train),
            len(X_val),
            len(X_test),
            np.abs(X_train_scaled.mean(axis=0)).max(),
            np.abs(X_test_scaled.mean(axis=0)).max(),
        ],
    }
)
checks["value"] = checks["value"].round(4)
checks.to_csv("outputs/part_b_checks.csv", index=False)
print(checks)


def make_metric_row(model_name, split_name, actual, predicted):
    """Calculate the required regression metrics for one model and split."""
    mse = mean_squared_error(actual, predicted)
    return {
        "model": model_name,
        "split": split_name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


# Baseline: predict the training target's mean for every validation parcel
baseline_predictions = np.full(len(y_val), y_train.mean())
metrics_rows = [
    make_metric_row("mean_baseline", "val", y_val, baseline_predictions)
]


# Compare decision trees with different maximum depths
tree_depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_rows = []
tree_validation_rmse = {}

for depth in tree_depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    val_predictions = tree.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    tree_validation_rmse[depth] = val_rmse
    tree_rows.append(
        {
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        }
    )

tree_table = pd.DataFrame(tree_rows)
tree_table[["train_rmse", "val_rmse"]] = tree_table[
    ["train_rmse", "val_rmse"]
].round(4)
tree_table.to_csv("outputs/part_b_tree_depth.csv", index=False)

# Choose the lowest validation RMSE; ties go to the smaller depth
best_tree_depth = min(
    tree_depths,
    key=lambda depth: (
        tree_validation_rmse[depth],
        depth if depth is not None else float("inf"),
    ),
)
print(tree_table)
print("chosen tree depth:", best_tree_depth)


# Fit the four candidate models on training rows and score validation rows
candidate_models = {
    "linear_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ),
    "decision_tree": DecisionTreeRegressor(
        max_depth=best_tree_depth,
        random_state=42,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

for model_name, model in candidate_models.items():
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    metrics_rows.append(
        make_metric_row(model_name, "val", y_val, val_predictions)
    )


# Compare the requested gradient boosting learning rates
learning_rate_rows = []

for learning_rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        random_state=42,
    )
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)

    learning_rate_rows.append(
        {
            "learning_rate": learning_rate,
            "val_rmse": np.sqrt(
                mean_squared_error(y_val, val_predictions)
            ),
            "val_r2": r2_score(y_val, val_predictions),
        }
    )

learning_rate_table = pd.DataFrame(learning_rate_rows)
learning_rate_table[["val_rmse", "val_r2"]] = learning_rate_table[
    ["val_rmse", "val_r2"]
].round(4)
learning_rate_table.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
)
print(learning_rate_table)


# Compare the two best validation candidates using the same five-fold split
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

candidate_val_rmse = {
    row["model"]: row["RMSE"]
    for row in metrics_rows
    if row["model"] in candidate_models
}
best_two_names = sorted(
    candidate_val_rmse,
    key=candidate_val_rmse.get,
)[:2]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for model_name in best_two_names:
    scores = cross_val_score(
        candidate_models[model_name],
        X_train_val,
        y_train_val,
        cv=cv,
    )
    cv_rows.append(
        {
            "model": model_name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
        }
    )

cv_table = pd.DataFrame(cv_rows)
cv_table[["mean_r2", "std_r2"]] = cv_table[
    ["mean_r2", "std_r2"]
].round(4)
cv_table.to_csv("outputs/part_b_cv.csv", index=False)
print(cv_table)


# Choose by validation RMSE, then evaluate that fitted model on test rows once
final_model_name = min(candidate_val_rmse, key=candidate_val_rmse.get)
final_model = candidate_models[final_model_name]
test_predictions = final_model.predict(X_test)

metrics_rows.append(
    make_metric_row(final_model_name, "test", y_test, test_predictions)
)

metrics_table = pd.DataFrame(metrics_rows)
metric_columns = ["MAE", "MSE", "RMSE", "R2"]
metrics_table[metric_columns] = metrics_table[metric_columns].round(4)
metrics_table.to_csv("outputs/part_b_metrics.csv", index=False)

print("final model:", final_model_name)
print(metrics_table)