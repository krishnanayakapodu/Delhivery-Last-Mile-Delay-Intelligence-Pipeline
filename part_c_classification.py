from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, NOISE_COLUMNS
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load Part A's modelling table
data = pd.read_csv("data/modelling_table.csv")

# Use the 17 required inputs and the delayed/not-delayed target
X = data[FEATURES]
y = data["is_delayed"]

# Split into training rows and temporary rows, preserving the delayed share
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# Split the temporary rows equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

print("train rows and delayed:", len(X_train), int(y_train.sum()))
print("validation rows and delayed:", len(X_val), int(y_val.sum()))
print("test rows and delayed:", len(X_test), int(y_test.sum()))


def make_classification_row(model_name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
    }


# Blind baseline: predict that every parcel is on time
model_rows = [
    make_classification_row(
        "blind_all_on_time",
        y_val,
        [0] * len(y_val),
    )
]

# Logistic regression, with scaling inside the pipeline
logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
logistic_model.fit(X_train, y_train)
model_rows.append(
    make_classification_row(
        "logistic_regression",
        y_val,
        logistic_model.predict(X_val),
    )
)

# Try each required tree depth and keep the one with highest validation F1
tree_depths = [2, 3, 4, 5, 6, 8, 10]
tree_results = []

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)
    tree_f1 = f1_score(y_val, predictions, zero_division=0)
    tree_results.append((depth, tree_f1))

best_tree_depth = min(
    tree_results,
    key=lambda result: (-result[1], result[0]),
)[0]

best_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_tree_depth,
    random_state=42,
)
best_tree.fit(X_train, y_train)
model_rows.append(
    make_classification_row(
        f"decision_tree_depth_{best_tree_depth}",
        y_val,
        best_tree.predict(X_val),
    )
)

# Default random forest
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
default_forest.fit(X_train, y_train)
model_rows.append(
    make_classification_row(
        "random_forest_default",
        y_val,
        default_forest.predict(X_val),
    )
)

models_table = pd.DataFrame(model_rows)
metric_columns = ["accuracy", "precision", "recall", "f1"]
models_table[metric_columns] = models_table[metric_columns].round(4)
models_table.to_csv("outputs/part_c_models.csv", index=False)

print("chosen tree depth:", best_tree_depth)
print(models_table)


# Score all features using training rows only
all_feature_selector = SelectKBest(score_func=f_classif, k="all")
all_feature_selector.fit(X_train, y_train)

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": all_feature_selector.scores_,
        "rf_importance": default_forest.feature_importances_,
        "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
    }
)
feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Save scores in the pinned feature order
feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]
feature_scores[["f_score", "rf_importance"]] = feature_scores[
    ["f_score", "rf_importance"]
].round(4)
feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)
print(feature_scores.sort_values("f_rank"))


# Check validation F1 when selecting different numbers of features
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selected_model = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )
    selected_model.fit(X_train, y_train)
    predictions = selected_model.predict(X_val)

    selectk_rows.append(
        {
            "k": k,
            "val_f1": f1_score(y_val, predictions, zero_division=0),
        }
    )

selectk_table = pd.DataFrame(selectk_rows)
selectk_table["val_f1"] = selectk_table["val_f1"].round(4)
selectk_table.to_csv("outputs/part_c_selectk.csv", index=False)
print(selectk_table)


# Fit feature selection on training rows only, then transform validation rows
selector_7 = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)

# Grid of forest settings to compare
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
)
grid_search.fit(X_train_7, y_train)
grid_predictions = grid_search.predict(X_val_7)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)
random_search.fit(X_train_7, y_train)
random_predictions = random_search.predict(X_val_7)

# Untuned default forest on the same seven selected features
default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
)
default_forest_7.fit(X_train_7, y_train)
default_predictions_7 = default_forest_7.predict(X_val_7)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": f1_score(y_val, grid_predictions, zero_division=0),
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": f1_score(y_val, random_predictions, zero_division=0),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": None,
        "val_f1": f1_score(y_val, default_predictions_7, zero_division=0),
    },
]

tuning_table = pd.DataFrame(tuning_rows)
tuning_table[["best_cv_accuracy", "val_f1"]] = tuning_table[
    ["best_cv_accuracy", "val_f1"]
].round(4)
tuning_table.to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_table)

# Choose highest validation F1; ties prefer default, then grid, then randomized
tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}
chosen_row = min(
    tuning_rows,
    key=lambda row: (-row["val_f1"], tie_priority[row["search"]]),
)
print("chosen tuning candidate:", chosen_row["search"])


# Get the parameters from the winning tuning candidate
chosen_from = chosen_row["search"]

if chosen_from == "grid_search":
    final_params = grid_search.best_params_.copy()
elif chosen_from == "randomized_search":
    final_params = random_search.best_params_.copy()
else:
    final_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

# Build the final model pipeline and fit it on training rows only
final_model = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **final_params,
            ),
        ),
    ]
)
final_model.fit(X_train, y_train)

# Choose the highest threshold that achieves at least 0.80 validation recall
val_probabilities = final_model.predict_proba(X_val)[:, 1]
threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
]
threshold_rows = []

for threshold in threshold_values:
    predictions = (val_probabilities >= threshold).astype(int)
    result = make_classification_row(
        "threshold_result",
        y_val,
        predictions,
    )
    threshold_rows.append(
        {
            "threshold": threshold,
            **{key: value for key, value in result.items() if key != "model"},
        }
    )

eligible_thresholds = [
    row["threshold"]
    for row in threshold_rows
    if row["recall"] >= 0.80
]
chosen_threshold = max(eligible_thresholds, default=0.1)

threshold_table = pd.DataFrame(threshold_rows)
metric_columns = ["accuracy", "precision", "recall", "f1"]
threshold_table[metric_columns] = threshold_table[
    metric_columns
].round(4)
threshold_table.to_csv("outputs/part_c_thresholds.csv", index=False)
print(threshold_table)
print("chosen threshold:", chosen_threshold)

# Evaluate the final model on test rows once, at the chosen threshold
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)
test_result = make_classification_row(
    "final_test",
    y_test,
    test_predictions,
)

final_test_table = pd.DataFrame(
    [
        {
            "threshold": chosen_threshold,
            **{
                key: value
                for key, value in test_result.items()
                if key != "model"
            },
        }
    ]
)
final_test_table[metric_columns] = final_test_table[
    metric_columns
].round(4)
final_test_table.to_csv("outputs/part_c_final_test.csv", index=False)
print(final_test_table)

# Save the chosen settings for the packaging step
max_depth_value = final_params.get("max_depth")
final_config = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": final_params["n_estimators"],
            "max_depth": (
                "unlimited" if max_depth_value is None else max_depth_value
            ),
            "min_samples_leaf": final_params["min_samples_leaf"],
            "threshold": chosen_threshold,
            "chosen_from": chosen_from,
        }
    ]
)
final_config.to_csv("outputs/part_c_final_config.csv", index=False)
print(final_config)