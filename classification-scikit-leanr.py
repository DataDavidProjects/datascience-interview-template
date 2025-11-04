from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd


# --- Configuration and Data Generation ---
np.random.seed(42)
N = 1000

# 1. Numerical Features
x1 = np.random.normal(10, 3, N)
x2 = np.random.normal(10, 3, N)
x3 = np.random.normal(10, 3, N)  # This feature is noise

# 2. Categorical Features
region = np.random.choice(["north", "south", "east", "west"], size=N)
color = np.random.choice(["red", "green", "blue"], size=N)  # This feature is noise

# 3. Target Variable (Now depends on x1, x2, and region)
# Create a linear combination of features that will influence Y
linear_predictor = 0.5 * x1 - 0.3 * x2 + np.random.normal(0, 1, N)

# Add a strong positive effect for 'north' region
region_effect = np.where(region == "north", 2.5, 0)
linear_predictor += region_effect

# Use the sigmoid function to convert the predictor to a probability
probability = 1 / (1 + np.exp(-linear_predictor))

# Generate the binary target Y based on the probability
Y = pd.Series((probability > np.random.rand(N)).astype(int))

# Input DataFrame
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": region, "color": color})

# Define feature lists
numerical_features = ["x1", "x2", "x3"]
categorical_features = ["region", "color"]


# Define your target and features
target = Y
matrix = X


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    matrix, target, train_size=0.6, stratify=target, random_state=42
)


# Classifier
classifier = LogisticRegression(solver="liblinear", max_iter=1000)

# Preprocessors
scaler = StandardScaler().set_output(transform="pandas")
one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore", sparse_output=False
).set_output(transform="pandas")

categorical_pipeline = Pipeline([("one_hot_encoder", one_hot_encoder)])

numerical_pipeline = Pipeline([("scaler", scaler)])

# Column transformer
numerical_columns = matrix.select_dtypes("number").columns
categorical_columns = matrix.select_dtypes("object").columns

processing = ColumnTransformer(
    [
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
).set_output(transform="pandas")

pipeline = Pipeline([("processing", processing), ("classifier", classifier)])

# Parameter grid
param_grid = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__penalty": ["l1", "l2"],
}

# Stratified group CV
cv = StratifiedKFold(n_splits=3, shuffle=False)

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)


grid_search.fit(X_train, y_train)


print("Best params:", grid_search.best_params_)
print("Best AUC:", grid_search.best_score_)
pipeline.set_params(**grid_search.best_params_)
pipeline.fit(X_train, y_train).set_output(transform="pandas")
pipeline.named_steps["classifier"]
y_pred = pd.Series(pipeline.predict_proba(X_test)[:, 1], name="y_pred")
roc_auc_score(y_test, y_pred)
