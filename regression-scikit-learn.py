import pandas
import numpy
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- 1. Data Loading and Preparation ---

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
# Clean column names for easier access (e.g., remove spaces)
df.columns = df.columns.str.replace(" ", "_")

matrix = df.drop(columns=["MedHouseVal"])
target = df["MedHouseVal"]

# Add synthetic categorical features for demonstration (Ocean Proximity and Gender)
matrix["Ocean_Proximity"] = numpy.random.choice(
    ["<1H OCEAN", "INLAND", "NEAR OCEAN"], size=len(matrix), p=[0.5, 0.3, 0.2]
)
matrix["Gender"] = numpy.random.choice(["M", "F"], size=len(matrix), p=[0.5, 0.5])

# Define the columns used for STRATIFICATION and feature transformation groups
STRATIFICATION_COLS = ["Gender", "Ocean_Proximity"]
num_standard_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
num_minmax_cols = ["Population", "AveOccup"]
num_no_impute_cols = [
    "Latitude",
    "Longitude",
]  # These are location features, often best left unscaled or handled separately
cat_onehot_cols = ["Ocean_Proximity", "Gender"]


# --- 2. Create Composite Key and Stratified Initial Split ---

# Create a composite key by combining categorical stratification columns
matrix["strat_key"] = matrix[STRATIFICATION_COLS].astype(str).agg("_".join, axis=1)
strat_key = matrix["strat_key"]
matrix = matrix.drop(columns=["strat_key"])

# Stratify the initial split (70% train / 30% test) to ensure representative feature group distributions
X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(
    matrix, target, strat_key, train_size=0.7, random_state=42, stratify=strat_key
)

# --- 3. Feature Preprocessing Pipelines ---

# Define reusable components for clear pipeline construction
imputer_median = SimpleImputer(strategy="median").set_output(transform="pandas")
standard_scaler = StandardScaler().set_output(transform="pandas")
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
    transform="pandas"
)

# Pipeline for Standard Scaling (Median Imputation followed by StandardScaler)
pipeline_standard = Pipeline(
    steps=[("impute_median", imputer_median), ("scale_standard", standard_scaler)]
).set_output(transform="pandas")
# Pipeline for One-Hot Encoding (Most Frequent Imputation followed by OneHotEncoder)
pipeline_nominal_cat = Pipeline(
    steps=[
        ("impute_mf", SimpleImputer(strategy="most_frequent")),
        ("encode_onehot", onehot_encoder),
    ]
).set_output(transform="pandas")

# ColumnTransformer orchestrates transformation for different feature groups
preprocessor = ColumnTransformer(
    transformers=[
        # Numerical features: Median Impute + Standard Scale (appropriate for most numeric features)
        ("median_impute_and_standard_scale", pipeline_standard, num_standard_cols),
        # Numerical features: Mean Impute + MinMax Scale (used for Count/Ratio features like Population)
        (
            "mean_impute_and_minmax_scale",
            Pipeline(
                steps=[
                    ("imputer_mean", SimpleImputer(strategy="mean")),
                    ("scaler_minmax", MinMaxScaler()),
                ]
            ).set_output(transform="pandas"),
            num_minmax_cols,
        ),
        # Location features: Passthrough (Latitude/Longitude should not be scaled/imputed)
        (
            "location_passthrough",
            Pipeline(steps=[("passthrough", "passthrough")]).set_output(
                transform="pandas"
            ),
            num_no_impute_cols,
        ),
        # Categorical features: Most Frequent Impute + One-Hot Encode
        ("cat_impute_and_onehot", pipeline_nominal_cat, cat_onehot_cols),
    ],
    remainder="drop",  # Drop any columns not explicitly transformed
    verbose_feature_names_out=False,
).set_output(transform="pandas")

# --- 4. Full Model Pipeline ---

regressor = RandomForestRegressor(random_state=42)
# Full pipeline: Preprocessor (Feature Engineering) -> Regressor (Model Training)
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", regressor)]
).set_output(transform="pandas")

# --- 5. Hyperparameter Tuning with GridSearchCV (Using Standard KFold) ---

# Define the hyperparameters to search for the RandomForestRegressor
params_grid = {
    "regressor__n_estimators": [50, 100],
    "regressor__max_depth": [5, 10],
}

# Use KFold for cross-validation on the training set (stratification was already applied in the initial split)
cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

# Set up GridSearchCV to find the best combination of hyperparameters
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv_strategy,
    scoring="neg_mean_squared_error",  # Optimize for minimizing MSE
    verbose=0,
    n_jobs=-1,  # Use all available cores for speed
)

# Fit the GridSearchCV on the training data (X_train and y_train)
grid.fit(X_train, y=y_train)

# --- 6. Results and Final Prediction ---

# Extract and store the best model found by GridSearchCV
best_estimator = grid.best_estimator_

# Predict the target values on the hold-out test set
y_pred_final = pandas.Series(best_estimator.predict(X_test), name="Predicted_Value")

# The results are now stored in the grid object for later analysis (e.g., grid.best_score_, grid.best_params_)
# The final predictions are stored in y_pred_final.
