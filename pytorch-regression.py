import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import math  # Used for setting the learning rate schedule

# --- PyTorch Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.01
# -----------------------------

# --- Data Generation ---
np.random.seed(42)
N = 1000
x1 = np.random.normal(10, 3, N)
x2 = np.random.normal(10, 3, N)
x3 = np.random.normal(10, 3, N)
region = np.random.choice(["north", "south", "east", "west"], size=N)
color = np.random.choice(["red", "green", "blue"], size=N)
Y_linear_part = 2.5 * x1 - 1.5 * x2
region_effect = np.where(region == "north", 5.0, np.where(region == "south", -3.0, 0))
Y = Y_linear_part + region_effect + np.random.normal(0, 5, N)
Y = pd.Series(Y, name="y")
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": region, "color": color})

# Define feature lists
numerical_columns = ["x1", "x2", "x3"]
categorical_columns = ["region", "color"]

# Split data (using the same 60/40 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.6, random_state=42
)

# 1. DATA PREPROCESSING (Using Scikit-learn to handle the transformation)
print("--- 1. Data Preprocessing ---")

# Define the full ColumnTransformer pipeline exactly as before, but without the final regressor
numerical_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        # Keeping degree=1 for simplicity in the FNN, but you can change this
        ("polynomial_features", PolynomialFeatures(degree=1, include_bias=False)),
    ]
)

categorical_pipeline = Pipeline(
    [("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

# Final preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
)

# Fit and Transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the final feature count after preprocessing
INPUT_SIZE = X_train_processed.shape[1]
print(f"Total input features after preprocessing: {INPUT_SIZE}")

# 2. CONVERT TO PYTORCH TENSORS
# Ensure the output is a float32 NumPy array
X_train_np = np.asarray(X_train_processed, dtype=np.float32)
X_test_np = np.asarray(X_test_processed, dtype=np.float32)
# Reshape for PyTorch
y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
y_test_np = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)

# Create PyTorch Tensors
X_train_tensor = torch.tensor(X_train_np).to(DEVICE)
X_test_tensor = torch.tensor(X_test_np).to(DEVICE)
y_train_tensor = torch.tensor(y_train_np).to(DEVICE)
y_test_tensor = torch.tensor(y_test_np).to(DEVICE)

# Create PyTorch DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 3. PYTORCH MODEL DEFINITION
class RegressionFNN(nn.Module):
    """A simple Feed-Forward Neural Network for regression."""

    def __init__(self, input_size):
        super(RegressionFNN, self).__init__()
        self.layer_stack = nn.Sequential(
            # Input layer: INPUT_SIZE -> 64
            nn.Linear(input_size, 64),
            nn.ReLU(),
            # Hidden layer: 64 -> 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layer_stack(x)


model = RegressionFNN(INPUT_SIZE).to(DEVICE)
print("\n--- 3. Model Architecture ---")
print(model)

# 4. LOSS FUNCTION AND OPTIMIZER
# Mean Squared Error (MSE) is standard for regression
criterion = nn.MSELoss()
# Adam optimizer is a good default choice
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Simple learning rate scheduler (reduces LR by 10x at 70% and 90% of epochs)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(0.7 * EPOCHS), int(0.9 * EPOCHS)], gamma=0.1
)

# 5. TRAINING LOOP
print("\n--- 5. Training Model ---")

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 2. Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradient
        optimizer.step()  # Update weights

        total_loss += loss.item() * batch_X.size(0)

    scheduler.step()  # Update learning rate

    avg_loss = total_loss / len(train_dataset)
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )


# 6. EVALUATION
print("\n--- 6. Evaluation on Test Set ---")
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

with torch.no_grad():  # Disable gradient calculation
    # Make predictions
    y_pred_tensor = model(X_test_tensor)

    # Convert tensors back to numpy for metric calculation
    y_test_np_eval = y_test_tensor.cpu().numpy()
    y_pred_np = y_pred_tensor.cpu().numpy()

    # Calculate metrics
    test_mse = mean_squared_error(y_test_np_eval, y_pred_np)
    test_r2 = r2_score(y_test_np_eval, y_pred_np)

print(f"Test Set Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Set R-squared ($R^2$): {test_r2:.4f}")
