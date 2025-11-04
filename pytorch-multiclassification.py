import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# --- PyTorch Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_CLASSES = 3
# -----------------------------

# --- Data Generation (Modified for Multiclass Classification) ---
np.random.seed(42)
N = 1000
x1 = np.random.normal(10, 3, N)
x2 = np.random.normal(10, 3, N)
x3 = np.random.normal(10, 3, N)
region = np.random.choice(["north", "south", "east", "west"], size=N)
color = np.random.choice(["red", "green", "blue"], size=N)
Y_linear_part = 2.5 * x1 - 1.5 * x2
region_effect = np.where(region == "north", 5.0, np.where(region == "south", -3.0, 0))


Y_base = Y_linear_part + region_effect + np.random.normal(0, 5, N)


Y = pd.qcut(Y_base, q=NUM_CLASSES, labels=False, duplicates="drop").astype(int)
Y = pd.Series(Y, name="y")  # Y is now 0, 1, or 2

X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": region, "color": color})

# Define feature lists
numerical_columns = ["x1", "x2", "x3"]
categorical_columns = ["region", "color"]

# Split data (using the same 60/40 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.6, random_state=42, stratify=Y
)

# 1. DATA PREPROCESSING (Scikit-learn)
print("--- 1. Data Preprocessing ---")

numerical_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=1, include_bias=False)),
    ]
)

categorical_pipeline = Pipeline(
    [("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

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

INPUT_SIZE = X_train_processed.shape[1]
print(f"Total input features after preprocessing: {INPUT_SIZE}")

# 2. CONVERT TO PYTORCH TENSORS
X_train_np = np.asarray(X_train_processed, dtype=np.float32)
X_test_np = np.asarray(X_test_processed, dtype=np.float32)
# !!! KEY CHANGE: Target must be LongTensor (int64) and 1D for CrossEntropyLoss
y_train_np = np.asarray(y_train, dtype=np.int64)
y_test_np = np.asarray(y_test, dtype=np.int64)

X_train_tensor = torch.tensor(X_train_np).to(DEVICE)
X_test_tensor = torch.tensor(X_test_np).to(DEVICE)
y_train_tensor = torch.tensor(y_train_np).to(DEVICE)
y_test_tensor = torch.tensor(y_test_np).to(DEVICE)

# Note: y_train_tensor is 1D, so TensorDataset handles it correctly
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 3. PYTORCH MODEL DEFINITION
class MulticlassClassificationFNN(nn.Module):
    """A simple Feed-Forward Neural Network for multiclass classification."""

    def __init__(self, input_size, num_classes):
        super(MulticlassClassificationFNN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # !!! KEY CHANGE: Output layer size must equal the number of classes (NUM_CLASSES)
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.layer_stack(x)


model = MulticlassClassificationFNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
print("\n--- 3. Model Architecture ---")
print(model)

# 4. LOSS FUNCTION AND OPTIMIZER
# !!! KEY CHANGE: Use CrossEntropyLoss for multiclass classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(0.7 * EPOCHS), int(0.9 * EPOCHS)], gamma=0.1
)

# 5. TRAINING LOOP (Functionally the same, but target shape is different)
print("\n--- 5. Training Model ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        # Note: batch_y is now [BATCH_SIZE]
        predictions_logits = model(batch_X)  # Output is [BATCH_SIZE, NUM_CLASSES]
        loss = criterion(
            predictions_logits, batch_y
        )  # Loss expects 1D LongTensor target

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    scheduler.step()

    avg_loss = total_loss / len(train_dataset)
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )


# 6. EVALUATION
print("\n--- 6. Evaluation on Test Set ---")
model.eval()

with torch.no_grad():
    # Make predictions (output is logits)
    predictions_logits = model(X_test_tensor)

    predicted_classes = torch.argmax(predictions_logits, dim=1)

    # Convert tensors back to numpy for metric calculation
    y_test_np_eval = y_test_tensor.cpu().numpy()
    y_pred_np = predicted_classes.cpu().numpy()


print("\n--- Test Set Classification Report ---")
# Display detailed metrics
print(
    classification_report(
        y_test_np_eval,
        y_pred_np,
        target_names=[f"Class {i}" for i in range(NUM_CLASSES)],
    )
)
