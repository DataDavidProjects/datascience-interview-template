import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# --- PyTorch Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.01
# -----------------------------

# --- Data Generation (Modified for Classification) ---
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


Y = (Y_base > np.median(Y_base)).astype(int)
Y = pd.Series(Y, name="y")  # Y is now 0 or 1

X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": region, "color": color})

# Define feature lists
numerical_columns = ["x1", "x2", "x3"]
categorical_columns = ["region", "color"]

# Split data (using the same 60/40 split)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    train_size=0.6,
    random_state=42,
    stratify=Y,  # stratify added for classification
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
# Target is now a LongTensor (int64) for Binary Cross-Entropy Loss with Logits (BCEWithLogitsLoss)
y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
y_test_np = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)

X_train_tensor = torch.tensor(X_train_np).to(DEVICE)
X_test_tensor = torch.tensor(X_test_np).to(DEVICE)
y_train_tensor = torch.tensor(y_train_np).to(DEVICE)
y_test_tensor = torch.tensor(y_test_np).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 3. PYTORCH MODEL DEFINITION
class ClassificationFNN(nn.Module):
    """A simple Feed-Forward Neural Network for binary classification."""

    def __init__(self, input_size):
        super(ClassificationFNN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layer_stack(x)


model = ClassificationFNN(INPUT_SIZE).to(DEVICE)
print("\n--- 3. Model Architecture ---")
print(model)

# 4. LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(0.7 * EPOCHS), int(0.9 * EPOCHS)], gamma=0.1
)

# 5. TRAINING LOOP (Remains structurally the same)
print("\n--- 5. Training Model ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        predictions_logits = model(batch_X)  # Output is LOGITS (raw scores)
        loss = criterion(predictions_logits, batch_y)

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
# Assuming 'model' is your trained MultiLabelFNN instance
MODEL_PATH = "fnn_weights.pth"
NUM_LABELS = 1

# Save only the learned parameters (state dictionary)
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model parameters saved to: {MODEL_PATH}")

# 1. Instantiate the model structure again (MUST match the saved model!)
model = ClassificationFNN(INPUT_SIZE).to(DEVICE)

# 2. Load the state dictionary from the file
model.load_state_dict(torch.load(MODEL_PATH))

# 3. Set the model to evaluation mode (important for inference/testing)
model.eval()

print("Model structure loaded and weights restored successfully.")

with torch.no_grad():
    # Make predictions (output is logits)
    predictions_logits = model(X_test_tensor)

    probabilities = nn.Sigmoid()(predictions_logits)
    predicted_classes = (probabilities >= 0.5).int()

    # Convert tensors back to numpy for metric calculation
    y_test_np_eval = y_test_tensor.cpu().numpy().flatten()
    y_pred_np = predicted_classes.cpu().numpy().flatten()
