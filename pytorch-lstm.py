import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# --- PyTorch Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.005
# --- LSTM Specific Parameters ---
SEQUENCE_LENGTH = 10  # Lookback window (how many past time steps to use)
INPUT_SIZE = 1  # Number of features per time step (just 1 for a univariate series)
HIDDEN_SIZE = 50  # Size of the hidden state in the LSTM
NUM_LAYERS = 2  # Number of stacked LSTM layers

# ==============================================================================
# --- 1. Data Generation, Preparation, and Sequence Creation ---
# ==============================================================================


def create_sequence_data(data, seq_length):
    """
    Transforms a time series array into X (sequences) and Y (targets).

    X shape: (N - seq_length, seq_length, 1)
    Y shape: (N - seq_length, 1)
    """
    X, Y = [], []
    for i in range(len(data) - seq_length):
        # Current sequence (seq_length elements)
        X.append(data[i : (i + seq_length)])
        # Target is the element immediately following the sequence
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y).reshape(-1, 1)


# --- Data Generation (Sine wave + Noise) ---
print("--- 1. Data Preparation ---")
np.random.seed(42)
time = np.arange(0, 500)
series = np.sin(0.1 * time) + np.random.normal(0, 0.2, len(time))
series = series.reshape(-1, 1)  # Must be 2D for StandardScaler

# 1.1. Scaling
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series)

# 1.2. Sequence Creation
X_seq, y_seq = create_sequence_data(series_scaled, SEQUENCE_LENGTH)

# 1.3. Split Data (No stratification needed for time series)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_seq,
    y_seq,
    train_size=0.8,
    shuffle=False,  # shuffle=False is CRUCIAL for time series
)

print(f"Train sequence shape: {X_train_np.shape}")
print(f"Test target shape: {y_test_np.shape}")


# ==============================================================================
# --- 2. PyTorch Model Definition (LSTM) ---
# ==============================================================================


class SequencePredictorLSTM(nn.Module):
    """LSTM model for sequence-to-one regression."""

    def __init__(self, input_size, hidden_size, num_layers):
        super(SequencePredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        # batch_first=True makes input shape: (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2,  # Dropout between LSTM layers
        ).to(DEVICE)

        # Define the fully connected output layer (maps LSTM output to a single prediction)
        self.fc = nn.Linear(hidden_size, 1).to(DEVICE)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Initialize hidden state (h_0) and cell state (c_0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        # Pass the input through the LSTM
        # out: (batch_size, seq_len, hidden_size)
        # (hn, cn): Final hidden and cell states
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the output from the LAST time step of the sequence (out[:, -1, :])
        # This is passed to the fully connected layer
        out = self.fc(out[:, -1, :])

        return out


# ==============================================================================
# --- 3. Training Preparation ---
# ==============================================================================

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
model = SequencePredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
print("\n--- 2. Model Architecture ---")
print(model)

# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(0.7 * EPOCHS), int(0.9 * EPOCHS)], gamma=0.1
)

# ==============================================================================
# --- 4. Training Loop ---
# ==============================================================================

print("\n--- 3. Training Model ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    scheduler.step()

    avg_loss = total_loss / len(train_dataset)
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )


# ==============================================================================
# --- 5. Evaluation ---
# ==============================================================================

print("\n--- 4. Evaluation on Test Set ---")
model.eval()

with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

    # Convert predictions back to original scale for MSE calculation (optional but good practice)
    # We use the scaler from step 1.1, inverted.
    y_test_np_eval = scaler.inverse_transform(y_test_tensor.cpu().numpy())
    y_pred_np = scaler.inverse_transform(y_pred_tensor.cpu().numpy())

    # Calculate metrics
    test_mse = mean_squared_error(y_test_np_eval, y_pred_np)

print(f"Test Set Mean Squared Error (MSE) on original scale: {test_mse:.6f}")
