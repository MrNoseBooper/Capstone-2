"""
LSTM (sequential deep learning model) and KAN (tabular deep learning model)
for sepsis early prediction, both implemented in PyTorch, sharing a common
training / prediction loop for a fair comparison.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.kan_layer import KAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


def _compute_pos_weight(y):
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return torch.tensor(max(n_neg / max(n_pos, 1), 1.0), dtype=torch.float32)


def _train_torch_binary_classifier(model, X_train, y_train, X_val, y_val,
                                    epochs, batch_size, lr, model_name="model"):
    model.to(DEVICE)
    pos_weight = _compute_pos_weight(y_train).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        print(f"[{model_name}] epoch {epoch + 1}/{epochs} "
              f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

    return model


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    p = config.LSTM_PARAMS
    model = LSTMClassifier(
        n_features=n_features,
        hidden_size=p["hidden_size"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
    )
    model = _train_torch_binary_classifier(
        model, X_train, y_train, X_val, y_val,
        epochs=p["epochs"], batch_size=p["batch_size"], lr=p["lr"],
        model_name="LSTM",
    )
    return model


def predict_proba_torch(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(DEVICE)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# --------------------------------------------------------------------
# KAN
# --------------------------------------------------------------------
class KANClassifier(nn.Module):
    def __init__(self, n_features, hidden_sizes, grid_size=5, spline_order=3):
        super().__init__()
        layer_sizes = [n_features] + list(hidden_sizes) + [1]
        self.kan = KAN(layer_sizes, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        return self.kan(x).squeeze(-1)


def train_kan(X_train, y_train, X_val, y_val, n_features):
    p = config.KAN_PARAMS
    model = KANClassifier(
        n_features=n_features,
        hidden_sizes=p["hidden_sizes"],
        grid_size=p["grid_size"],
        spline_order=p["spline_order"],
    )
    model = _train_torch_binary_classifier(
        model, X_train, y_train, X_val, y_val,
        epochs=p["epochs"], batch_size=p["batch_size"], lr=p["lr"],
        model_name="KAN",
    )
    return model
