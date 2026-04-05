import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# CONFIG
# -----------------------------
WIDGETS = [
    "dropdown",
    "radio",
    "slider",
    "textbox",
    "autocomplete",
    "table",
    "chart"
]

INPUT_DIM = 6
HIDDEN_DIM = 64
NUM_WIDGETS = len(WIDGETS)
EPOCHS = 25
LR = 0.001
DATASET_SIZE = 5000

# -----------------------------
# SYNTHETIC DATA GENERATOR
# -----------------------------
def generate_sample():
    data_type = np.random.choice([0, 1, 2])  # 0=nominal,1=ordinal,2=numeric
    cardinality = np.random.randint(1, 10000)
    volume = np.random.randint(10, 100000)
    aggregation = np.random.choice([0, 1])
    task_type = np.random.choice([0, 1, 2])  # select, explore, compare
    device = np.random.choice([0, 1])  # mobile, desktop

    features = np.array([
        data_type,
        cardinality / 10000,
        volume / 100000,
        aggregation,
        task_type,
        device
    ], dtype=np.float32)

    # Heuristic labeling
    if data_type == 0:
        if cardinality < 5:
            label = "radio"
        elif cardinality < 50:
            label = "dropdown"
        else:
            label = "autocomplete"

    elif data_type == 2:
        if aggregation == 1:
            label = "chart"
        else:
            label = "slider"

    else:
        label = "textbox"

    return features, WIDGETS.index(label)


def generate_dataset(n):
    X, y = [], []
    for _ in range(n):
        f, label = generate_sample()
        X.append(f)
        y.append(label)
    return np.array(X), np.array(y)

# -----------------------------
# MODEL
# -----------------------------
class UIWidgetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_WIDGETS)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# TRAIN
# -----------------------------
def train():
    X, y = generate_dataset(DATASET_SIZE)

    X = torch.tensor(X)
    y = torch.tensor(y)

    model = UIWidgetNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y).float().mean()
            print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Acc {acc.item():.4f}")

    return model

# -----------------------------
# PREDICT
# -----------------------------
def predict(model, sample):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=0)
        idx = torch.argmax(probs).item()
        return WIDGETS[idx], probs.numpy()

# -----------------------------
# DEMO
# -----------------------------
if __name__ == "__main__":
    print("Training UI Widget Selection Model...\n")
    model = train()

    print("\n--- Inference Demo ---")

    # Example input
    sample = np.array([
        0,      # nominal
        0.0005, # low cardinality
        0.1,    # volume
        0,      # no aggregation
        0,      # selection task
        1       # desktop
    ], dtype=np.float32)

    widget, probs = predict(model, sample)

    print("\nInput Features:", sample)
    print("Predicted Widget:", widget)
    print("\nProbabilities:")
    for w, p in zip(WIDGETS, probs):
        print(f"{w:12s}: {p:.3f}")
