import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------
# 1) Load Abalone
# -------------------------
X, y = fetch_openml(data_id=183, as_frame=True, return_X_y=True)  # Abalone on OpenML
y = y.astype(int).to_numpy()

# -------------------------
# 2) Turn regression target into classes (binning)
# -------------------------
q = np.quantile(y, [0.25, 0.50, 0.75])
y_cls = np.digitize(y, bins=q, right=True)  # classes: 0,1,2,3
num_classes = int(y_cls.max() + 1)

# -------------------------
# 3) Preprocess features
# -------------------------
# One-hot encode Sex, standardize numeric columns
cat_cols = ["Sex"]
num_cols = [c for c in X.columns if c not in cat_cols]

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

X_cat = ohe.fit_transform(X[cat_cols])
X_num = scaler.fit_transform(X[num_cols])

X_all = np.concatenate([X_cat, X_num], axis=1).astype(np.float32)

# -------------------------
# 4) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# -------------------------
# 5) Torch dataset
# -------------------------
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print(np.bincount(y_train))
train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)
test_loader  = DataLoader(TabDataset(X_test, y_test), batch_size=256, shuffle=False)

# -------------------------
# 6) Lightweight MLP model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

class TinyMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyMLP(in_dim=X_all.shape[1], num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

# -------------------------
# 7) Train + evaluate
# -------------------------
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total

for epoch in range(1, 100 + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

    acc = evaluate()
    print(f"epoch={epoch:02d}  test_acc={acc:.4f}")