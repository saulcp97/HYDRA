import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -------------------------
# Super lightweight model
# -------------------------
class TinyMLP(nn.Module):
    def __init__(self, num_classes=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total

def main():
    # -------------------------
    # 1) Repro + device
    # -------------------------
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    # On Windows, simplest is num_workers=0 (no subprocesses, data loaded in main process).
    num_workers = 0

    # -------------------------
    # 2) Load FashionMNIST
    # -------------------------
    transform = transforms.ToTensor()

    train_full = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    num_classes = 10

    # -------------------------
    # 3) Train/val split (stratified)
    # -------------------------
    targets = np.asarray(train_full.targets, dtype=np.int64)
    idx = np.arange(len(train_full))

    val_frac = 0.2
    train_idx, val_idx = [], []

    for c in range(num_classes):
        c_idx = idx[targets == c]
        np.random.shuffle(c_idx)
        n_val = int(len(c_idx) * val_frac)
        val_idx.append(c_idx[:n_val])
        train_idx.append(c_idx[n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full, val_idx)

    # -------------------------
    # 4) DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_set, batch_size=1024, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=1024, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=1024, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # -------------------------
    # 5) Model + optim
    # -------------------------
    model = TinyMLP(num_classes=num_classes, hidden=64).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

    # -------------------------
    # 6) Train loop
    # -------------------------
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    test_acc = accuracy(model, test_loader, device)
    print(f"test_acc={test_acc:.4f}")

if __name__ == "__main__":
    # Needed on Windows when DataLoader uses workers; safe even with num_workers=0.
    import multiprocessing as mp
    mp.freeze_support()
    main()