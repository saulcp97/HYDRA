import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import matplotlib.pyplot as plt
import plotly.express as px


# ============================================================
# 0) Reproducibility
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# 1) Load + preprocess Dry Bean (UCI)
# ============================================================
dry_bean = fetch_ucirepo(id=602)
X = dry_bean.data.features
y = dry_bean.data.targets

# y -> 1D
if hasattr(y, "shape") and len(y.shape) == 2:
    y = y.iloc[:, 0]
y = y.astype(str).to_numpy()

# labels
le = LabelEncoder()
y_all = le.fit_transform(y)
class_names = list(le.classes_)
num_classes = len(class_names)

idx2class = {i: name for i, name in enumerate(class_names)}
class2idx = {name: i for i, name in enumerate(class_names)}

# features
cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
num_cols = [c for c in X.columns if c not in cat_cols]

scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols].to_numpy()) if len(num_cols) else np.empty((len(X), 0))

if len(cat_cols):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(X[cat_cols])
    X_all = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
else:
    X_all = X_num.astype(np.float32)

# split into train/test ONCE (global test set)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# ---- mapping + train distribution
print("idx2class =", idx2class)
print("class2idx =", class2idx)

counts = np.bincount(y_train, minlength=num_classes)
train_dist = {idx2class[i]: int(counts[i]) for i in range(num_classes)}
print("train_class_distribution =", train_dist)


# ============================================================
# 2) Torch datasets
# ============================================================
class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


test_loader = DataLoader(TabDataset(X_test, y_test), batch_size=1024, shuffle=False)


# ============================================================
# 3) Partitioning: IID and Dirichlet non-IID
# ============================================================
def partition_iid(y, n_clients, seed=0):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y))
    rng.shuffle(idxs)
    splits = np.array_split(idxs, n_clients)
    return [s.copy() for s in splits]


def partition_dirichlet(y, n_clients, alpha=0.3, seed=0, min_size=32):
    """
    Label-distribution skew using Dirichlet over classes.
    Each sample is assigned to exactly one client (no overlap).
    alpha small -> more non-IID
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = int(y.max() + 1)

    idx_by_class = [np.where(y == k)[0] for k in range(K)]
    for k in range(K):
        rng.shuffle(idx_by_class[k])

    while True:
        client_idxs = [[] for _ in range(n_clients)]
        for k in range(K):
            p = rng.dirichlet(alpha=np.ones(n_clients) * alpha)

            nk = len(idx_by_class[k])
            counts = (p * nk).astype(int)

            diff = nk - counts.sum()
            if diff > 0:
                counts[np.argsort(-p)[:diff]] += 1
            elif diff < 0:
                counts[np.argsort(p)[:(-diff)]] -= 1

            start = 0
            for i in range(n_clients):
                c = counts[i]
                if c > 0:
                    client_idxs[i].extend(idx_by_class[k][start:start + c].tolist())
                    start += c

        sizes = np.array([len(ci) for ci in client_idxs])
        if sizes.min() >= min_size:
            return [np.array(ci, dtype=int) for ci in client_idxs]

        seed += 1
        rng = np.random.default_rng(seed)


# ============================================================
# Sanity check for FL partitions (i.e. no overlap, total size matches)
# ============================================================
def sanity_check_partitions(client_indices, total_samples):
    all_indices = np.concatenate(client_indices)

    assert len(all_indices) == total_samples, (
        f"Partition size mismatch: {len(all_indices)} != {total_samples}"
    )

    unique = np.unique(all_indices)
    assert len(unique) == total_samples, (
        f"Duplicate samples detected across clients: {len(unique)} unique vs {total_samples} total"
    )

    assert unique.min() == 0 and unique.max() == total_samples - 1, (
        "Missing sample indices detected"
    )

    print("partition_sanity_check = PASSED")


# ============================================================
# 3b) Bubble chart of class distribution per client (Plotly)
# ============================================================
def plot_client_class_bubbles(y, client_indices, idx2class, title):
    y = np.asarray(y)
    K = int(y.max() + 1)

    rows = []
    for cid, idxs in enumerate(client_indices):
        counts = np.bincount(y[idxs], minlength=K)
        for k in range(K):
            c = int(counts[k])
            if c > 0:
                rows.append(
                    {
                        "client": f"client_{cid:02d}",
                        "class": idx2class[k],
                        "count": c,
                        "text": str(c),
                    }
                )

    if not rows:
        print("No data to plot for bubble chart.")
        return

    fig = px.scatter(
        rows,
        x="class",
        y="client",
        size="count",
        text="text",
        size_max=55,
        title=title,
    )
    fig.update_traces(textposition="middle center")
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Client",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.show()


# ============================================================
# 4) Model: simple lightweight NN
# ============================================================
class TinyMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5) FedAvg utilities
# ============================================================
@torch.no_grad()
def get_model_params(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def set_model_params(model: nn.Module, params):
    model.load_state_dict(params, strict=True)

@torch.no_grad()
def fedavg_aggregate(client_params, client_sizes):
    total = float(np.sum(client_sizes))
    agg = {}
    for k in client_params[0].keys():
        agg[k] = sum((client_params[i][k] * (client_sizes[i] / total)) for i in range(len(client_params)))
    return agg

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)

def train_one_client(model, loader, device, epochs=1, lr=5e-4, weight_decay=2e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    n_samples = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

    return running_loss / max(1, n_samples)


# ============================================================
# 6) FL Simulator (FedAvg)
# ============================================================
def run_fedavg(
    mode="dirichlet",          # "iid" or "dirichlet"
    n_clients=10,
    rounds=100,
    client_frac=1.0,           # fraction of clients per round
    local_epochs=1,
    batch_size=128,
    dirichlet_alpha=0.3,
    seed=42,
    lr=5e-4,
    weight_decay=2e-3,
    show_bubbles=True,
):
    # Partition train data
    if mode == "iid":
        client_indices = partition_iid(y_train, n_clients, seed=seed)
    elif mode == "dirichlet":
        client_indices = partition_dirichlet(y_train, n_clients, alpha=dirichlet_alpha, seed=seed, min_size=64)
    else:
        raise ValueError("mode must be 'iid' or 'dirichlet'")

    # Sanity check (no overlap, full coverage)
    sanity_check_partitions(client_indices, len(y_train))

    # Per-client sizes print
    sizes = [len(ci) for ci in client_indices]
    print("client_sizes =", sizes)

    # Bubble chart for this partition
    if show_bubbles:
        if mode == "iid":
            title = f"Dry Bean - IID class distribution per client (n_clients={n_clients})"
        else:
            title = f"Dry Bean - Dirichlet non-IID class distribution (alpha={dirichlet_alpha}, n_clients={n_clients})"
        plot_client_class_bubbles(y_train, client_indices, idx2class, title=title)

    # Build client loaders
    client_loaders = []
    client_sizes = []
    for idxs in client_indices:
        ds = TabDataset(X_train[idxs], y_train[idxs])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        client_loaders.append(dl)
        client_sizes.append(len(ds))
    client_sizes = np.array(client_sizes)

    # Global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = TinyMLP(in_dim=X_all.shape[1], num_classes=num_classes).to(device)

    # Track
    acc_hist = []
    loss_hist = []

    rng = np.random.default_rng(seed)

    for r in range(1, rounds + 1):
        m = max(1, int(np.ceil(client_frac * n_clients)))
        selected = rng.choice(n_clients, size=m, replace=False)

        global_params = get_model_params(global_model)

        client_params = []
        client_ns = []
        client_losses = []

        for cid in selected:
            client_model = TinyMLP(in_dim=X_all.shape[1], num_classes=num_classes).to(device)
            set_model_params(client_model, global_params)

            avg_loss = train_one_client(
                client_model,
                client_loaders[cid],
                device=device,
                epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
            )
            client_losses.append(avg_loss)

            client_params.append(get_model_params(client_model))
            client_ns.append(client_sizes[cid])

        new_params = fedavg_aggregate(client_params, client_ns)
        set_model_params(global_model, new_params)

        test_acc = evaluate(global_model, test_loader, device=device)

        acc_hist.append(test_acc)
        loss_hist.append(float(np.mean(client_losses)))

        print(f"round={r:03d}  avg_client_loss={loss_hist[-1]:.4f}  test_acc={test_acc:.4f}")

    return acc_hist, loss_hist


# ============================================================
# 7) Run IID + Dirichlet non-IID + accuracy plot
# ============================================================
ROUNDS = 100
N_CLIENTS = 10

acc_iid, loss_iid = run_fedavg(
    mode="iid",
    n_clients=N_CLIENTS,
    rounds=ROUNDS,
    client_frac=1.0,
    local_epochs=1,
    batch_size=128,
    seed=SEED,
    lr=5e-4,
    weight_decay=2e-3,
    show_bubbles=True,
)

acc_dir, loss_dir = run_fedavg(
    mode="dirichlet",
    n_clients=N_CLIENTS,
    rounds=ROUNDS,
    client_frac=1.0,
    local_epochs=1,
    batch_size=128,
    dirichlet_alpha=0.3,
    seed=SEED,
    lr=5e-4,
    weight_decay=2e-3,
    show_bubbles=True,
)

# Accuracy over rounds
plt.figure(figsize=(9, 5))
plt.plot(range(1, ROUNDS + 1), acc_iid, label="IID")
plt.plot(range(1, ROUNDS + 1), acc_dir, label="Dirichlet non-IID (alpha=0.3)")
plt.xlabel("Federated Rounds")
plt.ylabel("Test Accuracy")
plt.title("FedAvg on Dry Bean: Test Accuracy vs Rounds")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()