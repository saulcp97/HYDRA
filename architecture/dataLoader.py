# dataLoader for a non-IID partition of data between agents.

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
import Config

#custom_distribution = [
#    [0, 1], # Agente 0 recibe solo ceros y unos
#    [2, 3, 4], # Agente 1 recibe dos, tres y cuatros
#    [5, 6, 7, 8, 9] # Agente 2 recibe el resto
#]


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
    X_all, y_all, test_size=0.2, random_state=Config.SIMULATION_SEED, stratify=y_all
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


test_loader = DataLoader(TabDataset(X_test, y_test), batch_size=Config.BATCH_SIZE, shuffle=False)

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


def get_DryBeanDS(agentId, local_val_frac, config=None):
    config = config or Config
    client_indices = None
    if config.IID: # iid mode
        client_indices = partition_iid(y_train, config.NUMBER_OF_AGENTS, seed=config.SIMULATION_SEED)
    else:
        client_indices = partition_dirichlet(
            y_train,
            config.NUMBER_OF_AGENTS,
            alpha=config.dirichlet_alpha,
            seed=config.SIMULATION_SEED,
            min_size=64,
        )
 
    # Sanity check (no overlap, full coverage)
    sanity_check_partitions(client_indices, len(y_train))

    indices_del_agente = client_indices[agentId]

    ds = TabDataset(X_train[indices_del_agente], y_train[indices_del_agente])
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
   
    return dl, test_loader, test_loader

def getModelsParams():
    return X_all.shape[1], num_classes