import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from architecture.neuralNetwork import TinyMLP
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, f1_score
from architecture.dataLoader import get_DryBeanDS, getModelsParams

use_accel = torch.accelerator.is_available()
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


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

@torch.no_grad()
def auc_score(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # IMPORTANTE: Usamos Softmax para obtener probabilidades, no argmax
        outputs = model(xb)
        probs = torch.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())
    
    # Concatenamos todo en dos tensores grandes
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Calculamos el AUC multiclase (One-vs-Rest)
    return roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

@torch.no_grad()
def f1_scores(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # F1 por etiqueta (array) y F1 macro (float)
    f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f1_per_label, f1_macro

class singleModel():
    def __init__(self):
        self.model = TinyMLP(*getModelsParams()).to(device)
        config = Config
        config.NUMBER_OF_AGENTS = 1

        self.dataset, self.valset, self.testset = get_DryBeanDS(0, 1, config)
        #torch.utils.data.DataLoader(train_set, 512, False)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-3, weight_decay=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loss = 0

    def step(self):
        localLoss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.dataset):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            localLoss += loss.item() * target.size(0)
            
        self.train_loss += localLoss / len(self.dataset.dataset)
        

    def accuracyTest(self):
        return accuracy(self.model, self.testset, device)
    
    def aucTest(self):
        return auc_score(self.model, self.testset, device)
    
    def f1scoreTest(self):
        return f1_scores(self.model, self.testset, device)

    def testResults(self):
        aT = self.accuracyTest()
        aucT = self.aucTest()
        f1s = self.f1scoreTest()[0].mean()
        return [aT, aucT, f1s]