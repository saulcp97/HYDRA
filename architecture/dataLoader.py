# dataLoader for a non-IID partition of data between agents.

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split

import Config

#custom_distribution = [
#    [0, 1], # Agente 0 recibe solo ceros y unos
#    [2, 3, 4], # Agente 1 recibe dos, tres y cuatros
#    [5, 6, 7, 8, 9] # Agente 2 recibe el resto
#]

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_set = datasets.MNIST('../data', train=True, download=True, transform=transform)

train_len = int(len(data_set)*0.95)
#Test set isnt used at the moment
train_set, test_set = random_split(data_set, [train_len, len(data_set)- train_len])

#If classes_per_agent is None, then IID default partition is used even if iid=False
#Classes per agent is the list of classes that id_agent will recieve, on custom_distribution id_agent = 0, would be [0,1]
def get_federated_dataset(dataset:Subset, iid=True, classes_per_agent=None):
    if iid or classes_per_agent is None:
        # IID: Mezcla total y división equitativa
        return dataset
    else:
        # Non-IID: Cada agente recibe un subconjunto específico de clases

        targets = np.array(dataset.dataset.targets)[dataset.indices]
        # Seleccionar clases específicas para este agente
        idx_by_label = {k: np.where(targets == k)[0].tolist() for k in range(len(dataset.dataset.classes))}

        selected_indices = []
        for cls in classes_per_agent:
            selected_indices.extend(idx_by_label[cls])
        # Crear un Subset del dataset original con los índices seleccionados
        subset = Subset(dataset.dataset, selected_indices)
        print(f"Subset creado con {len(subset)} muestras para clases {classes_per_agent}")

        return subset


# Agent from architecture/agents.py
# Rest of the options for the data Loader are default or from Config.py
def get_data_loaders(agent):
    train_setA = get_federated_dataset(train_set, iid=False, classes_per_agent=Config.iid_distribution[agent.selfID])
    train_setB = get_federated_dataset(test_set, iid=False, classes_per_agent=Config.iid_distribution[agent.selfID])

    train_loader = torch.utils.data.DataLoader(train_setA, Config.BATCH_SIZE, False)
    test_loader = torch.utils.data.DataLoader(train_setB, Config.BATCH_SIZE, False)

    return train_loader, test_loader