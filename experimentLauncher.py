#Attempt to make a 'clean' experiment launcher
import Config

import random
import numpy as np
import torch

from architecture.federatedAverage import FederatedAverageModel

def set_seed(seed: int):
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(Config.SIMULATION_SEED)

    starter_model = FederatedAverageModel(len(Config.AGENT_NAMES))
    print("Ammount of agents: ", len(starter_model.agents))
    for epoch in range(Config.EPOCH_NUM):
        starter_model.step()