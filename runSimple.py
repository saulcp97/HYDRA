#Singular experiment
import Config

import random
import numpy as np
import torch

import mesa

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

set_seed(Config.SIMULATION_SEED)

from architecture.model import FederatedModel
from utils.plotting import *
from utils.experiments import *

if __name__ == "__main__":
    print("Experimental runs")
    
    # For fixed threshold experiments: Config.threshold_similarity
    # For manage the dynamic percentage (range 0, 1) Config.coalition_percentage

    set_seed(Config.SIMULATION_SEED)
    #The case of no coalitions:
    Config.coalition_percentage = 0.25
    experimentRows = []
    
    combinatoria = []
    
    for i in range(2):
        starter_model = FederatedModel(len(Config.AGENT_NAMES))
        print("Ammount of agents: ", len(starter_model.agents))
        for epoch in range(Config.EPOCH_NUM):
            starter_model.step()

        results = calculateExperimentResults()

        combinatoria.append([Config.SIMILARITY_MEASURE, Config.coalition_percentage])
        experimentRows.append(results)

    experimentResultsCSV(combinatoria, experimentRows, "compromised")