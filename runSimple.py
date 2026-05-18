
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

    starter_model = FederatedModel(len(Config.AGENT_NAMES))
    print("Ammount of agents: ", len(starter_model.agents))
    for epoch in range(Config.EPOCH_NUM):
        starter_model.step()


    #Experiment interpretation:

    results = calculateExperimentResults()
    
    # Unpack results: metrics are returned as a tuple
    (finalCoalitionSize, finalCoalitionSizeSTD, finalTrainError, finalTrainErrorSTD,
     intraCoalitionDist, intraCoalitionDistSTD, coalitionDivergence, coalitionDivergenceSTD,
     convergedDistance, convergedDistanceSTD, reciprocityPercentage,
     finalAccuracy, finalAccuracySTD, finalAuc, finalAucSTD) = results
    
    # Print loaded accuracy and AUC with standard deviation
    if finalAccuracy > 0:
        print(f"Average Accuracy {finalAccuracy:.4f} +- STD {finalAccuracySTD:.4f}")
    if finalAuc > 0:
        print(f"Average Area Under the Curve {finalAuc:.4f} +- STD {finalAucSTD:.4f}")
    
    #plotCoalitionsOfExperimentList()
    #plotCoalitionsOfExperimentList()
    if Config.SIMULATION_MODE:
        #plot_fast_federated_3d()
        #plot_spread_optimized()
        pass
        #plot_MDS_3D_Simulation_csv()
    else:
        pass
        #plot_MDS_3D()
#f.write('%s\n' %items)