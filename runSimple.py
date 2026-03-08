
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

    calculateExperimentResults()
    import numpy as np
    rates, aucs = starter_model.rate_global_scores()
    avgAcuracy = np.mean(rates)
    stdAccuracy = np.std(rates)

    avgAucs = np.mean(aucs)
    stdAucs = np.std(aucs)
    print(f"Average Accuracy {avgAcuracy} +- STD {stdAccuracy}")
    print(f"Average Area Under the Curve {avgAucs} +- STD {stdAucs}")
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