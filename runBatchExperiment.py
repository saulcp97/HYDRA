#Launch massive batch experiments
import Config

import random
import numpy as np
import torch
import pandas as pd
import mesa
from codecarbon import EmissionsTracker

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

from architecture.model import FederatedModel
from utils.plotting import *
from utils.experiments import *

if __name__ == "__main__":
    set_seed(Config.SIMULATION_SEED)
    #We build the combinatory of the changes

    #Añadir un cambio de dataset tambien.

    #Test de codecarbon.
    #https://docs.codecarbon.io/latest/tutorials/first-tracking/#step-3-inspect-the-results

    #Set the experiment distribution, en lugar de ciclar 

    experperimentalFamilies = ["control", "dynamicThreshold" "fixedThreshold"]


    """
    #change the project_name 
    with EmissionsTracker(project_name="my-first-tracking") as tracker:
        # Simulate some computation
        total = 0
        for i in range(10_000_000):
            total += i

    df = pd.read_csv("emissions.csv")
    df[["project_name", "duration", "emissions", "emissions_rate", "cpu_power", "ram_power", "energy_consumed"]]
    """