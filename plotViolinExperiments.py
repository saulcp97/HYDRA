#Only Run violin Plots


import Config

import random
import numpy as np
import torch
import pandas as pd
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

    column_names = [
        "Similarity", 
        "Percentage", 

        "Avg. Coalition Size",
        "std Avg. Coalition Size",

        "Intra-Coalition Distance",
        "std Intra-Coalition Distance",

        "Coalition Divergence",
        "std Coalition Divergence",
        "Avg. Distance Between Convergence Points",
        "std Avg. Distance Between Convergence Points",
        
        "Avg. Training Error",
        "std Avg. Training Error",
        "Reciprocity (%)",
        "std Reciprocity (%)",
        
        "Isolated Agents",
        "std Isolated Agents",
        
        "Frequence of Changes",
        "std Frequence of Changes",

        "Reprocity",
        "std Reprocity",


        "Accuracy",
        "std Accuracy", 
        "AUC",
        "std AUC",
    ]


    #Config.experiment_name = "Cosine_0.25_2"
    #plotCoalitionStability(0.25, 0)

    NumberExperiments = 1
    nameSetExperiments = "multiTableResults"
    experimentsName = ["Cosine", "Euclidean", "Normalized Euclidean", "Manhattan", "Pearson Correlation", "Angular"]
    similarityM = ["COSINE_SIMILARITY", "EUCLIDEAN_DISTANCE", "NORM_EUCLIDEAN", "MANHATTAN", "PEARSON", "ANGULAR"]
    coalPercentage = [0.1, 0.25, 0.5] #[0] 
    Config.FIXED_THRESHOLD = False
    Config.IID = True
    Config.EPOCH_NUM = 100
    RELOAD = True
    for k in range(NumberExperiments):
        all_rows = []
        for i in range(len(experimentsName)):
            Config.SIMILARITY_MEASURE = similarityM[i]
            Config.IS_SIMILARITY = not Similarities.SimilarityMeasures[similarityM[i]].is_distance
            for j in range(len(coalPercentage)):
                Config.experiment_name = experimentsName[i] + "_" + str(coalPercentage[j]) + "_" + str(k)
                Config.coalition_percentage = coalPercentage[j]

                accuracyPlotViolin(Config.coalition_percentage)