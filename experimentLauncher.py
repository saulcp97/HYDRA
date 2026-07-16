#Attempt to make a 'clean' experiment launcher
import os
import pandas as pd
import Config

import random
import numpy as np
import torch

from architecture.federatedAverage import FederatedAverageModel

from architecture.singleTrainingModel import singleModel


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


def singleLaunch(epochs = Config.EPOCH_NUM, seed = Config.SIMULATION_SEED, expName = "single_"+Config.experiment_name):
    set_seed(seed)
    sModel = singleModel()
    results = []
    for _ in range(epochs):
        sModel.step()
        results.append(sModel.testResults())

    #Save the output of the experiment with the following measures
    if not os.path.exists("outputs\\" + expName):
        os.makedirs("outputs\\" + expName)

    fileName = "outputs\\" + expName + "\\expedient.csv"
    if not os.path.exists(fileName):
        originalFile = pd.DataFrame(results, columns=["Accuracy", "AUC", "F1Score"])
        originalFile.to_csv(fileName, index=False, mode='w')

def launchFederatedAverage(agents = Config.NUMBER_OF_AGENTS, epochs = Config.EPOCH_NUM, seed = Config.SIMULATION_SEED, expName = "fedAvg_"+Config.experiment_name):
    set_seed(seed)
    #change the config variables
    Config.NUMBER_OF_AGENTS = agents
    Config.SIMULATION_SEED = seed
    #change IIP and Non_IIP

    #modify the config, in this instance the inicialization of number of agents.
    Config.calcNewGraph()
    #with the seed and all configurations modified, 
    


    results = []
    federatedModel = FederatedAverageModel(agents)
    for _ in range(epochs):
        federatedModel.step()
        results.append(federatedModel.testResults())

    #Save the output of the experiment with the following measures
    if not os.path.exists("outputs\\" + expName):
        os.makedirs("outputs\\" + expName)

    fileName = "outputs\\" + expName + "\\expedient.csv"
    if not os.path.exists(fileName):
        originalFile = pd.DataFrame(results, columns=["Accuracy", "AUC", "F1Score"])
        originalFile.to_csv(fileName, index=False, mode='w')

def launchACOL():

    pass

if __name__ == "__main__":
    singleLaunch()
    launchFederatedAverage()

    launchACOL()