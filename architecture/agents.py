import mesa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import utils.Similarities as Similarities

from architecture.dataLoader import get_data_loaders

import os
import Config

import random

use_accel = torch.accelerator.is_available()

#torch.manual_seed(args.seed)
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


def rateSimilarity(A, B):
    return Similarities.applySimilarity(A, B, Config.SIMILARITY_MEASURE)
    """
    DEPRECATED FUNCTION
    #Similarity Normalization
    dA, dB = Similarities.applyNormalization(A, B)
    #Introduce the switch to read from config the similarity measure to use.
    euclideanDistance = Similarities.euclideanDistance(dA, dB).item()
    return euclideanDistance
    """

#Implementación de FLaMAS (Average)
def average_weights(A, B, eps = 0.5):
    average_weights = B
    for key in B.keys():
        if len(B[key]) != len(A[key]):
            print("Error - consensus can only be applied to arrays of same length")
            return None
        average_weights[key] = B[key] + eps*(A[key] - B[key])
    print("Pesos influenciados")
    return average_weights

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


originalWeights = Net().to(device).state_dict()

class nnAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.nnModel:nn.Module = Net().to(device)
        self.nnModel.load_state_dict(originalWeights)
        self.inWeightBuffer = []

        self.selfID = self.unique_id-1
        self.agent_name = "scp_" + str(self.selfID)

        #Only used for the random weight changes
        self.neighbors = None

        #Time to load the initial coalitions
        self.coalitionIndex = -1
        for idx, coalition in enumerate(Config.coalitions):
            if str(self.selfID) in coalition:
                self.coalitionIndex = idx
                break
        #In theory we know before hand the coalition of all the indexes, but we do it on the first iteration to have acces t
        self.neighbor_info = {}
        self.coallitionNeighbors = None
        self.neighborhood = None

        self.dataset, _ = get_data_loaders(self)
        #torch.utils.data.DataLoader(train_set, 512, False)
        self.optimizer = optim.Adadelta(self.nnModel.parameters(), lr=0.5)


    def calibrateNeihborhood(self):
        #IN the first iteration the values 
        self.neighborhood = self.model.neighbors[self.agent_name]
        self.coallitionNeighbors = []
        self.neighbors = []
        for name, agent, coalition in self.neighborhood:
            self.neighbors.append(agent)
            if self.coalitionIndex == coalition:
                self.coallitionNeighbors.append((name, agent))
            self.neighbor_info[name] = (1 if self.coalitionIndex == coalition else 100, coalition)

    def receiveWeight(self, input):
        self.inWeightBuffer.append(input)

    def mixing_Weights(self):
        #while inWeightBuffer > 0 pop 
        while len(self.inWeightBuffer) > 0:
            aux = self.inWeightBuffer.pop(0)

            similarity = rateSimilarity(aux[1], self.nnModel.state_dict())

            self.neighbor_info[aux[0]] = (similarity, aux[2])

            weight = average_weights(aux[1], self.nnModel.state_dict())
            self.nnModel.load_state_dict(weight)

    def train_model(self):
        self.nnModel.train()
        localLoss = 0
        for batch_idx, (data, target) in enumerate(self.dataset):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.nnModel(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            localLoss += loss.item()
            if batch_idx % 100 == 0:
                print('Train Iteration of agent {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.unique_id,
                    batch_idx * len(data), len(self.dataset.dataset),
                    100. * batch_idx / len(self.dataset), loss.item()))
        
        #Now check the experiment and logger
        if Config.log_Experiment:
            #Save on the output folder the weights and loss of the agent this epoch.
            #Check if the experiment folder exists and if not create it
            if not os.path.exists("outputs\\" + Config.experiment_name):
                os.makedirs("outputs\\" + Config.experiment_name)

            #Check if the agent folder exists and if not create it
            if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name):
                os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name)

            #Now do the same for the iteration folder inside the agent folder
            iteration = self.model.epoch
            if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration)):
                os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration))

            torch.save(self.nnModel.state_dict(), "outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\weights.pth")
            with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\loss.txt", "w") as f:
                f.write(str(localLoss))

            #Save also the coalition index of the agent this iteration
            with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\coalition.txt", "w") as f:
                f.write(str(self.coalitionIndex))

    def checkSelfCoalition(self):
        #Check if the neighbor info is empty, if it is, we dont change the coalition because we dont have any information to change it
        if len(self.neighbor_info) == 0:
            print(f"Agent {self.agent_name} - No se ha recibido información de vecinos, manteniendo coalición actual: {self.coalitionIndex}")
        aux = [(self.neighbor_info[k][0], k, self.neighbor_info[k][1]) for k in self.neighbor_info.keys()]
        aux.sort()
        #Half is the braket in this case
        aux = aux[:int(len(aux)/2)]

        max_freq = -1
        most_frequent_id = self.coalitionIndex

        counts = {}
        for item in aux:
            c_id = item[2] # El coalition_id es el tercer elemento
            if c_id in counts:
                counts[c_id] += 1
            else:
                counts[c_id] = 1

        for c_id in counts:
            if counts[c_id] > max_freq:
                max_freq = counts[c_id]
                most_frequent_id = c_id

        if most_frequent_id != self.coalitionIndex:
            self.coallitionNeighbors = []
            for name, agent, _ in self.neighborhood:
                if self.neighbor_info[name][1] == self.coalitionIndex:
                    self.coallitionNeighbors.append((name, agent))

        self.coalitionIndex = most_frequent_id

    def pass_weights(self):
        #print(self.agent_name + " - Vecinos: " + str(len(self.neighbors)) + " - Vecinos de coalición: " + str(len(self.coallitionNeighbors)))
        #print(self.neighbor_info)


        #First of all decide what coallition the agent it is:
        self.checkSelfCoalition()
        #Now the real pass weights
        other_agent:nnAgent = None
        if random.random() < Config.coalition_probability:
            other_agent = self.random.choice(self.coallitionNeighbors)[1]
        else:
            other_agent:nnAgent = self.random.choice(self.neighbors)
        
        
        if other_agent is not None:
            other_agent.receiveWeight((self.agent_name, self.nnModel.state_dict(), self.coalitionIndex))