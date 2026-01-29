#Mesa

# Has multi-dimensional arrays and matrices.
# Has a large collection of mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

# Data visualization tools.
import seaborn as sns

import mesa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms

import utils.Similarities as Similarities

from utils.plotting import plot_MDS_Points

import random


import Config

use_accel = torch.accelerator.is_available()

#torch.manual_seed(args.seed)
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


#Provisional function using only euclidian distance until the vectorNormalization function is implemented correctly
def rateSimilarity(A, B):
    sim = 0
    for e in list(A.keys()):
        if len(A[e].shape) > 1: #Esto elimina el Bias
            sim += Similarities.euclideanDistance(A[e], B[e]).item()
    return sim

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

transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])


data_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_len = int(len(data_set)*0.9)
#Test set isnt used at the moment
train_set, test_set = random_split(data_set, [train_len, len(data_set)- train_len])

#train_loader = torch.utils.data.DataLoader(dataset1, 8, False)

originalWeights = Net().to(device).state_dict()

class nnAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.nnModel:nn.Module = Net().to(device)
        self.nnModel.load_state_dict(originalWeights)
        self.inWeightBuffer = []

        self.dataset = torch.utils.data.DataLoader(train_set, 512, False)
        self.optimizer = optim.Adadelta(self.nnModel.parameters(), lr=0.5)

        self.selfID = self.unique_id -1
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
        for batch_idx, (data, target) in enumerate(self.dataset):
            data, target = data.to(device), target.to(device)
            output = self.nnModel(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Iteration of agent {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.unique_id,
                    batch_idx * len(data), len(self.dataset.dataset),
                    100. * batch_idx / len(self.dataset), loss.item()))


    def checkSelfCoalition(self):
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

class FederatedModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n=10, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        # Create agents
        nnAgent.create_agents(model=self, n=n)

        self.neighbors = {}
        for teAgent in Config.NEIGHBOURS.keys():
            #teAgent: teory or graph ag
            self.neighbors[teAgent] = []
            for teoAgent in Config.NEIGHBOURS[teAgent]:
                acAgent = [ac for ac in self.agents if ac.agent_name == teoAgent][0]
                self.neighbors[teAgent].append([acAgent.agent_name, acAgent, acAgent.coalitionIndex])
                #format agent unofficial name, direccion, variable auxiliar distancia, del agente para poder comunicarse

            #teAgent: teory or graph agent, not a real one
            #accAgent = [ac for ac in self.agents if ac.agent_name == teAgent][0]
            #Actual Agent, solo debería haber uno almenos que la cosa haya explotado pero por si a caso
            #podemos pasarle con accAgent.neighbors = self.neighbors[teAgent] una copia la lista de vecinos que tiene para que la mantenga
            #pero si lo dejamos como un servicio que l 
        for ac in self.agents:
           print(ac.agent_name) 

        self.epoch = 0

    def step(self):
        """Advance the model by one step."""
        # This function psuedo-randomly reorders the list of agent objects and
        # then iterates through calling the function passed in as the parameter

        #Can use either do or shuffle do because we separated the steps to do in 3 states so they dont have priority order
        if self.epoch == 0:
            self.agents.do("calibrateNeihborhood")
        else:
            self.agents.do("mixing_Weights")
        self.agents.shuffle_do("train_model")
        self.agents.shuffle_do("pass_weights")

        self.epoch += 1


if __name__ == "__main__":
    print("Experiment simple connection")

    starter_model = FederatedModel(len(Config.AGENT_NAMES))

    print("Ammount of agents: ", len(starter_model.agents))
    
    for epoch in range(Config.EPOCH_NUM):
        starter_model.step()

    #Calculate euclidian distance matrixç
    distancias = []
    for x in starter_model.agents:
        distancias.append([])
        A = x.nnModel.state_dict()
        for y in starter_model.agents:
            B = y.nnModel.state_dict()
            sim = 0
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    sim += Similarities.euclideanDistance(A[e], B[e]).item()

            distancias[-1].append(sim)

    mis_colores = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    plot_MDS_Points(distancias, mis_colores)