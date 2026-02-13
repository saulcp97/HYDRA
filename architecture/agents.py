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
import numpy as np
import random

import pandas as pd


use_accel = torch.accelerator.is_available()

#torch.manual_seed(args.seed)
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

def rateSimilarity(A, B):
    return Similarities.applySimilarity(A, B, Config.SIMILARITY_MEASURE)

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


from architecture.neuralNetwork import Net
originalWeights = Net().to(device).state_dict()

class nnAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        if Config.SIMULATION_MODE:
            #To ensure the same random weights for all the agents, we can use the unique_id as a seed for the random generator.
            np.random.seed(Config.SIMULATION_SEED)
            self.nnModel = np.random.rand(Config.VECTOR_DIMENSION)
            #Dataset is not a real dataset, but the random target vector that the agent will try to reach with its weights.
            self.dataset = self.nnModel + np.random.rand(Config.VECTOR_DIMENSION) * (Config.EPOCH_NUM/Config.VECTOR_DIMENSION)
            self.optimizer = None
        else:
            self.nnModel = Net().to(device)
            self.nnModel.load_state_dict(originalWeights)
        
            self.dataset, _ = get_data_loaders(self)
            #torch.utils.data.DataLoader(train_set, 512, False)
            self.optimizer = optim.Adadelta(self.nnModel.parameters(), lr=0.5)
        
        self.inWeightBuffer = []

        self.selfID = self.unique_id-1
        self.agent_name = "scp_" + str(self.selfID)

        #Only used for the random weight changes
        self.neighbors = None

        #In theory we know before hand the coalition of all the indexes, but we do it on the first iteration to have acces t
        self.neighbor_info = {}
        self.coallitionNeighbors = []
        self.neighborhood = None

        self.averageSimilarity = -1

    def calibrateNeihborhood(self):
        #IN the first iteration the values 
        self.neighborhood = self.model.neighbors[self.agent_name]
        self.coallitionNeighbors = []
        self.neighbors = []
        for name, agent in self.neighborhood:
            self.neighbors.append(agent)
            self.coallitionNeighbors.append((name, agent))
            self.neighbor_info[name] = (-1)

        #Now calibrate the coa

    def receiveWeight(self, input):
        self.inWeightBuffer.append(input)

    def mixing_Weights(self):
        #while inWeightBuffer > 0 pop

        while len(self.inWeightBuffer) > 0:
            aux = self.inWeightBuffer.pop(0)
            if Config.SIMULATION_MODE:
                similarity = rateSimilarity(aux[1], self.nnModel)
                self.neighbor_info[aux[0]] = (similarity)
                #We dont need the averaage weights because is just a plain vector.
                epsilon = 1/len(self.neighborhood)
                self.nnModel = self.nnModel + epsilon*(aux[1] - self.nnModel)
            else:
                similarity = rateSimilarity(aux[1], self.nnModel.state_dict())
                self.neighbor_info[aux[0]] = (similarity)
                #Epsilon is maximum 1 / number of neighbors.
                weight = average_weights(aux[1], self.nnModel.state_dict(), eps= 1/len(self.neighborhood))
                self.nnModel.load_state_dict(weight)


    def train_model(self):
        localLoss = 0

        if Config.SIMULATION_MODE:
            random_noise = np.random.rand(Config.VECTOR_DIMENSION) * Config.RANDOMNESS_SCALE
            
            movement = self.dataset - self.nnModel + random_noise
            normalizedMove = movement / np.linalg.norm(movement)

            self.nnModel += normalizedMove

            #Euclidean distance as a loss.
            localLoss = np.linalg.norm(self.dataset - self.nnModel)
        else:
            self.nnModel.train()
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

            if Config.SIMULATION_MODE:
                self.save_csv_iteration(localLoss)
                #Save also the coalition members of the agent this iteration
                with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\coalition.txt", "w") as f:
                    result = "Coalition members: \n"
                    for name, _ in self.coallitionNeighbors:
                        result += name + "\n"
            else:
                iteration = self.model.epoch
                if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration)):
                    os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration))

                torch.save(self.nnModel.state_dict(), "outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\weights.pth")
            
                with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\loss.txt", "w") as f:
                    f.write(str(localLoss))

                #Save also the coalition members of the agent this iteration
                with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\coalition.txt", "w") as f:
                    result = "Coalition members: \n"
                    for name, _ in self.coallitionNeighbors:
                        result += name + "\n"
                    #Add the self.averageSimilarity to the log for analysis
                    result += "Average Similarity: " + str(self.averageSimilarity) + "\n"
                    f.write(result)

    def save_csv_iteration(self, loss):
        #Check if the file exists, if not, we create it.
        if not os.path.exists("outputs\\" + Config.experiment_name):
            os.makedirs("outputs\\" + Config.experiment_name)

        #Check if the agent folder exists and if not create it
        if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name):
            os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name)

        #The format for the pandas is loss, density coalition, weights for simulated.
        rowData = [loss, self.averageSimilarity] + list(self.nnModel)

        columnsName = ['loss', 'density'] + [f'w{i}' for i in range(len(self.nnModel))]
        df_file = pd.DataFrame([rowData], columns=columnsName)

        fileName = "outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\expedient.csv"
        if not os.path.exists(fileName):
            df_file.to_csv(fileName, index=False, mode='w')
        else:
            df_file.to_csv(fileName, index=False, mode='a', header=False)


    def checkSelfCoalition(self):
        #We check first of all if the neighbors similarity have been updated, if not we do not update the coalition
        if -1 in self.neighbor_info.values():
            return
        #We calculate the average similarity of the neighbors and if the influence of the neighbors is greater than the threshold we update the coalition excluding it.
        #That case only works if the distance is greater, because a distance 0 would be technically ideal.
        #self.averageSimilarity = sum(self.neighbor_info.values())/len(self.neighbor_info)
        self.averageSimilarity = sum([similarity for name, similarity in self.neighbor_info.items() if name in [n[0] for n in self.coallitionNeighbors]])/len(self.coallitionNeighbors)

        #Now we loop through the neighbors and see, all who are lower are automatically included. Anything above 1.5 times excluded, that value is on Config.threshold, but we can change it to be more or less strict.
        new_coalition = []
        for name, similarity in self.neighbor_info.items():
            #We check what type of measure we are using.
            if Config.IS_SIMILARITY:
                if similarity > self.averageSimilarity * Config.threshold_similarity:
                    new_coalition.append(name)
            else:
                if similarity < self.averageSimilarity * Config.threshold_similarity:
                    new_coalition.append(name)
        self.coallitionNeighbors = [(name, agent) for name, agent in self.coallitionNeighbors if name in new_coalition]

    def pass_weights(self):
        #First of all decide what coallition the agent it is:
        self.checkSelfCoalition()
        #Now the real pass weights
        other_agent:nnAgent = None
        
        #Need to precalculate this to avoid choosing as an option if there is no neighbor outside the coalition.
        electableNeighbors = [agent for agent in self.neighbors if agent not in [n[1] for n in self.coallitionNeighbors]]
        if random.random() < Config.coalition_probability and len(self.coallitionNeighbors) > 0 or (len(electableNeighbors) == 0 and len(self.coallitionNeighbors) > 0):
            other_agent = self.random.choice(self.coallitionNeighbors)[1]
        elif len(self.neighbors) > 0 and len(electableNeighbors) > 0:
            other_agent:nnAgent = self.random.choice(electableNeighbors)
        
        if other_agent is not None:
            if Config.SIMULATION_MODE:
                other_agent.receiveWeight((self.agent_name, self.nnModel))
            else:
                other_agent.receiveWeight((self.agent_name, self.nnModel.state_dict()))