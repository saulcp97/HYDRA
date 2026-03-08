import mesa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import utils.Similarities as Similarities

from architecture.dataLoader import get_DryBeanDS, getModelsParams

from sklearn.metrics import roc_auc_score

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


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total

@torch.no_grad()
def auc_score(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # IMPORTANTE: Usamos Softmax para obtener probabilidades, no argmax
        outputs = model(xb)
        probs = torch.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu())
        all_labels.append(yb.cpu())
    
    # Concatenamos todo en dos tensores grandes
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Calculamos el AUC multiclase (One-vs-Rest)
    return roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

#Implementación de FLaMAS (Average)
#A is invited weight
#B is own weight
def average_weights(A, B):
    eps = 1 / Config.GRAPH_GRADE
    average_weights = B
    for key in B.keys():
        if len(B[key]) != len(A[key]):
            print("Error - consensus can only be applied to arrays of same length")
            return None
        average_weights[key] = (1-eps) * B[key] + eps * A[key]
    print("Pesos influenciados")
    return average_weights


from architecture.neuralNetwork import TinyMLP

originalWeights = None
if Config.SIMULATION_MODE:
    np.random.seed(Config.SIMULATION_SEED)
    originalWeights = np.random.rand(Config.VECTOR_DIMENSION)

    #Dataset is not a real dataset, but the random target vector that the agent will try to reach with its weights.
    objective = np.ones(Config.VECTOR_DIMENSION) - np.random.rand(Config.VECTOR_DIMENSION) * 2
    objectivePoint = originalWeights + objective/np.linalg.norm(objective) * (Config.EPOCH_NUM)
else:
    originalWeights = TinyMLP(*getModelsParams()).to(device).state_dict()


class nnAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.selfID = self.unique_id-1
        self.agent_name = "scp_" + str(self.selfID)

        if Config.SIMULATION_MODE:
            #To ensure the same random weights for all the agents, we can use the unique_id as a seed for the random generator.
            np.random.seed(Config.SIMULATION_SEED)
            self.nnModel = np.copy(originalWeights)
            #Dataset is not a real dataset, but the random target vector that the agent will try to reach with its weights.
            self.dataset = np.copy(objectivePoint)
            self.objective = self.dataset + np.ones(Config.VECTOR_DIMENSION) - np.random.rand(Config.VECTOR_DIMENSION) * 2
            self.optimizer = None
        else:
            self.nnModel = TinyMLP(*getModelsParams()).to(device)
            self.nnModel.load_state_dict(originalWeights)
        
            self.dataset, self.valset, self.testset = get_DryBeanDS(self.selfID, 0.2)
            #torch.utils.data.DataLoader(train_set, 512, False)
            self.optimizer = optim.AdamW(self.nnModel.parameters(), lr=3e-3, weight_decay=1e-3)
            self.loss_fn = nn.CrossEntropyLoss()
        self.inWeightBuffer = []

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
            self.neighbor_info[name] = (0)

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

                weight = average_weights(aux[1], self.nnModel.state_dict())
                self.nnModel.load_state_dict(weight)

                #Now pick do consensus with the other
                otherWeight = average_weights(self.nnModel.state_dict(), aux[1])
                agentB = [agent for agent in self.neighbors if agent.agent_name == aux[0]][0]

                agentB.nnModel.load_state_dict(otherWeight)


    def train_model(self):
        localLoss = 0
        train_loss = 0
        for i in range(Config.EPOCH_SHARE):
            if Config.SIMULATION_MODE:
                random_noise = (np.ones(Config.VECTOR_DIMENSION) - np.random.rand(Config.VECTOR_DIMENSION)*2) * Config.RANDOMNESS_SCALE
                #Dataset is objective point, and self.nnModel is the actual vector.
                
                direction = self.objective - self.nnModel
                movement = Config.ETA * direction + random_noise
                
                #movement = (self.dataset - self.objective) + random_noise
                #normalizedMove = movement / np.linalg.norm(movement) if np.linalg.norm(movement) > 1 else movement

                self.nnModel += movement #normalizedMove

                #Euclidean distance as a loss.
                localLoss += np.linalg.norm(self.dataset - self.nnModel)
            else:

                self.nnModel.train()
                for batch_idx, (data, target) in enumerate(self.dataset):
                    data, target = data.to(device), target.to(device)
                    self.optimizer.zero_grad()
                    output = self.nnModel(data)
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    self.optimizer.step()
                    localLoss += loss.item() * target.size(0)
                    """
                    if batch_idx % 100 == 0:
                        print('Train Iteration of agent {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            self.unique_id,
                            batch_idx * len(data), len(self.dataset.dataset),
                            100. * batch_idx / len(self.dataset), loss.item()))
                    """
                train_loss += localLoss / len(self.dataset.dataset)
                val_acc = accuracy(self.nnModel, self.valset, device)
                print(f"Agent {self.unique_id}, Epoch {self.model.epoch}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        
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
                self.save_csv_iteration(train_loss)
                #Save also the coalition members of the agent this iteration
                """
                iteration = self.model.epoch
                if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration)):
                    os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration))
                with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\coalition.txt", "w") as f:
                    result = "Coalition members: \n"
                    for name, _ in self.coallitionNeighbors:
                        result += name + "\n"
                        """

            else:
                self.save_csv_iteration(train_loss, False)
                iteration = self.model.epoch
                if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration)):
                    os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration))

                torch.save(self.nnModel.state_dict(), "outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\weights.pth")

                #Save also the coalition members of the agent this iteration

                """
                with open("outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\coalition.txt", "w") as f:
                    result = "Coalition members: \n"
                    for name, _ in self.coallitionNeighbors:
                        result += name + "\n"
                    #Add the self.averageSimilarity to the log for analysis
                    result += "Average Similarity: " + str(self.averageSimilarity) + "\n"
                    f.write(result)
                """

    def save_csv_iteration(self, loss, saveWeights=True):
        #Check if the file exists, if not, we create it.
        if not os.path.exists("outputs\\" + Config.experiment_name):
            os.makedirs("outputs\\" + Config.experiment_name)

        #Check if the agent folder exists and if not create it
        if not os.path.exists("outputs\\" + Config.experiment_name + "\\" + self.agent_name):
            os.makedirs("outputs\\" + Config.experiment_name + "\\" + self.agent_name)


        #Add a colum with the number of neighbors, the neighbors, the similarity with them, the actual length of the coalition and Config.GRAPH_GRADE columns (can be empty) for the members of the coalition
        num_neighbors = len(self.neighbors)
        neighbor_names = [agent.agent_name for agent in self.neighbors]
        neighbor_sims = [self.neighbor_info.get(name, -1) for name in neighbor_names]

        coalition_names = [n[0] for n in self.coallitionNeighbors]
        coalition_len = len(coalition_names)

        coalition_mask = [1 if name in coalition_names else 0 for name in neighbor_names]

        #Making it like this makes the tests reading afterwards more easy to do as they dont need to alternate
        rowData = [loss, self.averageSimilarity, num_neighbors] + neighbor_names + neighbor_sims + [coalition_len] + coalition_mask
        columnsName = ['loss', 'density', 'num_neighbors'] + [f'nNames{i}' for i in range(len(neighbor_names))] + [f'nRates{i}' for i in range(len(neighbor_sims))] + ["coalition_length"] + [f'isCoalition{i}' for i in range(len(coalition_mask))]
        
        if saveWeights:
            rowData = [loss, self.averageSimilarity, num_neighbors] + neighbor_names + neighbor_sims + [coalition_len] + coalition_mask + list(self.nnModel)
            columnsName = ['loss', 'density', 'num_neighbors'] + [f'nNames{i}' for i in range(len(neighbor_names))] + [f'nRates{i}' for i in range(len(neighbor_sims))] + ["coalition_length"] + [f'isCoalition{i}' for i in range(len(coalition_mask))] + [f'w{i}' for i in range(len(self.nnModel))]
        #print(len(rowData), len(columnsName))
        df_file = pd.DataFrame([rowData], columns=columnsName)

        fileName = "outputs\\" + Config.experiment_name + "\\" + self.agent_name + "\\expedient.csv"
        saved = False
        if not os.path.exists(fileName):
            if Config.SAVE_NEGATIVE_EPOCH:
                originalData = [-1, -1, num_neighbors]  + neighbor_names + [-1]*num_neighbors + [num_neighbors] + [1]*num_neighbors
                if saveWeights:
                    originalData = [-1, -1, num_neighbors]  + neighbor_names + [-1]*num_neighbors + [num_neighbors] + [1]*num_neighbors + list(np.copy(originalWeights))
                
                originalFile = pd.DataFrame([originalData], columns=columnsName)
                originalFile.to_csv(fileName, index=False, mode='w')
            else:
                originalFile = pd.DataFrame([rowData], columns=columnsName)
                originalFile.to_csv(fileName, index=False, mode='w')
        if not saved:
            df_file.to_csv(fileName, index=False, mode='a', header=False)


    def checkSelfCoalition(self):
        #We check first of all if the neighbors similarity have been updated, if not we do not update the coalition
        if -1 in self.neighbor_info.values():
            return
        #We calculate the average similarity of the neighbors and if the influence of the neighbors is greater than the threshold we update the coalition excluding it.
        #That case only works if the distance is greater, because a distance 0 would be technically ideal.
        #self.averageSimilarity = sum(self.neighbor_info.values())/len(self.neighbor_info)
        
        if len(self.coallitionNeighbors) > 0:
            if Config.SIMILARITY_MEASURE == "COSINE_SIMILARITY":
                self.averageSimilarity = sum([similarity for name, similarity in self.neighbor_info.items()])/len(self.neighbor_info)
            else:
                self.averageSimilarity = abs(sum([similarity for name, similarity in self.neighbor_info.items()]))/len(self.neighbor_info)
        
            #Now we loop through the neighbors and see, all who are lower are automatically included. Anything above 1.5 times excluded, that value is on Config.threshold, but we can change it to be more or less strict.
            new_coalition = []
            for name, similarity in self.neighbor_info.items():
                #We check what type of measure we are using.
                if Config.IS_SIMILARITY:
                    if similarity > self.averageSimilarity / Config.threshold_similarity:
                        new_coalition.append(name)
                else:
                    if similarity < self.averageSimilarity * Config.threshold_similarity:
                        new_coalition.append(name)
            self.coallitionNeighbors = [(name, agent) for name, agent in self.coallitionNeighbors if name in new_coalition]
        else:
            #If there is no one in the coallition everybody is in the coalition
            #Shouldnt be done light that but this is a quick patch
            self.coallitionNeighbors = [(agent.agent_name, agent) for agent in self.neighbors]



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

    def accuracyTest(self):
        return accuracy(self.nnModel, self.testset, device)
    
    def aucTest(self):
        return auc_score(self.nnModel, self.testset, device)