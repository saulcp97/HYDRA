import mesa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
import utils.Similarities as Similarities

from architecture.dataLoader import get_DryBeanDS, getModelsParams

from sklearn.metrics import roc_auc_score, f1_score

import os
import numpy as np
import random

import pandas as pd

import math

use_accel = torch.accelerator.is_available()

#torch.manual_seed(args.seed)
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

def rateSimilarity(A, B, config):
    return Similarities.applySimilarity(A, B, config.SIMILARITY_MEASURE, config.SIMULATION_MODE)


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

@torch.no_grad()
def f1_scores(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # F1 por etiqueta (array) y F1 macro (float)
    f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f1_per_label, f1_macro




#Implementación de FLaMAS (Average)
#A is invited weight
#B is own weight
def average_weights(A, B, graph_grade):
    eps = 1 / graph_grade
    average_weights = B
    for key in B.keys():
        if len(B[key]) != len(A[key]):
            print("Error - consensus can only be applied to arrays of same length")
            return None
        average_weights[key] = (1-eps) * B[key] + eps * A[key]
    print("Pesos influenciados")
    return average_weights

import math
from architecture.neuralNetwork import TinyMLP


class nnAgent(mesa.Agent):
    """An agent with fixed initial wealth."""
    originalWeights = None
    objectivePoint = None
    shared_state_config = None

    @classmethod
    def prepare_shared_state(cls, config):
        if cls.shared_state_config is config:
            return

        cls.shared_state_config = config
        if config.SIMULATION_MODE:
            rng = np.random.default_rng(config.SIMULATION_SEED)
            cls.originalWeights = rng.random(config.VECTOR_DIMENSION)
            objective = np.ones(config.VECTOR_DIMENSION) - rng.random(config.VECTOR_DIMENSION) * 2
            cls.objectivePoint = cls.originalWeights + objective / np.linalg.norm(objective) * config.EPOCH_NUM
        else:
            cls.originalWeights = TinyMLP(*getModelsParams()).to(device).state_dict()

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        self.selfID = self.unique_id - 1
        self.agent_name = "scp_" + str(self.selfID)
        config = self.model.config
        self.train_loss = 0

        if config.SIMULATION_MODE:
            rng = np.random.default_rng(config.SIMULATION_SEED)
            self.nnModel = np.copy(self.__class__.originalWeights)
            # Dataset is not a real dataset, but the random target vector that the agent will try to reach with its weights.
            self.dataset = np.copy(self.__class__.objectivePoint)
            self.objective = self.dataset + np.ones(config.VECTOR_DIMENSION) - rng.random(config.VECTOR_DIMENSION) * 2
            self.optimizer = None
        else:
            self.nnModel = TinyMLP(*getModelsParams()).to(device)
            self.nnModel.load_state_dict(self.__class__.originalWeights)

            self.dataset, self.valset, self.testset = get_DryBeanDS(self.selfID, 0.2, config)
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
            self.neighbor_info[name] = 1 if self.model.config.IS_SIMILARITY else 0

        #Now calibrate the coa

    def receiveWeight(self, input):
        self.inWeightBuffer.append(input)

    def mixing_Weights(self):
        #while inWeightBuffer > 0 pop

        while len(self.inWeightBuffer) > 0:
            aux = self.inWeightBuffer.pop(0)
            if self.model.config.SIMULATION_MODE:
                similarity = rateSimilarity(aux[1], self.nnModel, self.model.config)
                self.neighbor_info[aux[0]] = similarity
                # We don't need average weights because it is just a plain vector.
                epsilon = 1 / len(self.neighborhood)
                self.nnModel = self.nnModel + epsilon * (aux[1] - self.nnModel)
            else:
                similarity = rateSimilarity(aux[1], self.nnModel.state_dict(), self.model.config)
                self.neighbor_info[aux[0]] = similarity
                #local_grade = min(len(self.neighbors))


                weight = average_weights(aux[1], self.nnModel.state_dict(), self.model.config.GRAPH_GRADE)
                self.nnModel.load_state_dict(weight)

                # Now pick do consensus with the other
                otherWeight = average_weights(self.nnModel.state_dict(), aux[1], self.model.config.GRAPH_GRADE)
                agentB = [agent for agent in self.neighbors if agent.agent_name == aux[0]][0]

                agentB.nnModel.load_state_dict(otherWeight)


    def train_model(self):
        config = self.model.config
        localLoss = 0
        self.train_loss = 0
        for i in range(config.EPOCH_SHARE):
            if config.SIMULATION_MODE:
                random_noise = (np.ones(config.VECTOR_DIMENSION) - np.random.rand(config.VECTOR_DIMENSION) * 2) * config.RANDOMNESS_SCALE
                # Dataset is objective point, and self.nnModel is the actual vector.
                
                direction = self.objective - self.nnModel
                movement = config.ETA * direction + random_noise
                
                #movement = (self.dataset - self.objective) + random_noise
                #normalizedMove = movement / np.linalg.norm(movement) if np.linalg.norm(movement) > 1 else movement

                self.nnModel += movement #normalizedMove

                # Euclidean distance as a loss.
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
                self.train_loss += localLoss / len(self.dataset.dataset)
                val_acc = accuracy(self.nnModel, self.valset, device)
                print(f"Agent {self.unique_id}, Epoch {self.model.epoch}  train_loss={self.train_loss:.4f}  val_acc={val_acc:.4f}")


    def logger(self):
        config = self.model.config
        # Now check the experiment and logger
        if config.log_Experiment:
            # Save on the output folder the weights and loss of the agent this epoch.
            # Check if the experiment folder exists and if not create it
            if not os.path.exists("outputs\\" + config.experiment_name):
                os.makedirs("outputs\\" + config.experiment_name)

            # Check if the agent folder exists and if not create it
            if not os.path.exists("outputs\\" + config.experiment_name + "\\" + self.agent_name):
                os.makedirs("outputs\\" + config.experiment_name + "\\" + self.agent_name)

            if config.SIMULATION_MODE:
                #In simulation mode there is only a vector, so Accuracy and AUC are kinda irrelevant.
                self.save_csv_iteration(self.train_loss)
            else:
                # Calculate accuracy and AUC on test set
                test_accuracy = self.accuracyTest()
                test_auc = self.aucTest()
                self.save_csv_iteration(self.train_loss, False, test_accuracy, test_auc)
                iteration = self.model.epoch
                #We dont have the the certainty to be able to manage all the weight structure, so we use a conditional.
                if config.SAVE_FULL_WEIGHTS:    
                    if not os.path.exists("outputs\\" + config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration)):
                        os.makedirs("outputs\\" + config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration))
                    torch.save(self.nnModel.state_dict(), "outputs\\" + config.experiment_name + "\\" + self.agent_name + "\\iteration_" + str(iteration) + "\\weights.pth")            

    def save_csv_iteration(self, loss, saveWeights=True, accuracy=None, auc=None):
        config = self.model.config
        # Check if the file exists, if not, we create it.
        if not os.path.exists("outputs\\" + config.experiment_name):
            os.makedirs("outputs\\" + config.experiment_name)

        # Check if the agent folder exists and if not create it
        if not os.path.exists("outputs\\" + config.experiment_name + "\\" + self.agent_name):
            os.makedirs("outputs\\" + config.experiment_name + "\\" + self.agent_name)

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
        
        if accuracy:
            rowData.append(accuracy)
            columnsName.append('accuracy')

        if auc:
            rowData.append(auc)
            columnsName.append('auc')
        
        if saveWeights:
            #This would only be used in the case of simulated run because the pytorch model NEVER should be atemptet to be saved on csv.
            rowData += list(self.nnModel)
            columnsName += + [f'w{i}' for i in range(len(self.nnModel))]
        df_file = pd.DataFrame([rowData], columns=columnsName)

        fileName = "outputs\\" + config.experiment_name + "\\" + self.agent_name + "\\expedient.csv"
        saved = False
        if not os.path.exists(fileName):
            if config.SAVE_NEGATIVE_EPOCH:
                originalData = [-1, -1, num_neighbors]  + neighbor_names + [-1]*num_neighbors + [num_neighbors] + [1]*num_neighbors
                
                if accuracy:
                    originalData.append(-1)

                if auc:
                    originalData.append(-1)

                if saveWeights:
                    originalData += list(np.copy(self.__class__.originalWeights))

                originalFile = pd.DataFrame([originalData], columns=columnsName)
                originalFile.to_csv(fileName, index=False, mode='w')
                saved = True
            else:
                originalFile = pd.DataFrame([rowData], columns=columnsName)
                originalFile.to_csv(fileName, index=False, mode='w')
            saved = True
        if not saved:
            #TBH (To be honest) this could be done cleaner but not simpler to read.
            df_file.to_csv(fileName, index=False, mode='a', header=False)


    def checkSelfCoalition(self):
        if not self.neighbor_info:
            # no similarity/distance information yet; leave coalition untouched
            return

        if len(self.coallitionNeighbors) == 0:
            # initial state: everybody is in the coalition
            self.coallitionNeighbors = [(agent.agent_name, agent) for agent in self.neighbors]
            return

        # compute average of the raw values stored in neighbour_info
        self.averageSimilarity = sum(self.neighbor_info.values()) / len(self.neighbor_info)

        config = self.model.config
        new_coalition = []

        if config.FIXED_THRESHOLD:
            # Use fixed threshold: include all neighbors meeting the threshold
            threshold = config.threshold_similarity
            if config.IS_SIMILARITY:
                # Similarity: higher values are better, include >= threshold
                new_coalition = [name for name, sim in self.neighbor_info.items() if sim >= threshold]
            else:
                # Distance: lower values are better, include <= threshold
                new_coalition = [name for name, sim in self.neighbor_info.items() if sim <= threshold]
        else:
            # K-Neighbors algorithm: select top k% neighbors based on similarity/distance
            num_neighbors = len(self.neighbor_info)
            # Calculate k as percentage of neighbors, round up, minimum 1
            k = max(1, math.ceil(num_neighbors * config.coalition_percentage))  # Ceiling division
            
            if config.IS_SIMILARITY:
                # higher values are better; sort descending and take top k
                sorted_neighbors = sorted(self.neighbor_info.items(), key=lambda x: x[1], reverse=True)
            else:
                # lower values are better; sort ascending and take top k
                sorted_neighbors = sorted(self.neighbor_info.items(), key=lambda x: x[1])

            # Select top k neighbors
            for name, similarity in sorted_neighbors[:k]:
                new_coalition.append(name)

        # filter the stored neighbour tuple list so it contains only the allowed
        # names.  We keep the original agent objects intact.
        self.coallitionNeighbors = [
            (agent.agent_name, agent)
            for agent in self.neighbors
            if agent.agent_name in new_coalition
        ]

    def pass_weights(self):
        #First of all decide what coallition the agent it is:
        self.checkSelfCoalition()
        #Now the real pass weights
        other_agent:nnAgent = None
        
        #Need to precalculate this to avoid choosing as an option if there is no neighbor outside the coalition.
        electableNeighbors = [agent for agent in self.neighbors if agent not in [n[1] for n in self.coallitionNeighbors]]
        config = self.model.config
        if random.random() < config.coalition_probability and len(self.coallitionNeighbors) > 0 or (len(electableNeighbors) == 0 and len(self.coallitionNeighbors) > 0):
            other_agent = self.random.choice(self.coallitionNeighbors)[1]
        elif len(self.neighbors) > 0 and len(electableNeighbors) > 0:
            other_agent:nnAgent = self.random.choice(electableNeighbors)
        
        if other_agent is not None:
            if config.SIMULATION_MODE:
                other_agent.receiveWeight((self.agent_name, self.nnModel))
            else:
                other_agent.receiveWeight((self.agent_name, self.nnModel.state_dict()))

    def accuracyTest(self):
        return accuracy(self.nnModel, self.testset, device)
    
    def aucTest(self):
        return auc_score(self.nnModel, self.testset, device)
    
    def f1scoreTest(self):
        return f1_scores(self.nnModel, self.testset, device)