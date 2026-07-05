#Federated Average

import os

import mesa
import numpy as np
import pandas as pd

from architecture.agents import nnAgent
import Config
import torch

class FederatedAverageModel(mesa.Model):
    """This model is a lot 'simpler' than the FederatedModel version"""

    def __init__(self, n=10, seed=None, config=Config):
        self.config = config
        if seed is None:
            seed = self.config.SIMULATION_SEED
        super().__init__(seed=seed)
        self.num_agents = n
        print(n)
        #Config.refresh_runtime_state(self.config)
        # Prepare shared model state using the provided config before creating agents
        nnAgent.prepare_shared_state(self.config)

        # Create agents
        nnAgent.create_agents(model=self, n=n)
        self.network = self.config.H

        self.neighbors = {}

        for teAgent in self.config.NEIGHBOURS.keys():
            #teAgent: teory or graph ag
            self.neighbors[teAgent] = []
            for teoAgent in self.config.NEIGHBOURS[teAgent]:
                acAgent = [ac for ac in self.agents if ac.agent_name == teoAgent][0]
                self.neighbors[teAgent].append([acAgent.agent_name, acAgent])
                #format agent unofficial name, direccion, variable auxiliar distancia, del agente para poder comunicarse


        self.averagednnModel:dict = None

        self.epoch = 0


    def federatedAverage(self):
        aux = {}
        agentDictionary = self.agents[0].nnModel.state_dict()
        for key in agentDictionary.keys():
            aux[key] = agentDictionary[key]

        for i in range(1, len(self.agents)):
            agentDictionary = self.agents[i].nnModel.state_dict()
            for key in agentDictionary.keys():
                aux[key] += agentDictionary[key]
            
        for key in aux.keys():
            aux[key] /= len(self.agents)

        self.averageWeights = aux

        for agent in self.agents:
            agent.nnModel.load_state_dict(self.averageWeights)


    def step(self):
        """Advance the model by one step."""
        # This function psuedo-randomly reorders the list of agent objects and
        # then iterates through calling the function passed in as the parameter

        #Can use either do or shuffle do because we separated the steps to do in 3 states so they dont have priority order
        
        self.agents.do("train_model")
        if self.epoch == 0:
            self.agents.do("calibrateNeihborhood")

        self.federatedAverage()
        self.testResults()
        #And now the changes are saved and the loggers are moved to their own function from here.
        self.agents.do("logger")

        print(f"Epoch {self.epoch} completed.")
        self.epoch += 1


    def testResults(self):
        agent = self.agents[0]

        aT = agent.accuracyTest()
        aucT = agent.aucTest()
        f1s = agent.f1scoreTest()[0].mean()
    
        print(f"Epoch {self.epoch} Accuracy: {aT:.4f}, AUC: {aucT:.4f}, f1Score: {f1s:.4f}")