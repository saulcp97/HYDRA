import os

import mesa
import numpy as np
import pandas as pd

from architecture.agents import nnAgent
import Config

from sklearn.metrics import roc_auc_score


class FederatedModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n=10, seed=None, config=Config):
        if seed is None:
            seed = config.SIMULATION_SEED
        super().__init__(seed=seed)
        self.config = config
        self.num_agents = n

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
        
        self.agents.shuffle_do("train_model")
        
        if self.epoch == 0:
            self.agents.do("calibrateNeihborhood")
        
        self.agents.shuffle_do("pass_weights")

        self.agents.do("mixing_Weights")

        #And now the changes are saved and the loggers are moved to their own function from here.
        self.agents.do("logger")

        self.saveDistances()
        print(f"Epoch {self.epoch} completed.")
        self.epoch += 1


    def saveDistances(self):
        # In the case of not being able to save the full weights of the network (SAVE_FULL_WEIGHTS = False),
        # save the pairwise distances between an agent and its neighbors so experiments can recover distance
        # information without serializing the full model.
        config = self.config

        if config.SAVE_FULL_WEIGHTS:
            return

        if not config.log_Experiment:
            return

        for agent in self.agents:
            file_path = os.path.join("outputs", config.experiment_name, agent.agent_name, "expedient.csv")
            if not os.path.exists(file_path):
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue

            if config.SIMULATION_MODE:
                my_weights = np.asarray(agent.nnModel, dtype=float)
            else:
                my_weights = np.concatenate(
                    [param.detach().cpu().numpy().ravel() for param in agent.nnModel.state_dict().values()]
                ).astype(float)

            distance_cols = []
            distances = []
            for neighbor in agent.neighbors:
                if config.SIMULATION_MODE:
                    neighbor_weights = np.asarray(neighbor.nnModel, dtype=float)
                else:
                    neighbor_weights = np.concatenate(
                        [param.detach().cpu().numpy().ravel() for param in neighbor.nnModel.state_dict().values()]
                    ).astype(float)

                dist = np.linalg.norm(my_weights - neighbor_weights)
                col_name = f"distance_{neighbor.agent_name}"
                distance_cols.append(col_name)
                distances.append(dist)

            for col in distance_cols:
                if col not in df.columns:
                    df[col] = np.nan

            if len(df):
                last_index = df.index[-1]
                df.loc[last_index, distance_cols] = distances
                df.to_csv(file_path, index=False)


    def rate_global_scores(self):
        accuracy = []
        auc = []
        f1score = []
        for agent in self.agents:
            accuracy.append(agent.accuracyTest())
            auc.append(agent.aucTest())
            f1score.append(agent.f1scoreTest())
        return accuracy, auc