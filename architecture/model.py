import mesa

from architecture.agents import nnAgent
import Config


class FederatedModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n=10, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        # Create agents
        nnAgent.create_agents(model=self, n=n)

        self.network = Config.H

        self.neighbors = {}
        for teAgent in Config.NEIGHBOURS.keys():
            #teAgent: teory or graph ag
            self.neighbors[teAgent] = []
            for teoAgent in Config.NEIGHBOURS[teAgent]:
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
        if self.epoch == 0:
            self.agents.do("calibrateNeihborhood")
        else:
            self.agents.do("mixing_Weights")
        self.agents.shuffle_do("train_model")
        self.agents.shuffle_do("pass_weights")

        print(f"Epoch {self.epoch} completed.")
        self.epoch += 1