#App.py serves the porpuse of starting the mesaHYDRA application and setting up the web-app interface.


# federated model, the visualization app doesnt import the agent's code.

from platform import node
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
import solara
from architecture import model
import networkx as nx

from mesa.mesa_logging import INFO, log_to_stderr
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter

log_to_stderr(INFO)

import Config
model_params = {
    "n": {
        "type": "SliderInt",
        "min": 2,
        "max": 100,
        "value": 10,
        "step": 1,
        "label": "Number of Agents"
    },
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed"
    }
}

@solara.component
def plot_network(model):
    #TODO plot the network of agents
    update_counter.get()
    #g = model.network

    g = Config.H
    
    pos = nx.spring_layout(g, seed=3113794652) 
    fig = Figure()
    ax = fig.subplots()
    labels = {agent.selfID: agent.agent_name for agent in model.agents}

    #we convert the undirected graph to a directed one, so we can have different colors for the edges in each direction, to represent the communication between agents.
    g = g.to_directed()
    coallition_edges = []
    #We check the coalition list of each agent and color the edges accordingly.
    for node in g.nodes():
        agent_name = "scp_" + str(node)
        agent = [agent for agent in model.agents if agent.agent_name == agent_name][0]
        coalition = agent.coallitionNeighbors
        print(coalition)
        print(f"Agent {agent_name} is in coalition with {[name for name, _ in coalition]}")
        for neighbor_name, neighbor_agent in coalition:
            coallition_edges.append((node, neighbor_name))
    #print(labels)
    print(pos)
    edge_colors = ['red' if (u, v) in coallition_edges else 'gray' for u, v in g.edges()]       
    nx.draw(g, pos=pos, labels=labels, ax = ax, node_color="gray", arrows=True, with_labels=True, edge_color=edge_colors)
    
    """
    nx.draw(
        g,
        pos,
        #node_size=node_sizes,
        #node_color=node_colors,
        cmap=plt.cm.coolwarm,
        labels=labels,
        ax=ax,
    )

    solara.FigureMatplotlib(fig)
    """

    solara.FigureMatplotlib(fig)


@solara.component
def loss_horizon_evolution(model):
    #Try the plot multiContourPlot in a dynamic way.
    pass

model = model.FederatedModel(n=Config.NUMBER_OF_AGENTS, seed=42)
page = SolaraViz(
    model,
    components=[plot_network, loss_horizon_evolution],
    model_params=model_params,
    name="Dinamic coallitions visualization",
)
page  # noqa


"""
if __name__ == "__main__":
    print("Run the Solara visualization server for the web-UI.")

    #Simple test run of the model, based on the code on mesaRun.py
    federated_model = model.FederatedModel(n=10, seed=42)
    for i in range(10):
        federated_model.step()
    print("Model run complete.")
"""