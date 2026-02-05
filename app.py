#App.py serves the porpuse of starting the mesaHYDRA application and setting up the web-app interface.


# federated model, the visualization app doesnt import the agent's code.

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
    g = nx.read_graphml(Config.network_graph)
    
    pos = nx.spring_layout(g, seed=3113794652) 
    fig = Figure()
    ax = fig.subplots()
    labels = {str(agent.selfID): agent.agent_name for agent in model.agents}

    #node_sizes = [g.nodes[node]["size"] for node in g.nodes]
    #node_colors = [g.nodes[node]["size"] for node in g.nodes()]

    #Node Colors, depends on the actual coalition index of the agent.
    node_colors = []
    for node in g.nodes():
        agent_name = "scp_" + node
        agent = [ag for ag in model.agents if ag.agent_name == agent_name][0]
        if agent.coalitionIndex == 0:
            node_colors.append("blue")
        elif agent.coalitionIndex == 1:
            node_colors.append("red")
        else:
            node_colors.append("gray") #Default color for agents that are not in any coalition, shouldnt happen but just in case

    nx.draw(g, pos=pos, labels=labels, ax = ax, node_color=node_colors)
    
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

model = model.FederatedModel(n=10, seed=42)
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