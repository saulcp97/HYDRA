import numpy as np

#Code of result Metrics.
all_agents = []

#Intra-Coalition Distance: promedios del promedio de distancia euclidiana entre un agente y los miembros de su coalición.
def getIntraCoalitionDistance():
    agent_intra_coalition_distances = []
    for agent in all_agents:
        distances = []
        for coalitonNeighbor in agent.coalitionList:
            dist = np.linalg.norm(agent.weights - coalitonNeighbor.weights) #Euclidean distance
            distances.append(dist)
        agent_intra_coalition_distances.append(np.mean(distances) if distances else 0.0) #Mean of distances
    intraCoalitionDistance = np.mean(agent_intra_coalition_distances)
    return intraCoalitionDistance

#Coalition Divergence (Energy Divergence for some reason):
for agent in all_agents:
    distances = []
    for coalitonNeighbor in agent.coalitionList:
        dist = np.linalg.norm(agent.weights - coalitonNeighbor.weights) #Euclidean distance
        distances.append(dist)
    agent_intra_coalition_distances.append(np.mean(distances) if distances else 0.0) #Mean of distances
intraCoalitionDistance = np.mean(agent_intra_coalition_distances)




#Avg. Distance Between Convergence Points:

#Avg. Training Error:

#Reciprocit